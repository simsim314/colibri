#include "gguf_reader.h"
#include "compat.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef COMPAT_O_RDONLY
#define COMPAT_O_RDONLY O_RDONLY
#endif

#define COLI_GGUF_MAX_NAME_BYTES (1024u * 1024u)
#define COLI_GGUF_MAX_TABLE_ITEMS (10000000ull)

typedef struct {
    ColiGgufFile *g;
    uint64_t pos;
} GgufCursor;

static char *gguf_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static int gguf_fail(ColiGgufFile *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g->error, sizeof(g->error), fmt, ap);
    va_end(ap);
    return 0;
}

static int add_overflow_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (UINT64_MAX - a < b) return 1;
    *out = a + b;
    return 0;
}

static int mul_overflow_size(uint64_t a, size_t b, size_t *out) {
    if (a > (uint64_t)(SIZE_MAX / b)) return 1;
    *out = (size_t)a * b;
    return 0;
}

static uint32_t load_u32_le(const unsigned char p[4]) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t load_u64_le(const unsigned char p[8]) {
    return (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

int coli_gguf_read_at(const ColiGgufFile *g, uint64_t offset, void *dst, size_t bytes) {
    uint64_t end;
    if (!g || g->fd < 0 || (!dst && bytes)) return 0;
    if (add_overflow_u64(offset, (uint64_t)bytes, &end) || end > g->file_size) return 0;

    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(g->fd, (unsigned char *)dst + done, bytes - done,
                          (off_t)(offset + (uint64_t)done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static int cursor_read(GgufCursor *c, void *dst, size_t bytes) {
    if (!coli_gguf_read_at(c->g, c->pos, dst, bytes)) {
        return gguf_fail(c->g, "truncated GGUF at byte %llu while reading %zu bytes",
                         (unsigned long long)c->pos, bytes);
    }
    c->pos += (uint64_t)bytes;
    return 1;
}

static int cursor_skip(GgufCursor *c, uint64_t bytes) {
    uint64_t end;
    if (add_overflow_u64(c->pos, bytes, &end) || end > c->g->file_size) {
        return gguf_fail(c->g, "truncated GGUF at byte %llu while skipping %llu bytes",
                         (unsigned long long)c->pos, (unsigned long long)bytes);
    }
    c->pos = end;
    return 1;
}

static int cursor_u8(GgufCursor *c, uint8_t *out) {
    return cursor_read(c, out, 1);
}

static int cursor_u16(GgufCursor *c, uint16_t *out) {
    unsigned char p[2];
    if (!cursor_read(c, p, sizeof(p))) return 0;
    *out = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return 1;
}

static int cursor_u32(GgufCursor *c, uint32_t *out) {
    unsigned char p[4];
    if (!cursor_read(c, p, sizeof(p))) return 0;
    *out = load_u32_le(p);
    return 1;
}

static int cursor_u64(GgufCursor *c, uint64_t *out) {
    unsigned char p[8];
    if (!cursor_read(c, p, sizeof(p))) return 0;
    *out = load_u64_le(p);
    return 1;
}

static int cursor_string_alloc(GgufCursor *c, size_t max_bytes, char **out) {
    uint64_t n64;
    if (!cursor_u64(c, &n64)) return 0;
    if (n64 > max_bytes || n64 > SIZE_MAX - 1) {
        return gguf_fail(c->g, "GGUF string is too large: %llu bytes",
                         (unsigned long long)n64);
    }
    char *s = (char *)malloc((size_t)n64 + 1);
    if (!s) return gguf_fail(c->g, "out of memory allocating GGUF string");
    if (!cursor_read(c, s, (size_t)n64)) {
        free(s);
        return 0;
    }
    s[n64] = '\0';
    *out = s;
    return 1;
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
        case COLI_GGUF_TYPE_UINT8:
        case COLI_GGUF_TYPE_INT8:
        case COLI_GGUF_TYPE_BOOL: return 1;
        case COLI_GGUF_TYPE_UINT16:
        case COLI_GGUF_TYPE_INT16: return 2;
        case COLI_GGUF_TYPE_UINT32:
        case COLI_GGUF_TYPE_INT32:
        case COLI_GGUF_TYPE_FLOAT32: return 4;
        case COLI_GGUF_TYPE_UINT64:
        case COLI_GGUF_TYPE_INT64:
        case COLI_GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static int cursor_skip_string(GgufCursor *c) {
    uint64_t n;
    if (!cursor_u64(c, &n)) return 0;
    return cursor_skip(c, n);
}

static int cursor_skip_value(GgufCursor *c, uint32_t type, uint32_t *array_type,
                             uint64_t *array_count, uint64_t *value_offset,
                             uint64_t *value_size) {
    if (type > COLI_GGUF_TYPE_FLOAT64) {
        return gguf_fail(c->g, "unsupported GGUF metadata type %u", type);
    }

    if (type == COLI_GGUF_TYPE_ARRAY) {
        uint32_t elem_type;
        uint64_t count;
        if (!cursor_u32(c, &elem_type) || !cursor_u64(c, &count)) return 0;
        if (elem_type > COLI_GGUF_TYPE_FLOAT64 || elem_type == COLI_GGUF_TYPE_ARRAY) {
            return gguf_fail(c->g, "unsupported GGUF array element type %u", elem_type);
        }
        *array_type = elem_type;
        *array_count = count;
        *value_offset = c->pos;

        if (elem_type == COLI_GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < count; ++i) {
                if (!cursor_skip_string(c)) return 0;
            }
        } else {
            size_t sz = scalar_size(elem_type);
            if (!sz || count > UINT64_MAX / sz) {
                return gguf_fail(c->g, "invalid GGUF array size");
            }
            if (!cursor_skip(c, count * (uint64_t)sz)) return 0;
        }
    } else {
        *array_type = UINT32_MAX;
        *array_count = 0;
        *value_offset = c->pos;
        if (type == COLI_GGUF_TYPE_STRING) {
            if (!cursor_skip_string(c)) return 0;
        } else {
            size_t sz = scalar_size(type);
            if (!sz || !cursor_skip(c, (uint64_t)sz)) return 0;
        }
    }

    *value_size = c->pos - *value_offset;
    return 1;
}

static int read_raw(const ColiGgufFile *g, uint64_t offset, void *dst, size_t bytes) {
    return coli_gguf_read_at(g, offset, dst, bytes);
}

static int read_u16_at(const ColiGgufFile *g, uint64_t off, uint16_t *out) {
    unsigned char p[2];
    if (!read_raw(g, off, p, sizeof(p))) return 0;
    *out = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return 1;
}

static int read_u32_at(const ColiGgufFile *g, uint64_t off, uint32_t *out) {
    unsigned char p[4];
    if (!read_raw(g, off, p, sizeof(p))) return 0;
    *out = load_u32_le(p);
    return 1;
}

static int read_u64_at(const ColiGgufFile *g, uint64_t off, uint64_t *out) {
    unsigned char p[8];
    if (!read_raw(g, off, p, sizeof(p))) return 0;
    *out = load_u64_le(p);
    return 1;
}

const ColiGgufKV *coli_gguf_find_kv(const ColiGgufFile *g, const char *key) {
    if (!g || !key) return NULL;
    for (uint64_t i = 0; i < g->metadata_count; ++i) {
        if (strcmp(g->metadata[i].key, key) == 0) return &g->metadata[i];
    }
    return NULL;
}

const ColiGgufTensorInfo *coli_gguf_find_tensor(const ColiGgufFile *g, const char *name) {
    if (!g || !name) return NULL;
    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

int coli_gguf_kv_read_u64(const ColiGgufFile *g, const ColiGgufKV *kv, uint64_t *out) {
    if (!g || !kv || !out || kv->type == COLI_GGUF_TYPE_ARRAY) return 0;
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    switch (kv->type) {
        case COLI_GGUF_TYPE_UINT8:
        case COLI_GGUF_TYPE_BOOL:
            if (!read_raw(g, kv->value_offset, &u8, 1)) return 0;
            *out = u8;
            return 1;
        case COLI_GGUF_TYPE_UINT16:
            if (!read_u16_at(g, kv->value_offset, &u16)) return 0;
            *out = u16;
            return 1;
        case COLI_GGUF_TYPE_UINT32:
            if (!read_u32_at(g, kv->value_offset, &u32)) return 0;
            *out = u32;
            return 1;
        case COLI_GGUF_TYPE_UINT64:
            if (!read_u64_at(g, kv->value_offset, &u64)) return 0;
            *out = u64;
            return 1;
        default:
            return 0;
    }
}

int coli_gguf_kv_read_i64(const ColiGgufFile *g, const ColiGgufKV *kv, int64_t *out) {
    if (!g || !kv || !out || kv->type == COLI_GGUF_TYPE_ARRAY) return 0;
    uint8_t p8;
    uint16_t p16;
    uint32_t p32;
    uint64_t p64;
    switch (kv->type) {
        case COLI_GGUF_TYPE_INT8:
            if (!read_raw(g, kv->value_offset, &p8, 1)) return 0;
            *out = (int8_t)p8;
            return 1;
        case COLI_GGUF_TYPE_INT16:
            if (!read_u16_at(g, kv->value_offset, &p16)) return 0;
            *out = (int16_t)p16;
            return 1;
        case COLI_GGUF_TYPE_INT32:
            if (!read_u32_at(g, kv->value_offset, &p32)) return 0;
            *out = (int32_t)p32;
            return 1;
        case COLI_GGUF_TYPE_INT64:
            if (!read_u64_at(g, kv->value_offset, &p64)) return 0;
            *out = (int64_t)p64;
            return 1;
        default:
            return 0;
    }
}

int coli_gguf_kv_read_f64(const ColiGgufFile *g, const ColiGgufKV *kv, double *out) {
    if (!g || !kv || !out || kv->type == COLI_GGUF_TYPE_ARRAY) return 0;
    if (kv->type == COLI_GGUF_TYPE_FLOAT32) {
        uint32_t bits;
        float value;
        if (!read_u32_at(g, kv->value_offset, &bits)) return 0;
        memcpy(&value, &bits, sizeof(value));
        *out = value;
        return 1;
    }
    if (kv->type == COLI_GGUF_TYPE_FLOAT64) {
        uint64_t bits;
        double value;
        if (!read_u64_at(g, kv->value_offset, &bits)) return 0;
        memcpy(&value, &bits, sizeof(value));
        *out = value;
        return 1;
    }
    return 0;
}

int coli_gguf_kv_read_bool(const ColiGgufFile *g, const ColiGgufKV *kv, int *out) {
    uint8_t value;
    if (!g || !kv || !out || kv->type != COLI_GGUF_TYPE_BOOL) return 0;
    if (!read_raw(g, kv->value_offset, &value, 1)) return 0;
    *out = value != 0;
    return 1;
}

int coli_gguf_kv_read_string(const ColiGgufFile *g, const ColiGgufKV *kv, char **out) {
    if (!g || !kv || !out || kv->type != COLI_GGUF_TYPE_STRING) return 0;
    uint64_t n;
    if (!read_u64_at(g, kv->value_offset, &n) || n > SIZE_MAX - 1) return 0;
    uint64_t start;
    if (add_overflow_u64(kv->value_offset, 8, &start)) return 0;
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) return 0;
    if (!read_raw(g, start, s, (size_t)n)) {
        free(s);
        return 0;
    }
    s[n] = '\0';
    *out = s;
    return 1;
}

int coli_gguf_read_tensor_bytes(const ColiGgufFile *g,
                                const ColiGgufTensorInfo *tensor,
                                uint64_t relative_offset,
                                void *dst,
                                size_t bytes) {
    uint64_t off;
    if (!g || !tensor) return 0;
    if (add_overflow_u64(tensor->absolute_offset, relative_offset, &off)) return 0;
    return coli_gguf_read_at(g, off, dst, bytes);
}

static int is_power_of_two_u32(uint32_t x) {
    return x && !(x & (x - 1));
}

static int align_up_u64(uint64_t x, uint32_t alignment, uint64_t *out) {
    uint64_t mask = (uint64_t)alignment - 1;
    if (UINT64_MAX - x < mask) return 0;
    *out = (x + mask) & ~mask;
    return 1;
}

int coli_gguf_open(ColiGgufFile *g, const char *path) {
    if (!g || !path) return 0;
    memset(g, 0, sizeof(*g));
    g->fd = -1;
    g->alignment = COLI_GGUF_DEFAULT_ALIGNMENT;

    g->path = gguf_strdup(path);
    if (!g->path) return gguf_fail(g, "out of memory copying path");

    g->fd = open(path, COMPAT_O_RDONLY);
    if (g->fd < 0) {
        gguf_fail(g, "cannot open '%s': %s", path, strerror(errno));
        goto fail;
    }

    struct stat st;
    if (fstat(g->fd, &st) != 0 || st.st_size < 0) {
        gguf_fail(g, "cannot stat '%s': %s", path, strerror(errno));
        goto fail;
    }
    g->file_size = (uint64_t)st.st_size;

    GgufCursor c = { g, 0 };
    unsigned char magic[4];
    if (!cursor_read(&c, magic, sizeof(magic))) goto fail;
    if (memcmp(magic, "GGUF", 4) != 0) {
        gguf_fail(g, "not a GGUF file: bad magic");
        goto fail;
    }
    if (!cursor_u32(&c, &g->version)) goto fail;
    if (g->version != 2 && g->version != 3) {
        gguf_fail(g, "unsupported GGUF version %u (supported: 2 and 3)", g->version);
        goto fail;
    }
    if (!cursor_u64(&c, &g->tensor_count) || !cursor_u64(&c, &g->metadata_count)) goto fail;
    if (g->tensor_count > COLI_GGUF_MAX_TABLE_ITEMS ||
        g->metadata_count > COLI_GGUF_MAX_TABLE_ITEMS) {
        gguf_fail(g, "unreasonable GGUF table size");
        goto fail;
    }

    size_t metadata_bytes = 0;
    size_t tensor_bytes = 0;
    if (g->metadata_count &&
        mul_overflow_size(g->metadata_count, sizeof(*g->metadata), &metadata_bytes)) {
        gguf_fail(g, "GGUF metadata allocation overflow");
        goto fail;
    }
    if (g->tensor_count &&
        mul_overflow_size(g->tensor_count, sizeof(*g->tensors), &tensor_bytes)) {
        gguf_fail(g, "GGUF tensor allocation overflow");
        goto fail;
    }
    g->metadata = metadata_bytes ? (ColiGgufKV *)calloc(1, metadata_bytes) : NULL;
    g->tensors = tensor_bytes ? (ColiGgufTensorInfo *)calloc(1, tensor_bytes) : NULL;
    if ((metadata_bytes && !g->metadata) || (tensor_bytes && !g->tensors)) {
        gguf_fail(g, "out of memory allocating GGUF tables");
        goto fail;
    }

    for (uint64_t i = 0; i < g->metadata_count; ++i) {
        ColiGgufKV *kv = &g->metadata[i];
        if (!cursor_string_alloc(&c, COLI_GGUF_MAX_NAME_BYTES, &kv->key)) goto fail;
        if (!cursor_u32(&c, &kv->type)) goto fail;
        if (!cursor_skip_value(&c, kv->type, &kv->array_type, &kv->array_count,
                               &kv->value_offset, &kv->value_size)) goto fail;
    }

    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        ColiGgufTensorInfo *t = &g->tensors[i];
        if (!cursor_string_alloc(&c, COLI_GGUF_MAX_NAME_BYTES, &t->name)) goto fail;
        if (!cursor_u32(&c, &t->n_dims)) goto fail;
        if (t->n_dims == 0 || t->n_dims > COLI_GGUF_MAX_DIMS) {
            gguf_fail(g, "tensor '%s' has unsupported dimension count %u", t->name, t->n_dims);
            goto fail;
        }
        for (uint32_t d = 0; d < t->n_dims; ++d) {
            if (!cursor_u64(&c, &t->dims[d])) goto fail;
            if (t->dims[d] == 0) {
                gguf_fail(g, "tensor '%s' has a zero dimension", t->name);
                goto fail;
            }
        }
        if (!cursor_u32(&c, &t->type) || !cursor_u64(&c, &t->offset)) goto fail;
    }

    const ColiGgufKV *alignment_kv = coli_gguf_find_kv(g, "general.alignment");
    if (alignment_kv) {
        uint64_t alignment;
        if (!coli_gguf_kv_read_u64(g, alignment_kv, &alignment) ||
            alignment > UINT32_MAX || !is_power_of_two_u32((uint32_t)alignment)) {
            gguf_fail(g, "invalid general.alignment metadata");
            goto fail;
        }
        g->alignment = (uint32_t)alignment;
    }

    if (g->tensor_count) {
        if (!align_up_u64(c.pos, g->alignment, &g->data_offset) ||
            g->data_offset > g->file_size) {
            gguf_fail(g, "invalid GGUF tensor-data offset");
            goto fail;
        }
    } else {
        g->data_offset = c.pos;
    }

    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        ColiGgufTensorInfo *t = &g->tensors[i];
        if (t->offset % g->alignment != 0) {
            gguf_fail(g, "tensor '%s' has unaligned data offset %llu", t->name,
                      (unsigned long long)t->offset);
            goto fail;
        }
        if (add_overflow_u64(g->data_offset, t->offset, &t->absolute_offset) ||
            t->absolute_offset > g->file_size) {
            gguf_fail(g, "tensor '%s' points outside the file", t->name);
            goto fail;
        }
    }

    g->error[0] = '\0';
    return 1;

fail:
    coli_gguf_close(g);
    return 0;
}

void coli_gguf_close(ColiGgufFile *g) {
    if (!g) return;
    char saved_error[sizeof(g->error)];
    memcpy(saved_error, g->error, sizeof(saved_error));

    if (g->metadata) {
        for (uint64_t i = 0; i < g->metadata_count; ++i) free(g->metadata[i].key);
    }
    if (g->tensors) {
        for (uint64_t i = 0; i < g->tensor_count; ++i) free(g->tensors[i].name);
    }
    free(g->metadata);
    free(g->tensors);
    free(g->path);
    if (g->fd >= 0) close(g->fd);

    memset(g, 0, sizeof(*g));
    g->fd = -1;
    memcpy(g->error, saved_error, sizeof(g->error));
}

const char *coli_gguf_error(const ColiGgufFile *g) {
    return g && g->error[0] ? g->error : "unknown GGUF error";
}

const char *coli_gguf_value_type_name(uint32_t type) {
    static const char *names[] = {
        "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32",
        "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64"
    };
    return type < sizeof(names) / sizeof(names[0]) ? names[type] : "UNKNOWN";
}

const char *coli_ggml_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 4: return "REMOVED_Q4_2";
        case 5: return "REMOVED_Q4_3";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 31: return "REMOVED_Q4_0_4_4";
        case 32: return "REMOVED_Q4_0_4_8";
        case 33: return "REMOVED_Q4_0_8_8";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        case 36: return "REMOVED_IQ4_NL_4_4";
        case 37: return "REMOVED_IQ4_NL_4_8";
        case 38: return "REMOVED_IQ4_NL_8_8";
        case 39: return "MXFP4";
        case 40: return "NVFP4";
        case 41: return "Q1_0";
        case 42: return "Q2_0";
        default: return "UNKNOWN";
    }
}
