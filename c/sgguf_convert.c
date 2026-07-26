#include "gguf_reader.h"
#include "ggml_quants.h"
#include "ggml_types.h"
#include "sgguf.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#endif

#ifdef _WIN32
#include <io.h>
#define FSEEK64 _fseeki64
#define FTELL64 _ftelli64
#else
#include <sys/types.h>
#include <unistd.h>
#define FSEEK64 fseeko
#define FTELL64 ftello
#endif

typedef struct {
    uint32_t storage_kind;
    uint32_t codec_id;
    int32_t moe_layer;
    uint32_t moe_projection;
    uint32_t expert_count;
    uint32_t rows_per_expert;
    uint64_t offset_patch;
    uint64_t payload_patch;
    uint64_t index_offset_patch;
    uint64_t index_size_patch;
} TensorPlan;

typedef struct {
    float threshold;
    uint32_t forced_codec;
    int verify;
    int verbose;
    uint32_t jobs;
    uint64_t max_output_bytes;
} ConvertOptions;

typedef struct {
    uint64_t retained_total;
    uint64_t tree_total;
    uint64_t value_total;
    uint64_t aux_total;
} SparseStats;

typedef struct {
    uint32_t first_expert;
    uint32_t end_expert;
    uint64_t local_bytes;
    SparseStats stats;
    uint64_t failed_block;
    int ok;
} SparseWorkerResult;

typedef struct {
    volatile uint32_t experts_done;
} SparseProgress;

static int failf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("sgguf-convert: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    return 0;
}

static uint64_t tell64(FILE *f) {
    long long p = (long long)FTELL64(f);
    return p < 0 ? UINT64_MAX : (uint64_t)p;
}

static int seek64(FILE *f, uint64_t off) {
    return off <= INT64_MAX && FSEEK64(f, (long long)off, SEEK_SET) == 0;
}

static int put_bytes(FILE *f, const void *p, size_t n) {
    return n == 0 || fwrite(p, 1, n, f) == n;
}
static int put_u16(FILE *f, uint16_t v) {
    uint8_t p[2]; coli_sgguf_store_u16_le(p, v); return put_bytes(f, p, sizeof(p));
}
static int put_u32(FILE *f, uint32_t v) {
    uint8_t p[4]; coli_sgguf_store_u32_le(p, v); return put_bytes(f, p, sizeof(p));
}
static int put_u64(FILE *f, uint64_t v) {
    uint8_t p[8]; coli_sgguf_store_u64_le(p, v); return put_bytes(f, p, sizeof(p));
}
static int put_f32(FILE *f, float v) {
    uint32_t u; memcpy(&u, &v, sizeof(u)); return put_u32(f, u);
}
static int put_string(FILE *f, const char *s) {
    size_t n = strlen(s);
    return put_u64(f, (uint64_t)n) && put_bytes(f, s, n);
}

static int write_zeros(FILE *f, uint64_t n) {
    static const uint8_t zeros[65536] = {0};
    while (n) {
        size_t chunk = n > sizeof(zeros) ? sizeof(zeros) : (size_t)n;
        if (!put_bytes(f, zeros, chunk)) return 0;
        n -= chunk;
    }
    return 1;
}

static int pad_to(FILE *f, uint32_t alignment) {
    uint64_t pos = tell64(f);
    if (pos == UINT64_MAX || !alignment || (alignment & (alignment - 1u))) return 0;
    uint64_t aligned = (pos + alignment - 1u) & ~((uint64_t)alignment - 1u);
    return write_zeros(f, aligned - pos);
}

static int patch_u32(FILE *f, uint64_t at, uint32_t v) {
    uint64_t end = tell64(f);
    return end != UINT64_MAX && seek64(f, at) && put_u32(f, v) && seek64(f, end);
}
static int patch_u64(FILE *f, uint64_t at, uint64_t v) {
    uint64_t end = tell64(f);
    return end != UINT64_MAX && seek64(f, at) && put_u64(f, v) && seek64(f, end);
}

static int copy_mapped(FILE *out, const ColiGgufFile *src, uint64_t off, uint64_t bytes) {
    const uint8_t *p = (const uint8_t *)coli_gguf_mapped_at(src, off, bytes);
    if (!p) return 0;
    while (bytes) {
        size_t chunk = bytes > (8u << 20) ? (8u << 20) : (size_t)bytes;
        if (!put_bytes(out, p, chunk)) return 0;
        p += chunk;
        bytes -= chunk;
    }
    return 1;
}

static int write_metadata_copy(FILE *out, const ColiGgufFile *src, const ColiGgufKV *kv) {
    if (!put_string(out, kv->key) || !put_u32(out, kv->type)) return 0;
    if (kv->type == COLI_GGUF_TYPE_ARRAY) {
        if (!put_u32(out, kv->array_type) || !put_u64(out, kv->array_count)) return 0;
    }
    return copy_mapped(out, src, kv->value_offset, kv->value_size);
}

static int parse_moe_name(const char *name, int32_t *layer, uint32_t *projection) {
    int l = -1, n = 0;
    if (sscanf(name, "blk.%d.ffn_gate_exps.weight%n", &l, &n) == 1 && name[n] == '\0') {
        *layer = l; *projection = COLI_SGGUF_MOE_GATE; return 1;
    }
    n = 0;
    if (sscanf(name, "blk.%d.ffn_up_exps.weight%n", &l, &n) == 1 && name[n] == '\0') {
        *layer = l; *projection = COLI_SGGUF_MOE_UP; return 1;
    }
    n = 0;
    if (sscanf(name, "blk.%d.ffn_down_exps.weight%n", &l, &n) == 1 && name[n] == '\0') {
        *layer = l; *projection = COLI_SGGUF_MOE_DOWN; return 1;
    }
    return 0;
}

static uint32_t choose_codec(uint32_t type, uint32_t forced_codec) {
    if (!coli_dtype_traits((ColiDType)type)) return COLI_SGGUF_CODEC_NONE;
    if (forced_codec) return forced_codec;
    switch ((ColiDType)type) {
        case COLI_DTYPE_F32:  return COLI_SGGUF_CODEC_RETAINED_F32;
        case COLI_DTYPE_F16:  return COLI_SGGUF_CODEC_RETAINED_F16;
        case COLI_DTYPE_BF16: return COLI_SGGUF_CODEC_RETAINED_BF16;
        case COLI_DTYPE_Q4_0: return COLI_SGGUF_CODEC_Q4_0_EXACT;
        case COLI_DTYPE_Q4_1: return COLI_SGGUF_CODEC_Q4_1_EXACT;
        case COLI_DTYPE_Q5_0: return COLI_SGGUF_CODEC_Q5_0_EXACT;
        case COLI_DTYPE_Q5_1: return COLI_SGGUF_CODEC_Q5_1_EXACT;
        case COLI_DTYPE_Q8_0: return COLI_SGGUF_CODEC_Q8_0_EXACT;
        case COLI_DTYPE_Q8_1: return COLI_SGGUF_CODEC_Q8_1_EXACT;
        case COLI_DTYPE_Q3_K: return COLI_SGGUF_CODEC_Q3_K_EXACT;
        case COLI_DTYPE_Q4_K: return COLI_SGGUF_CODEC_Q4_K_EXACT;
        case COLI_DTYPE_Q5_K: return COLI_SGGUF_CODEC_Q5_K_EXACT;
        case COLI_DTYPE_Q6_K: return COLI_SGGUF_CODEC_Q6_K_EXACT;
        case COLI_DTYPE_Q8_K: return COLI_SGGUF_CODEC_Q8_K_EXACT;
        case COLI_DTYPE_MXFP4: return COLI_SGGUF_CODEC_MXFP4_EXACT;
        case COLI_DTYPE_IQ4_XS: return COLI_SGGUF_CODEC_IQ4_XS_EXACT;
        default: return COLI_SGGUF_CODEC_RETAINED_F16;
    }
}

static uint16_t fp32_to_fp16(float value) {
    uint32_t x; memcpy(&x, &value, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mantissa = x & 0x7fffffu;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    if (((x >> 23) & 0xffu) == 0xffu) {
        if (mantissa) return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t rounded = (mantissa + (1u << (shift - 1u)) - 1u + ((mantissa >> shift) & 1u)) >> shift;
        return (uint16_t)(sign | rounded);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    mantissa += 0xfffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) {
        mantissa = 0;
        if (++exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mantissa >> 13));
}

static uint16_t fp32_to_bf16(float value) {
    uint32_t u;
    memcpy(&u, &value, sizeof(u));
    /* Round to nearest, ties to even. */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static uint32_t load_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint8_t q4_0_or_q4_1_code(const uint8_t *p, uint32_t idx, uint32_t header) {
    uint32_t native = idx >> 5;
    uint32_t local = idx & 31u;
    const uint8_t *q = p + native * (header + 16u) + header;
    uint8_t v = q[local & 15u];
    return local < 16u ? (v & 15u) : (v >> 4);
}

static uint8_t q5_code(const uint8_t *p, uint32_t idx, int has_min) {
    uint32_t native = idx >> 5;
    uint32_t local = idx & 31u;
    uint32_t block_bytes = has_min ? 24u : 22u;
    uint32_t qh_off = has_min ? 4u : 2u;
    uint32_t qs_off = has_min ? 8u : 6u;
    const uint8_t *block = p + native * block_bytes;
    uint32_t qh = load_u32_le(block + qh_off);
    const uint8_t *q = block + qs_off;
    uint8_t low = local < 16u ? (q[local] & 15u) : (q[local - 16u] >> 4);
    uint8_t high = local < 16u
        ? (uint8_t)(((qh >> local) << 4) & 0x10u)
        : (uint8_t)((qh >> (local - 4u)) & 0x10u);
    return (uint8_t)(low | high);
}

static uint8_t q3_k_code(const uint8_t *p, uint32_t idx) {
    const uint8_t *hmask = p;
    const uint8_t *q = p + 32;
    uint32_t half = idx >> 7;
    uint32_t r = idx & 127u;
    uint32_t group = r >> 5;
    uint32_t local = r & 31u;
    uint8_t low = (uint8_t)((q[half * 32u + local] >> (group * 2u)) & 3u);
    uint8_t mask = (uint8_t)(1u << (half * 4u + group));
    return (uint8_t)(low | ((hmask[local] & mask) ? 4u : 0u));
}

static uint8_t q5_k_code(const uint8_t *p, uint32_t idx) {
    const uint8_t *qh = p + 16;
    const uint8_t *ql = p + 48;
    uint32_t chunk = idx >> 6;
    uint32_t local = idx & 63u;
    uint8_t low = local < 32u
        ? (ql[chunk * 32u + local] & 15u)
        : (ql[chunk * 32u + local - 32u] >> 4);
    uint8_t bit = (uint8_t)((local < 32u ? 1u : 2u) << (2u * chunk));
    uint8_t high = (qh[local & 31u] & bit) ? 16u : 0u;
    return (uint8_t)(low | high);
}

static uint8_t q6_k_code(const uint8_t *p, uint32_t idx) {
    uint32_t half = idx >> 7;
    uint32_t r = idx & 127u;
    uint32_t group = r >> 5;
    uint32_t l = r & 31u;
    const uint8_t *ql = p + half * 64u;
    const uint8_t *qh = p + 128u + half * 32u;
    uint8_t low, high;
    if (group == 0) { low = ql[l] & 15u; high = (qh[l] >> 0) & 3u; }
    else if (group == 1) { low = ql[l + 32] & 15u; high = (qh[l] >> 2) & 3u; }
    else if (group == 2) { low = ql[l] >> 4; high = (qh[l] >> 4) & 3u; }
    else { low = ql[l + 32] >> 4; high = (qh[l] >> 6) & 3u; }
    return (uint8_t)(low | (high << 4));
}

static uint8_t q4_k_code(const uint8_t *p, uint32_t idx) {
    const uint8_t *q = p + 16;
    uint32_t chunk = idx >> 6;
    uint32_t local = idx & 63u;
    uint8_t byte = q[chunk * 32u + (local & 31u)];
    return local < 32 ? (byte & 15u) : (byte >> 4);
}

static void pack_bits(uint8_t *dst, uint32_t bit_offset, uint32_t value, uint32_t bits) {
    for (uint32_t i = 0; i < bits; ++i) {
        if ((value >> i) & 1u) dst[(bit_offset + i) >> 3] |= (uint8_t)(1u << ((bit_offset + i) & 7u));
    }
}

static int write_sparse_block(FILE *out, uint32_t source_type, uint32_t codec, const uint8_t *encoded,
                              const float decoded[256], uint32_t logical_count, float threshold,
                              uint64_t *retained_total, uint64_t *bitmap_bytes_total,
                              uint64_t *value_bytes_total, uint64_t *aux_bytes_total) {
    if (!logical_count || logical_count > COLI_SGGUF_GROUP_SIZE) return 0;
    uint8_t keep[256] = {0};
    uint8_t bitmap[COLI_SGGUF_BITMAP_BYTES];
    uint32_t retained = 0;
    for (uint32_t i = 0; i < logical_count; ++i) {
        /* Values strictly below threshold are omitted. NaN/Inf are retained. */
        keep[i] = !(fabsf(decoded[i]) < threshold);
        retained += keep[i] != 0;
    }
    coli_sgguf_bitmap_from_keep(keep, bitmap);

    uint8_t aux[32];
    uint16_t aux_bytes = 0, value_bits = 0;
    uint8_t values[1024];
    memset(values, 0, sizeof(values));
    uint32_t value_bytes = 0;

    if (codec == COLI_SGGUF_CODEC_Q4_0_EXACT) {
        aux_bytes = 16; value_bits = 4; memset(aux, 0, aux_bytes);
        for (uint32_t n = 0; n < (logical_count + 31u) / 32u; ++n) memcpy(aux + n * 2u, encoded + n * 18u, 2);
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 4u, q4_0_or_q4_1_code(encoded, i, 2), 4);
        value_bytes = (retained * 4u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q4_1_EXACT) {
        aux_bytes = 32; value_bits = 4; memset(aux, 0, aux_bytes);
        for (uint32_t n = 0; n < (logical_count + 31u) / 32u; ++n) memcpy(aux + n * 4u, encoded + n * 20u, 4);
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 4u, q4_0_or_q4_1_code(encoded, i, 4), 4);
        value_bytes = (retained * 4u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q5_0_EXACT) {
        aux_bytes = 16; value_bits = 5; memset(aux, 0, aux_bytes);
        for (uint32_t n = 0; n < (logical_count + 31u) / 32u; ++n) memcpy(aux + n * 2u, encoded + n * 22u, 2);
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 5u, q5_code(encoded, i, 0), 5);
        value_bytes = (retained * 5u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q5_1_EXACT) {
        aux_bytes = 32; value_bits = 5; memset(aux, 0, aux_bytes);
        for (uint32_t n = 0; n < (logical_count + 31u) / 32u; ++n) memcpy(aux + n * 4u, encoded + n * 24u, 4);
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 5u, q5_code(encoded, i, 1), 5);
        value_bytes = (retained * 5u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q8_0_EXACT) {
        aux_bytes = 16; value_bits = 8; memset(aux, 0, aux_bytes);
        for (uint32_t n = 0; n < (logical_count + 31u) / 32u; ++n) memcpy(aux + n * 2u, encoded + n * 34u, 2);
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            values[k++] = encoded[(i >> 5) * 34u + 2u + (i & 31u)];
        value_bytes = retained;
    } else if (codec == COLI_SGGUF_CODEC_Q8_1_EXACT) {
        aux_bytes = 32; value_bits = 8; memset(aux, 0, aux_bytes);
        for (uint32_t n = 0; n < (logical_count + 31u) / 32u; ++n) memcpy(aux + n * 4u, encoded + n * 36u, 4);
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            values[k++] = encoded[(i >> 5) * 36u + 4u + (i & 31u)];
        value_bytes = retained;
    } else if (codec == COLI_SGGUF_CODEC_Q3_K_EXACT) {
        memcpy(aux, encoded + 96, 14); aux_bytes = 14; value_bits = 3;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 3u, q3_k_code(encoded, i), 3);
        value_bytes = (retained * 3u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q6_K_EXACT) {
        memcpy(aux, encoded + 192, 18); aux_bytes = 18; value_bits = 6;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 6u, q6_k_code(encoded, i), 6);
        value_bytes = (retained * 6u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q4_K_EXACT) {
        memcpy(aux, encoded, 16); aux_bytes = 16; value_bits = 4;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 4u, q4_k_code(encoded, i), 4);
        value_bytes = (retained * 4u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q5_K_EXACT) {
        memcpy(aux, encoded, 16); aux_bytes = 16; value_bits = 5;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            pack_bits(values, k++ * 5u, q5_k_code(encoded, i), 5);
        value_bytes = (retained * 5u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_Q8_K_EXACT) {
        memcpy(aux, encoded, 4); aux_bytes = 4; value_bits = 8;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i])
            values[k++] = encoded[4u + i];
        value_bytes = retained;
    } else if (codec == COLI_SGGUF_CODEC_MXFP4_EXACT) {
        aux_bytes = 8; value_bits = 4;
        memset(aux, 0, aux_bytes);
        uint32_t native_blocks = (logical_count + 31u) / 32u;
        for (uint32_t n = 0; n < native_blocks; ++n) aux[n] = encoded[n * 17u];
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i]) {
            uint32_t native = i >> 5;
            uint32_t local = i & 31u;
            uint8_t packed = encoded[native * 17u + 1u + (local & 15u)];
            uint8_t code = local < 16u ? (packed & 15u) : (packed >> 4);
            pack_bits(values, k++ * 4u, code, 4);
        }
        value_bytes = (retained * 4u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_IQ4_XS_EXACT) {
        /* One IQ4_XS source block is exactly one 256-value SGGUF group.
         * Preserve its FP16 super-scale and packed 6-bit subgroup scales in
         * the fixed auxiliary bytes, then retain the original nonlinear
         * 4-bit code for every occupied logical position. */
        if (logical_count != COLI_SGGUF_GROUP_SIZE) return 0;
        aux_bytes = 8; value_bits = 4;
        memcpy(aux, encoded, aux_bytes);
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i]) {
            uint32_t subgroup = i >> 5;
            uint32_t local = i & 31u;
            uint8_t packed = encoded[8u + subgroup * 16u + (local & 15u)];
            uint8_t code = local < 16u ? (packed & 15u) : (packed >> 4);
            pack_bits(values, k++ * 4u, code, 4);
        }
        value_bytes = (retained * 4u + 7u) / 8u;
    } else if (codec == COLI_SGGUF_CODEC_RETAINED_F16) {
        value_bits = 16;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i]) {
            uint16_t h;
            if (source_type == COLI_DTYPE_F16)
                h = coli_sgguf_load_u16_le(encoded + i * 2u);
            else
                h = fp32_to_fp16(decoded[i]);
            coli_sgguf_store_u16_le(values + k * 2u, h); ++k;
        }
        value_bytes = retained * 2u;
    } else if (codec == COLI_SGGUF_CODEC_RETAINED_BF16) {
        value_bits = 16;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i]) {
            uint16_t h;
            if (source_type == COLI_DTYPE_BF16)
                h = coli_sgguf_load_u16_le(encoded + i * 2u);
            else
                h = fp32_to_bf16(decoded[i]);
            coli_sgguf_store_u16_le(values + k * 2u, h); ++k;
        }
        value_bytes = retained * 2u;
    } else if (codec == COLI_SGGUF_CODEC_RETAINED_F32) {
        value_bits = 32;
        uint32_t k = 0;
        for (uint32_t i = 0; i < logical_count; ++i) if (keep[i]) {
            uint32_t u;
            if (source_type == COLI_DTYPE_F32)
                u = coli_sgguf_load_u32_le(encoded + i * 4u);
            else
                memcpy(&u, decoded + i, sizeof(u));
            coli_sgguf_store_u32_le(values + k * 4u, u); ++k;
        }
        value_bytes = retained * 4u;
    } else {
        return 0;
    }


    if (aux_bytes != coli_sgguf_codec_aux_bytes(codec) ||
        value_bits != coli_sgguf_codec_value_bits(codec)) return 0;
    if (!put_bytes(out, bitmap, sizeof(bitmap)) ||
        !put_bytes(out, aux, aux_bytes) ||
        !put_bytes(out, values, value_bytes)) return 0;

    if (retained_total) *retained_total += retained;
    if (bitmap_bytes_total) *bitmap_bytes_total += sizeof(bitmap);
    if (value_bytes_total) *value_bytes_total += value_bytes;
    if (aux_bytes_total) *aux_bytes_total += aux_bytes;
    return 1;
}

static double monotonic_seconds(void) {
#ifdef _WIN32
    return (double)clock() / (double)CLOCKS_PER_SEC;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static uint32_t online_cpu_count(void) {
#ifdef _WIN32
    return 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > 256) n = 256;
    return (uint32_t)n;
#endif
}

static int copy_stream(FILE *dst, FILE *src, uint64_t bytes) {
    uint8_t *buffer = (uint8_t *)malloc(8u << 20);
    if (!buffer) return 0;
    int ok = 1;
    while (bytes) {
        size_t chunk = bytes > (8u << 20) ? (8u << 20) : (size_t)bytes;
        if (fread(buffer, 1, chunk, src) != chunk || !put_bytes(dst, buffer, chunk)) {
            ok = 0;
            break;
        }
        bytes -= chunk;
    }
    free(buffer);
    return ok;
}

static FILE *open_worker_part(const char *tmp_prefix, uint64_t tensor_id, uint32_t worker) {
#ifdef _WIN32
    (void)tmp_prefix; (void)tensor_id; (void)worker;
    return tmpfile();
#else
    size_t n = strlen(tmp_prefix) + 96u;
    char *path = (char *)malloc(n);
    if (!path) return NULL;
    snprintf(path, n, "%s.mp.%llu.%u.XXXXXX", tmp_prefix,
             (unsigned long long)tensor_id, worker);
    int fd = mkstemp(path);
    if (fd < 0) {
        free(path);
        return NULL;
    }
    /* The file remains accessible through its descriptor but cannot be left
     * behind after interruption or a failed conversion. */
    unlink(path);
    free(path);
    FILE *f = fdopen(fd, "wb+");
    if (!f) {
        close(fd);
        return NULL;
    }
    return f;
#endif
}

static int encode_expert_range(FILE *part,
                               const ColiGgufTensorInfo *t,
                               const ColiDTypeTraits *traits,
                               const uint8_t *tensor_data,
                               uint64_t row_bytes,
                               uint32_t blocks_per_row,
                               uint32_t codec,
                               float threshold,
                               uint32_t first_expert,
                               uint32_t end_expert,
                               uint32_t *offsets,
                               SparseStats *stats,
                               SparseProgress *progress,
                               uint64_t *failed_block) {
    uint64_t part_start = tell64(part);
    if (part_start == UINT64_MAX) return 0;
    float decoded[256];

    for (uint32_t expert = first_expert; expert < end_expert; ++expert) {
        uint64_t first_row = (uint64_t)expert * t->dims[1];
        uint64_t last_row = first_row + t->dims[1];
        for (uint64_t row = first_row; row < last_row; ++row) {
            const uint8_t *row_data = tensor_data + row * row_bytes;
            for (uint32_t b = 0; b < blocks_per_row; ++b) {
                uint64_t block_index = row * blocks_per_row + b;
                uint64_t now = tell64(part);
                if (now == UINT64_MAX) {
                    if (failed_block) *failed_block = block_index;
                    return 0;
                }
                uint64_t relative = now - part_start;
                if (relative > UINT32_MAX) {
                    if (failed_block) *failed_block = block_index;
                    return 0;
                }
                offsets[block_index] = (uint32_t)relative;
                const uint32_t first_col = b * COLI_SGGUF_GROUP_SIZE;
                const uint32_t logical_count = first_col + COLI_SGGUF_GROUP_SIZE <= t->dims[0]
                    ? COLI_SGGUF_GROUP_SIZE : (uint32_t)t->dims[0] - first_col;
                const uint64_t source_block = first_col / traits->block_values;
                const uint8_t *group = row_data + source_block * traits->block_bytes;
                memset(decoded, 0, sizeof(decoded));
                if (!coli_dtype_dequantize_row((ColiDType)t->type, group, logical_count, decoded) ||
                    !write_sparse_block(part, t->type, codec, group, decoded, logical_count, threshold,
                                        &stats->retained_total, &stats->tree_total,
                                        &stats->value_total, &stats->aux_total)) {
                    if (failed_block) *failed_block = block_index;
                    return 0;
                }
            }
        }
        if (progress) __atomic_add_fetch(&progress->experts_done, 1u, __ATOMIC_RELAXED);
    }
    return 1;
}

#ifndef _WIN32
static int encode_tensor_multiprocess(FILE *out,
                                      const char *tmp_prefix,
                                      uint64_t tensor_id,
                                      const ColiGgufTensorInfo *t,
                                      const ColiDTypeTraits *traits,
                                      const uint8_t *tensor_data,
                                      uint64_t row_bytes,
                                      uint32_t blocks_per_row,
                                      uint32_t codec,
                                      float threshold,
                                      uint32_t jobs,
                                      uint32_t *offsets,
                                      SparseStats *stats,
                                      int verbose) {
    uint32_t expert_count = (uint32_t)t->dims[2];
    if (jobs > expert_count) jobs = expert_count;
    if (jobs < 2) return 0;

    FILE **parts = (FILE **)calloc(jobs, sizeof(*parts));
    pid_t *pids = (pid_t *)calloc(jobs, sizeof(*pids));
    uint8_t *reaped = (uint8_t *)calloc(jobs, 1);
    size_t result_bytes = (size_t)jobs * sizeof(SparseWorkerResult);
    SparseWorkerResult *results = (SparseWorkerResult *)mmap(
        NULL, result_bytes, PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    SparseProgress *progress = (SparseProgress *)mmap(
        NULL, sizeof(*progress), PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    int ok = parts && pids && reaped && results != MAP_FAILED && progress != MAP_FAILED;
    if (!ok) goto cleanup;
    memset(results, 0, result_bytes);
    memset(progress, 0, sizeof(*progress));

    for (uint32_t w = 0; w < jobs; ++w) {
        parts[w] = open_worker_part(tmp_prefix, tensor_id, w);
        if (!parts[w]) { ok = 0; goto cleanup; }
        setvbuf(parts[w], NULL, _IOFBF, 4u << 20);
        results[w].first_expert = (uint32_t)(((uint64_t)expert_count * w) / jobs);
        results[w].end_expert = (uint32_t)(((uint64_t)expert_count * (w + 1u)) / jobs);
        results[w].failed_block = UINT64_MAX;
    }

    if (fflush(out) != 0) { ok = 0; goto cleanup; }
    if (verbose) {
        fprintf(stderr, "[SGGUF-MP] %s using %u worker processes for %u experts\n",
                t->name, jobs, expert_count);
    }

    uint32_t launched = 0;
    for (uint32_t w = 0; w < jobs; ++w) {
        pid_t pid = fork();
        if (pid < 0) {
            ok = 0;
            break;
        }
        if (pid == 0) {
            SparseStats local = {0, 0, 0, 0};
            uint64_t failed = UINT64_MAX;
            int child_ok = encode_expert_range(
                parts[w], t, traits, tensor_data, row_bytes,
                blocks_per_row, codec, threshold,
                results[w].first_expert, results[w].end_expert,
                offsets, &local, progress, &failed);
            uint64_t end = tell64(parts[w]);
            if (end == UINT64_MAX || fflush(parts[w]) != 0) child_ok = 0;
            results[w].local_bytes = end == UINT64_MAX ? 0 : end;
            results[w].stats = local;
            results[w].failed_block = failed;
            results[w].ok = child_ok;
            fclose(parts[w]);
            _exit(child_ok ? 0 : 1);
        }
        pids[w] = pid;
        ++launched;
    }

    if (!ok) {
        for (uint32_t w = 0; w < launched; ++w) kill(pids[w], SIGTERM);
    }

    uint32_t remaining = launched;
    uint32_t last_report = 0;
    double start_time = monotonic_seconds();
    while (remaining) {
        for (uint32_t w = 0; w < launched; ++w) {
            if (reaped[w]) continue;
            int status = 0;
            pid_t got = waitpid(pids[w], &status, WNOHANG);
            if (got == pids[w]) {
                reaped[w] = 1;
                --remaining;
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) ok = 0;
            } else if (got < 0 && errno != EINTR) {
                reaped[w] = 1;
                --remaining;
                ok = 0;
            }
        }
        if (verbose && progress != MAP_FAILED) {
            uint32_t done = __atomic_load_n(&progress->experts_done, __ATOMIC_RELAXED);
            if (done == expert_count || done >= last_report + 16u) {
                double elapsed = monotonic_seconds() - start_time;
                double rate = elapsed > 0.0 ? done / elapsed : 0.0;
                double eta = rate > 0.0 ? (expert_count - done) / rate : 0.0;
                fprintf(stderr,
                    "[SGGUF-MP] %s expert %u/%u rate=%.2f/s eta=%.1fs\n",
                    t->name, done, expert_count, rate, eta);
                last_report = done;
            }
        }
        if (remaining) usleep(100000);
    }

    if (launched != jobs) ok = 0;
    for (uint32_t w = 0; w < jobs; ++w) {
        if (!results[w].ok) {
            if (verbose && results[w].failed_block != UINT64_MAX) {
                fprintf(stderr, "[SGGUF-MP] %s worker %u failed at block %llu\n",
                        t->name, w,
                        (unsigned long long)results[w].failed_block);
            }
            ok = 0;
        }
    }
    if (!ok) goto cleanup;

    uint64_t base = 0;
    for (uint32_t w = 0; w < jobs; ++w) {
        uint64_t first_block =
            (uint64_t)results[w].first_expert * t->dims[1] * blocks_per_row;
        uint64_t end_block =
            (uint64_t)results[w].end_expert * t->dims[1] * blocks_per_row;
        if (base > UINT32_MAX) { ok = 0; goto cleanup; }
        for (uint64_t b = first_block; b < end_block; ++b) {
            uint64_t adjusted = (uint64_t)offsets[b] + base;
            if (adjusted > UINT32_MAX) { ok = 0; goto cleanup; }
            offsets[b] = (uint32_t)adjusted;
        }

        if (FSEEK64(parts[w], 0, SEEK_SET) != 0 ||
            !copy_stream(out, parts[w], results[w].local_bytes)) {
            ok = 0;
            goto cleanup;
        }
        base += results[w].local_bytes;
        stats->retained_total += results[w].stats.retained_total;
        stats->tree_total += results[w].stats.tree_total;
        stats->value_total += results[w].stats.value_total;
        stats->aux_total += results[w].stats.aux_total;
    }
    if (base > UINT32_MAX) { ok = 0; goto cleanup; }
    offsets[(uint64_t)t->dims[1] * t->dims[2] * blocks_per_row] = (uint32_t)base;

cleanup:
    if (parts) {
        for (uint32_t w = 0; w < jobs; ++w) if (parts[w]) fclose(parts[w]);
    }
    if (results && results != MAP_FAILED) munmap(results, result_bytes);
    if (progress && progress != MAP_FAILED) munmap(progress, sizeof(*progress));
    free(reaped);
    free(pids);
    free(parts);
    return ok;
}
#endif

static int write_sparse_tensor(FILE *out, const ColiGgufFile *src,
                               const ColiGgufTensorInfo *t, uint32_t codec,
                               float threshold, uint64_t *payload_size_out,
                               uint64_t *index_rel_out, uint64_t *index_size_out,
                               int verbose, uint32_t requested_jobs,
                               const char *tmp_prefix, uint64_t tensor_id) {
    const ColiDTypeTraits *traits = coli_dtype_traits((ColiDType)t->type);
    if (!traits || t->n_dims != 3 || t->dims[0] == 0 ||
        t->dims[0] % traits->block_values ||
        t->dims[1] > UINT32_MAX || t->dims[2] > UINT32_MAX || t->dims[0] > UINT32_MAX)
        return 0;

    uint64_t row_bytes = 0;
    if (!coli_dtype_row_size((ColiDType)t->type, t->dims[0], &row_bytes)) return 0;
    uint64_t total_rows = t->dims[1] * t->dims[2];
    uint32_t blocks_per_row = (uint32_t)((t->dims[0] + 255u) / 256u);
    if (total_rows > UINT64_MAX / blocks_per_row) return 0;
    uint64_t total_blocks = total_rows * blocks_per_row;
    if (total_blocks > (SIZE_MAX / sizeof(uint32_t)) - 1u) return 0;

    size_t offsets_alloc = (size_t)(total_blocks + 1u) * sizeof(uint32_t);
    uint32_t *offsets = NULL;
#ifndef _WIN32
    uint32_t jobs = requested_jobs ? requested_jobs : online_cpu_count();
    if (jobs > t->dims[2]) jobs = (uint32_t)t->dims[2];
    if (jobs > 1) {
        offsets = (uint32_t *)mmap(NULL, offsets_alloc,
            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (offsets == MAP_FAILED) offsets = NULL;
    }
#else
    uint32_t jobs = 1;
    (void)requested_jobs; (void)tmp_prefix; (void)tensor_id;
#endif
    if (!offsets) {
        jobs = 1;
        offsets = (uint32_t *)malloc(offsets_alloc);
    }
    if (!offsets) return 0;

    uint64_t tensor_start = tell64(out);
    if (tensor_start == UINT64_MAX || !write_zeros(out, COLI_SGGUF_SPARSE_TENSOR_HEADER_BYTES)) {
        goto fail;
    }
    uint64_t offsets_offset = COLI_SGGUF_SPARSE_TENSOR_HEADER_BYTES;
    uint64_t offsets_bytes = (total_blocks + 1u) * 4u;
    if (!write_zeros(out, offsets_bytes) || !pad_to(out, 4)) goto fail;
    uint64_t blocks_start_abs = tell64(out);
    if (blocks_start_abs == UINT64_MAX) goto fail;
    uint64_t blocks_offset = blocks_start_abs - tensor_start;

    const uint8_t *tensor_data = (const uint8_t *)coli_gguf_mapped_at(src, t->absolute_offset, t->payload_size);
    if (!tensor_data) goto fail;
    SparseStats stats = {0, 0, 0, 0};
    double started = monotonic_seconds();
    int encoded_ok = 0;

#ifndef _WIN32
    if (jobs > 1) {
        encoded_ok = encode_tensor_multiprocess(
            out, tmp_prefix, tensor_id, t, traits, tensor_data,
            row_bytes, blocks_per_row,
            codec, threshold, jobs, offsets, &stats, verbose);
    } else
#endif
    {
        SparseProgress progress = {0};
        encoded_ok = encode_expert_range(
            out, t, traits, tensor_data, row_bytes,
            blocks_per_row, codec, threshold,
            0, (uint32_t)t->dims[2], offsets, &stats, &progress, NULL);
        uint64_t end_serial = tell64(out);
        if (encoded_ok && end_serial != UINT64_MAX) {
            uint64_t final_bytes = end_serial - blocks_start_abs;
            if (final_bytes > UINT32_MAX) encoded_ok = 0;
            else offsets[total_blocks] = (uint32_t)final_bytes;
        }
        if (verbose && encoded_ok)
            fprintf(stderr, "[SGGUF] %s expert %llu/%llu\n", t->name,
                    (unsigned long long)t->dims[2],
                    (unsigned long long)t->dims[2]);
    }
    if (!encoded_ok) goto fail;

    uint64_t end = tell64(out);
    if (end == UINT64_MAX) goto fail;
    uint64_t payload_size = end - tensor_start;

    uint8_t header[COLI_SGGUF_SPARSE_TENSOR_HEADER_BYTES];
    memset(header, 0, sizeof(header));
    memcpy(header, "SPB3", 4);
    coli_sgguf_store_u32_le(header + 4, 3);
    coli_sgguf_store_u32_le(header + 8, codec);
    coli_sgguf_store_u32_le(header + 12, 256);
    coli_sgguf_store_u32_le(header + 16, (uint32_t)t->dims[0]);
    coli_sgguf_store_u32_le(header + 20, (uint32_t)t->dims[1]);
    coli_sgguf_store_u32_le(header + 24, (uint32_t)t->dims[2]);
    coli_sgguf_store_u32_le(header + 28, blocks_per_row);
    coli_sgguf_store_u64_le(header + 32, total_rows);
    coli_sgguf_store_u64_le(header + 40, total_blocks);
    coli_sgguf_store_u64_le(header + 48, offsets_offset);
    coli_sgguf_store_u64_le(header + 56, blocks_offset);
    coli_sgguf_store_u64_le(header + 64, payload_size);
    coli_sgguf_store_u16_le(header + 72, coli_sgguf_codec_aux_bytes(codec));
    coli_sgguf_store_u16_le(header + 74, coli_sgguf_codec_value_bits(codec));

    if (!seek64(out, tensor_start) || !put_bytes(out, header, sizeof(header))) goto fail;
    for (uint64_t i = 0; i <= total_blocks; ++i) {
        if (!put_u32(out, offsets[i])) goto fail;
    }
    if (!seek64(out, end)) goto fail;

    if (payload_size_out) *payload_size_out = payload_size;
    if (index_rel_out) *index_rel_out = offsets_offset;
    if (index_size_out) *index_size_out = offsets_bytes;
    if (verbose) {
        double dense_mib = t->payload_size / (1024.0 * 1024.0);
        double sparse_mib = payload_size / (1024.0 * 1024.0);
        uint64_t logical_values = total_rows * t->dims[0];
        double keep = logical_values ? (double)stats.retained_total / (double)logical_values : 0.0;
        double elapsed = monotonic_seconds() - started;
        fprintf(stderr,
            "[SGGUF] %s codec=%s dense=%.2f MiB sparse=%.2f MiB ratio=%.3f retained=%.2f%% bitmap=%.2f MiB values=%.2f MiB aux=%.2f MiB index=%.2f MiB jobs=%u time=%.2fs\n",
            t->name, coli_sgguf_codec_name(codec), dense_mib, sparse_mib,
            t->payload_size ? (double)payload_size / (double)t->payload_size : 0.0,
            keep * 100.0, stats.tree_total / (1024.0 * 1024.0),
            stats.value_total / (1024.0 * 1024.0), stats.aux_total / (1024.0 * 1024.0),
            offsets_bytes / (1024.0 * 1024.0), jobs, elapsed);
    }
#ifndef _WIN32
    if (jobs > 1) munmap(offsets, offsets_alloc); else free(offsets);
#else
    free(offsets);
#endif
    return 1;

fail:
#ifndef _WIN32
    if (offsets) {
        if (jobs > 1) munmap(offsets, offsets_alloc); else free(offsets);
    }
#else
    free(offsets);
#endif
    return 0;
}

static float verify_expected_value(float source, float threshold, uint32_t codec) {
    if (fabsf(source) < threshold) return 0.0f;
    if (codec == COLI_SGGUF_CODEC_RETAINED_F16)
        return coli_fp16_to_fp32(fp32_to_fp16(source));
    if (codec == COLI_SGGUF_CODEC_RETAINED_BF16)
        return coli_bf16_to_fp32(fp32_to_bf16(source));
    return source;
}

static int same_float_value(float a, float b) {
    if (isnan(a) && isnan(b)) return 1;
    uint32_t ua, ub;
    memcpy(&ua, &a, sizeof(ua));
    memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

static int verify_sgguf(const char *path, const ColiGgufFile *source, float threshold) {
    ColiGgufFile f;
    if (!coli_gguf_open(&f, path)) return failf("verification open failed: %s", coli_gguf_error(&f));
    if (f.container_kind != COLI_MODEL_CONTAINER_SGGUF) {
        coli_gguf_close(&f); return failf("verification did not open an SGGUF container");
    }
    for (uint64_t i = 0; i < f.tensor_count; ++i) {
        const ColiGgufTensorInfo *t = &f.tensors[i];
        if (t->storage_kind != COLI_TENSOR_STORAGE_SPARSE_TREE) continue;
        const uint8_t *p = (const uint8_t *)coli_gguf_mapped_at(&f, t->absolute_offset, t->payload_size);
        ColiSggufSparseTensor st;
        char err[256];
        if (!coli_sgguf_sparse_tensor_parse(p, t->payload_size, &st, err, sizeof(err))) {
            coli_gguf_close(&f); return failf("verification of %s failed: %s", t->name, err);
        }
        ColiSggufSparseBlock b;
        if (!coli_sgguf_sparse_block_get(&st, 0, &b, err, sizeof(err)) ||
            !coli_sgguf_sparse_block_get(&st, st.total_blocks - 1u, &b, err, sizeof(err))) {
            coli_gguf_close(&f); return failf("verification of %s blocks failed: %s", t->name, err);
        }

        const ColiGgufTensorInfo *src = source ? coli_gguf_find_tensor(source, t->name) : NULL;
        const ColiDTypeTraits *traits = src ? coli_dtype_traits((ColiDType)src->type) : NULL;
        uint64_t row_bytes = 0;
        if (!src || !traits || !coli_dtype_row_size((ColiDType)src->type, src->dims[0], &row_bytes)) {
            coli_gguf_close(&f); return failf("verification source tensor missing/unsupported: %s", t->name);
        }
        const uint8_t *src_data = (const uint8_t *)coli_gguf_mapped_at(
            source, src->absolute_offset, src->payload_size);
        uint64_t samples[3] = {0, st.total_blocks / 2u, st.total_blocks - 1u};
        for (int si = 0; si < 3; ++si) {
            uint64_t bi = samples[si];
            if (si && bi == samples[si - 1]) continue;
            uint64_t row = bi / st.blocks_per_row;
            uint64_t block_in_row = bi % st.blocks_per_row;
            uint32_t first_col = (uint32_t)block_in_row * COLI_SGGUF_GROUP_SIZE;
            uint32_t logical_count = first_col + COLI_SGGUF_GROUP_SIZE <= src->dims[0]
                ? COLI_SGGUF_GROUP_SIZE : (uint32_t)src->dims[0] - first_col;
            uint64_t source_block = first_col / traits->block_values;
            const uint8_t *group = src_data + row * row_bytes + source_block * traits->block_bytes;
            float decoded[256] = {0}, materialized[256];
            if (!coli_dtype_dequantize_row((ColiDType)src->type, group, logical_count, decoded) ||
                !coli_sgguf_sparse_block_get(&st, bi, &b, err, sizeof(err)) ||
                !coli_sgguf_sparse_block_materialize_f32(&b, materialized)) {
                coli_gguf_close(&f); return failf("semantic verification failed for %s block %llu",
                    t->name, (unsigned long long)bi);
            }
            if (b.logical_count != logical_count) {
                coli_gguf_close(&f); return failf("tail-size mismatch in %s block %llu",
                    t->name, (unsigned long long)bi);
            }
            for (uint32_t j = 0; j < logical_count; ++j) {
                float expected = verify_expected_value(decoded[j], threshold, b.codec_id);
                if (!same_float_value(expected, materialized[j])) {
                    coli_gguf_close(&f);
                    return failf("value mismatch in %s block %llu position %u: expected %.9g got %.9g",
                        t->name, (unsigned long long)bi, j, expected, materialized[j]);
                }
            }
        }
    }
    coli_gguf_close(&f);
    return 1;
}

static int convert_file(const char *input, const char *output, const ConvertOptions *opts) {
    ColiGgufFile src;
    if (!coli_gguf_open(&src, input)) return failf("cannot open input: %s", coli_gguf_error(&src));
    if (src.container_kind != COLI_MODEL_CONTAINER_GGUF) {
        coli_gguf_close(&src); return failf("input must be a standard GGUF");
    }

    TensorPlan *plans = (TensorPlan *)calloc((size_t)src.tensor_count, sizeof(*plans));
    if (!plans) { coli_gguf_close(&src); return failf("out of memory allocating tensor plans"); }
    uint64_t sparse_count = 0;
    for (uint64_t i = 0; i < src.tensor_count; ++i) {
        const ColiGgufTensorInfo *t = &src.tensors[i];
        TensorPlan *p = &plans[i];
        p->storage_kind = COLI_TENSOR_STORAGE_DENSE;
        p->moe_layer = -1;
        const ColiDTypeTraits *traits = coli_dtype_traits((ColiDType)t->type);
        if (parse_moe_name(t->name, &p->moe_layer, &p->moe_projection) && t->n_dims == 3 &&
            traits && t->dims[0] && t->dims[0] % traits->block_values == 0 &&
            t->dims[0] <= UINT32_MAX && t->dims[1] <= UINT32_MAX && t->dims[2] <= UINT32_MAX) {
            p->codec_id = choose_codec(t->type, opts->forced_codec);
            if (p->codec_id != COLI_SGGUF_CODEC_NONE) {
                p->storage_kind = COLI_TENSOR_STORAGE_SPARSE_TREE;
                p->expert_count = (uint32_t)t->dims[2];
                p->rows_per_expert = (uint32_t)t->dims[1];
                ++sparse_count;
            }
        }
    }
    if (!sparse_count && opts->verbose)
        fprintf(stderr, "[SGGUF] no eligible routed-MoE tensors; writing a dense-compatible SGGUF container\n");

    size_t tmp_len = strlen(output) + 16;
    char *tmp = (char *)malloc(tmp_len);
    if (!tmp) { free(plans); coli_gguf_close(&src); return failf("out of memory"); }
    snprintf(tmp, tmp_len, "%s.tmp", output);
    FILE *out = fopen(tmp, "wb+");
    if (!out) {
        int saved_errno = errno;
        failf("cannot create '%s': %s", tmp, strerror(saved_errno));
        free(tmp); free(plans); coli_gguf_close(&src);
        return 0;
    }

    int ok = put_bytes(out, COLI_SGGUF_MAGIC, 4) && put_u32(out, COLI_SGGUF_VERSION) &&
             put_u64(out, src.tensor_count) && put_u64(out, src.metadata_count + 5u);
    for (uint64_t i = 0; ok && i < src.metadata_count; ++i)
        ok = write_metadata_copy(out, &src, &src.metadata[i]);
    ok = ok && put_string(out, "sgguf.version") && put_u32(out, COLI_GGUF_TYPE_UINT32) && put_u32(out, 3);
    ok = ok && put_string(out, "sgguf.pruning.threshold") && put_u32(out, COLI_GGUF_TYPE_FLOAT32) && put_f32(out, opts->threshold);
    ok = ok && put_string(out, "sgguf.pruning.scope") && put_u32(out, COLI_GGUF_TYPE_STRING) && put_string(out, "routed-moe-experts");
    ok = ok && put_string(out, "sgguf.value.order") && put_u32(out, COLI_GGUF_TYPE_STRING) && put_string(out, "kth-retained-value-is-kth-one-bit");
    ok = ok && put_string(out, "sgguf.sparse.occupancy") && put_u32(out, COLI_GGUF_TYPE_STRING) && put_string(out, "bitmap256-v3");

    for (uint64_t i = 0; ok && i < src.tensor_count; ++i) {
        const ColiGgufTensorInfo *t = &src.tensors[i];
        TensorPlan *p = &plans[i];
        ok = put_string(out, t->name) && put_u32(out, t->n_dims);
        for (uint32_t d = 0; ok && d < t->n_dims; ++d) ok = put_u64(out, t->dims[d]);
        ok = ok && put_u32(out, t->type) && put_u32(out, p->storage_kind) &&
             put_u32(out, p->codec_id) && put_u32(out, 0);
        p->offset_patch = tell64(out); ok = ok && p->offset_patch != UINT64_MAX && put_u64(out, 0);
        p->payload_patch = tell64(out); ok = ok && p->payload_patch != UINT64_MAX && put_u64(out, 0);
        p->index_offset_patch = tell64(out); ok = ok && p->index_offset_patch != UINT64_MAX && put_u64(out, 0);
        p->index_size_patch = tell64(out); ok = ok && p->index_size_patch != UINT64_MAX && put_u64(out, 0);
        ok = ok && put_u32(out, (uint32_t)p->moe_layer) && put_u32(out, p->moe_projection) &&
             put_u32(out, p->expert_count) && put_u32(out, p->rows_per_expert);
    }
    ok = ok && pad_to(out, src.alignment);
    uint64_t data_offset = tell64(out);
    if (!ok || data_offset == UINT64_MAX) goto done;

    for (uint64_t i = 0; ok && i < src.tensor_count; ++i) {
        const ColiGgufTensorInfo *t = &src.tensors[i];
        TensorPlan *p = &plans[i];
        ok = pad_to(out, src.alignment);
        uint64_t absolute = tell64(out);
        if (!ok || absolute == UINT64_MAX || absolute < data_offset) { ok = 0; break; }
        uint64_t relative = absolute - data_offset;
        uint64_t payload_size = 0, index_rel = 0, index_size = 0;

        if (p->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
            ok = write_sparse_tensor(out, &src, t, p->codec_id, opts->threshold,
                                     &payload_size, &index_rel, &index_size, opts->verbose,
                                     opts->jobs, tmp, i);
        } else {
            payload_size = t->payload_size;
            ok = copy_mapped(out, &src, t->absolute_offset, payload_size);
        }
        if (!ok) break;
        uint64_t output_pos = tell64(out);
        if (output_pos == UINT64_MAX ||
            (opts->max_output_bytes && output_pos > opts->max_output_bytes)) {
            ok = failf("output exceeded --max-output-gb after tensor %s: %.3f GiB > %.3f GiB",
                       t->name, output_pos / (1024.0 * 1024.0 * 1024.0),
                       opts->max_output_bytes / (1024.0 * 1024.0 * 1024.0));
            break;
        }
        ok = patch_u64(out, p->offset_patch, relative) &&
             patch_u64(out, p->payload_patch, payload_size) &&
             patch_u64(out, p->index_offset_patch,
                       p->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE ? relative + index_rel : 0) &&
             patch_u64(out, p->index_size_patch, index_size);
    }

done:
    if (ok && fflush(out) != 0) ok = 0;
    if (fclose(out) != 0) ok = 0;
    if (ok && opts->verify) ok = verify_sgguf(tmp, &src, opts->threshold);
    if (ok) {
        if (rename(tmp, output) != 0) ok = failf("cannot rename '%s' to '%s': %s", tmp, output, strerror(errno));
    } else {
        remove(tmp);
    }
    free(tmp);
    free(plans);
    coli_gguf_close(&src);
    return ok;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s INPUT.gguf OUTPUT.sgguf [--threshold N] "
        "[--codec auto|preserve|f16|bf16|f32] [--jobs N|auto|--mp N|auto] "
        "[--max-output-gb N] [--no-verify] [--quiet]\n",
        argv0);
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    ConvertOptions opts = { 0.01f, 0, 1, 1, 0, 0 };
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            char *end = NULL;
            double v = strtod(argv[++i], &end);
            if (!end || *end || !isfinite(v) || v < 0.0 || v > 1e6) {
                usage(argv[0]); return 2;
            }
            opts.threshold = (float)v;
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "auto") == 0 || strcmp(v, "preserve") == 0)
                opts.forced_codec = 0;
            else if (strcmp(v, "f16") == 0)
                opts.forced_codec = COLI_SGGUF_CODEC_RETAINED_F16;
            else if (strcmp(v, "bf16") == 0)
                opts.forced_codec = COLI_SGGUF_CODEC_RETAINED_BF16;
            else if (strcmp(v, "f32") == 0)
                opts.forced_codec = COLI_SGGUF_CODEC_RETAINED_F32;
            else { usage(argv[0]); return 2; }
        } else if ((strcmp(argv[i], "--jobs") == 0 || strcmp(argv[i], "--mp") == 0 ||
                    strcmp(argv[i], "-j") == 0) && i + 1 < argc) {
            const char *arg = argv[++i];
            if (strcmp(arg, "auto") == 0) {
                opts.jobs = 0;
            } else {
                char *end = NULL;
                unsigned long v = strtoul(arg, &end, 10);
                if (!end || *end || v > 256u) { usage(argv[0]); return 2; }
                opts.jobs = (uint32_t)v; /* zero also means automatic CPU count */
            }
        } else if (strcmp(argv[i], "--max-output-gb") == 0 && i + 1 < argc) {
            char *end = NULL;
            double gb = strtod(argv[++i], &end);
            if (!end || *end || !isfinite(gb) || gb <= 0.0 ||
                gb > (double)UINT64_MAX / (1024.0 * 1024.0 * 1024.0)) {
                usage(argv[0]); return 2;
            }
            opts.max_output_bytes = (uint64_t)(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (strcmp(argv[i], "--no-verify") == 0) opts.verify = 0;
        else if (strcmp(argv[i], "--quiet") == 0) opts.verbose = 0;
        else { usage(argv[0]); return 2; }
    }
    return convert_file(argv[1], argv[2], &opts) ? 0 : 1;
}
