#include "../gguf_reader.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#define unlink _unlink
#define getpid _getpid
#else
#include <unistd.h>
#endif

static int put_u8(FILE *f, uint8_t v) { return fwrite(&v, 1, 1, f) == 1; }
static int put_u16(FILE *f, uint16_t v) {
    unsigned char p[2] = { (unsigned char)v, (unsigned char)(v >> 8) };
    return fwrite(p, 1, sizeof(p), f) == sizeof(p);
}
static int put_u32(FILE *f, uint32_t v) {
    unsigned char p[4] = {
        (unsigned char)v, (unsigned char)(v >> 8),
        (unsigned char)(v >> 16), (unsigned char)(v >> 24)
    };
    return fwrite(p, 1, sizeof(p), f) == sizeof(p);
}
static int put_u64(FILE *f, uint64_t v) {
    unsigned char p[8];
    for (int i = 0; i < 8; ++i) p[i] = (unsigned char)(v >> (8 * i));
    return fwrite(p, 1, sizeof(p), f) == sizeof(p);
}
static int put_string(FILE *f, const char *s) {
    size_t n = strlen(s);
    return put_u64(f, (uint64_t)n) && fwrite(s, 1, n, f) == n;
}
static int pad_to(FILE *f, long alignment) {
    long pos = ftell(f);
    if (pos < 0) return 0;
    long end = (pos + alignment - 1) & ~(alignment - 1);
    while (pos++ < end) if (!put_u8(f, 0)) return 0;
    return 1;
}

static int write_fixture(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    int ok = fwrite("GGUF", 1, 4, f) == 4 &&
             put_u32(f, 3) && put_u64(f, 2) && put_u64(f, 5);

    ok = ok && put_string(f, "general.architecture") && put_u32(f, COLI_GGUF_TYPE_STRING) &&
         put_string(f, "qwen3next");
    ok = ok && put_string(f, "general.alignment") && put_u32(f, COLI_GGUF_TYPE_UINT32) &&
         put_u32(f, 64);
    ok = ok && put_string(f, "qwen3next.block_count") && put_u32(f, COLI_GGUF_TYPE_UINT32) &&
         put_u32(f, 48);
    ok = ok && put_string(f, "tokenizer.ggml.tokens") && put_u32(f, COLI_GGUF_TYPE_ARRAY) &&
         put_u32(f, COLI_GGUF_TYPE_STRING) && put_u64(f, 2) &&
         put_string(f, "hello") && put_string(f, "world");
    ok = ok && put_string(f, "test.flag") && put_u32(f, COLI_GGUF_TYPE_BOOL) && put_u8(f, 1);

    ok = ok && put_string(f, "token_embd.weight") && put_u32(f, 2) &&
         put_u64(f, 4) && put_u64(f, 3) && put_u32(f, 0) && put_u64(f, 0);
    ok = ok && put_string(f, "blk.0.ffn_gate_inp.weight") && put_u32(f, 2) &&
         put_u64(f, 4) && put_u64(f, 8) && put_u32(f, 12) && put_u64(f, 64);

    ok = ok && pad_to(f, 64);
    for (int i = 0; ok && i < 128; ++i) ok = put_u8(f, (uint8_t)i);

    if (fclose(f) != 0) ok = 0;
    return ok;
}

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #x); goto fail; } } while (0)

int main(void) {
    char path[256];
#ifdef _WIN32
    snprintf(path, sizeof(path), "test_gguf_reader_%ld.gguf", (long)getpid());
#else
    snprintf(path, sizeof(path), "/tmp/test_gguf_reader_%ld.gguf", (long)getpid());
#endif

    CHECK(write_fixture(path));

    ColiGgufFile g;
    CHECK(coli_gguf_open(&g, path));
    CHECK(g.version == 3);
    CHECK(g.tensor_count == 2);
    CHECK(g.metadata_count == 5);
    CHECK(g.alignment == 64);
    CHECK(g.data_offset % 64 == 0);

    const ColiGgufKV *arch = coli_gguf_find_kv(&g, "general.architecture");
    CHECK(arch != NULL);
    char *arch_s = NULL;
    CHECK(coli_gguf_kv_read_string(&g, arch, &arch_s));
    CHECK(strcmp(arch_s, "qwen3next") == 0);
    free(arch_s);

    const ColiGgufKV *layers = coli_gguf_find_kv(&g, "qwen3next.block_count");
    uint64_t layer_count = 0;
    CHECK(layers != NULL && coli_gguf_kv_read_u64(&g, layers, &layer_count));
    CHECK(layer_count == 48);

    const ColiGgufKV *tokens = coli_gguf_find_kv(&g, "tokenizer.ggml.tokens");
    CHECK(tokens != NULL);
    CHECK(tokens->type == COLI_GGUF_TYPE_ARRAY);
    CHECK(tokens->array_type == COLI_GGUF_TYPE_STRING);
    CHECK(tokens->array_count == 2);

    const ColiGgufKV *flag = coli_gguf_find_kv(&g, "test.flag");
    int b = 0;
    CHECK(flag != NULL && coli_gguf_kv_read_bool(&g, flag, &b) && b == 1);

    const ColiGgufTensorInfo *t = coli_gguf_find_tensor(&g, "blk.0.ffn_gate_inp.weight");
    CHECK(t != NULL);
    CHECK(t->type == 12);
    CHECK(t->n_dims == 2 && t->dims[0] == 4 && t->dims[1] == 8);
    CHECK(t->offset == 64);
    CHECK(t->absolute_offset == g.data_offset + 64);

    unsigned char payload[4] = {0};
    CHECK(coli_gguf_read_tensor_bytes(&g, t, 0, payload, sizeof(payload)));
    CHECK(payload[0] == 64 && payload[1] == 65 && payload[2] == 66 && payload[3] == 67);

    CHECK(strcmp(coli_ggml_type_name(12), "Q4_K") == 0);

    coli_gguf_close(&g);

    {
        FILE *bad = fopen(path, "wb");
        CHECK(bad != NULL);
        CHECK(fwrite("NOPE", 1, 4, bad) == 4);
        CHECK(fclose(bad) == 0);
        CHECK(!coli_gguf_open(&g, path));
        CHECK(strstr(coli_gguf_error(&g), "bad magic") != NULL);
    }

    unlink(path);
    puts("gguf reader: ok");
    return 0;

fail:
    unlink(path);
    return 1;
}
