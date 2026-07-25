#include "../sgguf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *what) {
    fprintf(stderr, "test_sgguf: %s\n", what);
    return 1;
}

static uint32_t popcount_mask(const uint32_t mask[8]) {
    uint32_t n = 0;
    for (int i = 0; i < 8; ++i) {
#if defined(__GNUC__) || defined(__clang__)
        n += (uint32_t)__builtin_popcount(mask[i]);
#else
        uint32_t x = mask[i]; while (x) { x &= x - 1u; ++n; }
#endif
    }
    return n;
}

static int test_tree_pattern(int which) {
    uint8_t keep[256];
    for (int i = 0; i < 256; ++i) {
        if (which == 0) keep[i] = 0;
        else if (which == 1) keep[i] = 1;
        else if (which == 2) keep[i] = (uint8_t)(i & 1);
        else if (which == 3) keep[i] = (uint8_t)((i >= 32 && i < 96) || i >= 224);
        else keep[i] = (uint8_t)(((uint32_t)i * 1103515245u + 12345u) >> 31);
    }
    ColiSggufTreeBits tree;
    uint32_t mask[8], retained = 0;
    if (!coli_sgguf_tree_encode(keep, &tree) ||
        !coli_sgguf_tree_decode(tree.data, tree.bit_count, mask, &retained)) return 0;
    for (int i = 0; i < 256; ++i) {
        uint32_t got = (mask[i >> 5] >> (i & 31)) & 1u;
        if (got != (uint32_t)keep[i]) return 0;
    }
    return retained == popcount_mask(mask);
}


static int test_malformed_tree_rejected(void) {
    /* Eight left branches reach the one-value interval [0,1). A ninth branch
     * is invalid because a singleton cannot be split further. */
    const uint8_t invalid_singleton_branch[2] = {0, 0};
    uint32_t mask[8], retained = 0;
    if (coli_sgguf_tree_decode(invalid_singleton_branch, 9, mask, &retained))
        return 0;

    /* A branch with no child payload must also be rejected. */
    const uint8_t truncated_branch[1] = {0};
    return !coli_sgguf_tree_decode(truncated_branch, 1, mask, &retained);
}

static int make_f32_block(const uint8_t keep[256], const float source[256],
                          uint8_t **encoded_out, uint32_t *encoded_bytes_out,
                          ColiSggufTreeBits *tree_out, uint16_t *retained_out) {
    ColiSggufTreeBits tree;
    if (!coli_sgguf_tree_encode(keep, &tree)) return 0;
    uint32_t retained = 0;
    for (int i = 0; i < 256; ++i) retained += keep[i] != 0;
    uint32_t tree_bytes = (tree.bit_count + 7u) / 8u;
    uint32_t raw = COLI_SGGUF_SPARSE_BLOCK_HEADER_BYTES + tree_bytes + retained * 4u;
    uint32_t total = (raw + 3u) & ~3u;
    uint8_t *p = (uint8_t *)calloc(1, total);
    if (!p) return 0;
    coli_sgguf_store_u16_le(p + 0, (uint16_t)tree.bit_count);
    coli_sgguf_store_u16_le(p + 2, (uint16_t)retained);
    coli_sgguf_store_u16_le(p + 4, 0);
    coli_sgguf_store_u16_le(p + 6, 32);
    coli_sgguf_store_u32_le(p + 8, retained * 4u);
    coli_sgguf_store_u32_le(p + 12, total);
    memcpy(p + 16, tree.data, tree_bytes);
    uint8_t *values = p + 16 + tree_bytes;
    uint32_t k = 0;
    for (int i = 0; i < 256; ++i) if (keep[i]) {
        uint32_t u;
        memcpy(&u, &source[i], sizeof(u));
        coli_sgguf_store_u32_le(values + 4u * k++, u);
    }
    *encoded_out = p;
    *encoded_bytes_out = total;
    *tree_out = tree;
    *retained_out = (uint16_t)retained;
    return 1;
}

static int test_sparse_f32(void) {
    uint8_t keep[256];
    float source[256], x[256], expected[256];
    double expected_dot = 0.0;
    for (int i = 0; i < 256; ++i) {
        source[i] = ((float)i - 100.0f) / 37.0f;
        x[i] = ((float)(i % 17) - 8.0f) / 9.0f;
        keep[i] = (uint8_t)((i % 5) != 0 && (i < 70 || i >= 190));
        expected[i] = keep[i] ? source[i] : 0.0f;
        expected_dot += (double)expected[i] * x[i];
    }

    uint8_t *encoded = NULL;
    uint32_t encoded_bytes = 0;
    uint16_t retained = 0;
    ColiSggufTreeBits tree;
    if (!make_f32_block(keep, source, &encoded, &encoded_bytes, &tree, &retained)) return 0;

    uint32_t tree_bytes = (tree.bit_count + 7u) / 8u;
    ColiSggufSparseBlock block;
    memset(&block, 0, sizeof(block));
    block.codec_id = COLI_SGGUF_CODEC_RETAINED_F32;
    block.tree_bits = (uint16_t)tree.bit_count;
    block.retained_count = retained;
    block.retained_value_bits = 32;
    block.retained_payload_bytes = retained * 4u;
    block.encoded_block_bytes = encoded_bytes;
    block.tree = encoded + 16;
    block.auxiliary = block.tree + tree_bytes;
    block.retained_values = block.auxiliary;

    float materialized[256];
    int ok = 0;
    float got_dot = coli_sgguf_sparse_block_dot_f32(&block, x, &ok);
    if (!ok || !coli_sgguf_sparse_block_materialize_f32(&block, materialized)) {
        free(encoded); return 0;
    }
    for (int i = 0; i < 256; ++i) {
        if (memcmp(&materialized[i], &expected[i], sizeof(float)) != 0) {
            free(encoded); return 0;
        }
    }
    if (fabs((double)got_dot - expected_dot) > 2e-4) {
        free(encoded); return 0;
    }

    const uint64_t offsets_offset = 80;
    const uint64_t blocks_offset = 96;
    const uint64_t payload_size = blocks_offset + encoded_bytes;
    uint8_t *payload = (uint8_t *)calloc(1, (size_t)payload_size);
    if (!payload) { free(encoded); return 0; }
    memcpy(payload, "SPT1", 4);
    coli_sgguf_store_u32_le(payload + 4, 1);
    coli_sgguf_store_u32_le(payload + 8, COLI_SGGUF_CODEC_RETAINED_F32);
    coli_sgguf_store_u32_le(payload + 12, 256);
    coli_sgguf_store_u32_le(payload + 16, 256);
    coli_sgguf_store_u32_le(payload + 20, 1);
    coli_sgguf_store_u32_le(payload + 24, 1);
    coli_sgguf_store_u32_le(payload + 28, 1);
    coli_sgguf_store_u64_le(payload + 32, 1);
    coli_sgguf_store_u64_le(payload + 40, 1);
    coli_sgguf_store_u64_le(payload + 48, offsets_offset);
    coli_sgguf_store_u64_le(payload + 56, blocks_offset);
    coli_sgguf_store_u64_le(payload + 64, payload_size);
    coli_sgguf_store_u64_le(payload + 80, 0);
    coli_sgguf_store_u64_le(payload + 88, encoded_bytes);
    memcpy(payload + blocks_offset, encoded, encoded_bytes);

    ColiSggufSparseTensor tensor;
    ColiSggufSparseBlock parsed;
    char error[160];
    int parsed_ok = coli_sgguf_sparse_tensor_parse(payload, payload_size, &tensor,
                                                    error, sizeof(error)) &&
                    coli_sgguf_sparse_block_get(&tensor, 0, &parsed,
                                                error, sizeof(error)) &&
                    parsed.codec_id == COLI_SGGUF_CODEC_RETAINED_F32 &&
                    parsed.retained_count == retained;
    free(payload);
    free(encoded);
    return parsed_ok;
}


static int test_bitmap_f32(void) {
    uint8_t keep[256], bitmap[COLI_SGGUF_BITMAP_BYTES];
    float source[256], x[256], expected[256];
    uint32_t retained = 0;
    double expected_dot = 0.0;
    for (int i = 0; i < 256; ++i) {
        source[i] = ((float)i - 90.0f) / 31.0f;
        x[i] = ((float)(i % 13) - 6.0f) / 7.0f;
        keep[i] = (uint8_t)((i % 7) != 0 && (i < 83 || i > 171));
        retained += keep[i] != 0;
        expected[i] = keep[i] ? source[i] : 0.0f;
        expected_dot += (double)expected[i] * x[i];
    }
    coli_sgguf_bitmap_from_keep(keep, bitmap);
    uint32_t values_bytes = retained * 4u;
    uint32_t block_bytes = COLI_SGGUF_BITMAP_BYTES + values_bytes;
    uint64_t blocks_offset = 88; /* 80-byte header + two 4-byte offsets */
    uint64_t payload_size = blocks_offset + block_bytes;
    uint8_t *payload = (uint8_t *)calloc(1, (size_t)payload_size);
    if (!payload) return 0;
    memcpy(payload, "SPB2", 4);
    coli_sgguf_store_u32_le(payload + 4, 2);
    coli_sgguf_store_u32_le(payload + 8, COLI_SGGUF_CODEC_RETAINED_F32);
    coli_sgguf_store_u32_le(payload + 12, 256);
    coli_sgguf_store_u32_le(payload + 16, 256);
    coli_sgguf_store_u32_le(payload + 20, 1);
    coli_sgguf_store_u32_le(payload + 24, 1);
    coli_sgguf_store_u32_le(payload + 28, 1);
    coli_sgguf_store_u64_le(payload + 32, 1);
    coli_sgguf_store_u64_le(payload + 40, 1);
    coli_sgguf_store_u64_le(payload + 48, 80);
    coli_sgguf_store_u64_le(payload + 56, blocks_offset);
    coli_sgguf_store_u64_le(payload + 64, payload_size);
    coli_sgguf_store_u16_le(payload + 72, 0);
    coli_sgguf_store_u16_le(payload + 74, 32);
    coli_sgguf_store_u32_le(payload + 80, 0);
    coli_sgguf_store_u32_le(payload + 84, block_bytes);
    memcpy(payload + blocks_offset, bitmap, sizeof(bitmap));
    uint8_t *values = payload + blocks_offset + COLI_SGGUF_BITMAP_BYTES;
    uint32_t k = 0;
    for (int i = 0; i < 256; ++i) if (keep[i]) {
        uint32_t u; memcpy(&u, &source[i], sizeof(u));
        coli_sgguf_store_u32_le(values + 4u * k++, u);
    }

    ColiSggufSparseTensor tensor;
    ColiSggufSparseBlock block;
    char error[160];
    int good = coli_sgguf_sparse_tensor_parse(payload, payload_size, &tensor,
                                               error, sizeof(error)) &&
               tensor.layout == COLI_SGGUF_LAYOUT_BITMAP_V2 &&
               tensor.offset_width == 4 &&
               coli_sgguf_sparse_block_get(&tensor, 0, &block,
                                            error, sizeof(error));
    if (!good) { free(payload); return 0; }
    float materialized[256];
    int ok = 0;
    float got_dot = coli_sgguf_sparse_block_dot_f32(&block, x, &ok);
    good = ok && coli_sgguf_sparse_block_materialize_f32(&block, materialized);
    for (int i = 0; good && i < 256; ++i)
        if (memcmp(&materialized[i], &expected[i], sizeof(float)) != 0) good = 0;
    if (good && fabs((double)got_dot - expected_dot) > 2e-4) good = 0;
    free(payload);
    return good;
}

int main(void) {
    for (int i = 0; i < 5; ++i)
        if (!test_tree_pattern(i)) return fail("tree roundtrip failed");
    if (!test_malformed_tree_rejected()) return fail("malformed tree accepted");
    if (!test_sparse_f32()) return fail("legacy tree F32 block/tensor test failed");
    if (!test_bitmap_f32()) return fail("bitmap F32 block/tensor test failed");
    puts("test_sgguf: ok");
    return 0;
}
