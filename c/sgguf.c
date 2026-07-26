#include "sgguf.h"
#include "ggml_quants.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

uint16_t coli_sgguf_load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}
uint32_t coli_sgguf_load_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t coli_sgguf_load_u64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
    return v;
}
void coli_sgguf_store_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
void coli_sgguf_store_u32_le(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
void coli_sgguf_store_u64_le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

static int sgguf_fail(char *error, size_t cap, const char *fmt, ...) {
    if (error && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(error, cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

const char *coli_sgguf_codec_name(uint32_t codec_id) {
    switch (codec_id) {
        case COLI_SGGUF_CODEC_RETAINED_F16: return "retained-f16";
        case COLI_SGGUF_CODEC_Q6_K_EXACT: return "q6_k-exact";
        case COLI_SGGUF_CODEC_Q4_K_EXACT: return "q4_k-exact";
        case COLI_SGGUF_CODEC_RETAINED_F32: return "retained-f32";
        case COLI_SGGUF_CODEC_RETAINED_BF16: return "retained-bf16";
        case COLI_SGGUF_CODEC_Q4_0_EXACT: return "q4_0-exact";
        case COLI_SGGUF_CODEC_Q4_1_EXACT: return "q4_1-exact";
        case COLI_SGGUF_CODEC_Q5_0_EXACT: return "q5_0-exact";
        case COLI_SGGUF_CODEC_Q5_1_EXACT: return "q5_1-exact";
        case COLI_SGGUF_CODEC_Q8_0_EXACT: return "q8_0-exact";
        case COLI_SGGUF_CODEC_Q8_1_EXACT: return "q8_1-exact";
        case COLI_SGGUF_CODEC_Q3_K_EXACT: return "q3_k-exact";
        case COLI_SGGUF_CODEC_Q5_K_EXACT: return "q5_k-exact";
        case COLI_SGGUF_CODEC_Q8_K_EXACT: return "q8_k-exact";
        case COLI_SGGUF_CODEC_MXFP4_EXACT: return "mxfp4-exact";
        case COLI_SGGUF_CODEC_IQ4_XS_EXACT: return "iq4_xs-exact";
        case COLI_SGGUF_CODEC_IQ4_NL_EXACT: return "iq4_nl-exact";
        default: return "unknown";
    }
}

uint16_t coli_sgguf_codec_aux_bytes(uint32_t codec_id) {
    switch (codec_id) {
        case COLI_SGGUF_CODEC_Q4_0_EXACT: return 16;
        case COLI_SGGUF_CODEC_Q4_1_EXACT: return 32;
        case COLI_SGGUF_CODEC_Q5_0_EXACT: return 16;
        case COLI_SGGUF_CODEC_Q5_1_EXACT: return 32;
        case COLI_SGGUF_CODEC_Q8_0_EXACT: return 16;
        case COLI_SGGUF_CODEC_Q8_1_EXACT: return 32;
        case COLI_SGGUF_CODEC_Q3_K_EXACT: return 14;
        case COLI_SGGUF_CODEC_Q4_K_EXACT: return 16;
        case COLI_SGGUF_CODEC_Q5_K_EXACT: return 16;
        case COLI_SGGUF_CODEC_Q6_K_EXACT: return 18;
        case COLI_SGGUF_CODEC_Q8_K_EXACT: return 4;
        case COLI_SGGUF_CODEC_MXFP4_EXACT: return 8;
        case COLI_SGGUF_CODEC_IQ4_XS_EXACT: return 8;
        case COLI_SGGUF_CODEC_IQ4_NL_EXACT: return 16;
        case COLI_SGGUF_CODEC_RETAINED_F16:
        case COLI_SGGUF_CODEC_RETAINED_BF16:
        case COLI_SGGUF_CODEC_RETAINED_F32: return 0;
        default: return UINT16_MAX;
    }
}

uint16_t coli_sgguf_codec_value_bits(uint32_t codec_id) {
    switch (codec_id) {
        case COLI_SGGUF_CODEC_Q3_K_EXACT: return 3;
        case COLI_SGGUF_CODEC_Q4_0_EXACT:
        case COLI_SGGUF_CODEC_Q4_1_EXACT:
        case COLI_SGGUF_CODEC_Q4_K_EXACT: return 4;
        case COLI_SGGUF_CODEC_MXFP4_EXACT: return 4;
        case COLI_SGGUF_CODEC_IQ4_XS_EXACT: return 4;
        case COLI_SGGUF_CODEC_IQ4_NL_EXACT: return 4;
        case COLI_SGGUF_CODEC_Q5_0_EXACT:
        case COLI_SGGUF_CODEC_Q5_1_EXACT:
        case COLI_SGGUF_CODEC_Q5_K_EXACT: return 5;
        case COLI_SGGUF_CODEC_Q6_K_EXACT: return 6;
        case COLI_SGGUF_CODEC_Q8_0_EXACT:
        case COLI_SGGUF_CODEC_Q8_1_EXACT:
        case COLI_SGGUF_CODEC_Q8_K_EXACT: return 8;
        case COLI_SGGUF_CODEC_RETAINED_F16:
        case COLI_SGGUF_CODEC_RETAINED_BF16: return 16;
        case COLI_SGGUF_CODEC_RETAINED_F32: return 32;
        default: return 0;
    }
}

typedef struct {
    ColiSggufTreeBits *out;
} TreeWriter;

static int tree_put(TreeWriter *w, unsigned bit) {
    if (!w || !w->out || w->out->bit_count >= COLI_SGGUF_MAX_TREE_BYTES * 8u) return 0;
    uint32_t n = w->out->bit_count++;
    if (bit) w->out->data[n >> 3] |= (uint8_t)(1u << (n & 7u));
    return 1;
}

static int tree_encode_node(TreeWriter *w, const uint8_t *keep, int left, int right) {
    const int first = keep[left] != 0;
    int uniform = 1;
    for (int i = left + 1; i < right; ++i) {
        if ((keep[i] != 0) != first) { uniform = 0; break; }
    }
    if (uniform) return tree_put(w, 1u) && tree_put(w, (unsigned)first);
    if (!tree_put(w, 0u)) return 0;
    int middle = (left + right) >> 1;
    return tree_encode_node(w, keep, left, middle) &&
           tree_encode_node(w, keep, middle, right);
}

int coli_sgguf_tree_encode(const uint8_t keep[COLI_SGGUF_GROUP_SIZE],
                           ColiSggufTreeBits *out) {
    if (!keep || !out) return 0;
    memset(out, 0, sizeof(*out));
    TreeWriter w = { out };
    return tree_encode_node(&w, keep, 0, COLI_SGGUF_GROUP_SIZE);
}

typedef struct {
    const uint8_t *data;
    uint32_t bits;
    uint32_t pos;
} TreeReader;

static int tree_get(TreeReader *r, unsigned *out) {
    if (!r || !out || r->pos >= r->bits) return 0;
    uint32_t n = r->pos++;
    *out = (r->data[n >> 3] >> (n & 7u)) & 1u;
    return 1;
}

static int mask_set_range(uint32_t mask[8], uint32_t left, uint32_t right) {
    if (!mask || left >= right || right > COLI_SGGUF_GROUP_SIZE) return 0;

    const uint32_t first_word = left >> 5;
    const uint32_t last_word = (right - 1u) >> 5;
    const uint32_t first_bit = left & 31u;
    const uint32_t last_bit = (right - 1u) & 31u;

    if (first_word == last_word) {
        const uint32_t low = UINT32_MAX << first_bit;
        const uint32_t high = last_bit == 31u
            ? UINT32_MAX
            : ((1u << (last_bit + 1u)) - 1u);
        mask[first_word] |= low & high;
        return 1;
    }

    mask[first_word] |= UINT32_MAX << first_bit;
    for (uint32_t word = first_word + 1u; word < last_word; ++word)
        mask[word] = UINT32_MAX;
    mask[last_word] |= last_bit == 31u
        ? UINT32_MAX
        : ((1u << (last_bit + 1u)) - 1u);
    return 1;
}

static int tree_decode_node(TreeReader *r, uint32_t left, uint32_t right,
                            uint32_t mask[8]) {
    if (!r || !mask || left >= right || right > COLI_SGGUF_GROUP_SIZE) return 0;

    unsigned first;
    if (!tree_get(r, &first)) return 0;
    if (first == 0) {
        /* A one-value interval cannot be split. Reject malformed trees instead
         * of recurring forever on [left,left). */
        if (right - left <= 1u) return 0;
        const uint32_t middle = left + (right - left) / 2u;
        return tree_decode_node(r, left, middle, mask) &&
               tree_decode_node(r, middle, right, mask);
    }

    unsigned retained;
    if (!tree_get(r, &retained)) return 0;
    return !retained || mask_set_range(mask, left, right);
}

static uint32_t popcount32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcount(x);
#else
    uint32_t n = 0; while (x) { x &= x - 1u; ++n; } return n;
#endif
}

int coli_sgguf_tree_decode(const uint8_t *tree, uint32_t tree_bits,
                           uint32_t occupancy[8], uint32_t *retained_out) {
    if (!tree || !occupancy || tree_bits < 2 || tree_bits > COLI_SGGUF_MAX_TREE_BYTES * 8u) return 0;
    memset(occupancy, 0, 8 * sizeof(*occupancy));
    TreeReader r = { tree, tree_bits, 0 };
    if (!tree_decode_node(&r, 0, COLI_SGGUF_GROUP_SIZE, occupancy)) return 0;
    if (r.pos != tree_bits) return 0;
    uint32_t retained = 0;
    for (int i = 0; i < 8; ++i) retained += popcount32(occupancy[i]);
    if (retained_out) *retained_out = retained;
    return 1;
}

void coli_sgguf_bitmap_from_keep(const uint8_t keep[COLI_SGGUF_GROUP_SIZE],
                                 uint8_t bitmap[COLI_SGGUF_BITMAP_BYTES]) {
    if (!bitmap) return;
    memset(bitmap, 0, COLI_SGGUF_BITMAP_BYTES);
    if (!keep) return;
    for (uint32_t i = 0; i < COLI_SGGUF_GROUP_SIZE; ++i)
        if (keep[i]) bitmap[i >> 3] |= (uint8_t)(1u << (i & 7u));
}

int coli_sgguf_bitmap_decode(const uint8_t bitmap[COLI_SGGUF_BITMAP_BYTES],
                             uint32_t occupancy[8], uint32_t *retained_out) {
    if (!bitmap || !occupancy) return 0;
    uint32_t retained = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        occupancy[i] = coli_sgguf_load_u32_le(bitmap + i * 4u);
        retained += popcount32(occupancy[i]);
    }
    if (retained_out) *retained_out = retained;
    return 1;
}

uint64_t coli_sgguf_sparse_block_offset(const ColiSggufSparseTensor *tensor,
                                        uint64_t block_index) {
    if (!tensor || !tensor->block_offsets_le || block_index > tensor->total_blocks)
        return UINT64_MAX;
    if (tensor->offset_width == 4)
        return coli_sgguf_load_u32_le(tensor->block_offsets_le + block_index * 4u);
    if (tensor->offset_width == 8)
        return coli_sgguf_load_u64_le(tensor->block_offsets_le + block_index * 8u);
    return UINT64_MAX;
}

int coli_sgguf_sparse_tensor_parse(const uint8_t *payload, uint64_t payload_size,
                                   ColiSggufSparseTensor *out,
                                   char *error, size_t error_size) {
    if (!payload || !out || payload_size < COLI_SGGUF_SPARSE_TENSOR_HEADER_BYTES)
        return sgguf_fail(error, error_size, "truncated sparse tensor header");

    ColiSggufSparseTensor t;
    memset(&t, 0, sizeof(t));
    t.payload = payload;
    t.payload_size = payload_size;

    if (memcmp(payload, "SPT1", 4) == 0) {
        uint32_t version = coli_sgguf_load_u32_le(payload + 4);
        if (version != 1)
            return sgguf_fail(error, error_size, "unsupported SPT1 version %u", version);
        t.layout = COLI_SGGUF_LAYOUT_TREE_V1;
        t.offset_width = 8;
    } else if (memcmp(payload, "SPB2", 4) == 0) {
        uint32_t version = coli_sgguf_load_u32_le(payload + 4);
        if (version != 2)
            return sgguf_fail(error, error_size, "unsupported SPB2 version %u", version);
        t.layout = COLI_SGGUF_LAYOUT_BITMAP_V2;
        t.offset_width = 4;
        t.auxiliary_bytes_per_block = coli_sgguf_load_u16_le(payload + 72);
        t.retained_value_bits = coli_sgguf_load_u16_le(payload + 74);
    } else if (memcmp(payload, "SPB3", 4) == 0) {
        uint32_t version = coli_sgguf_load_u32_le(payload + 4);
        if (version != 3)
            return sgguf_fail(error, error_size, "unsupported SPB3 version %u", version);
        t.layout = COLI_SGGUF_LAYOUT_BITMAP_V3;
        t.offset_width = 4;
        t.auxiliary_bytes_per_block = coli_sgguf_load_u16_le(payload + 72);
        t.retained_value_bits = coli_sgguf_load_u16_le(payload + 74);
    } else {
        return sgguf_fail(error, error_size, "bad sparse tensor magic");
    }

    t.codec_id = coli_sgguf_load_u32_le(payload + 8);
    t.group_size = coli_sgguf_load_u32_le(payload + 12);
    t.cols = coli_sgguf_load_u32_le(payload + 16);
    t.rows_per_expert = coli_sgguf_load_u32_le(payload + 20);
    t.expert_count = coli_sgguf_load_u32_le(payload + 24);
    t.blocks_per_row = coli_sgguf_load_u32_le(payload + 28);
    t.total_rows = coli_sgguf_load_u64_le(payload + 32);
    t.total_blocks = coli_sgguf_load_u64_le(payload + 40);
    uint64_t offsets_offset = coli_sgguf_load_u64_le(payload + 48);
    uint64_t blocks_offset = coli_sgguf_load_u64_le(payload + 56);
    uint64_t encoded_size = coli_sgguf_load_u64_le(payload + 64);

    const uint32_t expected_blocks_per_row = t.layout == COLI_SGGUF_LAYOUT_BITMAP_V3
        ? (t.cols + t.group_size - 1u) / t.group_size
        : (t.cols / t.group_size);
    if (t.group_size != COLI_SGGUF_GROUP_SIZE || !t.cols ||
        (t.layout != COLI_SGGUF_LAYOUT_BITMAP_V3 && t.cols % t.group_size) ||
        !t.rows_per_expert || !t.expert_count ||
        t.total_rows != (uint64_t)t.rows_per_expert * t.expert_count ||
        t.blocks_per_row != expected_blocks_per_row ||
        t.total_blocks != t.total_rows * t.blocks_per_row)
        return sgguf_fail(error, error_size, "invalid sparse tensor dimensions/index counts");
    if (encoded_size != payload_size)
        return sgguf_fail(error, error_size, "sparse tensor payload size mismatch");

    if (t.layout == COLI_SGGUF_LAYOUT_BITMAP_V2 ||
        t.layout == COLI_SGGUF_LAYOUT_BITMAP_V3) {
        uint16_t expected_aux = coli_sgguf_codec_aux_bytes(t.codec_id);
        uint16_t expected_bits = coli_sgguf_codec_value_bits(t.codec_id);
        if (expected_aux == UINT16_MAX || !expected_bits ||
            t.auxiliary_bytes_per_block != expected_aux ||
            t.retained_value_bits != expected_bits)
            return sgguf_fail(error, error_size, "SPB2 codec/header mismatch");
    }

    if (t.total_blocks > (UINT64_MAX / t.offset_width) - 1u)
        return sgguf_fail(error, error_size, "sparse tensor offset count overflow");
    uint64_t offsets_bytes = (t.total_blocks + 1u) * t.offset_width;
    if (offsets_offset > payload_size || offsets_bytes > payload_size - offsets_offset ||
        blocks_offset > payload_size || blocks_offset < offsets_offset + offsets_bytes)
        return sgguf_fail(error, error_size, "sparse tensor indexes exceed payload");

    t.block_offsets_le = payload + offsets_offset;
    t.blocks = payload + blocks_offset;
    t.blocks_size = payload_size - blocks_offset;

    uint64_t first = coli_sgguf_sparse_block_offset(&t, 0);
    uint64_t last = coli_sgguf_sparse_block_offset(&t, t.total_blocks);
    if (first != 0 || last != t.blocks_size)
        return sgguf_fail(error, error_size, "sparse block offset boundary mismatch");
    *out = t;
    if (error && error_size) error[0] = 0;
    return 1;
}

int coli_sgguf_sparse_block_get(const ColiSggufSparseTensor *tensor,
                                uint64_t block_index,
                                ColiSggufSparseBlock *out,
                                char *error, size_t error_size) {
    if (!tensor || !out || block_index >= tensor->total_blocks)
        return sgguf_fail(error, error_size, "invalid sparse block index");
    uint64_t begin = coli_sgguf_sparse_block_offset(tensor, block_index);
    uint64_t end = coli_sgguf_sparse_block_offset(tensor, block_index + 1u);
    if (begin == UINT64_MAX || end == UINT64_MAX || begin > end || end > tensor->blocks_size)
        return sgguf_fail(error, error_size, "corrupt sparse block offsets");

    const uint8_t *p = tensor->blocks + begin;
    ColiSggufSparseBlock b;
    memset(&b, 0, sizeof(b));
    b.layout = tensor->layout;
    b.codec_id = tensor->codec_id;
    b.encoded_block_bytes = (uint32_t)(end - begin);
    {
        const uint32_t block_in_row = (uint32_t)(block_index % tensor->blocks_per_row);
        const uint64_t first_col = (uint64_t)block_in_row * tensor->group_size;
        uint64_t remain = tensor->cols - first_col;
        if (remain > tensor->group_size) remain = tensor->group_size;
        b.logical_count = (uint16_t)remain;
    }

    if (tensor->layout == COLI_SGGUF_LAYOUT_BITMAP_V2 ||
        tensor->layout == COLI_SGGUF_LAYOUT_BITMAP_V3) {
        if (end - begin < COLI_SGGUF_BITMAP_BYTES + tensor->auxiliary_bytes_per_block)
            return sgguf_fail(error, error_size, "truncated SPB2 block");
        b.bitmap = p;
        b.auxiliary_bytes = tensor->auxiliary_bytes_per_block;
        b.retained_value_bits = tensor->retained_value_bits;
        b.auxiliary = p + COLI_SGGUF_BITMAP_BYTES;
        b.retained_values = b.auxiliary + b.auxiliary_bytes;
        uint32_t occupancy[8], retained = 0;
        if (!coli_sgguf_bitmap_decode(b.bitmap, occupancy, &retained))
            return sgguf_fail(error, error_size, "invalid SPB2 bitmap");
        if (b.logical_count < COLI_SGGUF_GROUP_SIZE) {
            for (uint32_t i = b.logical_count; i < COLI_SGGUF_GROUP_SIZE; ++i) {
                if (b.bitmap[i >> 3] & (uint8_t)(1u << (i & 7u)))
                    return sgguf_fail(error, error_size, "SPB3 tail bitmap contains out-of-row value");
            }
        }
        b.retained_count = (uint16_t)retained;
        uint64_t value_bytes = ((uint64_t)retained * b.retained_value_bits + 7u) / 8u;
        uint64_t expected = COLI_SGGUF_BITMAP_BYTES + b.auxiliary_bytes + value_bytes;
        if (expected != end - begin || value_bytes > UINT32_MAX)
            return sgguf_fail(error, error_size, "SPB2 block length mismatch");
        b.retained_payload_bytes = (uint32_t)value_bytes;
    } else {
        if (end - begin < COLI_SGGUF_SPARSE_BLOCK_HEADER_BYTES)
            return sgguf_fail(error, error_size, "truncated SPT1 block");
        b.tree_bits = coli_sgguf_load_u16_le(p + 0);
        b.retained_count = coli_sgguf_load_u16_le(p + 2);
        b.auxiliary_bytes = coli_sgguf_load_u16_le(p + 4);
        b.retained_value_bits = coli_sgguf_load_u16_le(p + 6);
        b.retained_payload_bytes = coli_sgguf_load_u32_le(p + 8);
        b.encoded_block_bytes = coli_sgguf_load_u32_le(p + 12);
        if (b.encoded_block_bytes != end - begin || b.tree_bits < 2 ||
            b.tree_bits > COLI_SGGUF_MAX_TREE_BYTES * 8u || b.retained_count > 256)
            return sgguf_fail(error, error_size, "invalid SPT1 block header");
        uint64_t tree_bytes = ((uint64_t)b.tree_bits + 7u) / 8u;
        uint64_t need = COLI_SGGUF_SPARSE_BLOCK_HEADER_BYTES + tree_bytes +
                        b.auxiliary_bytes + b.retained_payload_bytes;
        if (need > b.encoded_block_bytes)
            return sgguf_fail(error, error_size, "SPT1 block payload is truncated");
        b.tree = p + COLI_SGGUF_SPARSE_BLOCK_HEADER_BYTES;
        b.auxiliary = b.tree + tree_bytes;
        b.retained_values = b.auxiliary + b.auxiliary_bytes;
        uint32_t occupancy[8], retained = 0;
        if (!coli_sgguf_tree_decode(b.tree, b.tree_bits, occupancy, &retained) ||
            retained != b.retained_count)
            return sgguf_fail(error, error_size, "SPT1 retained-count mismatch");
        uint64_t expected_bits = (uint64_t)b.retained_count * b.retained_value_bits;
        if ((expected_bits + 7u) / 8u != b.retained_payload_bytes)
            return sgguf_fail(error, error_size, "SPT1 retained payload size mismatch");
    }

    *out = b;
    if (error && error_size) error[0] = 0;
    return 1;
}

static int sparse_block_occupancy(const ColiSggufSparseBlock *block,
                                  uint32_t occupancy[8], uint32_t *retained) {
    if (!block) return 0;
    if (block->layout == COLI_SGGUF_LAYOUT_BITMAP_V2 ||
        block->layout == COLI_SGGUF_LAYOUT_BITMAP_V3)
        return coli_sgguf_bitmap_decode(block->bitmap, occupancy, retained);
    return coli_sgguf_tree_decode(block->tree, block->tree_bits, occupancy, retained);
}

static uint32_t read_packed_bits(const uint8_t *p, uint32_t bit_offset, uint32_t bits) {
    uint32_t byte = bit_offset >> 3;
    uint32_t shift = bit_offset & 7u;
    uint32_t need = (shift + bits + 7u) >> 3;
    uint64_t v = 0;
    for (uint32_t i = 0; i < need; ++i) v |= (uint64_t)p[byte + i] << (8u * i);
    return (uint32_t)((v >> shift) & ((1ull << bits) - 1ull));
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static int8_t q3_k_scale(const uint8_t packed[12], uint32_t index) {
    uint32_t aux[4] = {0, 0, 0, 0};
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    memcpy(aux, packed, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    return (int8_t)((const uint8_t *)aux)[index] - 32;
}

static float sparse_value(const ColiSggufSparseBlock *b,
                          uint32_t dense_position, uint32_t retained_index,
                          int *ok) {
    if (ok) *ok = 1;
    switch (b->codec_id) {
        case COLI_SGGUF_CODEC_RETAINED_F16: {
            uint16_t h = coli_sgguf_load_u16_le(b->retained_values + retained_index * 2u);
            return coli_fp16_to_fp32(h);
        }
        case COLI_SGGUF_CODEC_RETAINED_BF16: {
            uint16_t h = coli_sgguf_load_u16_le(b->retained_values + retained_index * 2u);
            return coli_bf16_to_fp32(h);
        }
        case COLI_SGGUF_CODEC_RETAINED_F32: {
            uint32_t u = coli_sgguf_load_u32_le(b->retained_values + retained_index * 4u);
            float f; memcpy(&f, &u, sizeof(f)); return f;
        }
        case COLI_SGGUF_CODEC_Q4_0_EXACT: {
            if (b->auxiliary_bytes != 16 || b->retained_value_bits != 4) break;
            uint32_t native = dense_position >> 5;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 4u, 4u);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary + native * 2u));
            return d * ((int)code - 8);
        }
        case COLI_SGGUF_CODEC_Q4_1_EXACT: {
            if (b->auxiliary_bytes != 32 || b->retained_value_bits != 4) break;
            uint32_t native = dense_position >> 5;
            const uint8_t *aux = b->auxiliary + native * 4u;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 4u, 4u);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(aux));
            float m = coli_fp16_to_fp32(coli_sgguf_load_u16_le(aux + 2));
            return d * code + m;
        }
        case COLI_SGGUF_CODEC_Q5_0_EXACT: {
            if (b->auxiliary_bytes != 16 || b->retained_value_bits != 5) break;
            uint32_t native = dense_position >> 5;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 5u, 5u);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary + native * 2u));
            return d * ((int)code - 16);
        }
        case COLI_SGGUF_CODEC_Q5_1_EXACT: {
            if (b->auxiliary_bytes != 32 || b->retained_value_bits != 5) break;
            uint32_t native = dense_position >> 5;
            const uint8_t *aux = b->auxiliary + native * 4u;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 5u, 5u);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(aux));
            float m = coli_fp16_to_fp32(coli_sgguf_load_u16_le(aux + 2));
            return d * code + m;
        }
        case COLI_SGGUF_CODEC_Q8_0_EXACT:
        case COLI_SGGUF_CODEC_Q8_1_EXACT: {
            uint16_t expected_aux = b->codec_id == COLI_SGGUF_CODEC_Q8_0_EXACT ? 16 : 32;
            uint32_t stride = b->codec_id == COLI_SGGUF_CODEC_Q8_0_EXACT ? 2u : 4u;
            if (b->auxiliary_bytes != expected_aux || b->retained_value_bits != 8) break;
            uint32_t native = dense_position >> 5;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 8u, 8u);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary + native * stride));
            return d * (int8_t)code;
        }
        case COLI_SGGUF_CODEC_Q3_K_EXACT: {
            if (b->auxiliary_bytes != 14 || b->retained_value_bits != 3) break;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 3u, 3u);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary + 12));
            return d * q3_k_scale(b->auxiliary, dense_position >> 4) * ((int)code - 4);
        }
        case COLI_SGGUF_CODEC_Q6_K_EXACT: {
            if (b->auxiliary_bytes != 18 || b->retained_value_bits != 6) break;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 6u, 6u);
            uint32_t half = dense_position >> 7;
            uint32_t r = dense_position & 127u;
            uint32_t group = r >> 5;
            uint32_t local = r & 31u;
            uint32_t scale_index = half * 8u + group * 2u + local / 16u;
            int8_t scale = (int8_t)b->auxiliary[scale_index];
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary + 16));
            return d * scale * ((int)code - 32);
        }
        case COLI_SGGUF_CODEC_Q4_K_EXACT: {
            if (b->auxiliary_bytes != 16 || b->retained_value_bits != 4) break;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 4u, 4u);
            uint32_t subgroup = dense_position / 32u;
            uint8_t sc = 0, m = 0;
            get_scale_min_k4((int)subgroup, b->auxiliary + 4, &sc, &m);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary));
            float dmin = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary + 2));
            return d * sc * code - dmin * m;
        }
        case COLI_SGGUF_CODEC_Q5_K_EXACT: {
            if (b->auxiliary_bytes != 16 || b->retained_value_bits != 5) break;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 5u, 5u);
            uint32_t subgroup = dense_position / 32u;
            uint8_t sc = 0, m = 0;
            get_scale_min_k4((int)subgroup, b->auxiliary + 4, &sc, &m);
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary));
            float dmin = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary + 2));
            return d * sc * code - dmin * m;
        }
        case COLI_SGGUF_CODEC_Q8_K_EXACT: {
            if (b->auxiliary_bytes != 4 || b->retained_value_bits != 8) break;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 8u, 8u);
            uint32_t u = coli_sgguf_load_u32_le(b->auxiliary);
            float d; memcpy(&d, &u, sizeof(d));
            return d * (int8_t)code;
        }
        case COLI_SGGUF_CODEC_MXFP4_EXACT: {
            if (b->auxiliary_bytes != 8 || b->retained_value_bits != 4) break;
            uint32_t native = dense_position >> 5;
            if (native >= 8u) break;
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 4u, 4u);
            return coli_mxfp4_code_to_fp32(b->auxiliary[native], (uint8_t)code);
        }
        case COLI_SGGUF_CODEC_IQ4_NL_EXACT: {
            if (b->auxiliary_bytes != 16 || b->retained_value_bits != 4) break;
            uint32_t native = dense_position >> 5;
            if (native >= 8u) break;
            float d = coli_fp16_to_fp32(
                coli_sgguf_load_u16_le(b->auxiliary + native * 2u));
            uint32_t code = read_packed_bits(
                b->retained_values, retained_index * 4u, 4u);
            static const int8_t iq4nl_values[16] = {
                -127, -104, -83, -65, -49, -35, -22, -10,
                   1,   13,  25,  38,  53,  69,  89, 113,
            };
            return d * (float)iq4nl_values[code & 15u];
        }
        case COLI_SGGUF_CODEC_IQ4_XS_EXACT: {
            if (b->auxiliary_bytes != 8 || b->retained_value_bits != 4) break;
            uint32_t subgroup = dense_position >> 5;
            if (subgroup >= 8u) break;
            float d = coli_fp16_to_fp32(coli_sgguf_load_u16_le(b->auxiliary));
            uint16_t scales_h = coli_sgguf_load_u16_le(b->auxiliary + 2);
            uint8_t scales_l = b->auxiliary[4u + (subgroup >> 1)];
            uint32_t low = (scales_l >> (4u * (subgroup & 1u))) & 15u;
            uint32_t high = (scales_h >> (2u * subgroup)) & 3u;
            float dl = d * (float)((int)(low | (high << 4)) - 32);
            uint32_t code = read_packed_bits(b->retained_values, retained_index * 4u, 4u);
            static const int8_t iq4nl_values[16] = {
                -127, -104, -83, -65, -49, -35, -22, -10,
                   1,   13,  25,  38,  53,  69,  89, 113,
            };
            return dl * (float)iq4nl_values[code & 15u];
        }
        default: break;
    }
    if (ok) *ok = 0;
    return 0.0f;
}

float coli_sgguf_sparse_block_dot_f32(const ColiSggufSparseBlock *block,
                                      const float *x, int *ok) {
    if (ok) *ok = 0;
    if (!block || !x) return 0.0f;
    uint32_t mask[8], retained = 0;
    if (!sparse_block_occupancy(block, mask, &retained) ||
        retained != block->retained_count) return 0.0f;
    uint32_t k = 0;
    double sum = 0.0;
    for (uint32_t w = 0; w < 8; ++w) {
        uint32_t bits = mask[w];
        while (bits) {
#if defined(__GNUC__) || defined(__clang__)
            uint32_t bit = (uint32_t)__builtin_ctz(bits);
#else
            uint32_t bit = 0; while (((bits >> bit) & 1u) == 0) ++bit;
#endif
            uint32_t dense = w * 32u + bit;
            uint32_t logical_count = block->logical_count ? block->logical_count : COLI_SGGUF_GROUP_SIZE;
            if (dense >= logical_count) return 0.0f;
            int value_ok = 0;
            float value = sparse_value(block, dense, k++, &value_ok);
            if (!value_ok) return 0.0f;
            sum += (double)value * x[dense];
            bits &= bits - 1u;
        }
    }
    if (k != block->retained_count) return 0.0f;
    if (ok) *ok = 1;
    return (float)sum;
}

int coli_sgguf_sparse_block_materialize_f32(const ColiSggufSparseBlock *block,
                                            float output[COLI_SGGUF_GROUP_SIZE]) {
    if (!block || !output) return 0;
    memset(output, 0, COLI_SGGUF_GROUP_SIZE * sizeof(*output));
    uint32_t mask[8], retained = 0;
    if (!sparse_block_occupancy(block, mask, &retained) ||
        retained != block->retained_count) return 0;
    uint32_t k = 0;
    for (uint32_t w = 0; w < 8; ++w) {
        uint32_t bits = mask[w];
        while (bits) {
#if defined(__GNUC__) || defined(__clang__)
            uint32_t bit = (uint32_t)__builtin_ctz(bits);
#else
            uint32_t bit = 0; while (((bits >> bit) & 1u) == 0) ++bit;
#endif
            uint32_t dense = w * 32u + bit;
            uint32_t logical_count = block->logical_count ? block->logical_count : COLI_SGGUF_GROUP_SIZE;
            if (dense >= logical_count) return 0.0f;
            int value_ok = 0;
            output[dense] = sparse_value(block, dense, k++, &value_ok);
            if (!value_ok) return 0;
            bits &= bits - 1u;
        }
    }
    return k == block->retained_count;
}
