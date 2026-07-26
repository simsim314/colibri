#ifndef COLIBRI_SGGUF_H
#define COLIBRI_SGGUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_SGGUF_MAGIC "SGUF"
#define COLI_SGGUF_VERSION 1u
#define COLI_SGGUF_GROUP_SIZE 256u
#define COLI_SGGUF_BITMAP_BYTES 32u
#define COLI_SGGUF_SPARSE_TENSOR_HEADER_BYTES 80u
#define COLI_SGGUF_SPARSE_BLOCK_HEADER_BYTES 16u /* legacy SPT1 only */
#define COLI_SGGUF_MAX_TREE_BYTES 128u

/* Tensor storage kind recorded in an SGGUF tensor directory entry.  The
 * historical TREE name is retained for source compatibility; SPT2 payloads
 * use a fixed 256-bit bitmap and contain no tree. */
typedef enum {
    COLI_SGGUF_STORAGE_DENSE = 0,
    COLI_SGGUF_STORAGE_SPARSE_TREE = 1,
} ColiSggufStorageKind;

typedef enum {
    COLI_SGGUF_LAYOUT_TREE_V1 = 1,
    COLI_SGGUF_LAYOUT_BITMAP_V2 = 2,
    COLI_SGGUF_LAYOUT_BITMAP_V3 = 3,
} ColiSggufSparseLayout;

/* Sparse payload codecs. The logical GGML tensor type remains in the tensor
 * directory. A codec describes only the retained-value stream. */
typedef enum {
    COLI_SGGUF_CODEC_NONE = 0,
    COLI_SGGUF_CODEC_RETAINED_F16 = 1,
    COLI_SGGUF_CODEC_Q6_K_EXACT = 2,
    COLI_SGGUF_CODEC_Q4_K_EXACT = 3,
    COLI_SGGUF_CODEC_RETAINED_F32 = 4,
    COLI_SGGUF_CODEC_RETAINED_BF16 = 5,
    COLI_SGGUF_CODEC_Q4_0_EXACT = 6,
    COLI_SGGUF_CODEC_Q4_1_EXACT = 7,
    COLI_SGGUF_CODEC_Q5_0_EXACT = 8,
    COLI_SGGUF_CODEC_Q5_1_EXACT = 9,
    COLI_SGGUF_CODEC_Q8_0_EXACT = 10,
    COLI_SGGUF_CODEC_Q8_1_EXACT = 11,
    COLI_SGGUF_CODEC_Q3_K_EXACT = 12,
    COLI_SGGUF_CODEC_Q5_K_EXACT = 13,
    COLI_SGGUF_CODEC_Q8_K_EXACT = 14,
    COLI_SGGUF_CODEC_MXFP4_EXACT = 15,
    COLI_SGGUF_CODEC_IQ4_XS_EXACT = 16,
} ColiSggufCodecId;

typedef enum {
    COLI_SGGUF_MOE_NONE = 0,
    COLI_SGGUF_MOE_GATE = 1,
    COLI_SGGUF_MOE_UP = 2,
    COLI_SGGUF_MOE_DOWN = 3,
} ColiSggufMoeProjection;

typedef struct {
    uint8_t data[COLI_SGGUF_MAX_TREE_BYTES];
    uint32_t bit_count;
} ColiSggufTreeBits;

typedef struct {
    const uint8_t *payload;
    uint64_t payload_size;
    uint32_t layout;
    uint32_t codec_id;
    uint32_t group_size;
    uint32_t cols;
    uint32_t rows_per_expert;
    uint32_t expert_count;
    uint32_t blocks_per_row;
    uint64_t total_rows;
    uint64_t total_blocks;
    uint32_t offset_width;                 /* 8 for SPT1, 4 for SPB2 */
    uint16_t auxiliary_bytes_per_block;    /* SPB2 fixed codec metadata */
    uint16_t retained_value_bits;          /* SPB2 fixed retained code width */
    const uint8_t *block_offsets_le;
    const uint8_t *blocks;
    uint64_t blocks_size;
} ColiSggufSparseTensor;

typedef struct {
    uint32_t layout;
    uint32_t codec_id;
    uint16_t tree_bits;                    /* legacy SPT1 only */
    uint16_t retained_count;
    uint16_t auxiliary_bytes;
    uint16_t retained_value_bits;
    uint16_t logical_count;               /* 1..256; SPB3 tail groups may be short */
    uint32_t retained_payload_bytes;
    uint32_t encoded_block_bytes;
    const uint8_t *tree;                   /* legacy SPT1 only */
    const uint8_t *bitmap;                 /* SPB2: exactly 32 bytes */
    const uint8_t *auxiliary;
    const uint8_t *retained_values;
} ColiSggufSparseBlock;

/* Legacy tree helpers remain available for reading old SPT1 files. */
int coli_sgguf_tree_encode(const uint8_t keep[COLI_SGGUF_GROUP_SIZE],
                           ColiSggufTreeBits *out);
int coli_sgguf_tree_decode(const uint8_t *tree, uint32_t tree_bits,
                           uint32_t occupancy[8], uint32_t *retained_out);

void coli_sgguf_bitmap_from_keep(const uint8_t keep[COLI_SGGUF_GROUP_SIZE],
                                 uint8_t bitmap[COLI_SGGUF_BITMAP_BYTES]);
int coli_sgguf_bitmap_decode(const uint8_t bitmap[COLI_SGGUF_BITMAP_BYTES],
                             uint32_t occupancy[8], uint32_t *retained_out);

int coli_sgguf_sparse_tensor_parse(const uint8_t *payload, uint64_t payload_size,
                                   ColiSggufSparseTensor *out,
                                   char *error, size_t error_size);
int coli_sgguf_sparse_block_get(const ColiSggufSparseTensor *tensor,
                                uint64_t block_index,
                                ColiSggufSparseBlock *out,
                                char *error, size_t error_size);
uint64_t coli_sgguf_sparse_block_offset(const ColiSggufSparseTensor *tensor,
                                        uint64_t block_index);

/* Direct sparse operations. No dense 256-value block is constructed. */
float coli_sgguf_sparse_block_dot_f32(const ColiSggufSparseBlock *block,
                                      const float *x, int *ok);
int coli_sgguf_sparse_block_materialize_f32(const ColiSggufSparseBlock *block,
                                            float output[COLI_SGGUF_GROUP_SIZE]);

const char *coli_sgguf_codec_name(uint32_t codec_id);
uint16_t coli_sgguf_codec_aux_bytes(uint32_t codec_id);
uint16_t coli_sgguf_codec_value_bits(uint32_t codec_id);
uint64_t coli_sgguf_load_u64_le(const uint8_t *p);
uint32_t coli_sgguf_load_u32_le(const uint8_t *p);
uint16_t coli_sgguf_load_u16_le(const uint8_t *p);
void coli_sgguf_store_u64_le(uint8_t *p, uint64_t v);
void coli_sgguf_store_u32_le(uint8_t *p, uint32_t v);
void coli_sgguf_store_u16_le(uint8_t *p, uint16_t v);

#ifdef __cplusplus
}
#endif

#endif
