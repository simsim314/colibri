#ifndef COLIBRI_GGML_TYPES_H
#define COLIBRI_GGML_TYPES_H

#include <stdint.h>

/* Native Colibri storage dtypes. Numeric values intentionally match GGML's
 * serialized tensor type IDs so a GGUF tensor can be bound without conversion. */
typedef enum {
    COLI_DTYPE_F32  = 0,
    COLI_DTYPE_F16  = 1,
    COLI_DTYPE_Q4_0 = 2,
    COLI_DTYPE_Q4_1 = 3,
    COLI_DTYPE_Q5_0 = 6,
    COLI_DTYPE_Q5_1 = 7,
    COLI_DTYPE_Q8_0 = 8,
    COLI_DTYPE_Q8_1 = 9,
    COLI_DTYPE_Q3_K = 11,
    COLI_DTYPE_Q4_K = 12,
    COLI_DTYPE_Q5_K = 13,
    COLI_DTYPE_Q6_K = 14,
    COLI_DTYPE_Q8_K = 15,
    COLI_DTYPE_BF16 = 30,
} ColiDType;

typedef struct {
    ColiDType type;
    const char *name;
    uint32_t block_values;
    uint32_t block_bytes;
    int quantized;
} ColiDTypeTraits;

/* Backward-compatible name retained for the GGUF parser/tests. */
typedef ColiDTypeTraits ColiGgmlTypeTraits;

const ColiDTypeTraits *coli_dtype_traits(ColiDType type);
int coli_dtype_row_size(ColiDType type, uint64_t element_count, uint64_t *size_out);
int coli_dtype_tensor_size(ColiDType type, const uint64_t *dims, uint32_t n_dims,
                           uint64_t *elements_out, uint64_t *size_out);

const ColiGgmlTypeTraits *coli_ggml_type_traits(uint32_t type);
int coli_ggml_row_size(uint32_t type, uint64_t element_count, uint64_t *size_out);
int coli_ggml_tensor_size(uint32_t type, const uint64_t *dims, uint32_t n_dims,
                          uint64_t *elements_out, uint64_t *size_out);

#endif
