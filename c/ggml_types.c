#include "ggml_types.h"

#include <limits.h>
#include <stddef.h>

static const ColiDTypeTraits k_types[] = {
    { 0,  "F32",  1,   4,   0 },
    { 1,  "F16",  1,   2,   0 },
    { 2,  "Q4_0", 32,  18,  1 },
    { 3,  "Q4_1", 32,  20,  1 },
    { 6,  "Q5_0", 32,  22,  1 },
    { 7,  "Q5_1", 32,  24,  1 },
    { 8,  "Q8_0", 32,  34,  1 },
    { 9,  "Q8_1", 32,  36,  1 },
    { 11, "Q3_K", 256, 110, 1 },
    { 12, "Q4_K", 256, 144, 1 },
    { 13, "Q5_K", 256, 176, 1 },
    { 14, "Q6_K", 256, 210, 1 },
    { 15, "Q8_K", 256, 292, 1 },
    { 23, "IQ4_XS", 256, 136, 1 },
    { 30, "BF16", 1,   2,   0 },
    { 39, "MXFP4", 32,  17,  1 },
};

const ColiDTypeTraits *coli_dtype_traits(ColiDType type) {
    size_t n = sizeof(k_types) / sizeof(k_types[0]);
    for (size_t i = 0; i < n; ++i) {
        if (k_types[i].type == type) return &k_types[i];
    }
    return NULL;
}

int coli_dtype_row_size(ColiDType type, uint64_t element_count, uint64_t *size_out) {
    const ColiDTypeTraits *t = coli_dtype_traits(type);
    if (!t || !size_out) return 0;
    if (element_count == 0) {
        *size_out = 0;
        return 1;
    }
    if (element_count % t->block_values != 0) return 0;
    uint64_t blocks = element_count / t->block_values;
    if (blocks > UINT64_MAX / t->block_bytes) return 0;
    *size_out = blocks * t->block_bytes;
    return 1;
}

int coli_dtype_tensor_size(ColiDType type, const uint64_t *dims, uint32_t n_dims,
                           uint64_t *elements_out, uint64_t *size_out) {
    if (!dims || n_dims == 0 || !size_out) return 0;
    uint64_t elements = 1;
    for (uint32_t i = 0; i < n_dims; ++i) {
        if (!dims[i] || elements > UINT64_MAX / dims[i]) return 0;
        elements *= dims[i];
    }
    uint64_t row_bytes;
    if (!coli_dtype_row_size(type, dims[0], &row_bytes)) return 0;
    uint64_t rows = elements / dims[0];
    if (rows > UINT64_MAX / row_bytes) return 0;
    *size_out = rows * row_bytes;
    if (elements_out) *elements_out = elements;
    return 1;
}

const ColiGgmlTypeTraits *coli_ggml_type_traits(uint32_t type) {
    return coli_dtype_traits((ColiDType)type);
}

int coli_ggml_row_size(uint32_t type, uint64_t element_count, uint64_t *size_out) {
    return coli_dtype_row_size((ColiDType)type, element_count, size_out);
}

int coli_ggml_tensor_size(uint32_t type, const uint64_t *dims, uint32_t n_dims,
                          uint64_t *elements_out, uint64_t *size_out) {
    return coli_dtype_tensor_size((ColiDType)type, dims, n_dims, elements_out, size_out);
}
