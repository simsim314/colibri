#ifndef COLIBRI_GGML_TYPES_H
#define COLIBRI_GGML_TYPES_H

#include <stdint.h>

typedef struct {
    uint32_t type;
    const char *name;
    uint32_t block_values;
    uint32_t block_bytes;
    int quantized;
} ColiGgmlTypeTraits;

const ColiGgmlTypeTraits *coli_ggml_type_traits(uint32_t type);
int coli_ggml_row_size(uint32_t type, uint64_t element_count, uint64_t *size_out);
int coli_ggml_tensor_size(uint32_t type, const uint64_t *dims, uint32_t n_dims,
                          uint64_t *elements_out, uint64_t *size_out);

#endif
