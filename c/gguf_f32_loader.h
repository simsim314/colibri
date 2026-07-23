#ifndef COLIBRI_GGUF_F32_LOADER_H
#define COLIBRI_GGUF_F32_LOADER_H

#include "gguf_reader.h"

#include <stdint.h>

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[COLI_GGUF_MAX_DIMS];
    uint64_t element_count;
    float *data;
} ColiF32Tensor;

int coli_gguf_tensor_to_f32(const ColiGgufFile *file,
                            const ColiGgufTensorInfo *tensor,
                            ColiF32Tensor *out,
                            char *error, uint64_t error_size);
void coli_f32_tensor_destroy(ColiF32Tensor *tensor);

#endif
