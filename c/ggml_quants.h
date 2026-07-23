#ifndef COLIBRI_GGML_QUANTS_H
#define COLIBRI_GGML_QUANTS_H

#include "ggml_types.h"
#include <stdint.h>

float coli_fp16_to_fp32(uint16_t h);
int coli_dtype_dequantize_row(ColiDType type, const void *encoded,
                              uint64_t element_count, float *output);
float coli_dtype_dot_f32(ColiDType type, const void *encoded,
                         const float *x, uint64_t element_count);

/* Compatibility wrappers. */
int coli_ggml_dequantize_row(uint32_t type, const void *encoded,
                             uint64_t element_count, float *output);

#endif
