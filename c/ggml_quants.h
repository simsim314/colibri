#ifndef COLIBRI_GGML_QUANTS_H
#define COLIBRI_GGML_QUANTS_H

#include "ggml_types.h"
#include <stdint.h>

float coli_fp16_to_fp32(uint16_t h);
float coli_bf16_to_fp32(uint16_t h);
int coli_dtype_dequantize_row(ColiDType type, const void *encoded,
                              uint64_t element_count, float *output);
float coli_dtype_dot_f32(ColiDType type, const void *encoded,
                         const float *x, uint64_t element_count);

/* Compatibility wrappers. */
int coli_ggml_dequantize_row(uint32_t type, const void *encoded,
                             uint64_t element_count, float *output);

/* Replace representable encoded values whose decoded magnitude is below
 * threshold with an exact encoded zero, without changing the tensor dtype or
 * storage size. Currently supports F32/F16/BF16, Q6_K, Q8_0/Q8_1, and Q8_K.
 * changed_out receives the number of nonzero values changed to zero. */
int coli_dtype_zero_below_inplace(ColiDType type, void *encoded,
                                  uint64_t element_count, float threshold,
                                  uint64_t *changed_out);

#endif
