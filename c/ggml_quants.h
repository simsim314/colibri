#ifndef COLIBRI_GGML_QUANTS_H
#define COLIBRI_GGML_QUANTS_H

#include <stdint.h>

float coli_fp16_to_fp32(uint16_t h);
int coli_ggml_dequantize_row(uint32_t type, const void *encoded,
                             uint64_t element_count, float *output);

#endif
