#include "gguf_f32_loader.h"
#include "ggml_quants.h"
#include "ggml_types.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *error, uint64_t cap, const char *fmt, ...) {
    if (error && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(error, (size_t)cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

static char *copy_string(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

int coli_gguf_tensor_to_f32(const ColiGgufFile *file,
                            const ColiGgufTensorInfo *tensor,
                            ColiF32Tensor *out,
                            char *error, uint64_t error_size) {
    if (!file || !tensor || !out) return fail(error, error_size, "invalid tensor loader arguments");
    memset(out, 0, sizeof(*out));

    const ColiGgmlTypeTraits *traits = coli_ggml_type_traits(tensor->type);
    if (!traits) return fail(error, error_size, "tensor '%s': unsupported GGML type %u",
                             tensor->name, tensor->type);

    uint64_t elements, encoded_bytes, row_bytes;
    if (!coli_ggml_tensor_size(tensor->type, tensor->dims, tensor->n_dims,
                               &elements, &encoded_bytes) ||
        !coli_ggml_row_size(tensor->type, tensor->dims[0], &row_bytes)) {
        return fail(error, error_size, "tensor '%s': invalid dimensions for %s",
                    tensor->name, traits->name);
    }
    if (tensor->absolute_offset > file->file_size ||
        encoded_bytes > file->file_size - tensor->absolute_offset) {
        return fail(error, error_size, "tensor '%s': encoded data exceeds GGUF file", tensor->name);
    }
    if (elements > SIZE_MAX / sizeof(float)) {
        return fail(error, error_size, "tensor '%s': F32 allocation overflows", tensor->name);
    }
    if (row_bytes > SIZE_MAX) {
        return fail(error, error_size, "tensor '%s': encoded row is too large", tensor->name);
    }

    float *data = (float *)malloc((size_t)elements * sizeof(float));
    uint8_t *encoded = (uint8_t *)malloc((size_t)row_bytes);
    char *name = copy_string(tensor->name);
    if (!data || !encoded || !name) {
        free(data); free(encoded); free(name);
        return fail(error, error_size, "tensor '%s': out of memory", tensor->name);
    }

    uint64_t rows = elements / tensor->dims[0];
    for (uint64_t row = 0; row < rows; ++row) {
        uint64_t rel = row * row_bytes;
        if (!coli_gguf_read_tensor_bytes(file, tensor, rel, encoded, (size_t)row_bytes) ||
            !coli_ggml_dequantize_row(tensor->type, encoded, tensor->dims[0],
                                      data + row * tensor->dims[0])) {
            free(data); free(encoded); free(name);
            return fail(error, error_size, "tensor '%s': failed decoding row %llu",
                        tensor->name, (unsigned long long)row);
        }
    }
    free(encoded);

    out->name = name;
    out->n_dims = tensor->n_dims;
    memcpy(out->dims, tensor->dims, sizeof(out->dims));
    out->element_count = elements;
    out->data = data;
    if (error && error_size) error[0] = 0;
    return 1;
}

void coli_f32_tensor_destroy(ColiF32Tensor *tensor) {
    if (!tensor) return;
    free(tensor->name);
    free(tensor->data);
    memset(tensor, 0, sizeof(*tensor));
}
