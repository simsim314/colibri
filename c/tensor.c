#include "tensor.h"
#include "ggml_quants.h"

#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *error, size_t cap, const char *fmt, ...) {
    if (error && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(error, cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

int coli_tensor_bind_gguf(const ColiGgufFile *file,
                          const ColiGgufTensorInfo *info,
                          ColiTensor *out,
                          char *error, size_t error_size) {
    if (!file || !info || !out) return fail(error, error_size, "invalid tensor binding arguments");
    memset(out, 0, sizeof(*out));

    const ColiDTypeTraits *traits = coli_dtype_traits((ColiDType)info->type);
    if (!traits) return fail(error, error_size, "tensor '%s': unsupported dtype %u", info->name, info->type);

    uint64_t elements = 0, bytes = 0, row_bytes = 0;
    if (!coli_dtype_tensor_size((ColiDType)info->type, info->dims, info->n_dims,
                                &elements, &bytes) ||
        !coli_dtype_row_size((ColiDType)info->type, info->dims[0], &row_bytes)) {
        return fail(error, error_size, "tensor '%s': invalid dimensions for %s", info->name, traits->name);
    }
    if (info->absolute_offset > file->file_size || bytes > file->file_size - info->absolute_offset)
        return fail(error, error_size, "tensor '%s': storage exceeds GGUF file", info->name);

    const uint8_t *data = (const uint8_t *)coli_gguf_mapped_at(file, info->absolute_offset, bytes);
    if (!data) return fail(error, error_size,
                           "tensor '%s': GGUF is not memory mapped (64-bit mmap required)", info->name);

    out->name = info->name;
    out->dtype = (ColiDType)info->type;
    out->n_dims = info->n_dims;
    memcpy(out->dims, info->dims, sizeof(out->dims));
    out->element_count = elements;
    out->row_count = elements / info->dims[0];
    out->row_bytes = row_bytes;
    out->storage_bytes = bytes;
    out->data = data;
    out->source_offset = info->absolute_offset;
    out->mmap_backed = 1;
    if (error && error_size) error[0] = 0;
    return 1;
}

int coli_tensor_rows_view(const ColiTensor *tensor,
                          uint64_t first_row, uint64_t row_count,
                          ColiTensor *out) {
    if (!tensor || !out || !tensor->data || !row_count ||
        first_row > tensor->row_count || row_count > tensor->row_count - first_row)
        return 0;
    if (first_row > UINT64_MAX / tensor->row_bytes) return 0;
    uint64_t byte_offset = first_row * tensor->row_bytes;
    if (row_count > UINT64_MAX / tensor->row_bytes) return 0;
    uint64_t bytes = row_count * tensor->row_bytes;
    if (byte_offset > tensor->storage_bytes || bytes > tensor->storage_bytes - byte_offset) return 0;

    memset(out, 0, sizeof(*out));
    out->name = tensor->name;
    out->dtype = tensor->dtype;
    out->n_dims = 2;
    out->dims[0] = tensor->dims[0];
    out->dims[1] = row_count;
    out->element_count = tensor->dims[0] * row_count;
    out->row_count = row_count;
    out->row_bytes = tensor->row_bytes;
    out->storage_bytes = bytes;
    out->data = tensor->data + byte_offset;
    out->source_offset = tensor->source_offset + byte_offset;
    out->mmap_backed = tensor->mmap_backed;
#ifdef COLI_CUDA
    if (tensor->cuda) {
        if (!coli_cuda_tensor_view_rows(tensor->cuda, first_row, row_count, &out->cuda)) {
            memset(out, 0, sizeof(*out));
            return 0;
        }
        out->cuda_device = tensor->cuda_device;
        out->owns_cuda = 1;
    }
#endif
    return 1;
}

void coli_tensor_destroy(ColiTensor *tensor) {
    if (!tensor) return;
#ifdef COLI_CUDA
    if (tensor->cuda && tensor->owns_cuda) coli_cuda_tensor_free(tensor->cuda);
#endif
    if (tensor->owns_data) free((void *)tensor->data);
    memset(tensor, 0, sizeof(*tensor));
}

int coli_tensor_reside(const ColiExec *exec, ColiTensor *tensor) {
    if (!tensor || !tensor->data) return 0;
    if (!exec || exec->kind == COLI_BACKEND_CPU) return 1;
#ifdef COLI_CUDA
    if (exec->kind == COLI_BACKEND_CUDA) {
        if (tensor->cuda) return tensor->cuda_device == exec->device;
        if (tensor->dims[0] > INT_MAX || tensor->row_count > INT_MAX) return 0;
        if (!coli_cuda_tensor_upload_ggml(&tensor->cuda, tensor->data,
                (uint32_t)tensor->dtype, tensor->storage_bytes,
                (int)tensor->dims[0], (int)tensor->row_count, exec->device)) return 0;
        tensor->cuda_device = exec->device;
        tensor->owns_cuda = 1;
        return 1;
    }
#endif
    return 0;
}

int coli_tensor_is_resident(const ColiExec *exec, const ColiTensor *tensor) {
    if (!tensor) return 0;
    if (!exec || exec->kind == COLI_BACKEND_CPU) return tensor->data != NULL;
#ifdef COLI_CUDA
    if (exec->kind == COLI_BACKEND_CUDA)
        return tensor->cuda && tensor->cuda_device == exec->device;
#endif
    return 0;
}

const void *coli_tensor_device_data(const ColiTensor *tensor) {
#ifdef COLI_CUDA
    return tensor && tensor->cuda ? coli_cuda_tensor_data(tensor->cuda) : NULL;
#else
    (void)tensor; return NULL;
#endif
}

int coli_tensor_read_row_f32(const ColiTensor *tensor, uint64_t row,
                             float *output, uint64_t output_count) {
    if (!tensor || !tensor->data || !output || row >= tensor->row_count ||
        output_count < tensor->dims[0]) return 0;
    const uint8_t *encoded = tensor->data + row * tensor->row_bytes;
    return coli_dtype_dequantize_row(tensor->dtype, encoded, tensor->dims[0], output);
}

static int matmul_cpu(float *y, const float *x, const ColiTensor *w,
                      int S, int I, int O) {
    if (!y || !x || !w || !w->data || S < 1 || I < 1 || O < 1 ||
        w->dims[0] != (uint64_t)I || w->row_count != (uint64_t)O) return 0;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
        const uint8_t *row = w->data + (uint64_t)o * w->row_bytes;
        for (int s = 0; s < S; ++s) {
            y[(int64_t)s * O + o] = coli_dtype_dot_f32(w->dtype, row,
                                                       x + (int64_t)s * I,
                                                       (uint64_t)I);
        }
    }
    return 1;
}

int coli_tensor_matmul(const ColiExec *exec,
                       float *y, const float *x,
                       ColiTensor *weights,
                       int S, int I, int O) {
    ColiBackendKind kind = exec ? exec->kind : COLI_BACKEND_CPU;
    if (kind == COLI_BACKEND_CUDA) {
#ifdef COLI_CUDA
        if (!coli_tensor_reside(exec, weights)) return 0;
        if (coli_cuda_tensor_matmul_host(weights->cuda, y, x, S)) return 1;
#endif
        return 0;
    }
    return matmul_cpu(y, x, weights, S, I, O);
}

int coli_tensor_matmul_device(const ColiExec *exec, float *y_dev,
                       const float *x_dev, ColiTensor *weights, int S) {
    if (!exec || exec->kind != COLI_BACKEND_CUDA || !y_dev || !x_dev || !weights || S < 1) return 0;
#ifdef COLI_CUDA
    return coli_tensor_reside(exec, weights) &&
           coli_cuda_pipe_gemm(weights->cuda, y_dev, x_dev, S);
#else
    return 0;
#endif
}

int coli_tensor_read_row_device(const ColiExec *exec, ColiTensor *tensor,
                       uint64_t row, float *out_dev, float scale) {
    if (!exec || exec->kind != COLI_BACKEND_CUDA || !tensor || !out_dev || row >= tensor->row_count) return 0;
#ifdef COLI_CUDA
    return coli_tensor_reside(exec, tensor) &&
           coli_cuda_pipe_decode_row(tensor->cuda, row, out_dev, scale);
#else
    (void)scale; return 0;
#endif
}
