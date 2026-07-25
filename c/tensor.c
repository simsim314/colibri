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
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/mman.h>
#include <unistd.h>
#endif

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
    if (!traits && info->storage_kind == COLI_TENSOR_STORAGE_DENSE)
        return fail(error, error_size, "tensor '%s': unsupported dtype %u", info->name, info->type);

    if (info->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
        const uint8_t *payload = (const uint8_t *)coli_gguf_mapped_at(
            file, info->absolute_offset, info->payload_size);
        if (!payload) return fail(error, error_size,
            "tensor '%s': sparse SGGUF payload is not memory mapped", info->name);
        ColiSggufSparseTensor sparse;
        if (!coli_sgguf_sparse_tensor_parse(payload, info->payload_size, &sparse,
                                            error, error_size)) return 0;
        uint64_t elements = 1;
        for (uint32_t i = 0; i < info->n_dims; ++i) {
            if (elements > UINT64_MAX / info->dims[i])
                return fail(error, error_size, "tensor '%s': element count overflow", info->name);
            elements *= info->dims[i];
        }
        if (sparse.cols != info->dims[0] || sparse.total_rows != elements / info->dims[0])
            return fail(error, error_size, "tensor '%s': sparse payload shape mismatch", info->name);
        out->name = info->name;
        out->dtype = (ColiDType)info->type;
        out->n_dims = info->n_dims;
        memcpy(out->dims, info->dims, sizeof(out->dims));
        out->element_count = elements;
        out->row_count = sparse.total_rows;
        out->row_bytes = 0;
        out->storage_bytes = info->payload_size;
        out->source_offset = info->absolute_offset;
        out->storage_kind = COLI_TENSOR_STORAGE_SPARSE_TREE;
        out->sparse_tensor = sparse;
        out->sparse_first_block = 0;
        out->sparse_block_count = sparse.total_blocks;
        out->mmap_backed = 1;
        if (error && error_size) error[0] = 0;
        return 1;
    }

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
    out->storage_kind = COLI_TENSOR_STORAGE_DENSE;
    if (error && error_size) error[0] = 0;
    return 1;
}

int coli_tensor_rows_view(const ColiTensor *tensor,
                          uint64_t first_row, uint64_t row_count,
                          ColiTensor *out) {
    if (!tensor || !out || !row_count ||
        first_row > tensor->row_count || row_count > tensor->row_count - first_row)
        return 0;
    if (tensor->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
        const uint64_t blocks_per_row = tensor->sparse_tensor.blocks_per_row;
        if (!blocks_per_row || first_row > UINT64_MAX / blocks_per_row ||
            row_count > UINT64_MAX / blocks_per_row) return 0;
        uint64_t first_block = tensor->sparse_first_block + first_row * blocks_per_row;
        uint64_t block_count = row_count * blocks_per_row;
        if (first_block < tensor->sparse_first_block ||
            first_block > tensor->sparse_tensor.total_blocks ||
            block_count > tensor->sparse_tensor.total_blocks - first_block) return 0;
        memset(out, 0, sizeof(*out));
        out->name = tensor->name;
        out->dtype = tensor->dtype;
        out->n_dims = 2;
        out->dims[0] = tensor->dims[0];
        out->dims[1] = row_count;
        out->element_count = tensor->dims[0] * row_count;
        out->row_count = row_count;
        out->storage_kind = COLI_TENSOR_STORAGE_SPARSE_TREE;
        out->sparse_tensor = tensor->sparse_tensor;
        out->sparse_first_block = first_block;
        out->sparse_block_count = block_count;
        uint64_t begin = coli_sgguf_sparse_block_offset(&tensor->sparse_tensor, first_block);
        uint64_t end = coli_sgguf_sparse_block_offset(&tensor->sparse_tensor,
                                                       first_block + block_count);
        out->storage_bytes = begin != UINT64_MAX && end >= begin
            ? (end - begin) + (block_count + 1u) * tensor->sparse_tensor.offset_width
            : 0;
        out->source_offset = tensor->source_offset;
        out->mmap_backed = 1;
        return 1;
    }
    if (!tensor->data) return 0;
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
    if (!tensor) return 0;
    if (tensor->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
        if (!exec || exec->kind == COLI_BACKEND_CPU) return 1;
#ifdef COLI_CUDA
        if (exec->kind == COLI_BACKEND_CUDA) {
            if (tensor->cuda) return tensor->cuda_device == exec->device;
            if (tensor->dims[0] > INT_MAX || tensor->row_count > INT_MAX) return 0;
            if (!coli_cuda_tensor_upload_sgguf(
                    &tensor->cuda,
                    tensor->sparse_tensor.block_offsets_le,
                    tensor->sparse_tensor.blocks,
                    tensor->sparse_first_block,
                    tensor->sparse_block_count,
                    tensor->sparse_tensor.codec_id,
                    tensor->sparse_tensor.layout,
                    tensor->sparse_tensor.offset_width,
                    tensor->sparse_tensor.auxiliary_bytes_per_block,
                    tensor->sparse_tensor.retained_value_bits,
                    (int)tensor->dims[0],
                    (int)tensor->row_count,
                    exec->device)) return 0;
            tensor->cuda_device = exec->device;
            tensor->owns_cuda = 1;
            return 1;
        }
#endif
        return 0;
    }
    if (!tensor->data) return 0;
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
    if (!exec || exec->kind == COLI_BACKEND_CPU)
        return tensor->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE || tensor->data != NULL;
#ifdef COLI_CUDA
    if (exec->kind == COLI_BACKEND_CUDA)
        return tensor->cuda && tensor->cuda_device == exec->device;
#endif
    return 0;
}

void coli_tensor_release_backend(const ColiExec *exec, ColiTensor *tensor) {
    if (!tensor || !exec || exec->kind == COLI_BACKEND_CPU) return;
#ifdef COLI_CUDA
    if (exec->kind == COLI_BACKEND_CUDA && tensor->cuda && tensor->owns_cuda &&
        tensor->cuda_device == exec->device) {
        coli_cuda_tensor_free(tensor->cuda);
        tensor->cuda = NULL;
        tensor->cuda_device = 0;
        tensor->owns_cuda = 0;
    }
#else
    (void)exec;
#endif
}

void coli_tensor_prefetch_host(const ColiTensor *tensor) {
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    if (!tensor || !tensor->mmap_backed || !tensor->storage_bytes) return;
    const uint8_t *host = tensor->data;
    size_t host_bytes = (size_t)tensor->storage_bytes;
    if (tensor->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
        uint64_t begin = coli_sgguf_sparse_block_offset(&tensor->sparse_tensor,
                                                        tensor->sparse_first_block);
        uint64_t end = coli_sgguf_sparse_block_offset(&tensor->sparse_tensor,
                                                      tensor->sparse_first_block +
                                                      tensor->sparse_block_count);
        if (end < begin || end - begin > SIZE_MAX) return;
        host = tensor->sparse_tensor.blocks + begin;
        host_bytes = (size_t)(end - begin);
    }
    if (!host) return;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    uintptr_t begin = (uintptr_t)host & ~((uintptr_t)page - 1u);
    uintptr_t end = ((uintptr_t)host + (uintptr_t)host_bytes +
                     (uintptr_t)page - 1u) & ~((uintptr_t)page - 1u);
    if (end > begin) (void)madvise((void *)begin, (size_t)(end - begin), MADV_WILLNEED);
#else
    (void)tensor;
#endif
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
    if (!tensor || !output || row >= tensor->row_count ||
        output_count < tensor->dims[0]) return 0;
    if (tensor->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
        memset(output, 0, (size_t)tensor->dims[0] * sizeof(*output));
        uint64_t base = tensor->sparse_first_block +
                        row * tensor->sparse_tensor.blocks_per_row;
        for (uint32_t b = 0; b < tensor->sparse_tensor.blocks_per_row; ++b) {
            ColiSggufSparseBlock block;
            if (!coli_sgguf_sparse_block_get(&tensor->sparse_tensor, base + b,
                                             &block, NULL, 0) ||
                !coli_sgguf_sparse_block_materialize_f32(&block,
                    output + (uint64_t)b * COLI_SGGUF_GROUP_SIZE)) return 0;
        }
        return 1;
    }
    if (!tensor->data) return 0;
    const uint8_t *encoded = tensor->data + row * tensor->row_bytes;
    return coli_dtype_dequantize_row(tensor->dtype, encoded, tensor->dims[0], output);
}

static int matmul_cpu(float *y, const float *x, const ColiTensor *w,
                      int S, int I, int O) {
    if (!y || !x || !w || S < 1 || I < 1 || O < 1 ||
        w->dims[0] != (uint64_t)I || w->row_count != (uint64_t)O) return 0;
    if (w->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
        if (I % (int)COLI_SGGUF_GROUP_SIZE ||
            w->sparse_tensor.blocks_per_row != (uint32_t)(I / (int)COLI_SGGUF_GROUP_SIZE)) return 0;
        int failed = 0;
#pragma omp parallel for schedule(static) reduction(|:failed)
        for (int o = 0; o < O; ++o) {
            uint64_t row_base = w->sparse_first_block +
                                (uint64_t)o * w->sparse_tensor.blocks_per_row;
            for (int ss = 0; ss < S; ++ss) {
                double sum = 0.0;
                const float *xs = x + (int64_t)ss * I;
                for (uint32_t b = 0; b < w->sparse_tensor.blocks_per_row; ++b) {
                    ColiSggufSparseBlock block;
                    int ok = 0;
                    if (!coli_sgguf_sparse_block_get(&w->sparse_tensor, row_base + b,
                                                     &block, NULL, 0)) { failed = 1; break; }
                    sum += coli_sgguf_sparse_block_dot_f32(
                        &block, xs + (uint64_t)b * COLI_SGGUF_GROUP_SIZE, &ok);
                    if (!ok) { failed = 1; break; }
                }
                y[(int64_t)ss * O + o] = (float)sum;
            }
        }
        return !failed;
    }
    if (!w->data) return 0;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
        const uint8_t *row = w->data + (uint64_t)o * w->row_bytes;
        for (int ss = 0; ss < S; ++ss) {
            y[(int64_t)ss * O + o] = coli_dtype_dot_f32(w->dtype, row,
                                                        x + (int64_t)ss * I,
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
