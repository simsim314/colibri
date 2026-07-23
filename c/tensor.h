#ifndef COLIBRI_TENSOR_H
#define COLIBRI_TENSOR_H

#include "ggml_types.h"
#include "gguf_reader.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    COLI_BACKEND_CPU = 0,
    COLI_BACKEND_CUDA = 1,
} ColiBackendKind;

typedef struct ColiCudaTensor ColiCudaTensor;

typedef struct {
    ColiBackendKind kind;
    int device;
} ColiExec;

/* Native Colibri tensor descriptor. The storage pointer may refer to owned RAM,
 * a non-owning mmap view of a GGUF file, or another backend-owned allocation.
 * GGUF tensors use the same ColiDType values as every other Colibri tensor. */
typedef struct {
    const char *name;
    ColiDType dtype;
    uint32_t n_dims;
    uint64_t dims[COLI_GGUF_MAX_DIMS];
    uint64_t element_count;
    uint64_t row_count;
    uint64_t row_bytes;
    uint64_t storage_bytes;
    const uint8_t *data;
    uint64_t source_offset;
    ColiCudaTensor *cuda;
    int cuda_device;
    unsigned owns_data : 1;
    unsigned mmap_backed : 1;
    unsigned owns_cuda : 1;
} ColiTensor;

int coli_tensor_bind_gguf(const ColiGgufFile *file,
                          const ColiGgufTensorInfo *info,
                          ColiTensor *out,
                          char *error, size_t error_size);

/* Creates a non-owning 2-D row slice. Rows are the product of dimensions 1..N.
 * This is used for one expert inside GGUF [I,O,E] tensors without copying it. */
int coli_tensor_rows_view(const ColiTensor *tensor,
                          uint64_t first_row, uint64_t row_count,
                          ColiTensor *out);

void coli_tensor_destroy(ColiTensor *tensor);

/* Materialize the tensor's existing encoded representation on the selected
 * backend. CUDA residency preserves the native dtype and never creates F32
 * weights. Row views become zero-copy device views of the resident parent. */
int coli_tensor_reside(const ColiExec *exec, ColiTensor *tensor);
int coli_tensor_is_resident(const ColiExec *exec, const ColiTensor *tensor);
/* Release only the selected backend copy; mmap/host storage remains valid. */
void coli_tensor_release_backend(const ColiExec *exec, ColiTensor *tensor);
/* Hint the OS page cache for an mmap-backed encoded tensor. */
void coli_tensor_prefetch_host(const ColiTensor *tensor);
const void *coli_tensor_device_data(const ColiTensor *tensor);

/* Decode one logical row into caller-provided F32 storage. No decoded tensor is
 * retained. For F32 input this is a direct memcpy from the mapped file. */
int coli_tensor_read_row_f32(const ColiTensor *tensor, uint64_t row,
                             float *output, uint64_t output_count);

/* y[S,O] = x[S,I] @ W[O,I]^T. Quantized weights remain encoded. CPU kernels
 * decode one quantization block at a time; CUDA uploads encoded bytes and
 * decodes inside the kernel. */
int coli_tensor_matmul(const ColiExec *exec,
                       float *y, const float *x,
                       ColiTensor *weights,
                       int S, int I, int O);

/* Device-resident operation entry points used by Colibri model pipelines. */
int coli_tensor_matmul_device(const ColiExec *exec, float *y_dev,
                       const float *x_dev, ColiTensor *weights, int S);
int coli_tensor_read_row_device(const ColiExec *exec, ColiTensor *tensor,
                       uint64_t row, float *out_dev, float scale);

#ifdef __cplusplus
}
#endif

#endif
