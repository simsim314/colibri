#define _FILE_OFFSET_BITS 64
#include "gguf_reader.h"
#include "ggml_types.h"
#include "ggml_quants.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ColiGgufFile gguf;
    char error[512];
} SimHandle;

static void set_error(SimHandle *h, const char *msg) {
    if (!h) return;
    snprintf(h->error, sizeof(h->error), "%s", msg ? msg : "unknown error");
}

static int is_expert_tensor(const ColiGgufTensorInfo *t) {
    if (!t || !t->name || t->n_dims != 3) return 0;
    return strstr(t->name, ".ffn_gate_exps.weight") != NULL ||
           strstr(t->name, ".ffn_up_exps.weight") != NULL ||
           strstr(t->name, ".ffn_down_exps.weight") != NULL;
}

void *sgsim_open(const char *path) {
    if (!path || !*path) return NULL;
    SimHandle *h = (SimHandle *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->gguf.fd = -1;
    if (!coli_gguf_open(&h->gguf, path)) {
        set_error(h, coli_gguf_error(&h->gguf));
        return h;
    }
    h->error[0] = '\0';
    return h;
}

void sgsim_close(void *handle) {
    SimHandle *h = (SimHandle *)handle;
    if (!h) return;
    if (h->gguf.fd >= 0 || h->gguf.mapping || h->gguf.tensors || h->gguf.metadata)
        coli_gguf_close(&h->gguf);
    free(h);
}

const char *sgsim_error(void *handle) {
    SimHandle *h = (SimHandle *)handle;
    if (!h) return "null GGUF handle";
    return h->error[0] ? h->error : "";
}

uint64_t sgsim_file_size(void *handle) {
    SimHandle *h = (SimHandle *)handle;
    return h ? h->gguf.file_size : 0;
}

uint64_t sgsim_tensor_count(void *handle) {
    SimHandle *h = (SimHandle *)handle;
    return h ? h->gguf.tensor_count : 0;
}

const char *sgsim_tensor_name(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return NULL;
    return h->gguf.tensors[ti].name;
}

int sgsim_tensor_is_expert(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return 0;
    return is_expert_tensor(&h->gguf.tensors[ti]);
}

uint32_t sgsim_tensor_type(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return UINT32_MAX;
    return h->gguf.tensors[ti].type;
}

const char *sgsim_tensor_type_name(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return NULL;
    const ColiDTypeTraits *tr = coli_dtype_traits((ColiDType)h->gguf.tensors[ti].type);
    return tr ? tr->name : "UNKNOWN";
}

uint32_t sgsim_tensor_ndim(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return 0;
    return h->gguf.tensors[ti].n_dims;
}

uint64_t sgsim_tensor_dim(void *handle, uint64_t ti, uint32_t dim) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count || dim >= COLI_GGUF_MAX_DIMS) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[ti];
    return dim < t->n_dims ? t->dims[dim] : 0;
}

uint32_t sgsim_tensor_block_values(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return 0;
    const ColiDTypeTraits *tr = coli_dtype_traits((ColiDType)h->gguf.tensors[ti].type);
    return tr ? tr->block_values : 0;
}

uint32_t sgsim_tensor_block_bytes(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return 0;
    const ColiDTypeTraits *tr = coli_dtype_traits((ColiDType)h->gguf.tensors[ti].type);
    return tr ? tr->block_bytes : 0;
}

uint64_t sgsim_tensor_encoded_bytes(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[ti];
    uint64_t elements = 0, bytes = 0;
    if (!coli_dtype_tensor_size((ColiDType)t->type, t->dims, t->n_dims, &elements, &bytes))
        return 0;
    return bytes;
}

uint64_t sgsim_tensor_expert_bytes(void *handle, uint64_t ti) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || ti >= h->gguf.tensor_count) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[ti];
    if (!is_expert_tensor(t)) return 0;
    uint64_t row_bytes = 0;
    if (!coli_dtype_row_size((ColiDType)t->type, t->dims[0], &row_bytes)) return 0;
    if (t->dims[1] && row_bytes > UINT64_MAX / t->dims[1]) return 0;
    return row_bytes * t->dims[1];
}

int sgsim_load_expert(void *handle, uint64_t ti, uint64_t expert_index,
                      float *output, uint64_t output_count) {
    SimHandle *h = (SimHandle *)handle;
    if (!h || !output || ti >= h->gguf.tensor_count) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[ti];
    if (!is_expert_tensor(t)) {
        set_error(h, "selected tensor is not a routed expert tensor");
        return 0;
    }
    if (expert_index >= t->dims[2]) {
        set_error(h, "expert index out of range");
        return 0;
    }
    uint64_t weights = t->dims[0] * t->dims[1];
    if (output_count < weights) {
        set_error(h, "output buffer too small");
        return 0;
    }
    uint64_t row_bytes = 0;
    if (!coli_dtype_row_size((ColiDType)t->type, t->dims[0], &row_bytes)) {
        set_error(h, "invalid row shape for dtype");
        return 0;
    }
    uint64_t expert_bytes = row_bytes * t->dims[1];
    uint64_t tensor_bytes = sgsim_tensor_encoded_bytes(handle, ti);
    const uint8_t *base = (const uint8_t *)coli_gguf_mapped_at(
        &h->gguf, t->absolute_offset, tensor_bytes);
    if (!base) {
        set_error(h, "could not map tensor bytes");
        return 0;
    }
    const uint8_t *expert = base + expert_index * expert_bytes;
    for (uint64_t row = 0; row < t->dims[1]; ++row) {
        const void *encoded = expert + row * row_bytes;
        float *decoded = output + row * t->dims[0];
        if (!coli_dtype_dequantize_row((ColiDType)t->type, encoded, t->dims[0], decoded)) {
            set_error(h, "dequantization failed");
            return 0;
        }
    }
    h->error[0] = '\0';
    return 1;
}
