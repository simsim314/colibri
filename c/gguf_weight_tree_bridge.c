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
} GwvHandle;

static void gwv_set_error(GwvHandle *h, const char *msg) {
    if (!h) return;
    snprintf(h->error, sizeof(h->error), "%s", msg ? msg : "unknown error");
}

static int gwv_is_expert_tensor(const ColiGgufTensorInfo *t) {
    if (!t || !t->name || t->n_dims != 3) return 0;
    return strstr(t->name, ".ffn_gate_exps.weight") ||
           strstr(t->name, ".ffn_up_exps.weight") ||
           strstr(t->name, ".ffn_down_exps.weight");
}

void *gwv_open(const char *path) {
    if (!path) return NULL;
    GwvHandle *h = (GwvHandle *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    if (!coli_gguf_open(&h->gguf, path)) {
        gwv_set_error(h, coli_gguf_error(&h->gguf));
        return h;
    }
    h->error[0] = '\0';
    return h;
}

void gwv_close(void *handle) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h) return;
    if (h->gguf.fd >= 0 || h->gguf.mapping || h->gguf.tensors || h->gguf.metadata)
        coli_gguf_close(&h->gguf);
    free(h);
}

const char *gwv_error(void *handle) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h) return "null GGUF handle";
    return h->error[0] ? h->error : "";
}

uint64_t gwv_tensor_count(void *handle) {
    GwvHandle *h = (GwvHandle *)handle;
    return h ? h->gguf.tensor_count : 0;
}

int gwv_tensor_is_expert(void *handle, uint64_t tensor_index) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count) return 0;
    return gwv_is_expert_tensor(&h->gguf.tensors[tensor_index]);
}

const char *gwv_tensor_name(void *handle, uint64_t tensor_index) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count) return NULL;
    return h->gguf.tensors[tensor_index].name;
}

uint32_t gwv_tensor_type(void *handle, uint64_t tensor_index) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count) return UINT32_MAX;
    return h->gguf.tensors[tensor_index].type;
}

const char *gwv_tensor_type_name(void *handle, uint64_t tensor_index) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count) return NULL;
    const ColiDTypeTraits *tr = coli_dtype_traits((ColiDType)h->gguf.tensors[tensor_index].type);
    return tr ? tr->name : "UNKNOWN";
}

uint32_t gwv_tensor_ndim(void *handle, uint64_t tensor_index) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count) return 0;
    return h->gguf.tensors[tensor_index].n_dims;
}

uint64_t gwv_tensor_dim(void *handle, uint64_t tensor_index, uint32_t dim) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count || dim >= COLI_GGUF_MAX_DIMS) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[tensor_index];
    if (dim >= t->n_dims) return 0;
    return t->dims[dim];
}

uint64_t gwv_tensor_expert_count(void *handle, uint64_t tensor_index) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[tensor_index];
    return gwv_is_expert_tensor(t) ? t->dims[2] : 0;
}

uint64_t gwv_tensor_weights_per_expert(void *handle, uint64_t tensor_index) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || tensor_index >= h->gguf.tensor_count) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[tensor_index];
    if (!gwv_is_expert_tensor(t)) return 0;
    if (t->dims[0] && t->dims[1] > UINT64_MAX / t->dims[0]) return 0;
    return t->dims[0] * t->dims[1];
}

int gwv_load_expert(void *handle, uint64_t tensor_index, uint64_t expert_index,
                    float *output, uint64_t output_count) {
    GwvHandle *h = (GwvHandle *)handle;
    if (!h || !output || tensor_index >= h->gguf.tensor_count) return 0;
    const ColiGgufTensorInfo *t = &h->gguf.tensors[tensor_index];
    if (!gwv_is_expert_tensor(t)) {
        gwv_set_error(h, "selected tensor is not a stacked expert tensor");
        return 0;
    }
    if (expert_index >= t->dims[2]) {
        gwv_set_error(h, "expert index is out of range");
        return 0;
    }
    const ColiDTypeTraits *tr = coli_dtype_traits((ColiDType)t->type);
    if (!tr) {
        gwv_set_error(h, "unsupported GGML tensor type");
        return 0;
    }
    uint64_t weights = t->dims[0] * t->dims[1];
    if (output_count < weights) {
        gwv_set_error(h, "output buffer is too small");
        return 0;
    }
    uint64_t row_bytes = 0;
    if (!coli_dtype_row_size((ColiDType)t->type, t->dims[0], &row_bytes)) {
        gwv_set_error(h, "invalid row shape for tensor dtype");
        return 0;
    }
    uint64_t expert_bytes = row_bytes * t->dims[1];
    uint64_t tensor_bytes = 0, tensor_elements = 0;
    if (!coli_dtype_tensor_size((ColiDType)t->type, t->dims, t->n_dims,
                                &tensor_elements, &tensor_bytes)) {
        gwv_set_error(h, "invalid tensor shape for dtype");
        return 0;
    }
    const uint8_t *base = (const uint8_t *)coli_gguf_mapped_at(
        &h->gguf, t->absolute_offset, tensor_bytes);
    if (!base) {
        gwv_set_error(h, "could not map tensor bytes");
        return 0;
    }
    const uint8_t *expert = base + expert_index * expert_bytes;
    for (uint64_t row = 0; row < t->dims[1]; ++row) {
        const void *encoded = expert + row * row_bytes;
        float *decoded = output + row * t->dims[0];
        if (!coli_dtype_dequantize_row((ColiDType)t->type, encoded, t->dims[0], decoded)) {
            gwv_set_error(h, "dequantization failed");
            return 0;
        }
    }
    h->error[0] = '\0';
    return 1;
}
