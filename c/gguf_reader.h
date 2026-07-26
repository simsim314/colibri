#ifndef COLIBRI_GGUF_READER_H
#define COLIBRI_GGUF_READER_H

#include <stddef.h>
#include <stdint.h>

#include "split_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_GGUF_DEFAULT_ALIGNMENT 32u
#define COLI_GGUF_MAX_DIMS 8u

#define COLI_MODEL_CONTAINER_GGUF 0u
#define COLI_MODEL_CONTAINER_SGGUF 1u
#define COLI_TENSOR_STORAGE_DENSE 0u
#define COLI_TENSOR_STORAGE_SPARSE_TREE 1u

typedef enum {
    COLI_GGUF_TYPE_UINT8   = 0,
    COLI_GGUF_TYPE_INT8    = 1,
    COLI_GGUF_TYPE_UINT16  = 2,
    COLI_GGUF_TYPE_INT16   = 3,
    COLI_GGUF_TYPE_UINT32  = 4,
    COLI_GGUF_TYPE_INT32   = 5,
    COLI_GGUF_TYPE_FLOAT32 = 6,
    COLI_GGUF_TYPE_BOOL    = 7,
    COLI_GGUF_TYPE_STRING  = 8,
    COLI_GGUF_TYPE_ARRAY   = 9,
    COLI_GGUF_TYPE_UINT64  = 10,
    COLI_GGUF_TYPE_INT64   = 11,
    COLI_GGUF_TYPE_FLOAT64 = 12,
} ColiGgufValueType;

typedef struct {
    char *key;
    uint32_t type;
    uint32_t array_type;
    uint64_t array_count;
    uint64_t value_offset;
    uint64_t value_size;
} ColiGgufKV;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[COLI_GGUF_MAX_DIMS];
    uint32_t type;                 /* logical/base GGML type */
    uint32_t storage_kind;         /* dense GGUF bytes or sparse tree */
    uint32_t codec_id;             /* sparse retained-value codec */
    uint32_t flags;
    uint64_t offset;
    uint64_t absolute_offset;
    uint64_t payload_size;
    uint64_t index_offset;
    uint64_t index_absolute_offset;
    uint64_t index_size;
    int32_t moe_layer;
    uint32_t moe_projection;
    uint32_t expert_count;
    uint32_t rows_per_expert;
    uint32_t split_location;       /* ColiSplitLocation */
    uint64_t split_shard_offset;
} ColiGgufTensorInfo;

typedef struct {
    int fd;
    char *path;
    uint64_t file_size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint32_t alignment;
    uint32_t container_kind;
    uint64_t data_offset;
    ColiGgufKV *metadata;
    ColiGgufTensorInfo *tensors;
    void *mapping;
    void *mapping_handle;
    uint64_t mapping_size;
    uint64_t physical_file_size;
    ColiSplitState *split;
    int preload_backend_kind;
    int preload_device;
    char error[256];
} ColiGgufFile;

/* Opens either a standard GGUF or an SGGUF container. The historical name is
 * retained so existing model loaders remain source-compatible. */
int coli_gguf_open(ColiGgufFile *g, const char *path);
void coli_gguf_set_preload_backend(ColiGgufFile *g, int backend_kind, int device);
int coli_gguf_is_split(const ColiGgufFile *g);
void coli_gguf_close(ColiGgufFile *g);

const char *coli_gguf_error(const ColiGgufFile *g);
const char *coli_gguf_value_type_name(uint32_t type);
const char *coli_ggml_type_name(uint32_t type);

const ColiGgufKV *coli_gguf_find_kv(const ColiGgufFile *g, const char *key);
const ColiGgufTensorInfo *coli_gguf_find_tensor(const ColiGgufFile *g, const char *name);
/* Storage-neutral routed-MoE lookup. SGGUF uses its explicit tensor-directory
 * index; ordinary GGUF falls back to canonical tensor names. */
const ColiGgufTensorInfo *coli_gguf_find_moe_tensor(const ColiGgufFile *g,
                                                    int32_t layer,
                                                    uint32_t projection);

int coli_gguf_kv_read_u64(const ColiGgufFile *g, const ColiGgufKV *kv, uint64_t *out);
int coli_gguf_kv_read_i64(const ColiGgufFile *g, const ColiGgufKV *kv, int64_t *out);
int coli_gguf_kv_read_f64(const ColiGgufFile *g, const ColiGgufKV *kv, double *out);
int coli_gguf_kv_read_bool(const ColiGgufFile *g, const ColiGgufKV *kv, int *out);
int coli_gguf_kv_read_string(const ColiGgufFile *g, const ColiGgufKV *kv, char **out);
int coli_gguf_kv_read_string_array(const ColiGgufFile *g, const ColiGgufKV *kv,
                                   char ***out, uint64_t *count_out);
int coli_gguf_kv_read_u32_array(const ColiGgufFile *g, const ColiGgufKV *kv,
                                uint32_t **out, uint64_t *count_out);
void coli_gguf_free_string_array(char **items, uint64_t count);

const void *coli_gguf_mapped_at(const ColiGgufFile *g, uint64_t offset, uint64_t bytes);
int coli_gguf_range_is_mmap_backed(const ColiGgufFile *g, uint64_t offset, uint64_t bytes);
void coli_gguf_drop_source_pages(const ColiGgufFile *g, uint64_t offset, uint64_t bytes);
int coli_gguf_read_at(const ColiGgufFile *g, uint64_t offset, void *dst, size_t bytes);
int coli_gguf_read_tensor_bytes(const ColiGgufFile *g,
                                const ColiGgufTensorInfo *tensor,
                                uint64_t relative_offset,
                                void *dst,
                                size_t bytes);

#ifdef __cplusplus
}
#endif

#endif
