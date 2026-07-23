#ifndef COLIBRI_GGUF_READER_H
#define COLIBRI_GGUF_READER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_GGUF_DEFAULT_ALIGNMENT 32u
#define COLI_GGUF_MAX_DIMS 8u

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
    uint32_t type;
    uint64_t offset;
    uint64_t absolute_offset;
} ColiGgufTensorInfo;

typedef struct {
    int fd;
    char *path;
    uint64_t file_size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint32_t alignment;
    uint64_t data_offset;
    ColiGgufKV *metadata;
    ColiGgufTensorInfo *tensors;
    char error[256];
} ColiGgufFile;

int coli_gguf_open(ColiGgufFile *g, const char *path);
void coli_gguf_close(ColiGgufFile *g);

const char *coli_gguf_error(const ColiGgufFile *g);
const char *coli_gguf_value_type_name(uint32_t type);
const char *coli_ggml_type_name(uint32_t type);

const ColiGgufKV *coli_gguf_find_kv(const ColiGgufFile *g, const char *key);
const ColiGgufTensorInfo *coli_gguf_find_tensor(const ColiGgufFile *g, const char *name);

int coli_gguf_kv_read_u64(const ColiGgufFile *g, const ColiGgufKV *kv, uint64_t *out);
int coli_gguf_kv_read_i64(const ColiGgufFile *g, const ColiGgufKV *kv, int64_t *out);
int coli_gguf_kv_read_f64(const ColiGgufFile *g, const ColiGgufKV *kv, double *out);
int coli_gguf_kv_read_bool(const ColiGgufFile *g, const ColiGgufKV *kv, int *out);
int coli_gguf_kv_read_string(const ColiGgufFile *g, const ColiGgufKV *kv, char **out);

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
