#ifndef COLIBRI_SPLIT_STORAGE_H
#define COLIBRI_SPLIT_STORAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_SPLIT_VERSION 2u
#define COLI_SPLIT_FOOTER_BYTES 64u
#define COLI_SPLIT_MANIFEST_HEADER_BYTES 64u
#define COLI_SPLIT_ENTRY_BYTES 40u
#define COLI_SPLIT_SHARD_HEADER_BYTES 4096u

#define COLI_SPLIT_FOOTER_MAGIC "COLISPF1"
#define COLI_SPLIT_MANIFEST_MAGIC "COLISPM1"
#define COLI_SPLIT_SHARD_MAGIC "COLISHD1"

typedef enum {
    COLI_SPLIT_FAST_MMAP = 0,
    COLI_SPLIT_PRELOAD_RAM = 1,
    COLI_SPLIT_PRELOAD_VRAM = 2,
    COLI_SPLIT_SLOW_TAIL_MMAP = 3,
    COLI_SPLIT_FRAGMENTED = 4,
} ColiSplitLocation;

typedef struct {
    uint64_t original_offset;
    uint64_t bytes;
    uint64_t shard_offset;
    uint32_t location;
    uint32_t tensor_index;
    uint64_t reserved;
    void *ram_data;
} ColiSplitEntry;

typedef struct ColiSplitState ColiSplitState;

/* Probe a primary file for a split footer. A non-split file is a successful
 * probe with *state_out == NULL and logical_size_out == physical_size. */
int coli_split_open_primary(int primary_fd,
                            const char *primary_path,
                            uint64_t physical_size,
                            ColiSplitState **state_out,
                            uint64_t *logical_size_out,
                            char *error,
                            size_t error_size);
void coli_split_close(ColiSplitState *state);
void coli_split_set_preload_backend(ColiSplitState *state, int backend_kind);

const void *coli_split_mapped_at(ColiSplitState *state,
                                 const void *primary_mapping,
                                 uint64_t primary_mapping_size,
                                 uint64_t offset,
                                 uint64_t bytes,
                                 int *handled,
                                 int *mmap_backed,
                                 char *error,
                                 size_t error_size);
int coli_split_read_at(ColiSplitState *state,
                       int primary_fd,
                       const void *primary_mapping,
                       uint64_t primary_mapping_size,
                       uint64_t logical_size,
                       uint64_t offset,
                       void *dst,
                       size_t bytes);

const ColiSplitEntry *coli_split_entry_for_tensor(const ColiSplitState *state,
                                                   uint64_t tensor_index);
int coli_split_validate_tensor(const ColiSplitState *state,
                               uint64_t tensor_index,
                               uint64_t original_offset,
                               uint64_t bytes,
                               uint32_t *location_out,
                               uint64_t *shard_offset_out);
uint32_t coli_split_location_at(const ColiSplitState *state,
                                uint64_t offset, uint64_t bytes);
const char *coli_split_location_name(uint32_t location);
const char *coli_split_preload_path(const ColiSplitState *state);
const char *coli_split_tail_path(const ColiSplitState *state);
uint64_t coli_split_ram_loaded_bytes(const ColiSplitState *state);
uint64_t coli_split_entry_count(const ColiSplitState *state);
void coli_split_drop_source_pages(ColiSplitState *state,
                                  uint64_t offset,
                                  uint64_t bytes);

uint64_t coli_split_fnv1a64(const void *data, size_t bytes);
uint16_t coli_split_load_u16(const uint8_t *p);
uint32_t coli_split_load_u32(const uint8_t *p);
uint64_t coli_split_load_u64(const uint8_t *p);
void coli_split_store_u16(uint8_t *p, uint16_t v);
void coli_split_store_u32(uint8_t *p, uint32_t v);
void coli_split_store_u64(uint8_t *p, uint64_t v);

#ifdef __cplusplus
}
#endif

#endif
