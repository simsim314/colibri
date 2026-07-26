#ifndef COLIBRI_SPLIT_PLAN_H
#define COLIBRI_SPLIT_PLAN_H

#include "gguf_reader.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t location;
    uint32_t tensor_index;
    uint64_t original_offset;
    uint64_t bytes;
    uint64_t shard_offset;
    uint64_t allocation_bytes;
    int32_t moe_layer;
    int32_t expert_id;
    int32_t bundle_id;
    uint32_t usage_count;
    int is_expert;
    int is_vram_candidate;
} ColiSplitDecision;

typedef struct {
    ColiSplitDecision *entries;
    uint64_t entry_count;
    uint64_t tensor_count;
    uint64_t usage_total;
    uint64_t usage_nonzero;
    uint64_t expert_bundle_count;
} ColiSplitPlan;

typedef struct {
    uint64_t fast_budget;
    uint64_t ram_budget;
    uint64_t vram_budget;
    uint64_t fast_allocated;
    uint64_t preload_ram_bytes;
    uint64_t preload_vram_bytes;
    uint64_t tail_bytes;
    uint64_t source_payload_bytes;
    uint64_t header_and_manifest_bytes;
    uint64_t vram_candidate_bytes;
    uint64_t fast_unused_bytes;
    uint64_t ram_unused_bytes;
    uint64_t vram_unused_bytes;
    uint64_t fast_fragment_count;
    uint64_t ram_fragment_count;
    uint64_t vram_fragment_count;
    uint64_t tail_fragment_count;
    uint64_t fragmented_tensor_count;
    uint64_t hot_ram_bundle_count;
    uint64_t fast_bundle_count;
    uint64_t tail_bundle_count;
} ColiSplitPlanSummary;

int coli_split_is_expert_tensor(const ColiGgufTensorInfo *tensor);
int coli_split_is_vram_candidate(const ColiGgufTensorInfo *tensor);
uint64_t coli_split_round_up(uint64_t value, uint64_t alignment);

int coli_split_make_plan(const ColiGgufFile *model,
                         const char *usage_path,
                         uint64_t fast_budget,
                         uint64_t ram_budget,
                         uint64_t vram_budget,
                         ColiSplitPlan *plan,
                         ColiSplitPlanSummary *summary,
                         char *error,
                         size_t error_size);
void coli_split_plan_destroy(ColiSplitPlan *plan);

#ifdef __cplusplus
}
#endif

#endif
