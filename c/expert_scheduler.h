#ifndef COLIBRI_EXPERT_SCHEDULER_H
#define COLIBRI_EXPERT_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t stride;
    size_t eid_offset;
    size_t used_offset;
} ColiExpertSlotLayout;

typedef struct {
    void *pin;
    int npin;
    void *cache;
    int *ncache;
    int cache_cap;
    int n_experts;
    ColiExpertSlotLayout layout;
    uint64_t *clock;
    uint32_t *heat;
    uint32_t *last;
    uint32_t *usage;
    uint32_t *access_clock;
} ColiExpertLayerStore;

typedef struct {
    void *slot;
    int index;
    int from_pin;
} ColiExpertLookup;

typedef struct {
    void *slot;
    int index;
    int is_new;
    int victim_eid;
    int reservation_eid;
} ColiExpertAdmission;

typedef struct {
    int (*load)(void *ctx, int layer, int eid, void *slot, int demand);
    void (*evict)(void *ctx, int layer, void *slot);
    size_t (*resident_bytes)(void *ctx, int layer, const void *slot);
} ColiExpertStorageOps;

typedef struct {
    uint64_t hits;
    uint64_t pin_hits;
    uint64_t cache_hits;
    uint64_t misses;
    uint64_t admissions;
    uint64_t evictions;
    uint64_t speculative_loads;
    uint64_t speculative_drops;
} ColiExpertSchedulerStats;

void *coli_expert_slot_at(void *slots, const ColiExpertSlotLayout *layout, int index);
const void *coli_expert_slot_at_const(const void *slots, const ColiExpertSlotLayout *layout, int index);
int coli_expert_slot_eid(const void *slot, const ColiExpertSlotLayout *layout);
uint64_t coli_expert_slot_used(const void *slot, const ColiExpertSlotLayout *layout);
void coli_expert_slot_set_eid(void *slot, const ColiExpertSlotLayout *layout, int eid);
void coli_expert_slot_set_used(void *slot, const ColiExpertSlotLayout *layout, uint64_t used);
void coli_expert_slot_touch(void *slot, const ColiExpertSlotLayout *layout, uint64_t *clock);

void coli_expert_record_demand(ColiExpertLayerStore *store, int eid);
ColiExpertLookup coli_expert_lookup(ColiExpertLayerStore *store, int eid, int include_reservations);
int coli_expert_resident(ColiExpertLayerStore *store, int eid, int include_reservations);
int coli_expert_choose_lru(const ColiExpertLayerStore *store);

/* Begin/finish split is used by the existing PILOT_REAL path: policy chooses and
 * reserves a slot under its existing lock, storage I/O happens outside the lock,
 * then publication completes under the same lock. */
int coli_expert_begin_admission(ColiExpertLayerStore *store, int eid,
                                int speculative, int evict_guard,
                                ColiExpertAdmission *out);
void coli_expert_finish_admission(ColiExpertLayerStore *store,
                                  const ColiExpertAdmission *admission,
                                  int eid, int success);

/* Synchronous admission for native tensor stores (GGUF). This is the same
 * lookup/LRU/guard/admission policy, with storage represented by callbacks. */
void *coli_expert_acquire(ColiExpertLayerStore *store, int layer, int eid,
                          int demand, int evict_guard,
                          const ColiExpertStorageOps *ops, void *ctx,
                          ColiExpertSchedulerStats *stats);

/* Same policy as coli_expert_acquire(), but callers may disable new
 * admissions while preserving demand accounting and resident hits. This is
 * used by memory-constrained backends after the first device-residency
 * failure: already-resident experts remain usable, misses fall back to a
 * different execution tier, and the scheduler does not evict a good slot for
 * an upload that is known to fail. */
void *coli_expert_acquire_controlled(ColiExpertLayerStore *store, int layer,
                                     int eid, int demand, int allow_admission,
                                     int evict_guard,
                                     const ColiExpertStorageOps *ops, void *ctx,
                                     ColiExpertSchedulerStats *stats);

/* Promote a fully loaded temporary slot into the existing LRU cache by swapping
 * complete slot payloads. This preserves the legacy asynchronous load pipeline. */
void *coli_expert_promote_loaded(ColiExpertLayerStore *store, void *loaded_slot,
                                 int *evicted_eid);

/* Shared LFRU pin replacement choice used by REPIN. Returns 1 when a candidate
 * clears the same hysteresis used by tier_pick_lfru. */
int coli_expert_repin_pick(const ColiExpertLayerStore *store,
                           int *pin_index, int *candidate_eid, long *gain);

/* Move an already cached REPIN candidate into a pinned slot without creating
 * two slots that reference the same backend storage. The replaced pin is
 * evicted and the former cache slot becomes empty. Returns 1 when moved, 0
 * when the candidate is not in LRU, and -1 on an internal failure. */
int coli_expert_repin_promote_cached(ColiExpertLayerStore *store,
                                     int pin_index, int candidate_eid,
                                     int layer,
                                     const ColiExpertStorageOps *ops,
                                     void *ctx);

void coli_expert_decay_heat(ColiExpertLayerStore *store);

/* Persistent expert-selection history. The text format is the existing Colibri
 * format: one "layer expert count" record per nonzero expert. */
int64_t coli_expert_usage_load(const char *path, uint32_t **usage,
                                int n_layers, int n_experts);
int coli_expert_usage_save(const char *path, uint32_t *const *usage,
                           int n_layers, int n_experts);
int coli_expert_usage_top(uint32_t *const *usage, int n_layers, int n_experts,
                          int *flat_ids, int output_cap);
int coli_expert_usage_top_file(const char *path, int n_layers, int n_experts,
                               int *flat_ids, int output_cap, int64_t *total);

/* Concentrated hot-tier selection. Choose up to focus_layers layers from the
 * usage history, then greedily fill output_cap expert pins inside those layers.
 * Every selected layer receives at least one pin when possible and no layer
 * receives more than ceil(output_cap / focus_layers), making the requested
 * layer count an actual concentration control rather than a display hint.
 * selected_layers/per_layer_counts are optional telemetry outputs. */
int coli_expert_usage_focus(uint32_t *const *usage, int n_layers, int n_experts,
                            int focus_layers, int *flat_ids, int output_cap,
                            int *selected_layers, int selected_cap,
                            int *per_layer_counts);
int coli_expert_usage_focus_file(const char *path, int n_layers, int n_experts,
                                 int focus_layers, int *flat_ids, int output_cap,
                                 int *selected_layers, int selected_cap,
                                 int *per_layer_counts, int64_t *total);

#define COLI_EXPERT_COUPLE_M 16
typedef struct {
    int n_layers, n_experts;
    int16_t *pred;
    float *score;
} ColiExpertCoupling;

int coli_expert_coupling_load(ColiExpertCoupling *coupling, const char *path,
                              int n_layers, int n_experts,
                              long *conditioning_entries);
void coli_expert_coupling_destroy(ColiExpertCoupling *coupling);
int coli_expert_coupling_predict(const ColiExpertCoupling *coupling,
                                 int layer, int depth,
                                 const int *routed, int routed_count,
                                 int *output, int output_cap);

#ifdef __cplusplus
}
#endif
#endif
