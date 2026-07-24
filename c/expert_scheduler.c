#include "expert_scheduler.h"
#include "tier.h"

#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static int *eid_ptr(void *slot, const ColiExpertSlotLayout *l) {
    return (int *)((unsigned char *)slot + l->eid_offset);
}
static const int *eid_ptr_const(const void *slot, const ColiExpertSlotLayout *l) {
    return (const int *)((const unsigned char *)slot + l->eid_offset);
}
static uint64_t *used_ptr(void *slot, const ColiExpertSlotLayout *l) {
    return (uint64_t *)((unsigned char *)slot + l->used_offset);
}
static const uint64_t *used_ptr_const(const void *slot, const ColiExpertSlotLayout *l) {
    return (const uint64_t *)((const unsigned char *)slot + l->used_offset);
}

void *coli_expert_slot_at(void *slots, const ColiExpertSlotLayout *l, int i) {
    return slots && l && i >= 0 ? (void *)((unsigned char *)slots + (size_t)i * l->stride) : NULL;
}
const void *coli_expert_slot_at_const(const void *slots, const ColiExpertSlotLayout *l, int i) {
    return slots && l && i >= 0 ? (const void *)((const unsigned char *)slots + (size_t)i * l->stride) : NULL;
}
int coli_expert_slot_eid(const void *slot, const ColiExpertSlotLayout *l) {
    return slot && l ? *eid_ptr_const(slot, l) : -1;
}
uint64_t coli_expert_slot_used(const void *slot, const ColiExpertSlotLayout *l) {
    return slot && l ? *used_ptr_const(slot, l) : 0;
}
void coli_expert_slot_set_eid(void *slot, const ColiExpertSlotLayout *l, int eid) {
    if (slot && l) *eid_ptr(slot, l) = eid;
}
void coli_expert_slot_set_used(void *slot, const ColiExpertSlotLayout *l, uint64_t used) {
    if (slot && l) *used_ptr(slot, l) = used;
}
void coli_expert_slot_touch(void *slot, const ColiExpertSlotLayout *l, uint64_t *clock) {
    if (slot && l && clock) *used_ptr(slot, l) = __atomic_add_fetch(clock, 1, __ATOMIC_RELAXED);
}

void coli_expert_record_demand(ColiExpertLayerStore *s, int eid) {
    if (!s || eid < 0 || eid >= s->n_experts) return;
    if (s->heat && s->heat[eid] != UINT32_MAX) s->heat[eid]++;
    if (s->usage && s->usage[eid] != UINT32_MAX) s->usage[eid]++;
    if (s->last && s->access_clock) s->last[eid] = ++*s->access_clock;
}

static int find_index(const void *slots, int n, const ColiExpertSlotLayout *l,
                      int eid, int reservations) {
    if (!slots || n <= 0 || !l) return -1;
    const int reservation = -(eid + 2);
    for (int i = 0; i < n; ++i) {
        const int v = coli_expert_slot_eid(coli_expert_slot_at_const(slots, l, i), l);
        if (v == eid || (reservations && v == reservation)) return i;
    }
    return -1;
}

ColiExpertLookup coli_expert_lookup(ColiExpertLayerStore *s, int eid, int reservations) {
    ColiExpertLookup r = {0}; r.index = -1;
    if (!s) return r;
    int i = find_index(s->pin, s->npin, &s->layout, eid, reservations);
    if (i >= 0) { r.slot = coli_expert_slot_at(s->pin, &s->layout, i); r.index = i; r.from_pin = 1; return r; }
    const int nc = s->ncache ? *s->ncache : 0;
    i = find_index(s->cache, nc, &s->layout, eid, reservations);
    if (i >= 0) { r.slot = coli_expert_slot_at(s->cache, &s->layout, i); r.index = i; }
    return r;
}
int coli_expert_resident(ColiExpertLayerStore *s, int eid, int reservations) {
    return coli_expert_lookup(s, eid, reservations).slot != NULL;
}

int coli_expert_choose_lru(const ColiExpertLayerStore *s) {
    if (!s || !s->cache || !s->ncache || *s->ncache <= 0) return -1;
    int best = -1;
    for (int i = 0; i < *s->ncache; ++i) {
        const void *slot = coli_expert_slot_at_const(s->cache, &s->layout, i);
        const int eid = coli_expert_slot_eid(slot, &s->layout);
        if (eid == -1) return i;
        if (eid < -1) continue;
        if (best < 0 || coli_expert_slot_used(slot, &s->layout) <
                        coli_expert_slot_used(coli_expert_slot_at_const(s->cache, &s->layout, best), &s->layout)) best = i;
    }
    return best;
}

static int speculative_may_evict(const ColiExpertLayerStore *s, int victim, int candidate, int guard) {
    if (!guard || !s || !s->heat || !s->last || !s->access_clock || victim < 0 || candidate < 0) return 1;
    const uint32_t vh = s->heat[victim];
    if (vh < 2) return 1;
    const uint64_t vs = tier_lfru_score(vh, s->last[victim], *s->access_clock);
    const uint64_t cs = tier_lfru_score(s->heat[candidate], s->last[candidate], *s->access_clock);
    return vs + (vs >> 2) + (4u << 8) <= cs;
}

int coli_expert_begin_admission(ColiExpertLayerStore *s, int eid,
                                int speculative, int guard,
                                ColiExpertAdmission *out) {
    if (!s || !out || !s->cache || !s->ncache || s->cache_cap <= 0 ||
        eid < 0 || eid >= s->n_experts) return 0;
    memset(out, 0, sizeof(*out)); out->index = -1; out->victim_eid = -1;
    if (coli_expert_resident(s, eid, 1)) return 0;
    int index, is_new = 0;
    if (*s->ncache < s->cache_cap) { index = *s->ncache; is_new = 1; }
    else index = coli_expert_choose_lru(s);
    if (index < 0) return 0;
    void *slot = coli_expert_slot_at(s->cache, &s->layout, index);
    const int victim = coli_expert_slot_eid(slot, &s->layout);
    if (speculative && !is_new && !speculative_may_evict(s, victim, eid, guard)) return 0;
    out->slot = slot; out->index = index; out->is_new = is_new; out->victim_eid = victim;
    out->reservation_eid = -(eid + 2);
    if (is_new) *s->ncache = index + 1; /* reserve the index immediately */
    coli_expert_slot_set_eid(slot, &s->layout, out->reservation_eid);
    return 1;
}

void coli_expert_finish_admission(ColiExpertLayerStore *s,
                                  const ColiExpertAdmission *a,
                                  int eid, int success) {
    if (!s || !a || !a->slot) return;
    if (!success) {
        coli_expert_slot_set_eid(a->slot, &s->layout, -1);
        coli_expert_slot_set_used(a->slot, &s->layout, 0);
        return;
    }
    coli_expert_slot_set_eid(a->slot, &s->layout, eid);
    coli_expert_slot_touch(a->slot, &s->layout, s->clock);
}

void *coli_expert_acquire_controlled(ColiExpertLayerStore *s, int layer, int eid,
                                     int demand, int allow_admission, int guard,
                                     const ColiExpertStorageOps *ops, void *ctx,
                                     ColiExpertSchedulerStats *stats) {
    if (!s || !ops || !ops->load) return NULL;
    if (demand) coli_expert_record_demand(s, eid);
    ColiExpertLookup h = coli_expert_lookup(s, eid, 0);
    if (h.slot) {
        if (demand) {
            if (stats) { stats->hits++; if (h.from_pin) stats->pin_hits++; else stats->cache_hits++; }
            if (!h.from_pin) coli_expert_slot_touch(h.slot, &s->layout, s->clock);
        }
        return h.slot;
    }
    if (demand && stats) stats->misses++;
    if (!allow_admission) return NULL;
    ColiExpertAdmission a;
    if (!coli_expert_begin_admission(s, eid, !demand, guard, &a)) {
        if (!demand && stats) stats->speculative_drops++;
        return NULL;
    }
    if (!a.is_new && a.victim_eid >= 0 && ops->evict) {
        /* Restore the visible victim id while the storage callback releases it. */
        coli_expert_slot_set_eid(a.slot, &s->layout, a.victim_eid);
        ops->evict(ctx, layer, a.slot);
        coli_expert_slot_set_eid(a.slot, &s->layout, a.reservation_eid);
        if (stats) stats->evictions++;
    }
    const int ok = ops->load(ctx, layer, eid, a.slot, demand);
    coli_expert_finish_admission(s, &a, eid, ok);
    if (!ok) { if (!demand && stats) stats->speculative_drops++; return NULL; }
    if (stats) { stats->admissions++; if (!demand) stats->speculative_loads++; }
    return a.slot;
}

void *coli_expert_acquire(ColiExpertLayerStore *s, int layer, int eid,
                          int demand, int guard,
                          const ColiExpertStorageOps *ops, void *ctx,
                          ColiExpertSchedulerStats *stats) {
    return coli_expert_acquire_controlled(s, layer, eid, demand, 1, guard,
                                          ops, ctx, stats);
}

void *coli_expert_promote_loaded(ColiExpertLayerStore *s, void *loaded, int *evicted_eid) {
    if (evicted_eid) *evicted_eid = -1;
    if (!s || !loaded || !s->cache || !s->ncache || s->cache_cap <= 0) return NULL;
    int index;
    if (*s->ncache < s->cache_cap) index = (*s->ncache)++;
    else index = coli_expert_choose_lru(s);
    if (index < 0) return NULL;
    void *dst = coli_expert_slot_at(s->cache, &s->layout, index);
    if (evicted_eid) *evicted_eid = coli_expert_slot_eid(dst, &s->layout);
    unsigned char *tmp = (unsigned char *)malloc(s->layout.stride);
    if (!tmp) return NULL;
    memcpy(tmp, dst, s->layout.stride);
    memcpy(dst, loaded, s->layout.stride);
    memcpy(loaded, tmp, s->layout.stride);
    free(tmp);
    coli_expert_slot_touch(dst, &s->layout, s->clock);
    return dst;
}

int coli_expert_repin_pick(const ColiExpertLayerStore *s,
                           int *pin_index, int *candidate_eid, long *gain) {
    if (!s || !s->pin || s->npin < 1 || !s->heat || !s->last || !s->access_clock) return 0;
    int ids_stack[4096];
    if (s->npin > (int)(sizeof(ids_stack)/sizeof(ids_stack[0]))) return 0;
    for (int i = 0; i < s->npin; ++i)
        ids_stack[i] = coli_expert_slot_eid(coli_expert_slot_at_const(s->pin, &s->layout, i), &s->layout);
    return tier_pick_lfru(s->heat, s->last, *s->access_clock, s->n_experts,
                          ids_stack, s->npin, pin_index, candidate_eid, gain);
}

int coli_expert_repin_promote_cached(ColiExpertLayerStore *s,
                                     int pin_index, int candidate_eid,
                                     int layer,
                                     const ColiExpertStorageOps *ops,
                                     void *ctx) {
    if (!s || !s->pin || pin_index < 0 || pin_index >= s->npin ||
        candidate_eid < 0 || candidate_eid >= s->n_experts) return -1;
    ColiExpertLookup h = coli_expert_lookup(s, candidate_eid, 0);
    if (!h.slot || h.from_pin) return 0;
    if (!ops || !ops->evict || !s->layout.stride) return -1;

    void *pin = coli_expert_slot_at(s->pin, &s->layout, pin_index);
    unsigned char *tmp = (unsigned char *)malloc(s->layout.stride);
    if (!pin || !tmp) { free(tmp); return -1; }
    memcpy(tmp, pin, s->layout.stride);
    memcpy(pin, h.slot, s->layout.stride);
    memcpy(h.slot, tmp, s->layout.stride);
    free(tmp);

    /* h.slot now owns the replaced pin payload. Release that storage while
     * the promoted candidate remains alive exclusively in the pin slot. */
    ops->evict(ctx, layer, h.slot);
    coli_expert_slot_set_eid(h.slot, &s->layout, -1);
    coli_expert_slot_set_used(h.slot, &s->layout, 0);
    coli_expert_slot_touch(pin, &s->layout, s->clock);
    return 1;
}

void coli_expert_decay_heat(ColiExpertLayerStore *s) {
    if (s && s->heat && s->n_experts > 0) tier_decay(s->heat, s->n_experts);
}


typedef struct {
    int flat_id;
    uint32_t count;
} ColiUsageRank;

static int usage_rank_cmp(const void *a, const void *b) {
    const ColiUsageRank *x = (const ColiUsageRank *)a;
    const ColiUsageRank *y = (const ColiUsageRank *)b;
    if (x->count != y->count) return x->count < y->count ? 1 : -1;
    return x->flat_id > y->flat_id ? 1 : (x->flat_id < y->flat_id ? -1 : 0);
}

int64_t coli_expert_usage_load(const char *path, uint32_t **usage,
                                int n_layers, int n_experts) {
    if (!path || !*path || !usage || n_layers <= 0 || n_experts <= 0) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int layer, eid;
    uint32_t count;
    int64_t total = 0;
    while (fscanf(f, "%d %d %u", &layer, &eid, &count) == 3) {
        if (layer < 0 || layer >= n_layers || eid < 0 || eid >= n_experts || !usage[layer]) continue;
        uint64_t sum = (uint64_t)usage[layer][eid] + count;
        usage[layer][eid] = sum > UINT32_MAX ? UINT32_MAX : (uint32_t)sum;
        total += count;
    }
    fclose(f);
    return total;
}

int coli_expert_usage_save(const char *path, uint32_t *const *usage,
                           int n_layers, int n_experts) {
    if (!path || !*path || !usage || n_layers <= 0 || n_experts <= 0) return 0;
    const size_t n = strlen(path);
    char *tmp = (char *)malloc(n + 5);
    if (!tmp) return 0;
    memcpy(tmp, path, n);
    memcpy(tmp + n, ".tmp", 5);
    FILE *f = fopen(tmp, "w");
    if (!f) { free(tmp); return 0; }
    int ok = 1;
    for (int layer = 0; layer < n_layers && ok; ++layer) {
        if (!usage[layer]) continue;
        for (int eid = 0; eid < n_experts; ++eid) {
            const uint32_t count = usage[layer][eid];
            if (count && fprintf(f, "%d %d %u\n", layer, eid, count) < 0) { ok = 0; break; }
        }
    }
    if (fclose(f) != 0) ok = 0;
    if (ok) ok = rename(tmp, path) == 0;
    if (!ok) remove(tmp);
    free(tmp);
    return ok;
}

int coli_expert_usage_top(uint32_t *const *usage, int n_layers, int n_experts,
                          int *flat_ids, int output_cap) {
    if (!usage || !flat_ids || output_cap <= 0 || n_layers <= 0 || n_experts <= 0) return 0;
    if ((size_t)n_layers > SIZE_MAX / (size_t)n_experts) return 0;
    const size_t max_records = (size_t)n_layers * (size_t)n_experts;
    if (max_records > SIZE_MAX / sizeof(ColiUsageRank)) return 0;
    ColiUsageRank *rank = (ColiUsageRank *)malloc(max_records * sizeof(*rank));
    if (!rank) return 0;
    size_t count = 0;
    for (int layer = 0; layer < n_layers; ++layer) {
        if (!usage[layer]) continue;
        for (int eid = 0; eid < n_experts; ++eid) if (usage[layer][eid]) {
            rank[count++] = (ColiUsageRank){layer * n_experts + eid, usage[layer][eid]};
        }
    }
    qsort(rank, count, sizeof(*rank), usage_rank_cmp);
    const int out = count < (size_t)output_cap ? (int)count : output_cap;
    for (int i = 0; i < out; ++i) flat_ids[i] = rank[i].flat_id;
    free(rank);
    return out;
}

int coli_expert_usage_top_file(const char *path, int n_layers, int n_experts,
                               int *flat_ids, int output_cap, int64_t *total) {
    if (total) *total = 0;
    if (!path || !flat_ids || output_cap <= 0 || n_layers <= 0 || n_experts <= 0) return 0;
    if ((size_t)n_layers > SIZE_MAX / (size_t)n_experts) return 0;
    const size_t cells = (size_t)n_layers * (size_t)n_experts;
    uint32_t **rows = (uint32_t **)calloc((size_t)n_layers, sizeof(*rows));
    uint32_t *data = (uint32_t *)calloc(cells, sizeof(*data));
    if (!rows || !data) { free(rows); free(data); return 0; }
    for (int l = 0; l < n_layers; ++l) rows[l] = data + (size_t)l * n_experts;
    const int64_t loaded = coli_expert_usage_load(path, rows, n_layers, n_experts);
    const int out = loaded > 0 ? coli_expert_usage_top(rows, n_layers, n_experts, flat_ids, output_cap) : 0;
    if (total) *total = loaded;
    free(data); free(rows);
    return out;
}


typedef struct {
    int eid;
    uint32_t count;
} ColiUsageExpertRank;

typedef struct {
    int layer;
    uint64_t score;
} ColiUsageLayerRank;

static int usage_expert_cmp(const void *a, const void *b) {
    const ColiUsageExpertRank *x = (const ColiUsageExpertRank *)a;
    const ColiUsageExpertRank *y = (const ColiUsageExpertRank *)b;
    if (x->count != y->count) return x->count < y->count ? 1 : -1;
    return x->eid > y->eid ? 1 : (x->eid < y->eid ? -1 : 0);
}

static int usage_layer_cmp(const void *a, const void *b) {
    const ColiUsageLayerRank *x = (const ColiUsageLayerRank *)a;
    const ColiUsageLayerRank *y = (const ColiUsageLayerRank *)b;
    if (x->score != y->score) return x->score < y->score ? 1 : -1;
    return x->layer > y->layer ? 1 : (x->layer < y->layer ? -1 : 0);
}

int coli_expert_usage_focus(uint32_t *const *usage, int n_layers, int n_experts,
                            int focus_layers, int *flat_ids, int output_cap,
                            int *selected_layers, int selected_cap,
                            int *per_layer_counts) {
    if (!usage || !flat_ids || output_cap <= 0 || n_layers <= 0 || n_experts <= 0)
        return 0;
    if (focus_layers <= 0)
        return coli_expert_usage_top(usage, n_layers, n_experts, flat_ids, output_cap);
    if (focus_layers > output_cap) focus_layers = output_cap;
    if (focus_layers > n_layers) focus_layers = n_layers;
    if (selected_cap < 0) selected_cap = 0;
    if (per_layer_counts) memset(per_layer_counts, 0, (size_t)n_layers * sizeof(*per_layer_counts));

    const int layer_cap = (output_cap + focus_layers - 1) / focus_layers;
    if ((size_t)n_layers > SIZE_MAX / (size_t)n_experts) return 0;
    const size_t cells = (size_t)n_layers * (size_t)n_experts;
    if (cells > SIZE_MAX / sizeof(ColiUsageExpertRank)) return 0;
    ColiUsageExpertRank *rank = (ColiUsageExpertRank *)malloc(cells * sizeof(*rank));
    ColiUsageLayerRank *layers = (ColiUsageLayerRank *)calloc((size_t)n_layers, sizeof(*layers));
    unsigned char *chosen = (unsigned char *)calloc((size_t)n_layers, 1);
    ColiUsageRank *candidates = (ColiUsageRank *)malloc(cells * sizeof(*candidates));
    int *counts = per_layer_counts ? per_layer_counts :
                  (int *)calloc((size_t)n_layers, sizeof(*counts));
    if (!rank || !layers || !chosen || !candidates || !counts) {
        free(rank); free(layers); free(chosen); free(candidates);
        if (!per_layer_counts) free(counts);
        return 0;
    }

    for (int l = 0; l < n_layers; ++l) {
        ColiUsageExpertRank *r = rank + (size_t)l * n_experts;
        for (int e = 0; e < n_experts; ++e) {
            r[e].eid = e;
            r[e].count = usage[l] ? usage[l][e] : 0;
        }
        qsort(r, (size_t)n_experts, sizeof(*r), usage_expert_cmp);
        uint64_t score = 0;
        const int lim = layer_cap < n_experts ? layer_cap : n_experts;
        for (int i = 0; i < lim; ++i) score += r[i].count;
        layers[l] = (ColiUsageLayerRank){l, score};
    }
    qsort(layers, (size_t)n_layers, sizeof(*layers), usage_layer_cmp);

    int selected_n = 0;
    for (int i = 0; i < n_layers && selected_n < focus_layers; ++i) {
        const int l = layers[i].layer;
        const ColiUsageExpertRank *r = rank + (size_t)l * n_experts;
        if (!r[0].count) continue;
        chosen[l] = 1;
        if (selected_layers && selected_n < selected_cap) selected_layers[selected_n] = l;
        ++selected_n;
    }
    if (!selected_n) {
        free(rank); free(layers); free(chosen); free(candidates);
        if (!per_layer_counts) free(counts);
        return 0;
    }

    /* Guarantee one hot expert in each chosen layer so the requested K is a
     * real policy parameter. Fill the rest globally by observed demand while
     * enforcing the equal-share ceiling. */
    int out = 0;
    for (int i = 0; i < n_layers && out < output_cap; ++i) if (chosen[i]) {
        ColiUsageExpertRank *r = rank + (size_t)i * n_experts;
        if (!r[0].count) continue;
        flat_ids[out++] = i * n_experts + r[0].eid;
        counts[i] = 1;
    }

    size_t ncand = 0;
    for (int l = 0; l < n_layers; ++l) if (chosen[l]) {
        ColiUsageExpertRank *r = rank + (size_t)l * n_experts;
        const int lim = layer_cap < n_experts ? layer_cap : n_experts;
        for (int i = 1; i < lim && r[i].count; ++i)
            candidates[ncand++] = (ColiUsageRank){l * n_experts + r[i].eid, r[i].count};
    }
    qsort(candidates, ncand, sizeof(*candidates), usage_rank_cmp);
    for (size_t i = 0; i < ncand && out < output_cap; ++i) {
        const int l = candidates[i].flat_id / n_experts;
        if (counts[l] >= layer_cap) continue;
        flat_ids[out++] = candidates[i].flat_id;
        ++counts[l];
    }

    free(rank); free(layers); free(chosen); free(candidates);
    if (!per_layer_counts) free(counts);
    return out;
}

int coli_expert_usage_focus_file(const char *path, int n_layers, int n_experts,
                                 int focus_layers, int *flat_ids, int output_cap,
                                 int *selected_layers, int selected_cap,
                                 int *per_layer_counts, int64_t *total) {
    if (total) *total = 0;
    if (!path || !flat_ids || output_cap <= 0 || n_layers <= 0 || n_experts <= 0) return 0;
    if ((size_t)n_layers > SIZE_MAX / (size_t)n_experts) return 0;
    const size_t cells = (size_t)n_layers * (size_t)n_experts;
    uint32_t **rows = (uint32_t **)calloc((size_t)n_layers, sizeof(*rows));
    uint32_t *data = (uint32_t *)calloc(cells, sizeof(*data));
    if (!rows || !data) { free(rows); free(data); return 0; }
    for (int l = 0; l < n_layers; ++l) rows[l] = data + (size_t)l * n_experts;
    const int64_t loaded = coli_expert_usage_load(path, rows, n_layers, n_experts);
    const int out = loaded > 0 ? coli_expert_usage_focus(rows, n_layers, n_experts,
            focus_layers, flat_ids, output_cap, selected_layers, selected_cap,
            per_layer_counts) : 0;
    if (total) *total = loaded;
    free(data); free(rows);
    return out;
}


int coli_expert_coupling_load(ColiExpertCoupling *c, const char *path,
                              int n_layers, int n_experts,
                              long *conditioning_entries) {
    if (conditioning_entries) *conditioning_entries = 0;
    if (!c || !path || n_layers <= 0 || n_experts <= 0) return 0;
    memset(c, 0, sizeof(*c));
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[16]; int version = 0; long declared = 0;
    if (fscanf(f, "%15s %d %ld", magic, &version, &declared) != 3 ||
        strcmp(magic, "COLIPAIRS") || version != 1) { fclose(f); return 0; }
    const size_t cells = (size_t)n_layers * 2u * (size_t)n_experts * COLI_EXPERT_COUPLE_M;
    if (cells > SIZE_MAX / sizeof(int16_t) || cells > SIZE_MAX / sizeof(float)) { fclose(f); return 0; }
    c->pred = (int16_t *)malloc(cells * sizeof(int16_t));
    c->score = (float *)calloc(cells, sizeof(float));
    if (!c->pred || !c->score) { fclose(f); coli_expert_coupling_destroy(c); return 0; }
    for (size_t i = 0; i < cells; ++i) c->pred[i] = -1;
    c->n_layers = n_layers; c->n_experts = n_experts;
    long used = 0; char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *q = line; int layer, depth, expert, consumed = 0;
        if (sscanf(q, "%d %d %d%n", &layer, &depth, &expert, &consumed) != 3) continue;
        q += consumed;
        if (layer < 0 || layer >= n_layers || (depth != 1 && depth != 2) ||
            expert < 0 || expert >= n_experts) continue;
        const size_t base = ((size_t)(layer * 2 + (depth - 1)) * n_experts + expert) * COLI_EXPERT_COUPLE_M;
        int j = 0;
        while (j < COLI_EXPERT_COUPLE_M) {
            int target; float weight;
            if (sscanf(q, " %d:%f%n", &target, &weight, &consumed) != 2) break;
            q += consumed;
            if (target >= 0 && target < n_experts) {
                c->pred[base + j] = (int16_t)target;
                c->score[base + j] = weight;
                ++j;
            }
        }
        if (j) ++used;
    }
    fclose(f);
    if (conditioning_entries) *conditioning_entries = used;
    return 1;
}

void coli_expert_coupling_destroy(ColiExpertCoupling *c) {
    if (!c) return;
    free(c->pred); free(c->score); memset(c, 0, sizeof(*c));
}

int coli_expert_coupling_predict(const ColiExpertCoupling *c,
                                 int layer, int depth,
                                 const int *routed, int routed_count,
                                 int *output, int output_cap) {
    if (!c || !c->pred || !c->score || !routed || routed_count <= 0 || !output ||
        output_cap <= 0 || layer < 0 || layer >= c->n_layers ||
        (depth != 1 && depth != 2)) return 0;
    float *sum = (float *)calloc((size_t)c->n_experts, sizeof(float));
    if (!sum) return 0;
    for (int k = 0; k < routed_count; ++k) {
        const int e = routed[k]; if (e < 0 || e >= c->n_experts) continue;
        const size_t base = ((size_t)(layer * 2 + (depth - 1)) * c->n_experts + e) * COLI_EXPERT_COUPLE_M;
        for (int j = 0; j < COLI_EXPERT_COUPLE_M && c->pred[base + j] >= 0; ++j)
            sum[c->pred[base + j]] += c->score[base + j];
    }
    int n = 0;
    while (n < output_cap) {
        int best = -1; float value = 0.f;
        for (int e = 0; e < c->n_experts; ++e) if (sum[e] > value) { value = sum[e]; best = e; }
        if (best < 0) break;
        output[n++] = best; sum[best] = 0.f;
    }
    free(sum); return n;
}
