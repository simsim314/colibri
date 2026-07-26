#include "split_plan.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int plan_fail(char *error, size_t cap, const char *fmt, ...) {
    if (error && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(error, cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

uint64_t coli_split_round_up(uint64_t value, uint64_t alignment) {
    if (!alignment) return value;
    uint64_t mask = alignment - 1u;
    if (value > UINT64_MAX - mask) return UINT64_MAX;
    return (value + mask) & ~mask;
}

static int contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle || !*needle) return 0;
    size_t n = strlen(needle);
    for (const char *p = haystack; *p; ++p) {
        size_t i = 0;
        while (i < n && p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i])) ++i;
        if (i == n) return 1;
    }
    return 0;
}

int coli_split_is_expert_tensor(const ColiGgufTensorInfo *t) {
    if (!t || !t->name) return 0;
    if (t->moe_layer >= 0 || t->expert_count > 0) return 1;
    return contains_ci(t->name, "_exps.") ||
           contains_ci(t->name, ".experts.") ||
           contains_ci(t->name, "expert.weight") ||
           contains_ci(t->name, "ffn_gate_exps") ||
           contains_ci(t->name, "ffn_up_exps") ||
           contains_ci(t->name, "ffn_down_exps");
}

int coli_split_is_vram_candidate(const ColiGgufTensorInfo *t) {
    if (!t || !t->name || coli_split_is_expert_tensor(t) || t->n_dims < 2 ||
        t->payload_size < 64u * 1024u) return 0;
    if (!contains_ci(t->name, "weight")) return 0;
    if (contains_ci(t->name, "token_embd") || contains_ci(t->name, "embed_tokens") ||
        contains_ci(t->name, "output.weight") || contains_ci(t->name, "lm_head") ||
        contains_ci(t->name, "norm") || contains_ci(t->name, "bias") ||
        contains_ci(t->name, "rope") || contains_ci(t->name, "position")) return 0;
    return 1;
}

static uint64_t kv_u64(const ColiGgufFile *g, const char *key, uint64_t fallback) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    uint64_t u = 0;
    int64_t i = 0;
    if (!kv) return fallback;
    if (coli_gguf_kv_read_u64(g, kv, &u)) return u;
    if (coli_gguf_kv_read_i64(g, kv, &i) && i >= 0) return (uint64_t)i;
    return fallback;
}

static int parse_layer(const ColiGgufTensorInfo *t) {
    if (!t) return -1;
    if (t->moe_layer >= 0) return t->moe_layer;
    if (!t->name) return -1;
    int layer = -1;
    if (sscanf(t->name, "blk.%d.", &layer) == 1 && layer >= 0) return layer;
    if (sscanf(t->name, "model.layers.%d.", &layer) == 1 && layer >= 0) return layer;
    return -1;
}

static int usage_load(const char *path, uint32_t *usage,
                      int n_layers, int n_experts,
                      uint64_t *total, uint64_t *nonzero,
                      char *error, size_t error_size) {
    *total = 0;
    *nonzero = 0;
    if (!path || !*path)
        return plan_fail(error, error_size,
                         "hot expert placement requires --usage-file PATH");
    FILE *f = fopen(path, "r");
    if (!f)
        return plan_fail(error, error_size, "cannot open usage file '%s': %s",
                         path, strerror(errno));
    int layer = 0, expert = 0;
    unsigned count = 0;
    while (fscanf(f, "%d %d %u", &layer, &expert, &count) == 3) {
        if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) continue;
        size_t idx = (size_t)layer * (size_t)n_experts + (size_t)expert;
        uint64_t sum = (uint64_t)usage[idx] + count;
        usage[idx] = sum > UINT32_MAX ? UINT32_MAX : (uint32_t)sum;
    }
    if (ferror(f)) {
        fclose(f);
        return plan_fail(error, error_size, "cannot read usage file '%s'", path);
    }
    fclose(f);
    for (int l = 0; l < n_layers; ++l) {
        for (int e = 0; e < n_experts; ++e) {
            uint32_t c = usage[(size_t)l * n_experts + e];
            if (c) ++*nonzero;
            *total += c;
        }
    }
    if (!*total)
        return plan_fail(error, error_size,
                         "usage file '%s' contains no GPT-OSS expert selections", path);
    return 1;
}

typedef struct {
    int layer;
    int expert;
    uint32_t usage;
    uint64_t bytes;
    uint64_t fast_bytes;
    uint32_t location;
    int present;
} ExpertBundle;

typedef struct {
    uint64_t decision_index;
    uint64_t bytes;
} DenseItem;

typedef struct {
    int bundle_id;
    uint32_t usage;
    int layer;
    int expert;
} BundleRank;

static int dense_desc(const void *a, const void *b) {
    const DenseItem *x = (const DenseItem *)a;
    const DenseItem *y = (const DenseItem *)b;
    if (x->bytes != y->bytes) return x->bytes > y->bytes ? -1 : 1;
    return x->decision_index < y->decision_index ? -1 :
           x->decision_index > y->decision_index ? 1 : 0;
}

static int bundle_hot_desc(const void *a, const void *b) {
    const BundleRank *x = (const BundleRank *)a;
    const BundleRank *y = (const BundleRank *)b;
    if (x->usage != y->usage) return x->usage > y->usage ? -1 : 1;
    if (x->layer != y->layer) return x->layer < y->layer ? -1 : 1;
    return x->expert < y->expert ? -1 : x->expert > y->expert ? 1 : 0;
}

static int append_decision(ColiSplitPlan *plan, uint64_t *capacity,
                           const ColiSplitDecision *value,
                           char *error, size_t error_size) {
    if (plan->entry_count == *capacity) {
        uint64_t next = *capacity ? *capacity * 2u : 1024u;
        if (next < *capacity || next > SIZE_MAX / sizeof(*plan->entries))
            return plan_fail(error, error_size, "too many split fragments");
        void *p = realloc(plan->entries, (size_t)next * sizeof(*plan->entries));
        if (!p) return plan_fail(error, error_size, "out of memory planning fragments");
        plan->entries = (ColiSplitDecision *)p;
        *capacity = next;
    }
    plan->entries[plan->entry_count++] = *value;
    return 1;
}

void coli_split_plan_destroy(ColiSplitPlan *plan) {
    if (!plan) return;
    free(plan->entries);
    memset(plan, 0, sizeof(*plan));
}

int coli_split_make_plan(const ColiGgufFile *model,
                         const char *usage_path,
                         uint64_t fast_budget,
                         uint64_t ram_budget,
                         uint64_t vram_budget,
                         ColiSplitPlan *plan,
                         ColiSplitPlanSummary *summary,
                         char *error,
                         size_t error_size) {
    if (!model || !plan || !summary || !model->tensor_count)
        return plan_fail(error, error_size, "invalid split planning arguments");
    if (model->split)
        return plan_fail(error, error_size, "re-splitting an already split model is not supported");

    memset(plan, 0, sizeof(*plan));
    memset(summary, 0, sizeof(*summary));
    summary->fast_budget = fast_budget;
    summary->ram_budget = ram_budget;
    summary->vram_budget = vram_budget;
    plan->tensor_count = model->tensor_count;

    const uint64_t layers64 = kv_u64(model, "gpt-oss.block_count", 0);
    const uint64_t experts64 = kv_u64(model, "gpt-oss.expert_count", 0);
    if (!layers64 || !experts64 || layers64 > INT_MAX || experts64 > INT_MAX)
        return plan_fail(error, error_size,
                         "hot split currently requires GPT-OSS block/expert metadata");
    const int n_layers = (int)layers64;
    const int n_experts = (int)experts64;
    if ((size_t)n_layers > SIZE_MAX / (size_t)n_experts)
        return plan_fail(error, error_size, "expert usage dimensions overflow");
    const size_t cells = (size_t)n_layers * (size_t)n_experts;
    uint32_t *usage = (uint32_t *)calloc(cells, sizeof(*usage));
    ExpertBundle *bundles = (ExpertBundle *)calloc(cells, sizeof(*bundles));
    BundleRank *rank = (BundleRank *)calloc(cells, sizeof(*rank));
    DenseItem *vram_items = (DenseItem *)calloc((size_t)model->tensor_count, sizeof(*vram_items));
    if (!usage || !bundles || !rank || !vram_items) {
        free(usage); free(bundles); free(rank); free(vram_items);
        return plan_fail(error, error_size, "out of memory planning hot experts");
    }
    if (!usage_load(usage_path, usage, n_layers, n_experts,
                    &plan->usage_total, &plan->usage_nonzero,
                    error, error_size)) {
        free(usage); free(bundles); free(rank); free(vram_items);
        return 0;
    }

    uint64_t fixed = coli_split_round_up(model->data_offset, 4096u);
    if (fixed == UINT64_MAX || fixed > UINT64_MAX - 2u * 1024u * 1024u) {
        free(usage); free(bundles); free(rank); free(vram_items);
        return plan_fail(error, error_size, "split header size overflow");
    }
    fixed += 2u * 1024u * 1024u;
    summary->header_and_manifest_bytes = fixed;
    if (fast_budget < fixed) {
        free(usage); free(bundles); free(rank); free(vram_items);
        return plan_fail(error, error_size,
                         "fast budget is smaller than model metadata/header requirement");
    }

    uint64_t capacity = 0;
    size_t nvram = 0;
    for (uint64_t ti = 0; ti < model->tensor_count; ++ti) {
        const ColiGgufTensorInfo *t = &model->tensors[ti];
        const int is_expert = coli_split_is_expert_tensor(t);
        const int layer = is_expert ? parse_layer(t) : -1;
        const int can_fragment = is_expert && layer >= 0 && layer < n_layers &&
            t->storage_kind == COLI_TENSOR_STORAGE_DENSE && t->n_dims == 3 &&
            t->dims[t->n_dims - 1] == (uint64_t)n_experts &&
            t->payload_size && t->payload_size % (uint64_t)n_experts == 0;
        const uint64_t pieces = can_fragment ? (uint64_t)n_experts : 1u;
        const uint64_t piece_bytes = can_fragment ? t->payload_size / pieces : t->payload_size;
        if (can_fragment) ++summary->fragmented_tensor_count;
        for (uint64_t p = 0; p < pieces; ++p) {
            ColiSplitDecision d;
            memset(&d, 0, sizeof(d));
            d.location = COLI_SPLIT_SLOW_TAIL_MMAP;
            d.tensor_index = (uint32_t)ti;
            d.original_offset = t->absolute_offset + p * piece_bytes;
            d.bytes = piece_bytes;
            d.allocation_bytes = coli_split_round_up(piece_bytes, 4096u);
            d.moe_layer = can_fragment ? layer : -1;
            d.expert_id = can_fragment ? (int32_t)p : -1;
            d.bundle_id = can_fragment ? layer * n_experts + (int)p : -1;
            d.usage_count = can_fragment ? usage[(size_t)layer * n_experts + p] : 0;
            d.is_expert = is_expert;
            d.is_vram_candidate = !can_fragment && coli_split_is_vram_candidate(t);
            if (d.allocation_bytes == UINT64_MAX ||
                !append_decision(plan, &capacity, &d, error, error_size)) {
                free(usage); free(bundles); free(rank); free(vram_items);
                coli_split_plan_destroy(plan);
                return 0;
            }
            summary->source_payload_bytes += piece_bytes;
            if (can_fragment) {
                ExpertBundle *b = &bundles[d.bundle_id];
                b->layer = layer;
                b->expert = (int)p;
                b->usage = d.usage_count;
                b->present = 1;
                b->bytes += d.bytes;
                b->fast_bytes += d.allocation_bytes;
            } else if (d.is_vram_candidate) {
                summary->vram_candidate_bytes += d.bytes;
                vram_items[nvram++] = (DenseItem){plan->entry_count - 1u, d.bytes};
            }
        }
    }

    qsort(vram_items, nvram, sizeof(*vram_items), dense_desc);
    uint64_t vram_left = vram_budget;
    for (size_t i = 0; i < nvram; ++i) {
        ColiSplitDecision *d = &plan->entries[vram_items[i].decision_index];
        if (d->bytes <= vram_left) {
            d->location = COLI_SPLIT_PRELOAD_VRAM;
            vram_left -= d->bytes;
        }
    }

    uint64_t fast_left = fast_budget - fixed;
    /* Always-used nonexpert tensors are placed before routed experts. */
    for (uint64_t i = 0; i < plan->entry_count; ++i) {
        ColiSplitDecision *d = &plan->entries[i];
        if (d->bundle_id >= 0 || d->location != COLI_SPLIT_SLOW_TAIL_MMAP) continue;
        if (d->allocation_bytes <= fast_left) {
            d->location = COLI_SPLIT_FAST_MMAP;
            fast_left -= d->allocation_bytes;
        }
    }

    uint64_t ram_left = ram_budget;
    /* Any always-used dense tensor that did not fit fast storage gets RAM first. */
    for (uint64_t i = 0; i < plan->entry_count; ++i) {
        ColiSplitDecision *d = &plan->entries[i];
        if (d->bundle_id >= 0 || d->location != COLI_SPLIT_SLOW_TAIL_MMAP) continue;
        if (d->bytes <= ram_left) {
            d->location = COLI_SPLIT_PRELOAD_RAM;
            ram_left -= d->bytes;
        }
    }

    size_t nrank = 0;
    for (size_t id = 0; id < cells; ++id) if (bundles[id].present) {
        rank[nrank++] = (BundleRank){(int)id, bundles[id].usage,
                                    bundles[id].layer, bundles[id].expert};
    }
    plan->expert_bundle_count = nrank;
    qsort(rank, nrank, sizeof(*rank), bundle_hot_desc);

    /* RAM is the fastest host tier: put the hottest complete experts there. */
    for (size_t i = 0; i < nrank; ++i) {
        ExpertBundle *b = &bundles[rank[i].bundle_id];
        if (b->bytes <= ram_left) {
            b->location = COLI_SPLIT_PRELOAD_RAM;
            ram_left -= b->bytes;
            ++summary->hot_ram_bundle_count;
        }
    }
    /* Next-hottest complete experts occupy the fast mmap primary. */
    for (size_t i = 0; i < nrank; ++i) {
        ExpertBundle *b = &bundles[rank[i].bundle_id];
        if (b->location) continue;
        if (b->fast_bytes <= fast_left) {
            b->location = COLI_SPLIT_FAST_MMAP;
            fast_left -= b->fast_bytes;
            ++summary->fast_bundle_count;
        } else {
            b->location = COLI_SPLIT_SLOW_TAIL_MMAP;
            ++summary->tail_bundle_count;
        }
    }
    for (uint64_t i = 0; i < plan->entry_count; ++i) {
        ColiSplitDecision *d = &plan->entries[i];
        if (d->bundle_id >= 0) d->location = bundles[d->bundle_id].location;
    }

    summary->fast_allocated = fixed;
    for (uint64_t i = 0; i < plan->entry_count; ++i) {
        const ColiSplitDecision *d = &plan->entries[i];
        switch (d->location) {
            case COLI_SPLIT_FAST_MMAP:
                summary->fast_allocated += d->allocation_bytes;
                ++summary->fast_fragment_count;
                break;
            case COLI_SPLIT_PRELOAD_RAM:
                summary->preload_ram_bytes += d->bytes;
                ++summary->ram_fragment_count;
                break;
            case COLI_SPLIT_PRELOAD_VRAM:
                summary->preload_vram_bytes += d->bytes;
                ++summary->vram_fragment_count;
                break;
            default:
                summary->tail_bytes += d->bytes;
                ++summary->tail_fragment_count;
                break;
        }
    }
    if (summary->fast_allocated > fast_budget ||
        summary->preload_ram_bytes > ram_budget ||
        summary->preload_vram_bytes > vram_budget) {
        free(usage); free(bundles); free(rank); free(vram_items);
        coli_split_plan_destroy(plan);
        return plan_fail(error, error_size, "internal hot split budget error");
    }
    summary->fast_unused_bytes = fast_budget - summary->fast_allocated;
    summary->ram_unused_bytes = ram_budget - summary->preload_ram_bytes;
    summary->vram_unused_bytes = vram_budget - summary->preload_vram_bytes;

    free(usage); free(bundles); free(rank); free(vram_items);
    if (error && error_size) error[0] = 0;
    return 1;
}
