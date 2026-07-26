#include "gguf_qwen35moe.h"
#include "tensor.h"
#include "ggml_quants.h"
#include "gguf_tokenizer.h"
#include "f32_kernels.h"
#include "expert_scheduler.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif

typedef struct {
    int eid;
    ColiTensor *gate, *up, *down;
    int owns_pruned;
    uint64_t used;
} Q35ExpertSlot;

static const ColiExpertSlotLayout g_q35_slot_layout = {
    sizeof(Q35ExpertSlot), offsetof(Q35ExpertSlot, eid), offsetof(Q35ExpertSlot, used)
};

typedef struct {
    int recurrent;
    int recurrent_index;
    int attention_index;

    ColiTensor attn_norm;
    ColiTensor post_norm;

    /* Full-attention tensors. q contains query followed by the output gate. */
    ColiTensor q, k, v, q_norm, k_norm, o;

    /* Gated DeltaNet tensors. The GGUF converter has already split QKV and Z. */
    ColiTensor qkv, z, beta, alpha, conv1d, dt_bias, a, ssm_norm, ssm_out;

    /* Optional NextN/MTP tensors (only present on appended decoder blocks). */
    ColiTensor nextn_eh_proj, nextn_enorm, nextn_hnorm;
    ColiTensor nextn_embed_tokens, nextn_head, nextn_head_norm;
    int has_nextn_embed, has_nextn_head, has_nextn_head_norm;

    /* Routed and shared experts. */
    ColiTensor router;
    ColiTensor gate_exps, up_exps, down_exps;
    ColiTensor shared_gate_inp, shared_gate, shared_up, shared_down;
    ColiTensor *gate_expert, *up_expert, *down_expert;
    int expert_views;

    Q35ExpertSlot *pin, *cache;
    int npin, ncache, cache_cap;
    uint32_t *heat, *last, *usage;
} Q35Layer;

typedef struct {
    ColiGgufFile gguf;
    ColiGgufTokenizer *tokenizer;
    char *architecture;

    int n_layers, n_layers_all, n_mtp_layers, n_exec_layers, hidden, vocab;
    int n_heads, n_kv_heads, head_dim, key_length, value_length, kv_dim;
    int rope_dims, rope_sections[4], context_length, full_attention_interval;

    int n_experts, n_expert_used, expert_ff, shared_ff;

    int conv_kernel, state_size, n_key_heads, n_value_heads;
    int inner_size, value_head_dim, key_dim, value_dim, conv_dim;
    int n_recurrent, n_attention, n_attention_all, n_attention_exec;

    float rope_base, eps, attention_scale;

    ColiTensor token_embd, output_norm, output;
    int tied_output;
    Q35Layer *layers;
    int mtp_enabled, mtp_draft_max, mtp_layer;
    uint64_t mtp_proposed, mtp_accepted, mtp_steps, mtp_kv_updates;
    uint64_t mtp_verify_batches, mtp_verify_tokens, mtp_verify_replays;

    int context_capacity;
    float *k_cache, *v_cache;
    float *conv_state, *recurrent_state;
#ifdef COLI_CUDA
    float *k_cache_dev, *v_cache_dev;
    float *conv_state_dev, *recurrent_state_dev;
#endif

    size_t cuda_dense_bytes, cuda_expert_bytes, cuda_weight_bytes;
    size_t expert_bytes;
    int cuda_requested;
    int cuda_dense_complete;
    int cuda_fell_back_to_cpu;
    const char *cuda_fallback_tensor;

    uint64_t expert_clock;
    uint32_t expert_access_clock;
    ColiExpertSchedulerStats scheduler_stats;
    int scheduler_evict_guard;
    int focus_layer_count, focus_layer_effective;
    unsigned char *focus_layer_mask;
    int focus_group_enabled, focus_group_min;
    uint64_t focus_group_calls, focus_group_experts, focus_group_failures;
    int repin_interval, tokens_since_repin;
    int pilot, pilot_real, pilot_real_downgraded, pilot_k;
    int couple, couple_k, couple_d;
    ColiExpertCoupling coupling;
    char usage_path[2048];
    int64_t usage_history;

    /* Qwen3.5-MoE's top-k routing commonly runs with only one replaceable slot per
     * layer on 6 GB cards. After the first CUDA residency failure, keep using
     * existing pin/LRU hits but stop destructive miss admissions and execute
     * misses directly from the mmap-backed tensors on CPU. */
    int cuda_expert_admission_disabled;
    int cuda_expert_compute_disabled;
    int cuda_expert_failure_reported;
    uint64_t cuda_expert_residency_failures;
    uint64_t cuda_expert_compute_failures;
    uint64_t cuda_expert_cpu_fallbacks;
    int cuda_router_gpu_enabled;
    uint64_t cuda_router_gpu_calls;
    uint64_t cuda_router_cpu_fallbacks;

    float expert_zero_threshold;
    uint64_t expert_prune_materializations;
    uint64_t expert_prune_values_seen;
    uint64_t expert_prune_values_zeroed;

    int verbose;
    ColiExec exec;
} Q35Model;

typedef struct {
    /* Host scratch. */
    float *x, *norm, *mixer, *post;
    float *qg, *attn_q, *attn_gate, *k, *v, *attn, *scores;
    float *qkv, *z, *beta, *alpha, *conv, *delta;
    float *router, *gate, *up, *expert_out, *moe;
    float *shared_gate, *shared_up, *shared_out;
    float *logits, *weight;
    float *hidden_norm, *mtp_hidden, *mtp_x, *mtp_cat, *mtp_logits;
    int *top_idx;
    float *top_w;

    /* Bounded target-verification batch (S <= mtp_draft_max <= 8). */
    int verify_cap;
    float *vx, *vnorm, *vmixer, *vpost;
    float *vqg, *vattn_q, *vattn_gate, *vk, *vv, *vattn;
    float *vqkv, *vz, *vbeta, *valpha, *vdelta;
    float *vrouter, *vgate, *vup, *vexpert_in, *vexpert_out, *vmoe;
    float *vshared_gate, *vshared_up, *vshared_out, *vlogits;
    int *vtop_idx; float *vtop_w;
    float *conv_backup, *state_backup, *hidden_backup;

#ifdef COLI_CUDA
    /* Device scratch. */
    float *dx, *dnorm, *dmixer, *dpost;
    float *dqg, *dattn_q, *dattn_gate, *dk, *dv, *dattn, *dscores;
    float *dqkv, *dz, *dbeta, *dalpha, *ddelta;
    float *drouter, *dgate, *dup, *dexpert_out, *dgroup_out, *dmoe;
    float *dshared_gate, *dshared_up, *dshared_out;
    float *dlogits, *dhidden_norm, *dmtp_hidden, *dmtp_x, *dmtp_cat, *dmtp_logits;
    float *dvx, *dvnorm, *dvmixer, *dvpost;
    float *dvqg, *dvattn_q, *dvattn_gate, *dvk, *dvv, *dvattn;
    float *dvqkv, *dvz, *dvbeta, *dvalpha, *dvdelta;
    float *dvrouter, *dvgate, *dvup, *dvexpert_in, *dvexpert_out, *dvmoe;
    float *dvshared_gate, *dvshared_up, *dvshared_out, *dvlogits;
    float *dconv_backup, *dstate_backup, *dhidden_backup;
    int cuda_device, cuda_allocated;
#endif
} Q35Scratch;

static double q35_now_sec(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int q35_errf(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

static int q35_kv_u64(const ColiGgufFile *g, const char *key, uint64_t *out,
                       int required, char *err, size_t cap) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    if (!kv) return required ? q35_errf(err, cap, "missing metadata: %s", key) : 0;
    return coli_gguf_kv_read_u64(g, kv, out) ? 1 :
           q35_errf(err, cap, "invalid integer metadata: %s", key);
}

static int q35_kv_f32(const ColiGgufFile *g, const char *key, float *out,
                       int required, char *err, size_t cap) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    double v;
    if (!kv) return required ? q35_errf(err, cap, "missing metadata: %s", key) : 0;
    if (!coli_gguf_kv_read_f64(g, kv, &v))
        return q35_errf(err, cap, "invalid float metadata: %s", key);
    *out = (float)v;
    return 1;
}

static int q35_to_int(uint64_t v, int *out, const char *name,
                       char *err, size_t cap) {
    if (v == 0 || v > INT_MAX)
        return q35_errf(err, cap, "invalid %s: %llu", name, (unsigned long long)v);
    *out = (int)v;
    return 1;
}

static void q35_tensor_free(ColiTensor *t) { coli_tensor_destroy(t); }

static int q35_tensor_dims(const ColiTensor *t, int nd, const uint64_t *dims) {
    if (!t || (int)t->n_dims != nd) return 0;
    for (int i = 0; i < nd; ++i) if (t->dims[i] != dims[i]) return 0;
    return 1;
}

static int q35_load_tensor(Q35Model *m, const char *name, ColiTensor *out,
                            int nd, const uint64_t *dims,
                            char *err, size_t cap) {
    const ColiGgufTensorInfo *ti = coli_gguf_find_tensor(&m->gguf, name);
    if (!ti) return q35_errf(err, cap, "missing tensor: %s", name);
    if (!coli_tensor_bind_gguf(&m->gguf, ti, out, err, cap)) return 0;
    if (!q35_tensor_dims(out, nd, dims)) {
        q35_tensor_free(out);
        return q35_errf(err, cap, "wrong dimensions for tensor: %s", name);
    }
    return 1;
}

static int q35_load_layer_tensor(Q35Model *m, int layer, const char *suffix,
                                  ColiTensor *out, int nd, const uint64_t *dims,
                                  char *err, size_t cap) {
    char name[160];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return q35_load_tensor(m, name, out, nd, dims, err, cap);
}

static int q35_load_moe_tensor(Q35Model *m, int layer, uint32_t projection,
                               ColiTensor *out, int nd, const uint64_t *dims,
                               char *err, size_t cap) {
    const ColiGgufTensorInfo *ti = coli_gguf_find_moe_tensor(&m->gguf, layer, projection);
    if (!ti) return q35_errf(err, cap, "missing routed-MoE tensor at layer %d projection %u",
                             layer, projection);
    if (!coli_tensor_bind_gguf(&m->gguf, ti, out, err, cap)) return 0;
    if (!q35_tensor_dims(out, nd, dims)) {
        q35_tensor_free(out);
        return q35_errf(err, cap, "wrong dimensions for routed-MoE tensor: %s", ti->name);
    }
    return 1;
}

static int q35_load_optional_tensor(Q35Model *m, const char *name, ColiTensor *out,
                                    int nd, const uint64_t *dims, int *present,
                                    char *err, size_t cap) {
    const ColiGgufTensorInfo *ti = coli_gguf_find_tensor(&m->gguf, name);
    if (!ti) { if (present) *present = 0; return 1; }
    if (!q35_load_tensor(m, name, out, nd, dims, err, cap)) return 0;
    if (present) *present = 1;
    return 1;
}

static int q35_load_optional_layer_tensor(Q35Model *m, int layer, const char *suffix,
                                          ColiTensor *out, int nd, const uint64_t *dims,
                                          int *present, char *err, size_t cap) {
    char name[192];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return q35_load_optional_tensor(m, name, out, nd, dims, present, err, cap);
}

static void q35_slot_clear(Q35ExpertSlot *s) {
    if (!s) return;
    if (s->owns_pruned) {
        if (s->gate) { coli_tensor_destroy(s->gate); free(s->gate); }
        if (s->up) { coli_tensor_destroy(s->up); free(s->up); }
        if (s->down) { coli_tensor_destroy(s->down); free(s->down); }
    }
    s->gate = s->up = s->down = NULL;
    s->owns_pruned = 0;
    s->eid = -1;
    s->used = 0;
}

static void q35_layer_slots_free(Q35Layer *l) {
    if (!l) return;
    for (int i = 0; i < l->npin; ++i) q35_slot_clear(&l->pin[i]);
    for (int i = 0; i < l->cache_cap; ++i) q35_slot_clear(&l->cache[i]);
}

static void q35_layer_free(Q35Layer *l) {
    if (!l) return;
    /* Destroy child views before their aggregate tensors. Today each expert
     * owns an independent CUDA upload, but this ordering also stays correct if
     * row views later become zero-copy views of a resident aggregate. */
    for (int e = 0; e < l->expert_views; ++e) {
        q35_tensor_free(&l->gate_expert[e]);
        q35_tensor_free(&l->up_expert[e]);
        q35_tensor_free(&l->down_expert[e]);
    }
#define QFREE(name) q35_tensor_free(&l->name)
    QFREE(attn_norm); QFREE(post_norm);
    QFREE(q); QFREE(k); QFREE(v); QFREE(q_norm); QFREE(k_norm); QFREE(o);
    QFREE(qkv); QFREE(z); QFREE(beta); QFREE(alpha); QFREE(conv1d); QFREE(dt_bias); QFREE(a);
    QFREE(nextn_eh_proj); QFREE(nextn_enorm); QFREE(nextn_hnorm);
    QFREE(nextn_embed_tokens); QFREE(nextn_head); QFREE(nextn_head_norm);
    QFREE(ssm_norm); QFREE(ssm_out);
    QFREE(router); QFREE(gate_exps); QFREE(up_exps); QFREE(down_exps);
    QFREE(shared_gate_inp); QFREE(shared_gate); QFREE(shared_up); QFREE(shared_down);
#undef QFREE
    free(l->gate_expert); free(l->up_expert); free(l->down_expert);
    free(l->pin); free(l->cache); free(l->heat); free(l->last); free(l->usage);
    memset(l, 0, sizeof(*l));
}

static void q35_release_state(Q35Model *m) {
    if (!m) return;
#ifdef COLI_CUDA
    if (m->k_cache_dev) coli_cuda_pipe_free(m->exec.device, m->k_cache_dev);
    if (m->v_cache_dev) coli_cuda_pipe_free(m->exec.device, m->v_cache_dev);
    if (m->conv_state_dev) coli_cuda_pipe_free(m->exec.device, m->conv_state_dev);
    if (m->recurrent_state_dev) coli_cuda_pipe_free(m->exec.device, m->recurrent_state_dev);
    m->k_cache_dev = m->v_cache_dev = NULL;
    m->conv_state_dev = m->recurrent_state_dev = NULL;
#endif
    free(m->k_cache); free(m->v_cache); free(m->conv_state); free(m->recurrent_state);
    m->k_cache = m->v_cache = NULL;
    m->conv_state = m->recurrent_state = NULL;
    m->context_capacity = 0;
}

static void q35_model_free(Q35Model *m) {
    if (!m) return;
    if (m->layers) for (int i = 0; i < m->n_layers_all; ++i) q35_layer_slots_free(&m->layers[i]);
    if (m->layers) for (int i = 0; i < m->n_layers_all; ++i) q35_layer_free(&m->layers[i]);
    free(m->layers);
    q35_tensor_free(&m->token_embd);
    q35_tensor_free(&m->output_norm);
    q35_tensor_free(&m->output);
    q35_release_state(m);
    free(m->focus_layer_mask);
    free(m->architecture);
    coli_expert_coupling_destroy(&m->coupling);
    coli_gguf_tokenizer_destroy(m->tokenizer);
    coli_gguf_close(&m->gguf);
    memset(m, 0, sizeof(*m));
}

static int q35_model_config(Q35Model *m, char *err, size_t cap) {
    const ColiGgufKV *akv = coli_gguf_find_kv(&m->gguf, "general.architecture");
    if (!akv || !coli_gguf_kv_read_string(&m->gguf, akv, &m->architecture))
        return q35_errf(err, cap, "general.architecture missing");
    if (strcmp(m->architecture, "qwen35moe") != 0)
        return q35_errf(err, cap, "unsupported GGUF architecture '%s' (expected qwen35moe)", m->architecture);

    uint64_t u;
#define QREADI(key, field) do { \
    if (!q35_kv_u64(&m->gguf, key, &u, 1, err, cap) || \
        !q35_to_int(u, &m->field, key, err, cap)) return 0; \
} while (0)
    QREADI("qwen35moe.block_count", n_layers_all);
    QREADI("qwen35moe.context_length", context_length);
    QREADI("qwen35moe.embedding_length", hidden);
    QREADI("qwen35moe.attention.head_count", n_heads);
    QREADI("qwen35moe.attention.head_count_kv", n_kv_heads);
    QREADI("qwen35moe.attention.key_length", key_length);
    QREADI("qwen35moe.attention.value_length", value_length);
    QREADI("qwen35moe.expert_count", n_experts);
    QREADI("qwen35moe.expert_used_count", n_expert_used);
    QREADI("qwen35moe.expert_feed_forward_length", expert_ff);
    QREADI("qwen35moe.expert_shared_feed_forward_length", shared_ff);
    QREADI("qwen35moe.ssm.conv_kernel", conv_kernel);
    QREADI("qwen35moe.ssm.state_size", state_size);
    QREADI("qwen35moe.ssm.group_count", n_key_heads);
    QREADI("qwen35moe.ssm.time_step_rank", n_value_heads);
    QREADI("qwen35moe.ssm.inner_size", inner_size);
    QREADI("qwen35moe.full_attention_interval", full_attention_interval);
    QREADI("qwen35moe.rope.dimension_count", rope_dims);
#undef QREADI

    m->n_mtp_layers = 0;
    if (q35_kv_u64(&m->gguf, "qwen35moe.nextn_predict_layers", &u, 0, err, cap)) {
        if (u > (uint64_t)m->n_layers_all || u > INT_MAX)
            return q35_errf(err, cap, "invalid qwen35moe.nextn_predict_layers");
        m->n_mtp_layers = (int)u;
    }
    m->n_layers = m->n_layers_all - m->n_mtp_layers;
    if (m->n_layers < 1 || m->n_mtp_layers > 1)
        return q35_errf(err, cap, "unsupported trunk/MTP layer split: %d/%d",
                        m->n_layers, m->n_mtp_layers);
    m->mtp_layer = m->n_mtp_layers ? m->n_layers : -1;

    for (int i = 0; i < 4; ++i) m->rope_sections[i] = 0;
    const ColiGgufKV *rskv = coli_gguf_find_kv(&m->gguf, "qwen35moe.rope.dimension_sections");
    if (rskv) {
        uint32_t *a = NULL; uint64_t n = 0;
        if (!coli_gguf_kv_read_u32_array(&m->gguf, rskv, &a, &n) || n < 3 || n > 4) {
            free(a); return q35_errf(err, cap, "invalid qwen35moe.rope.dimension_sections");
        }
        int sum = 0;
        for (uint64_t i = 0; i < n; ++i) { m->rope_sections[i] = (int)a[i]; sum += (int)a[i]; }
        free(a);
        /* GGUF sections count rotary pairs. For the 35B model 11+11+10=32,
         * corresponding to 64 rotary dimensions. */
        if (sum * 2 != m->rope_dims)
            return q35_errf(err, cap, "M-RoPE sections do not match rotary dimension: %d*2 != %d", sum, m->rope_dims);
    } else {
        m->rope_sections[0] = m->rope_dims / 2;
    }

    if (m->n_heads % m->n_kv_heads || m->key_length != m->value_length)
        return q35_errf(err, cap, "invalid full-attention head dimensions: heads=%d/%d key=%d value=%d",
                         m->n_heads, m->n_kv_heads, m->key_length, m->value_length);
    m->head_dim = m->key_length;
    m->kv_dim = m->n_kv_heads * m->head_dim;
    if (m->rope_dims > m->head_dim || (m->rope_dims & 1))
        return q35_errf(err, cap, "invalid partial M-RoPE dimension count");

    if (m->inner_size % m->n_value_heads)
        return q35_errf(err, cap, "SSM inner size is not divisible by value heads");
    m->value_head_dim = m->inner_size / m->n_value_heads;
    if (m->state_size != m->value_head_dim)
        return q35_errf(err, cap, "unsupported Qwen3.5 state/value head mismatch: %d/%d",
                         m->state_size, m->value_head_dim);
    if (m->n_value_heads % m->n_key_heads)
        return q35_errf(err, cap, "value-head count is not divisible by key-head count");
    m->key_dim = m->state_size * m->n_key_heads;
    m->value_dim = m->value_head_dim * m->n_value_heads;
    m->conv_dim = 2 * m->key_dim + m->value_dim;
    if (m->conv_kernel < 1 || m->conv_kernel > 16)
        return q35_errf(err, cap, "unsupported convolution width: %d", m->conv_kernel);
    if (m->full_attention_interval < 1)
        return q35_errf(err, cap, "invalid full-attention interval: %d", m->full_attention_interval);
    if (m->n_expert_used > m->n_experts)
        return q35_errf(err, cap, "expert_used_count exceeds expert_count");

    m->n_recurrent = 0;
    m->n_attention = 0;
    for (int i = 0; i < m->n_layers; ++i) {
        if ((i + 1) % m->full_attention_interval) ++m->n_recurrent;
        else ++m->n_attention;
    }
    m->n_attention_all = m->n_attention + m->n_mtp_layers;

    const ColiGgufTensorInfo *emb = coli_gguf_find_tensor(&m->gguf, "token_embd.weight");
    if (!emb || emb->n_dims != 2 || emb->dims[0] != (uint64_t)m->hidden || emb->dims[1] > INT_MAX)
        return q35_errf(err, cap, "invalid token_embd.weight dimensions");
    m->vocab = (int)emb->dims[1];

    m->rope_base = 10000000.0f;
    (void)q35_kv_f32(&m->gguf, "qwen35moe.rope.freq_base", &m->rope_base, 0, err, cap);
    m->eps = 1e-6f;
    (void)q35_kv_f32(&m->gguf, "qwen35moe.attention.layer_norm_rms_epsilon", &m->eps, 0, err, cap);
    m->attention_scale = 1.0f / sqrtf((float)m->head_dim);
    return 1;
}

static int q35_model_load_weights(Q35Model *m, char *err, size_t cap) {
    uint64_t d[3];
    d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->vocab;
    if (!q35_load_tensor(m, "token_embd.weight", &m->token_embd, 2, d, err, cap)) return 0;
    d[0] = (uint64_t)m->hidden;
    if (!q35_load_tensor(m, "output_norm.weight", &m->output_norm, 1, d, err, cap)) return 0;
    const ColiGgufTensorInfo *outi = coli_gguf_find_tensor(&m->gguf, "output.weight");
    if (outi) {
        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->vocab;
        if (!q35_load_tensor(m, "output.weight", &m->output, 2, d, err, cap)) return 0;
    } else {
        m->tied_output = 1;
    }

    m->layers = (Q35Layer *)calloc((size_t)m->n_layers_all, sizeof(*m->layers));
    if (!m->layers) return q35_errf(err, cap, "out of memory allocating Qwen3.5-MoE layers");

    int ri = 0, ai = 0;
    for (int l = 0; l < m->n_layers_all; ++l) {
        Q35Layer *x = &m->layers[l];
        const int is_mtp = l >= m->n_layers;
        x->recurrent = !is_mtp && (((l + 1) % m->full_attention_interval) != 0);
        x->recurrent_index = x->recurrent ? ri++ : -1;
        x->attention_index = x->recurrent ? -1 : ai++;

        d[0] = (uint64_t)m->hidden;
        if (!q35_load_layer_tensor(m, l, "attn_norm.weight", &x->attn_norm, 1, d, err, cap) ||
            !q35_load_layer_tensor(m, l, "post_attention_norm.weight", &x->post_norm, 1, d, err, cap)) return 0;

        if (x->recurrent) {
            d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->conv_dim;
            if (!q35_load_layer_tensor(m, l, "attn_qkv.weight", &x->qkv, 2, d, err, cap)) return 0;
            d[1] = (uint64_t)m->value_dim;
            if (!q35_load_layer_tensor(m, l, "attn_gate.weight", &x->z, 2, d, err, cap)) return 0;
            d[1] = (uint64_t)m->n_value_heads;
            if (!q35_load_layer_tensor(m, l, "ssm_beta.weight", &x->beta, 2, d, err, cap) ||
                !q35_load_layer_tensor(m, l, "ssm_alpha.weight", &x->alpha, 2, d, err, cap)) return 0;
            d[0] = (uint64_t)m->conv_kernel; d[1] = (uint64_t)m->conv_dim;
            if (!q35_load_layer_tensor(m, l, "ssm_conv1d.weight", &x->conv1d, 2, d, err, cap)) return 0;
            d[0] = (uint64_t)m->n_value_heads;
            if (!q35_load_layer_tensor(m, l, "ssm_dt.bias", &x->dt_bias, 1, d, err, cap) ||
                !q35_load_layer_tensor(m, l, "ssm_a", &x->a, 1, d, err, cap)) return 0;
            d[0] = (uint64_t)m->value_head_dim;
            if (!q35_load_layer_tensor(m, l, "ssm_norm.weight", &x->ssm_norm, 1, d, err, cap)) return 0;
            d[0] = (uint64_t)m->value_dim; d[1] = (uint64_t)m->hidden;
            if (!q35_load_layer_tensor(m, l, "ssm_out.weight", &x->ssm_out, 2, d, err, cap)) return 0;
        } else {
            d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)(2 * m->n_heads * m->head_dim);
            if (!q35_load_layer_tensor(m, l, "attn_q.weight", &x->q, 2, d, err, cap)) return 0;
            d[1] = (uint64_t)m->kv_dim;
            if (!q35_load_layer_tensor(m, l, "attn_k.weight", &x->k, 2, d, err, cap) ||
                !q35_load_layer_tensor(m, l, "attn_v.weight", &x->v, 2, d, err, cap)) return 0;
            d[0] = (uint64_t)m->head_dim;
            if (!q35_load_layer_tensor(m, l, "attn_q_norm.weight", &x->q_norm, 1, d, err, cap) ||
                !q35_load_layer_tensor(m, l, "attn_k_norm.weight", &x->k_norm, 1, d, err, cap)) return 0;
            d[0] = (uint64_t)(m->n_heads * m->head_dim); d[1] = (uint64_t)m->hidden;
            if (!q35_load_layer_tensor(m, l, "attn_output.weight", &x->o, 2, d, err, cap)) return 0;
        }

        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->n_experts;
        if (!q35_load_layer_tensor(m, l, "ffn_gate_inp.weight", &x->router, 2, d, err, cap)) return 0;
        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->expert_ff; d[2] = (uint64_t)m->n_experts;
        if (!q35_load_moe_tensor(m, l, COLI_SGGUF_MOE_GATE, &x->gate_exps, 3, d, err, cap) ||
            !q35_load_moe_tensor(m, l, COLI_SGGUF_MOE_UP, &x->up_exps, 3, d, err, cap)) return 0;
        d[0] = (uint64_t)m->expert_ff; d[1] = (uint64_t)m->hidden; d[2] = (uint64_t)m->n_experts;
        if (!q35_load_moe_tensor(m, l, COLI_SGGUF_MOE_DOWN, &x->down_exps, 3, d, err, cap)) return 0;

        d[0] = (uint64_t)m->hidden;
        if (!q35_load_layer_tensor(m, l, "ffn_gate_inp_shexp.weight", &x->shared_gate_inp, 1, d, err, cap)) return 0;
        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->shared_ff;
        if (!q35_load_layer_tensor(m, l, "ffn_gate_shexp.weight", &x->shared_gate, 2, d, err, cap) ||
            !q35_load_layer_tensor(m, l, "ffn_up_shexp.weight", &x->shared_up, 2, d, err, cap)) return 0;
        d[0] = (uint64_t)m->shared_ff; d[1] = (uint64_t)m->hidden;
        if (!q35_load_layer_tensor(m, l, "ffn_down_shexp.weight", &x->shared_down, 2, d, err, cap)) return 0;

        if (is_mtp) {
            d[0] = (uint64_t)(2 * m->hidden); d[1] = (uint64_t)m->hidden;
            if (!q35_load_layer_tensor(m, l, "nextn.eh_proj.weight", &x->nextn_eh_proj, 2, d, err, cap)) return 0;
            d[0] = (uint64_t)m->hidden;
            if (!q35_load_layer_tensor(m, l, "nextn.enorm.weight", &x->nextn_enorm, 1, d, err, cap) ||
                !q35_load_layer_tensor(m, l, "nextn.hnorm.weight", &x->nextn_hnorm, 1, d, err, cap)) return 0;
            if (!q35_load_optional_layer_tensor(m, l, "nextn.shared_head_norm.weight", &x->nextn_head_norm,
                                                1, d, &x->has_nextn_head_norm, err, cap)) return 0;
            d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->vocab;
            if (!q35_load_optional_layer_tensor(m, l, "nextn.embed_tokens.weight", &x->nextn_embed_tokens,
                                                2, d, &x->has_nextn_embed, err, cap) ||
                !q35_load_optional_layer_tensor(m, l, "nextn.shared_head_head.weight", &x->nextn_head,
                                                2, d, &x->has_nextn_head, err, cap)) return 0;
        }

        if (m->verbose && ((l + 1) % 8 == 0 || l + 1 == m->n_layers_all))
            fprintf(stderr, "[GGUF] bound Qwen3.5-MoE block %d/%d%s\n", l + 1, m->n_layers_all,
                    is_mtp ? " (MTP)" : "");
    }
    return 1;
}

static int q35_build_expert_views(Q35Model *m, char *err, size_t cap) {
    for (int l = 0; l < m->n_layers_all; ++l) {
        Q35Layer *x = &m->layers[l];
        x->gate_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->up_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->down_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->heat = (uint32_t *)calloc((size_t)m->n_experts, sizeof(uint32_t));
        x->last = (uint32_t *)calloc((size_t)m->n_experts, sizeof(uint32_t));
        x->usage = (uint32_t *)calloc((size_t)m->n_experts, sizeof(uint32_t));
        if (!x->gate_expert || !x->up_expert || !x->down_expert ||
            !x->heat || !x->last || !x->usage)
            return q35_errf(err, cap, "out of memory allocating Q35 expert views");
        for (int e = 0; e < m->n_experts; ++e) {
            if (!coli_tensor_rows_view(&x->gate_exps, (uint64_t)e * m->expert_ff,
                                       (uint64_t)m->expert_ff, &x->gate_expert[e]) ||
                !coli_tensor_rows_view(&x->up_exps, (uint64_t)e * m->expert_ff,
                                       (uint64_t)m->expert_ff, &x->up_expert[e]) ||
                !coli_tensor_rows_view(&x->down_exps, (uint64_t)e * m->hidden,
                                       (uint64_t)m->hidden, &x->down_expert[e]))
                return q35_errf(err, cap, "invalid Q35 expert view at layer %d expert %d", l, e);
            x->expert_views = e + 1;
        }
    }
    /* Quantization can vary by layer (the target GGUF mixes Q4_K and Q6_K
     * expert-down matrices). Budget every scheduler slot at the largest expert
     * triple so encoded VRAM residency can never exceed the requested tier. */
    m->expert_bytes = 0;
    for (int l = 0; l < m->n_layers_all; ++l) {
        Q35Layer *x = &m->layers[l];
        const size_t bytes = (size_t)x->gate_expert[0].storage_bytes +
                             (size_t)x->up_expert[0].storage_bytes +
                             (size_t)x->down_expert[0].storage_bytes;
        if (bytes > m->expert_bytes) m->expert_bytes = bytes;
    }
    return 1;
}

/* ---- CPU reference operations ------------------------------------------------ */

static float q35_sigmoid(float x) {
    if (x >= 0.0f) { float z = expf(-x); return 1.0f / (1.0f + z); }
    float z = expf(x); return z / (1.0f + z);
}

static float q35_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* Full-attention q_proj is laid out per head as [Q(head), gate(head)].
 * It is not family-major. Split it before Q normalization/RoPE and retain the
 * raw gate for the post-attention sigmoid. */
static void q35_split_q_gate(float *q, float *gate, const float *qg,
                              int n_heads, int head_dim) {
    for (int h = 0; h < n_heads; ++h) {
        const float *src = qg + (size_t)h * 2 * head_dim;
        memcpy(q + (size_t)h * head_dim, src, (size_t)head_dim * sizeof(float));
        memcpy(gate + (size_t)h * head_dim, src + head_dim,
               (size_t)head_dim * sizeof(float));
    }
}

static void q35_rope_neox(float *v, int n_heads, int head_dim,
                           int rope_dims, int position, float theta) {
    if (rope_dims > head_dim) rope_dims = head_dim;
    rope_dims &= ~1;
    const int half = rope_dims / 2;
    for (int h = 0; h < n_heads; ++h) {
        float *p = v + (size_t)h * head_dim;
        for (int i = 0; i < half; ++i) {
            const float angle = (float)position * powf(theta, -2.0f * i / rope_dims);
            const float cs = cosf(angle), sn = sinf(angle);
            const float a = p[i], b = p[half + i];
            p[i] = a * cs - b * sn;
            p[half + i] = a * sn + b * cs;
        }
    }
}

static void q35_l2norm_heads(float *x, int heads, int dim, float eps) {
    for (int h = 0; h < heads; ++h) {
        float *p = x + (size_t)h * dim;
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) sum += (double)p[i] * p[i];
        const float inv = 1.0f / sqrtf((float)sum + eps);
        for (int i = 0; i < dim; ++i) p[i] *= inv;
    }
}

static int q35_f32_vector(const ColiTensor *t, float *out, int n,
                           char *err, size_t cap) {
    if (!t || t->row_count != 1 || t->dims[0] != (uint64_t)n ||
        !coli_tensor_read_row_f32(t, 0, out, (uint64_t)n))
        return q35_errf(err, cap, "cannot decode vector %s", t && t->name ? t->name : "<unnamed>");
    return 1;
}

static int q35_mm_cpu(float *y, const float *x, ColiTensor *w,
                       int I, int O, char *err, size_t cap) {
    ColiExec cpu = { COLI_BACKEND_CPU, 0 };
    if (!coli_tensor_matmul(&cpu, y, x, w, 1, I, O))
        return q35_errf(err, cap, "CPU native matmul failed for %s", w->name ? w->name : "<unnamed>");
    return 1;
}

static int q35_mm_cpu_s(float *y, const float *x, ColiTensor *w, int S,
                         int I, int O, char *err, size_t cap) {
    ColiExec cpu = { COLI_BACKEND_CPU, 0 };
    if (!coli_tensor_matmul(&cpu, y, x, w, S, I, O))
        return q35_errf(err, cap, "CPU native batch matmul failed for %s",
                        w->name ? w->name : "<unnamed>");
    return 1;
}

static int q35_mm_cpu_thresholded_s(float *y, const float *x, ColiTensor *w,
                                      int S, int I, int O, float threshold,
                                      char *err, size_t cap) {
    const ColiDTypeTraits *traits = w ? coli_dtype_traits(w->dtype) : NULL;
    if (!y || !x || !w || !w->data || !traits || S < 1 || I < 1 || O < 1 ||
        w->dims[0] != (uint64_t)I || w->row_count != (uint64_t)O ||
        I % (int)traits->block_values)
        return q35_errf(err, cap, "invalid thresholded expert matmul for %s",
                        w && w->name ? w->name : "<unnamed>");
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
        const uint8_t *row = w->data + (uint64_t)o * w->row_bytes;
        for (int ss = 0; ss < S; ++ss) {
            const float *xs = x + (size_t)ss * I;
            const uint8_t *block = row;
            double sum = 0.0;
            float decoded[256];
            for (int base = 0; base < I; base += (int)traits->block_values) {
                if (!coli_dtype_dequantize_row(w->dtype, block, traits->block_values, decoded)) {
                    sum = NAN;
                    break;
                }
                for (uint32_t i = 0; i < traits->block_values; ++i) {
                    const float v = decoded[i];
                    if (v == 0.0f || fabsf(v) >= threshold)
                        sum += (double)xs[base + (int)i] * v;
                }
                block += traits->block_bytes;
            }
            y[(size_t)ss * O + o] = (float)sum;
        }
    }
    return 1;
}

static int q35_mm_cpu_expert_s(Q35Model *m, float *y, const float *x,
                                ColiTensor *w, int S, int I, int O,
                                char *err, size_t cap) {
    if (!m || m->expert_zero_threshold <= 0.0f || w->owns_data ||
        w->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE)
        return q35_mm_cpu_s(y, x, w, S, I, O, err, cap);
    return q35_mm_cpu_thresholded_s(y, x, w, S, I, O,
                                     m->expert_zero_threshold, err, cap);
}

static int q35_mm_cpu_expert(Q35Model *m, float *y, const float *x,
                              ColiTensor *w, int I, int O,
                              char *err, size_t cap) {
    return q35_mm_cpu_expert_s(m, y, x, w, 1, I, O, err, cap);
}

static int q35_conv_update_cpu(Q35Model *m, Q35Layer *l,
                                const float *input, float *output,
                                float *state, char *err, size_t cap) {
    if (l->conv1d.dtype != COLI_DTYPE_F32 || l->conv1d.dims[0] != (uint64_t)m->conv_kernel)
        return q35_errf(err, cap, "Q35 conv1d must be F32");
    const float *weights = (const float *)l->conv1d.data;
#pragma omp parallel for schedule(static)
    for (int c = 0; c < m->conv_dim; ++c) {
        float *s = state + (size_t)c * m->conv_kernel;
        for (int j = 0; j + 1 < m->conv_kernel; ++j) s[j] = s[j + 1];
        s[m->conv_kernel - 1] = input[c];
        const float *w = weights + (size_t)c * m->conv_kernel;
        float acc = 0.0f;
        for (int j = 0; j < m->conv_kernel; ++j) acc += s[j] * w[j];
        output[c] = coli_f32_silu(acc);
    }
    return 1;
}

static int q35_delta_decode_cpu(Q35Model *m, Q35Layer *l,
                                 float *conv, const float *z,
                                 const float *beta_raw, const float *alpha_raw,
                                 float *state, float *out,
                                 char *err, size_t cap) {
    float *q = conv;
    float *k = conv + m->key_dim;
    float *v = conv + 2 * m->key_dim;
    q35_l2norm_heads(q, m->n_key_heads, m->state_size, 1e-6f);
    q35_l2norm_heads(k, m->n_key_heads, m->state_size, 1e-6f);

    float *dt = (float *)malloc((size_t)m->n_value_heads * sizeof(float));
    float *a = (float *)malloc((size_t)m->n_value_heads * sizeof(float));
    float *nw = (float *)malloc((size_t)m->value_head_dim * sizeof(float));
    if (!dt || !a || !nw) { free(dt); free(a); free(nw); return q35_errf(err, cap, "DeltaNet vector scratch OOM"); }
    if (!q35_f32_vector(&l->dt_bias, dt, m->n_value_heads, err, cap) ||
        !q35_f32_vector(&l->a, a, m->n_value_heads, err, cap) ||
        !q35_f32_vector(&l->ssm_norm, nw, m->value_head_dim, err, cap)) {
        free(dt); free(a); free(nw); return 0;
    }

    /* Qwen3.5/3.6 GGUF conversion tiles value heads across key heads:
     * [K0-v0, K1-v0, ..., K0-v1, K1-v1, ...].  This differs from
     * Qwen3-Next's older grouped layout and therefore maps with modulo. */
    const float scale = 1.0f / sqrtf((float)m->state_size);
    for (int vh = 0; vh < m->n_value_heads; ++vh) {
        const int kh = vh % m->n_key_heads;
        const float beta = q35_sigmoid(beta_raw[vh]);
        const float decay_log = a[vh] * q35_softplus(alpha_raw[vh] + dt[vh]);
        const float decay = expf(decay_log);
        const float *qh = q + (size_t)kh * m->state_size;
        const float *khv = k + (size_t)kh * m->state_size;
        const float *vhv = v + (size_t)vh * m->value_head_dim;
        const float *zh = z + (size_t)vh * m->value_head_dim;
        float *S = state + (size_t)vh * m->state_size * m->value_head_dim;
        float *oh = out + (size_t)vh * m->value_head_dim;

        for (int col = 0; col < m->value_head_dim; ++col) {
            float *Sc = S + (size_t)col * m->state_size;
            float kv = 0.0f;
            for (int i = 0; i < m->state_size; ++i) kv += Sc[i] * khv[i];
            const float delta = (vhv[col] - decay * kv) * beta;
            float y = 0.0f;
            for (int i = 0; i < m->state_size; ++i) {
                Sc[i] = decay * Sc[i] + khv[i] * delta;
                y += Sc[i] * qh[i];
            }
            oh[col] = y * scale;
        }

        double sum2 = 0.0;
        for (int i = 0; i < m->value_head_dim; ++i) sum2 += (double)oh[i] * oh[i];
        const float inv = 1.0f / sqrtf((float)(sum2 / m->value_head_dim) + m->eps);
        for (int i = 0; i < m->value_head_dim; ++i)
            oh[i] = oh[i] * inv * nw[i] * coli_f32_silu(zh[i]);
    }
    free(dt); free(a); free(nw);
    return 1;
}

/* ---- Scheduler integration: one existing policy, Q35 storage callbacks ------- */

static ColiExpertLayerStore q35_store(Q35Model *m, int layer) {
    Q35Layer *l = &m->layers[layer];
    ColiExpertLayerStore s;
    memset(&s, 0, sizeof(s));
    s.pin = l->pin; s.npin = l->npin;
    s.cache = l->cache; s.ncache = &l->ncache; s.cache_cap = l->cache_cap;
    s.n_experts = m->n_experts; s.layout = g_q35_slot_layout;
    s.clock = &m->expert_clock; s.heat = l->heat; s.last = l->last; s.usage = l->usage;
    s.access_clock = &m->expert_access_clock;
    return s;
}

static void q35_slot_bind(Q35Model *m, int layer, int eid, Q35ExpertSlot *s) {
    Q35Layer *l = &m->layers[layer];
    s->eid = eid;
    s->gate = &l->gate_expert[eid];
    s->up = &l->up_expert[eid];
    s->down = &l->down_expert[eid];
}

static int q35_tensor_clone_pruned(const ColiTensor *src, float threshold,
                                    ColiTensor *dst, uint64_t *changed,
                                    char *err, size_t cap) {
    if (changed) *changed = 0;
    if (!src || !src->data || !dst || threshold <= 0.0f)
        return q35_errf(err, cap, "invalid Q35 pruned tensor clone");
    uint8_t *copy = (uint8_t *)malloc((size_t)src->storage_bytes);
    if (!copy) return q35_errf(err, cap, "Q35 expert pruning copy OOM");
    memcpy(copy, src->data, (size_t)src->storage_bytes);
    uint64_t local_changed = 0;
    if (!coli_dtype_zero_below_inplace(src->dtype, copy, src->element_count,
                                       threshold, &local_changed)) {
        free(copy);
        return q35_errf(err, cap,
            "Q35 expert pruning unsupported for dtype %s",
            coli_dtype_traits(src->dtype) ? coli_dtype_traits(src->dtype)->name : "unknown");
    }
    *dst = *src;
    dst->data = copy;
    dst->source_offset = 0;
    dst->cuda = NULL;
    dst->cuda_device = 0;
    dst->owns_data = 1;
    dst->mmap_backed = 0;
    dst->owns_cuda = 0;
    if (changed) *changed = local_changed;
    return 1;
}

static int q35_slot_materialize_pruned(Q35Model *m, Q35ExpertSlot *s,
                                        char *err, size_t cap) {
    uint64_t cg = 0, cu = 0, cd = 0;
    ColiTensor *gate = (ColiTensor *)calloc(1, sizeof(*gate));
    ColiTensor *up = (ColiTensor *)calloc(1, sizeof(*up));
    ColiTensor *down = (ColiTensor *)calloc(1, sizeof(*down));
    if (!gate || !up || !down) {
        free(gate); free(up); free(down);
        return q35_errf(err, cap, "Q35 expert pruning descriptor OOM");
    }
    if (!q35_tensor_clone_pruned(s->gate, m->expert_zero_threshold, gate, &cg, err, cap) ||
        !q35_tensor_clone_pruned(s->up, m->expert_zero_threshold, up, &cu, err, cap) ||
        !q35_tensor_clone_pruned(s->down, m->expert_zero_threshold, down, &cd, err, cap)) {
        coli_tensor_destroy(gate); coli_tensor_destroy(up); coli_tensor_destroy(down);
        free(gate); free(up); free(down);
        return 0;
    }
    s->gate = gate;
    s->up = up;
    s->down = down;
    s->owns_pruned = 1;
    ++m->expert_prune_materializations;
    m->expert_prune_values_seen += gate->element_count + up->element_count + down->element_count;
    m->expert_prune_values_zeroed += cg + cu + cd;
    return 1;
}

static int q35_storage_load(void *ctx, int layer, int eid, void *slot, int demand) {
    Q35Model *m = (Q35Model *)ctx;
    Q35ExpertSlot *s = (Q35ExpertSlot *)slot;
    (void)demand;
    q35_slot_bind(m, layer, eid, s);
    coli_tensor_prefetch_host(s->gate);
    coli_tensor_prefetch_host(s->up);
    coli_tensor_prefetch_host(s->down);
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        if (m->cuda_expert_admission_disabled) {
            q35_slot_clear(s);
            return 0;
        }
        if (m->expert_zero_threshold > 0.0f &&
            s->gate->storage_kind != COLI_TENSOR_STORAGE_SPARSE_TREE) {
            char prune_err[256];
            if (!q35_slot_materialize_pruned(m, s, prune_err, sizeof(prune_err))) {
                fprintf(stderr, "[Q35-PRUNE] %s\n", prune_err);
                q35_slot_clear(s);
                return 0;
            }
        }
        if (!coli_tensor_reside(&m->exec, s->gate) ||
            !coli_tensor_reside(&m->exec, s->up) ||
            !coli_tensor_reside(&m->exec, s->down)) {
            coli_tensor_release_backend(&m->exec, s->gate);
            coli_tensor_release_backend(&m->exec, s->up);
            coli_tensor_release_backend(&m->exec, s->down);
            q35_slot_clear(s);
            m->cuda_expert_admission_disabled = 1;
            ++m->cuda_expert_residency_failures;
            if (!m->cuda_expert_failure_reported) {
                m->cuda_expert_failure_reported = 1;
                fprintf(stderr,
                    "[Q35] CUDA expert residency failed; disabling new GPU admissions and using mmap CPU fallback for misses\n");
            }
            return 0;
        }
        m->cuda_expert_bytes += m->expert_bytes;
        m->cuda_weight_bytes = m->cuda_dense_bytes + m->cuda_expert_bytes;
    }
#endif
    return 1;
}

static void q35_storage_evict(void *ctx, int layer, void *slot) {
    Q35Model *m = (Q35Model *)ctx;
    Q35ExpertSlot *s = (Q35ExpertSlot *)slot;
    (void)layer; (void)m;
#ifdef COLI_CUDA
    if (s->eid >= 0 && m->exec.kind == COLI_BACKEND_CUDA) {
        coli_tensor_release_backend(&m->exec, s->gate);
        coli_tensor_release_backend(&m->exec, s->up);
        coli_tensor_release_backend(&m->exec, s->down);
        if (m->cuda_expert_bytes >= m->expert_bytes) m->cuda_expert_bytes -= m->expert_bytes;
        m->cuda_weight_bytes = m->cuda_dense_bytes + m->cuda_expert_bytes;
    }
#endif
    q35_slot_clear(s);
}

static size_t q35_storage_bytes(void *ctx, int layer, const void *slot) {
    (void)layer; (void)slot;
    return ((Q35Model *)ctx)->expert_bytes;
}

static const ColiExpertStorageOps g_q35_storage = {
    q35_storage_load, q35_storage_evict, q35_storage_bytes
};

static Q35ExpertSlot *q35_expert_acquire(Q35Model *m, int layer, int eid, int demand) {
    ColiExpertLayerStore st = q35_store(m, layer);
    int allow = 1;
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA && m->cuda_expert_admission_disabled) allow = 0;
#endif
    return (Q35ExpertSlot *)coli_expert_acquire_controlled(&st, layer, eid, demand,
            allow, m->scheduler_evict_guard, &g_q35_storage, m, &m->scheduler_stats);
}

static void q35_usage_path(Q35Model *m, const char *model_path) {
    const size_t n = strlen(model_path);
    if (n + sizeof(".coli_usage") > sizeof(m->usage_path)) return;
    memcpy(m->usage_path, model_path, n);
    memcpy(m->usage_path + n, ".coli_usage", sizeof(".coli_usage"));
}

static void q35_usage_rows(Q35Model *m, uint32_t **rows) {
    for (int l = 0; l < m->n_layers_all; ++l) rows[l] = m->layers[l].usage;
}

static int64_t q35_usage_load(Q35Model *m, const char *model_path) {
    q35_usage_path(m, model_path);
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers_all * sizeof(*rows));
    if (!rows) return 0;
    q35_usage_rows(m, rows);
    const int64_t total = coli_expert_usage_load(m->usage_path, rows, m->n_layers_all, m->n_experts);
    free(rows);
    m->usage_history = total;
    return total;
}

static void q35_usage_save(Q35Model *m) {
    if (!m || !m->usage_path[0]) return;
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers_all * sizeof(*rows));
    if (!rows) return;
    q35_usage_rows(m, rows);
    (void)coli_expert_usage_save(m->usage_path, rows, m->n_layers_all, m->n_experts);
    free(rows);
}

static int q35_usage_top(Q35Model *m, int *ids, int cap) {
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_exec_layers * sizeof(*rows));
    if (!rows) return 0;
    for (int l = 0; l < m->n_exec_layers; ++l) rows[l] = m->layers[l].usage;
    const int n = coli_expert_usage_top(rows, m->n_exec_layers, m->n_experts, ids, cap);
    free(rows);
    return n;
}

static int q35_usage_focus(Q35Model *m, int focus_layers, int *ids, int cap,
                            int *selected, int selected_cap, int *counts) {
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_exec_layers * sizeof(*rows));
    if (!rows) return 0;
    for (int l = 0; l < m->n_exec_layers; ++l) rows[l] = m->layers[l].usage;
    const int n = coli_expert_usage_focus(rows, m->n_exec_layers, m->n_experts,
            focus_layers, ids, cap, selected, selected_cap, counts);
    free(rows);
    return n;
}

static int q35_stats_fallback(Q35Model *m, char *path, size_t cap) {
    if (!m->usage_path[0] || !path || cap == 0) return 0;
    const char *slash = strrchr(m->usage_path, '/');
    if (!slash) return snprintf(path, cap, "stats.txt") > 0;
    const size_t dir = (size_t)(slash - m->usage_path);
    if (dir + sizeof("/stats.txt") > cap) return 0;
    memcpy(path, m->usage_path, dir);
    memcpy(path + dir, "/stats.txt", sizeof("/stats.txt"));
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); const long size = ftell(f); fclose(f);
    return size > 0;
}

static int q35_scheduler_init(Q35Model *m, char *err, size_t cap) {
    m->scheduler_evict_guard = getenv("PILOT_EVICT_GUARD") ? atoi(getenv("PILOT_EVICT_GUARD")) : 1;
    m->focus_layer_count = getenv("Q35_FOCUS_LAYER_COUNT") ?
                           atoi(getenv("Q35_FOCUS_LAYER_COUNT")) : 0;
    if (m->focus_layer_count < 0) m->focus_layer_count = 0;
    if (m->focus_layer_count > m->n_exec_layers) m->focus_layer_count = m->n_exec_layers;
    m->focus_group_enabled = m->focus_layer_count > 0;
    if (getenv("Q35_FOCUS_GROUP"))
        m->focus_group_enabled = atoi(getenv("Q35_FOCUS_GROUP")) != 0;
    m->focus_group_min = getenv("Q35_FOCUS_GROUP_MIN") ?
                         atoi(getenv("Q35_FOCUS_GROUP_MIN")) : 2;
    if (m->focus_group_min < 2) m->focus_group_min = 2;
    if (m->focus_group_min > m->n_expert_used) m->focus_group_min = m->n_expert_used;
    if (m->focus_layer_count > 0) {
        m->focus_layer_mask = (unsigned char *)calloc((size_t)m->n_layers_all, 1);
        if (!m->focus_layer_mask) return q35_errf(err, cap, "Q35 focus mask OOM");
    }
    {
        const char *v = getenv("Q35_ROUTER_GPU");
        m->cuda_router_gpu_enabled = v && atoi(v) != 0;
    }
    m->repin_interval = getenv("REPIN") ? atoi(getenv("REPIN")) : 0;
    m->pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    m->pilot_real = getenv("PILOT_REAL") ? atoi(getenv("PILOT_REAL")) : 0;
    if (m->pilot_real) m->pilot = 1;
    m->pilot_k = getenv("PILOT_K") ? atoi(getenv("PILOT_K")) : (m->pilot_real ? 6 : 8);
    if (m->pilot_k < 1) m->pilot_k = 1;
    if (m->pilot_k > m->n_experts) m->pilot_k = m->n_experts;
    m->couple_k = getenv("COUPLE_K") ? atoi(getenv("COUPLE_K")) : 8;
    if (m->couple_k < 1) m->couple_k = 1;
    if (m->couple_k > 32) m->couple_k = 32;
    m->couple_d = getenv("COUPLE_D") ? atoi(getenv("COUPLE_D")) : 1;
    if (m->couple_d < 1) m->couple_d = 1;
    if (m->couple_d > 2) m->couple_d = 2;
    if (getenv("COUPLE") && *getenv("COUPLE")) {
        long used = 0;
        m->couple = coli_expert_coupling_load(&m->coupling, getenv("COUPLE"),
                                              m->n_exec_layers, m->n_experts, &used);
        if (m->verbose) fprintf(stderr, "[COUPLE] Q35 GGUF: %ld conditioning entries\n", used);
    }

    const int total = m->n_exec_layers * m->n_experts;
    /* mmap-backed CPU experts need scheduler metadata, not a 48 GB eager
     * WILLNEED sweep. Keep enough LRU slots for the routed set per layer. */
    int slots = m->n_exec_layers * m->n_expert_used;
    if (slots < m->n_exec_layers) slots = m->n_exec_layers;
    if (slots > total) slots = total;
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        size_t free_b = 0, total_b = 0;
        double reserve = getenv("CUDA_RESERVE_GB") ? atof(getenv("CUDA_RESERVE_GB")) : 0.5;
        if (reserve < 0.0) reserve = 0.0;
        const int have_mem = coli_cuda_mem_info(m->exec.device, &free_b, &total_b);
        double available = have_mem ? (double)free_b - reserve * 1e9 : -1.0;
        if (available < 0.0 && have_mem) available = 0.0;
        double budget = getenv("CUDA_EXPERT_GB") ? atof(getenv("CUDA_EXPERT_GB")) * 1e9 : available;
        if (budget < 0.0) budget = 0.0;
        if (have_mem && budget > available) {
            if (m->verbose)
                fprintf(stderr, "[SCHED] CUDA_EXPERT_GB exceeds free VRAM; clamped to %.2f GB after reserve\n",
                        available / 1e9);
            budget = available;
        }
        slots = m->expert_bytes ? (int)(budget / (double)m->expert_bytes) : 0;
        if (slots > total) slots = total;
        if (slots < m->n_exec_layers)
            return q35_errf(err, cap,
                "CUDA expert budget fits %d slots; need at least %d after dense Q35 residency",
                slots, m->n_exec_layers);
    }
#endif

    int pin_total = slots == total ? total : 0;
    int *pinids = NULL, npinids = 0;
    const char *pinfile = getenv("PIN");
    if (slots < total) {
        const int maxpins = slots > m->n_exec_layers ? slots - m->n_exec_layers : 0;
        int want = 0;
        char auto_stats[2048];
        const char *source = pinfile;
        if (pinfile) {
            const char *pgs = getenv("PIN_GB");
            if (pgs && !strcmp(pgs, "all")) want = maxpins;
            else if (pgs && atof(pgs) > 0 && m->expert_bytes)
                want = (int)(atof(pgs) * 1e9 / m->expert_bytes);
            else want = slots / 2;
            if (want > maxpins) want = maxpins;
            if (!strcmp(pinfile, "auto")) {
                if (m->usage_history > 0) source = m->usage_path;
                else if (q35_stats_fallback(m, auto_stats, sizeof(auto_stats))) source = auto_stats;
                else source = NULL;
            }
        } else {
            const int autopin = getenv("AUTOPIN") ? atoi(getenv("AUTOPIN")) : 1;
            if (autopin && m->usage_history >= 5000) {
                double confidence = (double)m->usage_history / 200000.0;
                if (confidence > 1.0) confidence = 1.0;
                want = (int)(0.5 * confidence * slots);
                if (want > maxpins) want = maxpins;
                source = m->usage_path;
            }
        }
        if (want > 0 && source) {
            pinids = (int *)malloc((size_t)want * sizeof(int));
            if (!pinids) return q35_errf(err, cap, "Q35 scheduler pin ranking OOM");
            if (m->focus_layer_count > 0) {
                int *selected = (int *)malloc((size_t)m->focus_layer_count * sizeof(*selected));
                int *counts = (int *)calloc((size_t)m->n_exec_layers, sizeof(*counts));
                if (!selected || !counts) {
                    free(selected); free(counts); free(pinids);
                    return q35_errf(err, cap, "Q35 focused pin ranking OOM");
                }
                for (int i = 0; i < m->focus_layer_count; ++i) selected[i] = -1;
                npinids = source == m->usage_path ?
                    q35_usage_focus(m, m->focus_layer_count, pinids, want,
                                     selected, m->focus_layer_count, counts) :
                    coli_expert_usage_focus_file(source, m->n_exec_layers, m->n_experts,
                                                  m->focus_layer_count, pinids, want,
                                                  selected, m->focus_layer_count, counts, NULL);
                m->focus_layer_effective = 0;
                for (int l = 0; l < m->n_exec_layers; ++l)
                    if (counts[l] > 0) ++m->focus_layer_effective;
                if (m->verbose) {
                    fprintf(stderr, "[FOCUS] Q35 hot pins: requested=%d effective=%d budget=%d layers=",
                            m->focus_layer_count, m->focus_layer_effective, want);
                    int shown = 0;
                    for (int i = 0; i < m->focus_layer_count && selected[i] >= 0; ++i) {
                        const int l = selected[i];
                        if (m->focus_layer_mask) m->focus_layer_mask[l] = 1;
                        fprintf(stderr, "%s%d:%d", shown++ ? "," : "", l, counts[l]);
                    }
                    fputc('\n', stderr);
                } else if (m->focus_layer_mask) {
                    for (int i = 0; i < m->focus_layer_count && selected[i] >= 0; ++i)
                        m->focus_layer_mask[selected[i]] = 1;
                }
                free(selected); free(counts);
            } else {
                npinids = source == m->usage_path ? q35_usage_top(m, pinids, want) :
                           coli_expert_usage_top_file(source, m->n_exec_layers, m->n_experts,
                                                      pinids, want, NULL);
            }
            pin_total = npinids;
            if (m->verbose) fprintf(stderr, "[PIN] Q35 GGUF: %d experts from %s\n", npinids, source);
        }
    }

    int *pc = (int *)calloc((size_t)m->n_exec_layers, sizeof(int));
    if (!pc) { free(pinids); return q35_errf(err, cap, "Q35 scheduler OOM"); }
    if (pin_total == total) for (int l = 0; l < m->n_exec_layers; ++l) pc[l] = m->n_experts;
    else for (int i = 0; i < npinids; ++i) ++pc[pinids[i] / m->n_experts];

    const int cache_slots = slots - pin_total;
    const int base = cache_slots / m->n_exec_layers;
    const int extra = cache_slots % m->n_exec_layers;
    for (int l = 0; l < m->n_exec_layers; ++l) {
        Q35Layer *x = &m->layers[l];
        x->npin = pc[l];
        x->cache_cap = base + (l < extra);
        if (x->npin) x->pin = (Q35ExpertSlot *)calloc((size_t)x->npin, sizeof(*x->pin));
        if (x->cache_cap) x->cache = (Q35ExpertSlot *)calloc((size_t)x->cache_cap, sizeof(*x->cache));
        if ((x->npin && !x->pin) || (x->cache_cap && !x->cache)) {
            free(pc); free(pinids); return q35_errf(err, cap, "Q35 scheduler slot OOM");
        }
        for (int z = 0; z < x->npin; ++z) x->pin[z].eid = -1;
        for (int z = 0; z < x->cache_cap; ++z) x->cache[z].eid = -1;
    }

    /* A real speculative upload is counterproductive when the replaceable
     * tier cannot hold one complete routed set. It steals the layer's only
     * LRU slot before the real top-k demand arrives. Preserve PILOT as a
     * page-cache hint unless the user explicitly forces the old behavior. */
    if (m->pilot_real && !(getenv("PILOT_REAL_FORCE") && atoi(getenv("PILOT_REAL_FORCE")))) {
        int min_cache = INT_MAX;
        for (int l = 0; l < m->n_exec_layers; ++l)
            if (m->layers[l].cache_cap < min_cache) min_cache = m->layers[l].cache_cap;
        if (min_cache < m->n_expert_used) {
            m->pilot_real = 0;
            m->pilot_real_downgraded = 1;
            if (m->verbose)
                fprintf(stderr,
                    "[PILOT] Q35 real prefetch downgraded to mmap hint: min LRU/layer=%d < routed top-%d (set PILOT_REAL_FORCE=1 to override)\n",
                    min_cache, m->n_expert_used);
        }
    }

    if (pin_total == total) {
        for (int l = 0; l < m->n_exec_layers; ++l) for (int e = 0; e < m->n_experts; ++e) {
            Q35ExpertSlot *q = &m->layers[l].pin[e];
            if (!q35_storage_load(m, l, e, q, 0)) {
                free(pc); free(pinids); return q35_errf(err, cap, "Q35 expert pin residency failed");
            }
            q->used = ++m->expert_clock;
        }
    } else {
        int *next = (int *)calloc((size_t)m->n_exec_layers, sizeof(int));
        if (!next) { free(pc); free(pinids); return q35_errf(err, cap, "Q35 pin OOM"); }
        for (int i = 0; i < npinids; ++i) {
            const int l = pinids[i] / m->n_experts, e = pinids[i] % m->n_experts;
            Q35ExpertSlot *q = &m->layers[l].pin[next[l]++];
            if (!q35_storage_load(m, l, e, q, 0)) {
                free(next); free(pc); free(pinids);
                return q35_errf(err, cap, "Q35 expert pin residency failed");
            }
            q->used = ++m->expert_clock;
        }
        free(next);
    }
    free(pc); free(pinids);

    if (m->verbose)
        fprintf(stderr, "[SCHED] one native scheduler: Q35 %d pin + %d LRU slots, %.2f MiB/slot; FOCUS=%s GROUP=%s/%d PILOT=%s COUPLE=%s\n",
                pin_total, cache_slots, m->expert_bytes / (1024.0 * 1024.0),
                m->focus_layer_count > 0 ? "on" : "off",
                m->focus_group_enabled ? "on" : "off", m->focus_group_min,
                m->pilot ? (m->pilot_real ? "real" : "hint") : "off",
                m->couple ? "on" : "off");
    return 1;
}

static void q35_prefetch_ids(Q35Model *m, int layer, const int *ids, int n) {
    if (layer < 0 || layer >= m->n_exec_layers) return;
    Q35Layer *l = &m->layers[layer];
    for (int i = 0; i < n; ++i) {
        const int eid = ids[i];
        if (eid < 0 || eid >= m->n_experts) continue;
        if (m->pilot_real) (void)q35_expert_acquire(m, layer, eid, 0);
        else {
            coli_tensor_prefetch_host(&l->gate_expert[eid]);
            coli_tensor_prefetch_host(&l->up_expert[eid]);
            coli_tensor_prefetch_host(&l->down_expert[eid]);
        }
    }
}

static void q35_couple_prefetch(Q35Model *m, int layer, const int *routed, int nrouted) {
    if (!m->couple) return;
    for (int d = 1; d <= m->couple_d; ++d) {
        const int target = layer + d;
        if (target >= m->n_exec_layers) break;
        int pred[32];
        const int n = coli_expert_coupling_predict(&m->coupling, layer, d, routed, nrouted,
                                                   pred, m->couple_k);
        q35_prefetch_ids(m, target, pred, n);
    }
}

static void q35_repin(Q35Model *m) {
    if (m->repin_interval <= 0 || ++m->tokens_since_repin < m->repin_interval) return;
    m->tokens_since_repin = 0;
    for (int l = 0; l < m->n_exec_layers; ++l) {
        ColiExpertLayerStore st = q35_store(m, l);
        int pi, e; long gain;
        if (!coli_expert_repin_pick(&st, &pi, &e, &gain)) continue;
        Q35ExpertSlot *q = &m->layers[l].pin[pi];
        const int old = q->eid;
        const int moved = coli_expert_repin_promote_cached(&st, pi, e, l,
                                                           &g_q35_storage, m);
        if (moved > 0) {
            if (m->verbose)
                fprintf(stderr, "[REPIN] Q35 layer %d: %d <- cached %d (gain %ld)\n",
                        l, old, e, gain);
            coli_expert_decay_heat(&st);
            continue;
        }
        if (moved < 0) continue;
        q35_storage_evict(m, l, q);
        if (q35_storage_load(m, l, e, q, 0)) {
            q->used = ++m->expert_clock;
            if (m->verbose)
                fprintf(stderr, "[REPIN] Q35 layer %d: %d <- %d (gain %ld)\n", l, old, e, gain);
        } else if (old >= 0) {
            (void)q35_storage_load(m, l, old, q, 0);
            q->used = ++m->expert_clock;
        }
        coli_expert_decay_heat(&st);
    }
}

/* ---- Dense residency and state ------------------------------------------------ */

#ifdef COLI_CUDA
static int q35_reside_one(Q35Model *m, ColiTensor *t) {
    if (!t->data) return 1;
    if (!coli_tensor_reside(&m->exec, t)) return 0;
    m->cuda_dense_bytes += (size_t)t->storage_bytes;
    return 1;
}

static void q35_release_layer_cuda(Q35Model *m, Q35Layer *x) {
#define QREL(t) coli_tensor_release_backend(&m->exec, &(t))
    QREL(x->attn_norm); QREL(x->post_norm);
    QREL(x->q); QREL(x->k); QREL(x->v); QREL(x->q_norm); QREL(x->k_norm); QREL(x->o);
    QREL(x->qkv); QREL(x->z); QREL(x->beta); QREL(x->alpha); QREL(x->conv1d);
    QREL(x->dt_bias); QREL(x->a); QREL(x->ssm_norm); QREL(x->ssm_out);
    QREL(x->router); QREL(x->shared_gate_inp); QREL(x->shared_gate);
    QREL(x->shared_up); QREL(x->shared_down);
    QREL(x->nextn_eh_proj); QREL(x->nextn_enorm); QREL(x->nextn_hnorm);
    QREL(x->nextn_embed_tokens); QREL(x->nextn_head); QREL(x->nextn_head_norm);
#undef QREL
}

static void q35_release_dense_cuda(Q35Model *m) {
    if (!m || m->exec.kind != COLI_BACKEND_CUDA) return;
    coli_tensor_release_backend(&m->exec, &m->token_embd);
    coli_tensor_release_backend(&m->exec, &m->output_norm);
    coli_tensor_release_backend(&m->exec, &m->output);
    for (int l = 0; l < m->n_layers_all; ++l) q35_release_layer_cuda(m, &m->layers[l]);
    m->cuda_dense_bytes = 0;
    m->cuda_weight_bytes = 0;
}
#endif

/* Best-effort dense residency.  The token embedding is deliberately not made
 * resident when a separate output head exists: decoding one 2048-value row
 * from mmap and uploading 8 KiB per token saves about 398 MiB of VRAM.  State
 * is allocated before this function, so weights can never crowd out mandatory
 * recurrent/KV storage.  A failed allocation is reported to the caller as a
 * request to downgrade to the mmap-backed CPU path, never as a fatal load error. */
static int q35_model_reside_cuda(Q35Model *m, char *err, size_t cap) {
#ifndef COLI_CUDA
    (void)m; (void)err; (void)cap;
    return 1;
#else
    if (m->exec.kind != COLI_BACKEND_CUDA) return 1;
    m->cuda_dense_complete = 0;
    m->cuda_fallback_tensor = NULL;
#define QTRY(t) do { \
    if (!q35_reside_one(m, &(t))) { \
        const ColiDTypeTraits *qtry_traits = coli_dtype_traits((t).dtype); \
        size_t qtry_free = 0, qtry_total = 0; \
        const int qtry_mem = coli_cuda_mem_info(m->exec.device, &qtry_free, &qtry_total); \
        m->cuda_fallback_tensor = (t).name ? (t).name : "<unnamed>"; \
        if (err && cap) snprintf(err, cap, \
            "cannot make dense tensor resident: %s type=%s(%u) bytes=%.2f MiB%s%.2f MiB", \
            m->cuda_fallback_tensor, qtry_traits ? qtry_traits->name : "unknown", \
            (unsigned)(t).dtype, (double)(t).storage_bytes / (1024.0 * 1024.0), \
            qtry_mem ? " free_vram=" : "", \
            qtry_mem ? (double)qtry_free / (1024.0 * 1024.0) : 0.0); \
        return 0; \
    } \
} while (0)

    /* The output projection is used for every generated token and is much more
     * valuable resident than the input embedding, which only needs one row. */
    QTRY(m->output_norm);
    if (m->tied_output) QTRY(m->token_embd);
    else QTRY(m->output);

    for (int l = 0; l < m->n_exec_layers; ++l) {
        Q35Layer *x = &m->layers[l];
        QTRY(x->attn_norm); QTRY(x->post_norm);
        if (x->recurrent) {
            QTRY(x->qkv); QTRY(x->z); QTRY(x->beta); QTRY(x->alpha); QTRY(x->conv1d);
            QTRY(x->dt_bias); QTRY(x->a); QTRY(x->ssm_norm); QTRY(x->ssm_out);
        } else {
            QTRY(x->q); QTRY(x->k); QTRY(x->v); QTRY(x->q_norm); QTRY(x->k_norm); QTRY(x->o);
        }
        QTRY(x->router); QTRY(x->shared_gate_inp); QTRY(x->shared_gate);
        QTRY(x->shared_up); QTRY(x->shared_down);
        if (l >= m->n_layers) {
            QTRY(x->nextn_eh_proj); QTRY(x->nextn_enorm); QTRY(x->nextn_hnorm);
            /* Dedicated MTP embeddings are row-streamed for the same reason as
             * token_embd. A dedicated draft head, if present, must be resident. */
            if (x->has_nextn_head) QTRY(x->nextn_head);
            if (x->has_nextn_head_norm) QTRY(x->nextn_head_norm);
        }
        if (m->verbose && ((l + 1) % 8 == 0 || l + 1 == m->n_exec_layers))
            fprintf(stderr, "[CUDA] resident Q35 dense block %d/%d%s\n", l + 1, m->n_exec_layers,
                    l >= m->n_layers ? " (MTP)" : "");
    }
#undef QTRY
    m->cuda_weight_bytes = m->cuda_dense_bytes;
    m->cuda_dense_complete = 1;
    return 1;
#endif
}

static int q35_alloc_state(Q35Model *m, int context, char *err, size_t cap) {
    if (context <= 0 || context > m->context_length)
        return q35_errf(err, cap, "requested context %d exceeds model context %d", context, m->context_length);

    size_t kv_elems = (size_t)m->n_attention_exec * context * m->kv_dim;
    size_t conv_elems = (size_t)m->n_recurrent * m->conv_dim * m->conv_kernel;
    size_t state_elems = (size_t)m->n_recurrent * m->n_value_heads *
                         m->state_size * m->value_head_dim;
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        const size_t kv_bytes = kv_elems * sizeof(float);
        const size_t conv_bytes = conv_elems * sizeof(float);
        const size_t state_bytes = state_elems * sizeof(float);
        m->k_cache_dev = (float *)coli_cuda_pipe_alloc(m->exec.device, kv_bytes ? kv_bytes : sizeof(float));
        m->v_cache_dev = (float *)coli_cuda_pipe_alloc(m->exec.device, kv_bytes ? kv_bytes : sizeof(float));
        m->conv_state_dev = (float *)coli_cuda_pipe_alloc(m->exec.device, conv_bytes ? conv_bytes : sizeof(float));
        m->recurrent_state_dev = (float *)coli_cuda_pipe_alloc(m->exec.device, state_bytes ? state_bytes : sizeof(float));
        if (!m->k_cache_dev || !m->v_cache_dev || !m->conv_state_dev || !m->recurrent_state_dev) {
            q35_release_state(m);
            return q35_errf(err, cap, "out of VRAM allocating Q35 hybrid state");
        }
        if ((kv_elems && (!coli_cuda_pipe_zero(m->exec.device, m->k_cache_dev, kv_elems) ||
                          !coli_cuda_pipe_zero(m->exec.device, m->v_cache_dev, kv_elems))) ||
            (conv_elems && !coli_cuda_pipe_zero(m->exec.device, m->conv_state_dev, conv_elems)) ||
            (state_elems && !coli_cuda_pipe_zero(m->exec.device, m->recurrent_state_dev, state_elems))) {
            q35_release_state(m);
            return q35_errf(err, cap, "cannot initialize Q35 CUDA state");
        }
    } else
#endif
    {
        m->k_cache = kv_elems ? (float *)calloc(kv_elems, sizeof(float)) : NULL;
        m->v_cache = kv_elems ? (float *)calloc(kv_elems, sizeof(float)) : NULL;
        m->conv_state = conv_elems ? (float *)calloc(conv_elems, sizeof(float)) : NULL;
        m->recurrent_state = state_elems ? (float *)calloc(state_elems, sizeof(float)) : NULL;
        if ((kv_elems && (!m->k_cache || !m->v_cache)) ||
            (conv_elems && !m->conv_state) || (state_elems && !m->recurrent_state)) {
            q35_release_state(m);
            return q35_errf(err, cap, "out of memory allocating Q35 hybrid state");
        }
    }
    m->context_capacity = context;
    return 1;
}

static int q35_downgrade_to_cpu(Q35Model *m, int context,
                                  const char *reason, char *err, size_t cap) {
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        if (m->verbose)
            fprintf(stderr, "[CUDA] Q35 VRAM fallback: %s; releasing CUDA state/weights and continuing from mmap on CPU RAM\n",
                    reason && *reason ? reason : "insufficient VRAM");
        q35_release_dense_cuda(m);
        q35_release_state(m);
        m->exec.kind = COLI_BACKEND_CPU;
        m->exec.device = 0;
        m->cuda_fell_back_to_cpu = 1;
        m->cuda_dense_complete = 0;
        m->cuda_expert_admission_disabled = 1;
        m->cuda_expert_compute_disabled = 1;
    }
#else
    (void)reason;
#endif
    return q35_alloc_state(m, context, err, cap);
}

/* ---- Scratch ------------------------------------------------------------------ */

static void q35_scratch_free(Q35Scratch *s) {
#ifdef COLI_CUDA
    if (s->cuda_allocated) {
#define QDFREE(x) do { if (s->x) coli_cuda_pipe_free(s->cuda_device, s->x); } while (0)
        QDFREE(dx); QDFREE(dnorm); QDFREE(dmixer); QDFREE(dpost);
        QDFREE(dqg); QDFREE(dattn_q); QDFREE(dattn_gate); QDFREE(dk); QDFREE(dv); QDFREE(dattn); QDFREE(dscores);
        QDFREE(dqkv); QDFREE(dz); QDFREE(dbeta); QDFREE(dalpha); QDFREE(ddelta);
        QDFREE(drouter); QDFREE(dgate); QDFREE(dup); QDFREE(dexpert_out); QDFREE(dgroup_out); QDFREE(dmoe);
        QDFREE(dshared_gate); QDFREE(dshared_up); QDFREE(dshared_out); QDFREE(dlogits);
        QDFREE(dhidden_norm); QDFREE(dmtp_hidden); QDFREE(dmtp_x); QDFREE(dmtp_cat); QDFREE(dmtp_logits);
        QDFREE(dvx); QDFREE(dvnorm); QDFREE(dvmixer); QDFREE(dvpost);
        QDFREE(dvqg); QDFREE(dvattn_q); QDFREE(dvattn_gate); QDFREE(dvk); QDFREE(dvv); QDFREE(dvattn);
        QDFREE(dvqkv); QDFREE(dvz); QDFREE(dvbeta); QDFREE(dvalpha); QDFREE(dvdelta);
        QDFREE(dvrouter); QDFREE(dvgate); QDFREE(dvup); QDFREE(dvexpert_in); QDFREE(dvexpert_out); QDFREE(dvmoe);
        QDFREE(dvshared_gate); QDFREE(dvshared_up); QDFREE(dvshared_out); QDFREE(dvlogits);
        QDFREE(dconv_backup); QDFREE(dstate_backup); QDFREE(dhidden_backup);
#undef QDFREE
    }
#endif
#define QHFREE(x) free(s->x)
    QHFREE(x); QHFREE(norm); QHFREE(mixer); QHFREE(post);
    QHFREE(qg); QHFREE(attn_q); QHFREE(attn_gate); QHFREE(k); QHFREE(v); QHFREE(attn); QHFREE(scores);
    QHFREE(qkv); QHFREE(z); QHFREE(beta); QHFREE(alpha); QHFREE(conv); QHFREE(delta);
    QHFREE(router); QHFREE(gate); QHFREE(up); QHFREE(expert_out); QHFREE(moe);
    QHFREE(shared_gate); QHFREE(shared_up); QHFREE(shared_out);
    QHFREE(logits); QHFREE(weight); QHFREE(hidden_norm); QHFREE(mtp_hidden); QHFREE(mtp_x); QHFREE(mtp_cat); QHFREE(mtp_logits); QHFREE(top_idx); QHFREE(top_w);
    QHFREE(vx); QHFREE(vnorm); QHFREE(vmixer); QHFREE(vpost);
    QHFREE(vqg); QHFREE(vattn_q); QHFREE(vattn_gate); QHFREE(vk); QHFREE(vv); QHFREE(vattn);
    QHFREE(vqkv); QHFREE(vz); QHFREE(vbeta); QHFREE(valpha); QHFREE(vdelta);
    QHFREE(vrouter); QHFREE(vgate); QHFREE(vup); QHFREE(vexpert_in); QHFREE(vexpert_out); QHFREE(vmoe);
    QHFREE(vshared_gate); QHFREE(vshared_up); QHFREE(vshared_out); QHFREE(vlogits);
    QHFREE(vtop_idx); QHFREE(vtop_w); QHFREE(conv_backup); QHFREE(state_backup); QHFREE(hidden_backup);
#undef QHFREE
    memset(s, 0, sizeof(*s));
}

static int q35_scratch_alloc(const Q35Model *m, Q35Scratch *s, char *err, size_t cap) {
#define QALLOC(field, n, type) do { \
    s->field = (type *)calloc((size_t)(n), sizeof(type)); \
    if (!s->field) { q35_scratch_free(s); return q35_errf(err, cap, "Q35 inference scratch OOM"); } \
} while (0)
    QALLOC(x, m->hidden, float); QALLOC(norm, m->hidden, float);
    QALLOC(mixer, m->hidden, float); QALLOC(post, m->hidden, float);
    QALLOC(qg, 2 * m->n_heads * m->head_dim, float);
    QALLOC(attn_q, m->n_heads * m->head_dim, float);
    QALLOC(attn_gate, m->n_heads * m->head_dim, float);
    QALLOC(k, m->kv_dim, float); QALLOC(v, m->kv_dim, float);
    QALLOC(attn, m->n_heads * m->head_dim, float);
    QALLOC(scores, (size_t)m->n_heads * m->context_capacity, float);
    QALLOC(qkv, m->conv_dim, float); QALLOC(z, m->value_dim, float);
    QALLOC(beta, m->n_value_heads, float); QALLOC(alpha, m->n_value_heads, float);
 QALLOC(conv, m->conv_dim, float);
    QALLOC(delta, m->value_dim, float);
    QALLOC(router, m->n_experts, float);
    QALLOC(gate, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff, float);
    QALLOC(up, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff, float);
    QALLOC(expert_out, m->hidden, float); QALLOC(moe, m->hidden, float);
    QALLOC(shared_gate, 1, float); QALLOC(shared_up, m->shared_ff, float);
    QALLOC(shared_out, m->hidden, float);
    QALLOC(logits, m->vocab, float); QALLOC(weight, m->hidden > m->head_dim ? m->hidden : m->head_dim, float);
    QALLOC(hidden_norm, m->hidden, float);
    if (m->mtp_enabled) {
        QALLOC(mtp_hidden, m->hidden, float); QALLOC(mtp_x, m->hidden, float);
        QALLOC(mtp_cat, 2 * m->hidden, float); QALLOC(mtp_logits, m->vocab, float);
    }
    QALLOC(top_idx, m->n_expert_used, int); QALLOC(top_w, m->n_expert_used, float);
    s->verify_cap = m->mtp_enabled && m->mtp_draft_max > 0 ? m->mtp_draft_max + 1 : 0;
    if (s->verify_cap > 0) {
        const size_t B = (size_t)s->verify_cap;
        const size_t maxff = (size_t)(m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff);
        const size_t conv_elems = (size_t)m->n_recurrent * m->conv_dim * m->conv_kernel;
        const size_t state_elems = (size_t)m->n_recurrent * m->n_value_heads * m->state_size * m->value_head_dim;
        QALLOC(vx, B*m->hidden, float); QALLOC(vnorm, B*m->hidden, float);
        QALLOC(vmixer, B*m->hidden, float); QALLOC(vpost, B*m->hidden, float);
        QALLOC(vqg, B*2*m->n_heads*m->head_dim, float);
        QALLOC(vattn_q, B*m->n_heads*m->head_dim, float); QALLOC(vattn_gate, B*m->n_heads*m->head_dim, float);
        QALLOC(vk, B*m->kv_dim, float); QALLOC(vv, B*m->kv_dim, float); QALLOC(vattn, B*m->n_heads*m->head_dim, float);
        QALLOC(vqkv, B*m->conv_dim, float); QALLOC(vz, B*m->value_dim, float);
        QALLOC(vbeta, B*m->n_value_heads, float); QALLOC(valpha, B*m->n_value_heads, float); QALLOC(vdelta, B*m->value_dim, float);
        QALLOC(vrouter, B*m->n_experts, float); QALLOC(vgate, B*maxff, float); QALLOC(vup, B*maxff, float);
        QALLOC(vexpert_in, B*m->hidden, float); QALLOC(vexpert_out, B*m->hidden, float); QALLOC(vmoe, B*m->hidden, float);
        QALLOC(vshared_gate, B, float); QALLOC(vshared_up, B*m->shared_ff, float); QALLOC(vshared_out, B*m->hidden, float);
        QALLOC(vlogits, B*m->vocab, float); QALLOC(vtop_idx, B*m->n_expert_used, int); QALLOC(vtop_w, B*m->n_expert_used, float);
        QALLOC(conv_backup, conv_elems ? conv_elems : 1, float);
        QALLOC(state_backup, state_elems ? state_elems : 1, float);
        QALLOC(hidden_backup, m->hidden, float);
    }
#undef QALLOC

#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        s->cuda_device = m->exec.device;
#define QDALLOC(field, n) do { \
    s->field = (float *)coli_cuda_pipe_alloc(m->exec.device, (size_t)(n) * sizeof(float)); \
    if (!s->field) { q35_scratch_free(s); return q35_errf(err, cap, "Q35 CUDA scratch OOM: %s", #field); } \
} while (0)
        QDALLOC(dx, m->hidden); QDALLOC(dnorm, m->hidden); QDALLOC(dmixer, m->hidden); QDALLOC(dpost, m->hidden);
        QDALLOC(dqg, 2 * m->n_heads * m->head_dim);
        QDALLOC(dattn_q, m->n_heads * m->head_dim); QDALLOC(dattn_gate, m->n_heads * m->head_dim);
        QDALLOC(dk, m->kv_dim); QDALLOC(dv, m->kv_dim);
        QDALLOC(dattn, m->n_heads * m->head_dim); QDALLOC(dscores, (size_t)m->n_heads * m->context_capacity);
        QDALLOC(dqkv, m->conv_dim); QDALLOC(dz, m->value_dim); QDALLOC(dbeta, m->n_value_heads); QDALLOC(dalpha, m->n_value_heads);
        QDALLOC(ddelta, m->value_dim); QDALLOC(drouter, m->n_experts);
        QDALLOC(dgate, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff);
        QDALLOC(dup, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff);
        QDALLOC(dexpert_out, m->hidden); QDALLOC(dgroup_out, m->hidden); QDALLOC(dmoe, m->hidden);
        QDALLOC(dshared_gate, 1); QDALLOC(dshared_up, m->shared_ff); QDALLOC(dshared_out, m->hidden);
        QDALLOC(dlogits, m->vocab); QDALLOC(dhidden_norm, m->hidden);
        if (m->mtp_enabled) {
            QDALLOC(dmtp_hidden, m->hidden); QDALLOC(dmtp_x, m->hidden);
            QDALLOC(dmtp_cat, 2 * m->hidden); QDALLOC(dmtp_logits, m->vocab);
        }
        if (s->verify_cap > 0) {
            const size_t B = (size_t)s->verify_cap;
            const size_t maxff = (size_t)(m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff);
            const size_t conv_elems = (size_t)m->n_recurrent * m->conv_dim * m->conv_kernel;
            const size_t state_elems = (size_t)m->n_recurrent * m->n_value_heads * m->state_size * m->value_head_dim;
            QDALLOC(dvx, B*m->hidden); QDALLOC(dvnorm, B*m->hidden); QDALLOC(dvmixer, B*m->hidden); QDALLOC(dvpost, B*m->hidden);
            QDALLOC(dvqg, B*2*m->n_heads*m->head_dim); QDALLOC(dvattn_q, B*m->n_heads*m->head_dim);
            QDALLOC(dvattn_gate, B*m->n_heads*m->head_dim); QDALLOC(dvk, B*m->kv_dim); QDALLOC(dvv, B*m->kv_dim);
            QDALLOC(dvattn, B*m->n_heads*m->head_dim); QDALLOC(dvqkv, B*m->conv_dim); QDALLOC(dvz, B*m->value_dim);
            QDALLOC(dvbeta, B*m->n_value_heads); QDALLOC(dvalpha, B*m->n_value_heads); QDALLOC(dvdelta, B*m->value_dim);
            QDALLOC(dvrouter, B*m->n_experts); QDALLOC(dvgate, B*maxff); QDALLOC(dvup, B*maxff);
            QDALLOC(dvexpert_in, B*m->hidden); QDALLOC(dvexpert_out, B*m->hidden); QDALLOC(dvmoe, B*m->hidden);
            QDALLOC(dvshared_gate, B); QDALLOC(dvshared_up, B*m->shared_ff); QDALLOC(dvshared_out, B*m->hidden);
            QDALLOC(dvlogits, B*m->vocab); QDALLOC(dconv_backup, conv_elems ? conv_elems : 1);
            QDALLOC(dstate_backup, state_elems ? state_elems : 1);
            QDALLOC(dhidden_backup, m->hidden);
        }
#undef QDALLOC
        s->cuda_allocated = 1;
        /* MTP position zero pairs token[0] with the cross-batch pending hidden
         * row. At the start of a sequence that row is defined as all zeros. */
        if (!coli_cuda_pipe_zero(m->exec.device, s->dhidden_norm, (size_t)m->hidden)) {
            q35_scratch_free(s);
            return q35_errf(err, cap, "cannot initialize Q35 MTP pending hidden row");
        }
    }
#endif
    return 1;
}


static int q35_moe_cpu(Q35Model *m, Q35Scratch *s, Q35Layer *L,
                       int layer, const float *input, float *out,
                       int allow_pilot, char *err, size_t cap) {
    if (!q35_mm_cpu(s->router, input, &L->router, m->hidden, m->n_experts, err, cap)) return 0;
    const int nk = coli_f32_router_topk(s->router, m->n_experts, m->n_expert_used,
                                        s->top_idx, s->top_w);
    if (nk != m->n_expert_used)
        return q35_errf(err, cap, "Q35 CPU router top-k failed at layer %d", layer);
    q35_couple_prefetch(m, layer, s->top_idx, nk);
    if (allow_pilot && m->pilot && layer + 1 < m->n_layers) {
        int pidx[64]; float pw[64];
        const int keep = m->pilot_k < 64 ? m->pilot_k : 64;
        if (!q35_mm_cpu(s->router, input, &m->layers[layer + 1].router,
                         m->hidden, m->n_experts, err, cap)) return 0;
        const int pn = coli_f32_router_topk(s->router, m->n_experts, keep, pidx, pw);
        q35_prefetch_ids(m, layer + 1, pidx, pn);
    }

    memset(out, 0, (size_t)m->hidden * sizeof(float));
    if (!q35_mm_cpu(s->shared_gate, input, &L->shared_gate_inp, m->hidden, 1, err, cap) ||
        !q35_mm_cpu(s->gate, input, &L->shared_gate, m->hidden, m->shared_ff, err, cap) ||
        !q35_mm_cpu(s->shared_up, input, &L->shared_up, m->hidden, m->shared_ff, err, cap)) return 0;
    for (int i = 0; i < m->shared_ff; ++i)
        s->gate[i] = coli_f32_silu(s->gate[i]) * s->shared_up[i];
    if (!q35_mm_cpu(s->shared_out, s->gate, &L->shared_down,
                     m->shared_ff, m->hidden, err, cap)) return 0;
    const float sw = q35_sigmoid(s->shared_gate[0]);
    for (int i = 0; i < m->hidden; ++i) out[i] = sw * s->shared_out[i];

    for (int j = 0; j < nk; ++j) {
        const int e = s->top_idx[j];
        Q35ExpertSlot *slot = q35_expert_acquire(m, layer, e, 1);
        if (!slot) return q35_errf(err, cap, "Q35 expert admission failed at layer %d expert %d", layer, e);
        if (!q35_mm_cpu_expert(m, s->gate, input, slot->gate, m->hidden, m->expert_ff, err, cap) ||
            !q35_mm_cpu_expert(m, s->up, input, slot->up, m->hidden, m->expert_ff, err, cap)) return 0;
        for (int i = 0; i < m->expert_ff; ++i)
            s->gate[i] = coli_f32_silu(s->gate[i]) * s->up[i];
        if (!q35_mm_cpu_expert(m, s->expert_out, s->gate, slot->down,
                                m->expert_ff, m->hidden, err, cap)) return 0;
        for (int i = 0; i < m->hidden; ++i)
            out[i] += s->top_w[j] * s->expert_out[i];
    }
    return 1;
}

/* Verification-only MoE batch.  Each unique routed expert is admitted once and
 * evaluates all verification rows that selected it in one native matmul.  This
 * preserves the existing scheduler/storage policy while avoiding duplicate
 * expert uploads inside a speculative verification block. */
static int q35_moe_batch_cpu(Q35Model *m, Q35Scratch *s, Q35Layer *L,
                             int layer, const float *input, float *out, int S,
                             char *err, size_t cap) {
    if (S < 1 || S > s->verify_cap) return q35_errf(err, cap, "invalid Q35 CPU MoE batch");
    const int H = m->hidden, E = m->n_experts, K = m->n_expert_used;
    const int F = m->expert_ff, SF = m->shared_ff;

    if (!q35_mm_cpu_s(s->vrouter, input, &L->router, S, H, E, err, cap)) return 0;
    for (int r = 0; r < S; ++r) {
        int *idx = s->vtop_idx + (size_t)r * K;
        float *wt = s->vtop_w + (size_t)r * K;
        if (coli_f32_router_topk(s->vrouter + (size_t)r * E, E, K, idx, wt) != K)
            return q35_errf(err, cap, "Q35 CPU batch router top-k failed at layer %d row %d", layer, r);
        q35_couple_prefetch(m, layer, idx, K);
    }

    memset(out, 0, (size_t)S * H * sizeof(float));
    if (!q35_mm_cpu_s(s->vshared_gate, input, &L->shared_gate_inp, S, H, 1, err, cap) ||
        !q35_mm_cpu_s(s->vgate, input, &L->shared_gate, S, H, SF, err, cap) ||
        !q35_mm_cpu_s(s->vshared_up, input, &L->shared_up, S, H, SF, err, cap)) return 0;
    for (int r = 0; r < S; ++r)
        for (int i = 0; i < SF; ++i)
            s->vgate[(size_t)r * SF + i] = coli_f32_silu(s->vgate[(size_t)r * SF + i]) *
                                            s->vshared_up[(size_t)r * SF + i];
    if (!q35_mm_cpu_s(s->vshared_out, s->vgate, &L->shared_down, S, SF, H, err, cap)) return 0;
    for (int r = 0; r < S; ++r) {
        const float sw = q35_sigmoid(s->vshared_gate[r]);
        for (int i = 0; i < H; ++i) out[(size_t)r * H + i] = sw * s->vshared_out[(size_t)r * H + i];
    }

    unsigned char seen[512] = {0};
    int unique[512], nu = 0;
    if (E > (int)sizeof(seen)) return q35_errf(err, cap, "Q35 expert count exceeds verifier limit");
    for (int r = 0; r < S; ++r) {
        const int *idx = s->vtop_idx + (size_t)r * K;
        for (int j = 0; j < K; ++j) if (!seen[idx[j]]) {
            seen[idx[j]] = 1; unique[nu++] = idx[j];
        }
    }

    for (int u = 0; u < nu; ++u) {
        const int eid = unique[u];
        int rows[16], n = 0;
        float rw[16];
        Q35ExpertSlot *slot = NULL;
        for (int r = 0; r < S; ++r) {
            const int *idx = s->vtop_idx + (size_t)r * K;
            const float *wt = s->vtop_w + (size_t)r * K;
            for (int j = 0; j < K; ++j) if (idx[j] == eid) {
                /* Acquire once per routed occurrence so usage/heat and hit
                 * telemetry remain identical to ordinary token-by-token mode. */
                Q35ExpertSlot *cur = q35_expert_acquire(m, layer, eid, 1);
                if (!cur) return q35_errf(err, cap,
                    "Q35 batch expert admission failed at layer %d expert %d", layer, eid);
                if (!slot) slot = cur;
                rows[n] = r; rw[n] = wt[j]; ++n;
                memcpy(s->vexpert_in + (size_t)(n - 1) * H,
                       input + (size_t)r * H, (size_t)H * sizeof(float));
                break;
            }
        }
        if (!slot || n < 1) continue;
        if (!q35_mm_cpu_expert_s(m, s->vgate, s->vexpert_in, slot->gate, n, H, F, err, cap) ||
            !q35_mm_cpu_expert_s(m, s->vup, s->vexpert_in, slot->up, n, H, F, err, cap)) return 0;
        for (int r = 0; r < n; ++r)
            for (int i = 0; i < F; ++i)
                s->vgate[(size_t)r * F + i] = coli_f32_silu(s->vgate[(size_t)r * F + i]) *
                                               s->vup[(size_t)r * F + i];
        if (!q35_mm_cpu_expert_s(m, s->vexpert_out, s->vgate, slot->down, n, F, H, err, cap)) return 0;
        for (int r = 0; r < n; ++r) {
            float *dst = out + (size_t)rows[r] * H;
            const float *src = s->vexpert_out + (size_t)r * H;
            for (int i = 0; i < H; ++i) dst[i] += rw[r] * src[i];
        }
    }
    return 1;
}

/* ---- CPU forward --------------------------------------------------------------- */

static int q35_forward_cpu(Q35Model *m, Q35Scratch *s, int token, int pos,
                            char *err, size_t cap) {
    if (token < 0 || token >= m->vocab) return q35_errf(err, cap, "token id outside vocabulary");
    if (pos < 0 || pos >= m->context_capacity) return q35_errf(err, cap, "position outside context");
    if (!coli_tensor_read_row_f32(&m->token_embd, (uint64_t)token, s->x, (uint64_t)m->hidden))
        return q35_errf(err, cap, "embedding decode failed");

    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    const size_t conv_layer_stride = (size_t)m->conv_dim * m->conv_kernel;
    const size_t state_layer_stride = (size_t)m->n_value_heads * m->state_size * m->value_head_dim;

    for (int l = 0; l < m->n_layers; ++l) {
        Q35Layer *L = &m->layers[l];
        if (!q35_f32_vector(&L->attn_norm, s->weight, m->hidden, err, cap)) return 0;
        coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);

        if (L->recurrent) {
            if (!q35_mm_cpu(s->qkv, s->norm, &L->qkv, m->hidden, m->conv_dim, err, cap) ||
                !q35_mm_cpu(s->z, s->norm, &L->z, m->hidden, m->value_dim, err, cap) ||
                !q35_mm_cpu(s->beta, s->norm, &L->beta, m->hidden, m->n_value_heads, err, cap) ||
                !q35_mm_cpu(s->alpha, s->norm, &L->alpha, m->hidden, m->n_value_heads, err, cap)) return 0;
            float *conv_state = m->conv_state + (size_t)L->recurrent_index * conv_layer_stride;
            float *rec_state = m->recurrent_state + (size_t)L->recurrent_index * state_layer_stride;
            if (!q35_conv_update_cpu(m, L, s->qkv, s->conv, conv_state, err, cap) ||
                !q35_delta_decode_cpu(m, L, s->conv, s->z, s->beta, s->alpha, rec_state, s->delta, err, cap) ||
                !q35_mm_cpu(s->mixer, s->delta, &L->ssm_out, m->value_dim, m->hidden, err, cap)) return 0;
        } else {
            if (!q35_mm_cpu(s->qg, s->norm, &L->q, m->hidden, 2 * m->n_heads * m->head_dim, err, cap) ||
                !q35_mm_cpu(s->k, s->norm, &L->k, m->hidden, m->kv_dim, err, cap) ||
                !q35_mm_cpu(s->v, s->norm, &L->v, m->hidden, m->kv_dim, err, cap)) return 0;
            q35_split_q_gate(s->attn_q, s->attn_gate, s->qg, m->n_heads, m->head_dim);
            if (!q35_f32_vector(&L->q_norm, s->weight, m->head_dim, err, cap)) return 0;
            for (int h = 0; h < m->n_heads; ++h)
                coli_f32_rmsnorm(s->attn_q + (size_t)h * m->head_dim,
                                 s->attn_q + (size_t)h * m->head_dim,
                                 s->weight, m->head_dim, m->eps);
            if (!q35_f32_vector(&L->k_norm, s->weight, m->head_dim, err, cap)) return 0;
            for (int h = 0; h < m->n_kv_heads; ++h)
                coli_f32_rmsnorm(s->k + (size_t)h * m->head_dim,
                                 s->k + (size_t)h * m->head_dim,
                                 s->weight, m->head_dim, m->eps);
            q35_rope_neox(s->attn_q, m->n_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
            q35_rope_neox(s->k, m->n_kv_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
            float *kc = m->k_cache + (size_t)L->attention_index * kv_layer_stride;
            float *vc = m->v_cache + (size_t)L->attention_index * kv_layer_stride;
            memcpy(kc + (size_t)pos * m->kv_dim, s->k, (size_t)m->kv_dim * sizeof(float));
            memcpy(vc + (size_t)pos * m->kv_dim, s->v, (size_t)m->kv_dim * sizeof(float));
            coli_f32_gqa_attention(s->attn, s->attn_q, kc, vc, pos, m->n_heads, m->n_kv_heads,
                                   m->head_dim, m->attention_scale, s->scores);
            for (int i = 0; i < m->n_heads * m->head_dim; ++i)
                s->attn[i] *= q35_sigmoid(s->attn_gate[i]);
            if (!q35_mm_cpu(s->mixer, s->attn, &L->o,
                             m->n_heads * m->head_dim, m->hidden, err, cap)) return 0;
        }

        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->mixer[i];
        if (!q35_f32_vector(&L->post_norm, s->weight, m->hidden, err, cap)) return 0;
        coli_f32_rmsnorm(s->post, s->x, s->weight, m->hidden, m->eps);

        if (!q35_moe_cpu(m, s, L, l, s->post, s->moe, 1, err, cap)) return 0;
        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->moe[i];
    }

    if (!q35_f32_vector(&m->output_norm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
    memcpy(s->hidden_norm, s->norm, (size_t)m->hidden * sizeof(float));
    ColiTensor *outw = m->tied_output ? &m->token_embd : &m->output;
    return q35_mm_cpu(s->logits, s->norm, outw, m->hidden, m->vocab, err, cap);
}

static int q35_forward_batch_cpu(Q35Model *m, Q35Scratch *s,
                                  const int *tokens, int S, int pos0,
                                  char *err, size_t cap) {
    if (!tokens || S < 1 || S > s->verify_cap)
        return q35_errf(err, cap, "invalid Q35 CPU verification batch");
    if (pos0 < 0 || pos0 + S > m->context_capacity)
        return q35_errf(err, cap, "Q35 CPU verification batch outside context");
    const int H = m->hidden;
    const int QD = m->n_heads * m->head_dim;
    const int QG = 2 * QD;
    for (int r = 0; r < S; ++r) {
        if (tokens[r] < 0 || tokens[r] >= m->vocab ||
            !coli_tensor_read_row_f32(&m->token_embd, (uint64_t)tokens[r],
                                      s->vx + (size_t)r * H, (uint64_t)H))
            return q35_errf(err, cap, "Q35 CPU batch embedding decode failed at row %d", r);
    }

    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    const size_t conv_layer_stride = (size_t)m->conv_dim * m->conv_kernel;
    const size_t state_layer_stride = (size_t)m->n_value_heads * m->state_size * m->value_head_dim;

    for (int l = 0; l < m->n_layers; ++l) {
        Q35Layer *L = &m->layers[l];
        if (!q35_f32_vector(&L->attn_norm, s->weight, H, err, cap)) return 0;
        for (int r = 0; r < S; ++r)
            coli_f32_rmsnorm(s->vnorm + (size_t)r * H,
                             s->vx + (size_t)r * H, s->weight, H, m->eps);

        if (L->recurrent) {
            if (!q35_mm_cpu_s(s->vqkv, s->vnorm, &L->qkv, S, H, m->conv_dim, err, cap) ||
                !q35_mm_cpu_s(s->vz, s->vnorm, &L->z, S, H, m->value_dim, err, cap) ||
                !q35_mm_cpu_s(s->vbeta, s->vnorm, &L->beta, S, H, m->n_value_heads, err, cap) ||
                !q35_mm_cpu_s(s->valpha, s->vnorm, &L->alpha, S, H, m->n_value_heads, err, cap)) return 0;
            float *conv_state = m->conv_state + (size_t)L->recurrent_index * conv_layer_stride;
            float *rec_state = m->recurrent_state + (size_t)L->recurrent_index * state_layer_stride;
            for (int r = 0; r < S; ++r) {
                if (!q35_conv_update_cpu(m, L,
                        s->vqkv + (size_t)r * m->conv_dim, s->conv,
                        conv_state, err, cap) ||
                    !q35_delta_decode_cpu(m, L, s->conv,
                        s->vz + (size_t)r * m->value_dim,
                        s->vbeta + (size_t)r * m->n_value_heads,
                        s->valpha + (size_t)r * m->n_value_heads,
                        rec_state, s->vdelta + (size_t)r * m->value_dim,
                        err, cap)) return 0;
            }
            if (!q35_mm_cpu_s(s->vmixer, s->vdelta, &L->ssm_out,
                               S, m->value_dim, H, err, cap)) return 0;
        } else {
            if (!q35_mm_cpu_s(s->vqg, s->vnorm, &L->q, S, H, QG, err, cap) ||
                !q35_mm_cpu_s(s->vk, s->vnorm, &L->k, S, H, m->kv_dim, err, cap) ||
                !q35_mm_cpu_s(s->vv, s->vnorm, &L->v, S, H, m->kv_dim, err, cap)) return 0;
            if (!q35_f32_vector(&L->q_norm, s->weight, m->head_dim, err, cap)) return 0;
            for (int r = 0; r < S; ++r) {
                float *qr = s->vattn_q + (size_t)r * QD;
                float *gr = s->vattn_gate + (size_t)r * QD;
                q35_split_q_gate(qr, gr, s->vqg + (size_t)r * QG,
                                  m->n_heads, m->head_dim);
                for (int h = 0; h < m->n_heads; ++h)
                    coli_f32_rmsnorm(qr + (size_t)h * m->head_dim,
                                     qr + (size_t)h * m->head_dim,
                                     s->weight, m->head_dim, m->eps);
            }
            if (!q35_f32_vector(&L->k_norm, s->weight, m->head_dim, err, cap)) return 0;
            for (int r = 0; r < S; ++r) {
                float *kr = s->vk + (size_t)r * m->kv_dim;
                for (int h = 0; h < m->n_kv_heads; ++h)
                    coli_f32_rmsnorm(kr + (size_t)h * m->head_dim,
                                     kr + (size_t)h * m->head_dim,
                                     s->weight, m->head_dim, m->eps);
                const int pos = pos0 + r;
                float *qr = s->vattn_q + (size_t)r * QD;
                q35_rope_neox(qr, m->n_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
                q35_rope_neox(kr, m->n_kv_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
                float *kc = m->k_cache + (size_t)L->attention_index * kv_layer_stride;
                float *vc = m->v_cache + (size_t)L->attention_index * kv_layer_stride;
                memcpy(kc + (size_t)pos * m->kv_dim, kr, (size_t)m->kv_dim * sizeof(float));
                memcpy(vc + (size_t)pos * m->kv_dim,
                       s->vv + (size_t)r * m->kv_dim, (size_t)m->kv_dim * sizeof(float));
                float *ar = s->vattn + (size_t)r * QD;
                coli_f32_gqa_attention(ar, qr, kc, vc, pos,
                    m->n_heads, m->n_kv_heads, m->head_dim,
                    m->attention_scale, s->scores);
                const float *gr = s->vattn_gate + (size_t)r * QD;
                for (int i = 0; i < QD; ++i) ar[i] *= q35_sigmoid(gr[i]);
            }
            if (!q35_mm_cpu_s(s->vmixer, s->vattn, &L->o, S, QD, H, err, cap)) return 0;
        }

        for (int i = 0; i < S * H; ++i) s->vx[i] += s->vmixer[i];
        if (!q35_f32_vector(&L->post_norm, s->weight, H, err, cap)) return 0;
        for (int r = 0; r < S; ++r)
            coli_f32_rmsnorm(s->vpost + (size_t)r * H,
                             s->vx + (size_t)r * H, s->weight, H, m->eps);
        if (!q35_moe_batch_cpu(m, s, L, l, s->vpost, s->vmoe, S, err, cap)) return 0;
        for (int i = 0; i < S * H; ++i) s->vx[i] += s->vmoe[i];
    }

    if (!q35_f32_vector(&m->output_norm, s->weight, H, err, cap)) return 0;
    for (int r = 0; r < S; ++r)
        coli_f32_rmsnorm(s->vnorm + (size_t)r * H,
                         s->vx + (size_t)r * H, s->weight, H, m->eps);
    memcpy(s->hidden_norm, s->vnorm + (size_t)(S - 1) * H, (size_t)H * sizeof(float));
    ColiTensor *outw = m->tied_output ? &m->token_embd : &m->output;
    return q35_mm_cpu_s(s->vlogits, s->vnorm, outw, S, H, m->vocab, err, cap);
}


static int q35_mtp_forward_cpu(Q35Model *m, Q35Scratch *s, int token,
                                const float *hidden_in, int pos, int need_logits,
                                char *err, size_t cap) {
    if (!m->n_mtp_layers || m->mtp_layer < 0)
        return q35_errf(err, cap, "Q35 MTP head is not present");
    if (token < 0 || token >= m->vocab || pos < 0 || pos >= m->context_capacity)
        return q35_errf(err, cap, "invalid Q35 MTP token/position");
    Q35Layer *L = &m->layers[m->mtp_layer];
    ColiTensor *embed = L->has_nextn_embed ? &L->nextn_embed_tokens : &m->token_embd;
    if (!coli_tensor_read_row_f32(embed, (uint64_t)token, s->mtp_x, (uint64_t)m->hidden))
        return q35_errf(err, cap, "Q35 MTP embedding decode failed");

    if (!q35_f32_vector(&L->nextn_enorm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->norm, s->mtp_x, s->weight, m->hidden, m->eps);
    if (!q35_f32_vector(&L->nextn_hnorm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->post, hidden_in, s->weight, m->hidden, m->eps);
    memcpy(s->mtp_cat, s->norm, (size_t)m->hidden * sizeof(float));
    memcpy(s->mtp_cat + m->hidden, s->post, (size_t)m->hidden * sizeof(float));
    if (!q35_mm_cpu(s->mtp_x, s->mtp_cat, &L->nextn_eh_proj,
                     2 * m->hidden, m->hidden, err, cap)) return 0;

    if (!q35_f32_vector(&L->attn_norm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->norm, s->mtp_x, s->weight, m->hidden, m->eps);

    /* State-only MTP synchronization needs only the K/V rows.  K and V are
     * projections of the pre-attention MTP input and do not depend on Q,
     * attention, the MLP, output normalization, or the vocabulary head. */
    if (!q35_mm_cpu(s->k, s->norm, &L->k, m->hidden, m->kv_dim, err, cap) ||
        !q35_mm_cpu(s->v, s->norm, &L->v, m->hidden, m->kv_dim, err, cap)) return 0;
    if (!q35_f32_vector(&L->k_norm, s->weight, m->head_dim, err, cap)) return 0;
    for (int h = 0; h < m->n_kv_heads; ++h)
        coli_f32_rmsnorm(s->k + (size_t)h * m->head_dim,
                         s->k + (size_t)h * m->head_dim,
                         s->weight, m->head_dim, m->eps);
    q35_rope_neox(s->k, m->n_kv_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    float *kc = m->k_cache + (size_t)L->attention_index * kv_layer_stride;
    float *vc = m->v_cache + (size_t)L->attention_index * kv_layer_stride;
    memcpy(kc + (size_t)pos * m->kv_dim, s->k, (size_t)m->kv_dim * sizeof(float));
    memcpy(vc + (size_t)pos * m->kv_dim, s->v, (size_t)m->kv_dim * sizeof(float));
    if (!need_logits) {
        ++m->mtp_kv_updates;
        return 1;
    }

    if (!q35_mm_cpu(s->qg, s->norm, &L->q, m->hidden,
                     2 * m->n_heads * m->head_dim, err, cap)) return 0;
    q35_split_q_gate(s->attn_q, s->attn_gate, s->qg, m->n_heads, m->head_dim);
    if (!q35_f32_vector(&L->q_norm, s->weight, m->head_dim, err, cap)) return 0;
    for (int h = 0; h < m->n_heads; ++h)
        coli_f32_rmsnorm(s->attn_q + (size_t)h * m->head_dim,
                         s->attn_q + (size_t)h * m->head_dim,
                         s->weight, m->head_dim, m->eps);
    /* Text-only Qwen3.5 uses equal temporal/height/width positions, so IMRoPE
     * reduces exactly to the same partial NeoX rotation used by the trunk. */
    q35_rope_neox(s->attn_q, m->n_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
    coli_f32_gqa_attention(s->attn, s->attn_q, kc, vc, pos,
                           m->n_heads, m->n_kv_heads, m->head_dim,
                           m->attention_scale, s->scores);
    for (int i = 0; i < m->n_heads * m->head_dim; ++i)
        s->attn[i] *= q35_sigmoid(s->attn_gate[i]);
    if (!q35_mm_cpu(s->mixer, s->attn, &L->o,
                     m->n_heads * m->head_dim, m->hidden, err, cap)) return 0;
    for (int i = 0; i < m->hidden; ++i) s->mtp_x[i] += s->mixer[i];

    if (!q35_f32_vector(&L->post_norm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->post, s->mtp_x, s->weight, m->hidden, m->eps);
    if (!q35_moe_cpu(m, s, L, m->mtp_layer, s->post, s->moe, 0, err, cap)) return 0;
    for (int i = 0; i < m->hidden; ++i) s->mtp_x[i] += s->moe[i];

    ColiTensor *head_norm = L->has_nextn_head_norm ? &L->nextn_head_norm : &m->output_norm;
    if (!q35_f32_vector(head_norm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->mtp_hidden, s->mtp_x, s->weight, m->hidden, m->eps);
    ColiTensor *head = L->has_nextn_head ? &L->nextn_head :
                       (m->tied_output ? &m->token_embd : &m->output);
    if (!q35_mm_cpu(s->mtp_logits, s->mtp_hidden, head,
                     m->hidden, m->vocab, err, cap)) return 0;
    ++m->mtp_steps;
    return 1;
}

#ifdef COLI_CUDA
static int q35_row_to_device(Q35Model *m, ColiTensor *t, uint64_t row,
                              float *out_dev, float *host_tmp, int width,
                              char *err, size_t cap) {
    if (!t || !out_dev || !host_tmp || width < 1 || t->dims[0] != (uint64_t)width)
        return q35_errf(err, cap, "invalid Q35 row upload request");
    if (coli_tensor_is_resident(&m->exec, t)) {
        if (coli_tensor_read_row_device(&m->exec, t, row, out_dev, 1.0f)) return 1;
        return q35_errf(err, cap, "Q35 CUDA resident row decode failed for %s",
                        t->name ? t->name : "<unnamed>");
    }
    if (!coli_tensor_read_row_f32(t, row, host_tmp, (uint64_t)width) ||
        !coli_cuda_pipe_upload(m->exec.device, out_dev, host_tmp,
                               (size_t)width * sizeof(float)))
        return q35_errf(err, cap, "Q35 mmap row stream failed for %s",
                        t->name ? t->name : "<unnamed>");
    return 1;
}

static const float *q35_f32_device(Q35Model *m, ColiTensor *t, int n,
                                    char *err, size_t cap) {
    if (t->dtype != COLI_DTYPE_F32 || t->row_count != 1 || t->dims[0] != (uint64_t)n ||
        !coli_tensor_reside(&m->exec, t)) {
        q35_errf(err, cap, "invalid resident Q35 F32 vector: %s", t->name ? t->name : "<unnamed>");
        return NULL;
    }
    return (const float *)coli_tensor_device_data(t);
}

static int q35_mm_device(Q35Model *m, float *y, const float *x, ColiTensor *w,
                          char *err, size_t cap) {
    if (!coli_tensor_matmul_device(&m->exec, y, x, w, 1))
        return q35_errf(err, cap, "Q35 CUDA matmul failed for %s", w->name ? w->name : "<unnamed>");
    return 1;
}

static int q35_mm_device_s(Q35Model *m, float *y, const float *x, ColiTensor *w,
                            int S, char *err, size_t cap) {
    if (!coli_tensor_matmul_device(&m->exec, y, x, w, S))
        return q35_errf(err, cap, "Q35 CUDA batch matmul failed for %s",
                        w->name ? w->name : "<unnamed>");
    return 1;
}

/* Routed-expert CUDA execution is intentionally kept one expert at a time.
 * On small cards this lets a single LRU slot be reused immediately instead of
 * bulk-admitting ten experts that cannot coexist. Do not write an error here:
 * the caller can fall back to the mmap-backed CPU tensor path. */
static int q35_expert_device(Q35Model *m, Q35Scratch *s,
                              const Q35ExpertSlot *slot, float weight) {
    if (!slot || !slot->gate || !slot->up || !slot->down ||
        m->cuda_expert_compute_disabled) return 0;
    const int dev = m->exec.device;
    return coli_tensor_matmul_device(&m->exec, s->dgate, s->dpost, slot->gate, 1) &&
           coli_tensor_matmul_device(&m->exec, s->dup, s->dpost, slot->up, 1) &&
           coli_cuda_pipe_silu_mul(dev, s->dgate, s->dup, (size_t)m->expert_ff) &&
           coli_tensor_matmul_device(&m->exec, s->dexpert_out, s->dgate, slot->down, 1) &&
           coli_cuda_pipe_axpy(dev, s->dmoe, s->dexpert_out, weight, (size_t)m->hidden);
}

/* Record a routed demand only when it is already resident. Misses are left
 * untouched here and pass through q35_expert_acquire() later, which records
 * the demand exactly once while preserving the normal admission policy. */
static Q35ExpertSlot *q35_expert_lookup_demand_hit(Q35Model *m, int layer,
                                                      int eid, int *from_pin) {
    ColiExpertLayerStore st = q35_store(m, layer);
    ColiExpertLookup h = coli_expert_lookup(&st, eid, 0);
    if (!h.slot) return NULL;
    if (from_pin) *from_pin = h.from_pin;
    coli_expert_record_demand(&st, eid);
    ++m->scheduler_stats.hits;
    if (h.from_pin) ++m->scheduler_stats.pin_hits;
    else {
        ++m->scheduler_stats.cache_hits;
        coli_expert_slot_touch(h.slot, &st.layout, st.clock);
    }
    return (Q35ExpertSlot *)h.slot;
}

/* Group only focused pinned experts. Pin storage cannot be evicted by the
 * sequential miss admissions that follow, so the group never holds an LRU
 * pointer across a policy transition. */
static int q35_expert_group_device(Q35Model *m, Q35Scratch *s,
                                    Q35ExpertSlot *const *slots,
                                    const float *weights, int count) {
    if (count < m->focus_group_min || count > 64) return 0;
    ColiCudaTensor *g[64], *u[64], *d[64];
    for (int i = 0; i < count; ++i) {
        Q35ExpertSlot *q = slots[i];
        if (!q || !q->gate || !q->up || !q->down ||
            q->gate->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE ||
            q->up->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE ||
            q->down->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE ||
            !q->gate->cuda || !q->up->cuda || !q->down->cuda) return 0;
        g[i] = q->gate->cuda; u[i] = q->up->cuda; d[i] = q->down->cuda;
    }
    const int dev = m->exec.device, devices[1] = { dev };
    if (!coli_cuda_expert_group_resident_issue(g, u, d, weights, count,
                                                dev, s->dpost, s->dexpert_out) ||
        !coli_cuda_expert_group_resident_take(dev, devices, 1,
                                               s->dexpert_out, s->dgroup_out,
                                               m->hidden) ||
        !coli_cuda_pipe_add(dev, s->dmoe, s->dgroup_out, (size_t)m->hidden))
        return 0;
    ++m->focus_group_calls;
    m->focus_group_experts += (uint64_t)count;
    return 1;
}

static int q35_expert_cpu_accumulate(Q35Model *m, Q35Scratch *s,
                                      Q35Layer *L, int eid, float weight,
                                      int *post_ready, int *moe_ready,
                                      char *err, size_t cap) {
    if (eid < 0 || eid >= m->n_experts) return q35_errf(err, cap, "invalid Q35 expert id");
    if (!*post_ready) {
        if (!coli_cuda_pipe_download(m->exec.device, s->dpost, s->post,
                                     (size_t)m->hidden * sizeof(float)))
            return q35_errf(err, cap, "cannot download Q35 expert input for CPU fallback");
        *post_ready = 1;
    }
    if (!*moe_ready) {
        memset(s->moe, 0, (size_t)m->hidden * sizeof(float));
        *moe_ready = 1;
    }
    ColiTensor *gate = &L->gate_expert[eid];
    ColiTensor *up = &L->up_expert[eid];
    ColiTensor *down = &L->down_expert[eid];
    coli_tensor_prefetch_host(gate);
    coli_tensor_prefetch_host(up);
    coli_tensor_prefetch_host(down);
    if (!q35_mm_cpu_expert(m, s->gate, s->post, gate, m->hidden, m->expert_ff, err, cap) ||
        !q35_mm_cpu_expert(m, s->up, s->post, up, m->hidden, m->expert_ff, err, cap)) return 0;
    for (int i = 0; i < m->expert_ff; ++i)
        s->gate[i] = coli_f32_silu(s->gate[i]) * s->up[i];
    if (!q35_mm_cpu_expert(m, s->expert_out, s->gate, down,
                            m->expert_ff, m->hidden, err, cap)) return 0;
    for (int i = 0; i < m->hidden; ++i) s->moe[i] += weight * s->expert_out[i];
    ++m->cuda_expert_cpu_fallbacks;
    return 1;
}

static int q35_router_topk_cuda(Q35Model *m, Q35Scratch *s, int k,
                                 int *idx, float *weights) {
    if (m->cuda_router_gpu_enabled &&
        coli_cuda_pipe_qwen_topk(m->exec.device, s->drouter,
                                 m->n_experts, k, idx, weights)) {
        ++m->cuda_router_gpu_calls;
        return k;
    }
    ++m->cuda_router_cpu_fallbacks;
    if (!coli_cuda_pipe_download(m->exec.device, s->drouter, s->router,
                                 (size_t)m->n_experts * sizeof(float))) return 0;
    return coli_f32_router_topk(s->router, m->n_experts, k, idx, weights);
}


static int q35_moe_cuda(Q35Model *m, Q35Scratch *s, Q35Layer *L,
                        int layer, int allow_pilot, char *err, size_t cap) {
    const int dev = m->exec.device;
    if (!q35_mm_device(m, s->drouter, s->dpost, &L->router, err, cap)) return 0;
    const int nk = q35_router_topk_cuda(m, s, m->n_expert_used,
                                         s->top_idx, s->top_w);
    if (nk != m->n_expert_used)
        return q35_errf(err, cap, "Q35 CUDA router top-k failed at layer %d", layer);
    q35_couple_prefetch(m, layer, s->top_idx, nk);
    if (allow_pilot && m->pilot && layer + 1 < m->n_layers) {
        int pidx[64]; float pw[64];
        const int keep = m->pilot_k < 64 ? m->pilot_k : 64;
        if (!q35_mm_device(m, s->drouter, s->dpost,
                            &m->layers[layer + 1].router, err, cap)) return 0;
        const int pn = q35_router_topk_cuda(m, s, keep, pidx, pw);
        if (pn != keep) return q35_errf(err, cap, "Q35 pilot router top-k failed at layer %d", layer + 1);
        q35_prefetch_ids(m, layer + 1, pidx, pn);
    }

    if (!coli_cuda_pipe_zero(dev, s->dmoe, (size_t)m->hidden) ||
        !q35_mm_device(m, s->dshared_gate, s->dpost, &L->shared_gate_inp, err, cap) ||
        !q35_mm_device(m, s->dgate, s->dpost, &L->shared_gate, err, cap) ||
        !q35_mm_device(m, s->dshared_up, s->dpost, &L->shared_up, err, cap) ||
        !coli_cuda_pipe_silu_mul(dev, s->dgate, s->dshared_up, (size_t)m->shared_ff) ||
        !q35_mm_device(m, s->dshared_out, s->dgate, &L->shared_down, err, cap) ||
        !coli_cuda_pipe_sigmoid_scale(dev, s->dshared_out, s->dshared_gate, (size_t)m->hidden) ||
        !coli_cuda_pipe_add(dev, s->dmoe, s->dshared_out, (size_t)m->hidden)) return 0;

    int post_host_ready = 0;
    int host_moe_ready = 0;
    unsigned char handled[64] = {0};
    Q35ExpertSlot *resident[64] = {0};

    if (m->focus_group_enabled && m->focus_layer_mask && m->focus_layer_mask[layer]) {
        Q35ExpertSlot *group_slots[64];
        float group_weights[64];
        int group_index[64], gn = 0;
        for (int j = 0; j < nk; ++j) {
            int from_pin = 0;
            resident[j] = q35_expert_lookup_demand_hit(m, layer, s->top_idx[j], &from_pin);
            if (resident[j] && from_pin) {
                group_slots[gn] = resident[j];
                group_weights[gn] = s->top_w[j];
                group_index[gn++] = j;
            }
        }
        if (gn >= m->focus_group_min) {
            if (q35_expert_group_device(m, s, group_slots, group_weights, gn)) {
                for (int q = 0; q < gn; ++q) handled[group_index[q]] = 1;
            } else {
                ++m->focus_group_failures;
                m->focus_group_enabled = 0;
                if (m->verbose)
                    fprintf(stderr, "[FOCUS] Q35 resident group failed; disabling grouping and retaining sequential execution\n");
            }
        }
    }

    for (int j = 0; j < nk; ++j) {
        if (handled[j]) continue;
        const int e = s->top_idx[j];
        Q35ExpertSlot *slot = resident[j] ? resident[j] : q35_expert_acquire(m, layer, e, 1);
        if (slot && q35_expert_device(m, s, slot, s->top_w[j])) continue;
        if (slot && !m->cuda_expert_compute_disabled) {
            m->cuda_expert_compute_disabled = 1;
            m->cuda_expert_admission_disabled = 1;
            ++m->cuda_expert_compute_failures;
            if (!m->cuda_expert_failure_reported) {
                m->cuda_expert_failure_reported = 1;
                fprintf(stderr, "[Q35] CUDA expert execution failed; using mmap CPU fallback for remaining routed experts\n");
            }
        }
        if (!q35_expert_cpu_accumulate(m, s, L, e, s->top_w[j],
                                         &post_host_ready, &host_moe_ready,
                                         err, cap)) return 0;
    }
    if (host_moe_ready) {
        if (!coli_cuda_pipe_upload(dev, s->dexpert_out, s->moe,
                                   (size_t)m->hidden * sizeof(float)) ||
            !coli_cuda_pipe_add(dev, s->dmoe, s->dexpert_out, (size_t)m->hidden))
            return q35_errf(err, cap, "cannot merge Q35 CPU expert fallback at layer %d", layer);
    }
    return 1;
}

static int q35_moe_batch_cuda(Q35Model *m, Q35Scratch *s, Q35Layer *L,
                              int layer, const float *input_dev, float *out_dev,
                              int S, char *err, size_t cap) {
    if (S < 1 || S > s->verify_cap) return q35_errf(err, cap, "invalid Q35 CUDA MoE batch");
    const int dev = m->exec.device, H = m->hidden, E = m->n_experts;
    const int K = m->n_expert_used, F = m->expert_ff, SF = m->shared_ff;
    if (!q35_mm_device_s(m, s->dvrouter, input_dev, &L->router, S, err, cap) ||
        !coli_cuda_pipe_download(dev, s->dvrouter, s->vrouter,
                                 (size_t)S * E * sizeof(float))) return 0;
    for (int r = 0; r < S; ++r) {
        int *idx = s->vtop_idx + (size_t)r * K;
        float *wt = s->vtop_w + (size_t)r * K;
        if (coli_f32_router_topk(s->vrouter + (size_t)r * E, E, K, idx, wt) != K)
            return q35_errf(err, cap, "Q35 CUDA batch router top-k failed at layer %d row %d", layer, r);
        q35_couple_prefetch(m, layer, idx, K);
    }

    if (!coli_cuda_pipe_zero(dev, out_dev, (size_t)S * H) ||
        !q35_mm_device_s(m, s->dvshared_gate, input_dev, &L->shared_gate_inp, S, err, cap) ||
        !q35_mm_device_s(m, s->dvgate, input_dev, &L->shared_gate, S, err, cap) ||
        !q35_mm_device_s(m, s->dvshared_up, input_dev, &L->shared_up, S, err, cap) ||
        !coli_cuda_pipe_silu_mul(dev, s->dvgate, s->dvshared_up, (size_t)S * SF) ||
        !q35_mm_device_s(m, s->dvshared_out, s->dvgate, &L->shared_down, S, err, cap)) return 0;
    for (int r = 0; r < S; ++r) {
        if (!coli_cuda_pipe_sigmoid_scale(dev,
                s->dvshared_out + (size_t)r * H,
                s->dvshared_gate + r, (size_t)H)) return 0;
    }
    if (!coli_cuda_pipe_add(dev, out_dev, s->dvshared_out, (size_t)S * H)) return 0;

    unsigned char seen[512] = {0};
    int unique[512], nu = 0;
    if (E > (int)sizeof(seen)) return q35_errf(err, cap, "Q35 expert count exceeds verifier limit");
    for (int r = 0; r < S; ++r) {
        const int *idx = s->vtop_idx + (size_t)r * K;
        for (int j = 0; j < K; ++j) if (!seen[idx[j]]) {
            seen[idx[j]] = 1; unique[nu++] = idx[j];
        }
    }

    for (int u = 0; u < nu; ++u) {
        const int eid = unique[u];
        int rows[16], n = 0;
        float rw[16];
        Q35ExpertSlot *slot = NULL;
        for (int r = 0; r < S; ++r) {
            const int *idx = s->vtop_idx + (size_t)r * K;
            const float *wt = s->vtop_w + (size_t)r * K;
            for (int j = 0; j < K; ++j) if (idx[j] == eid) {
                Q35ExpertSlot *cur = q35_expert_acquire(m, layer, eid, 1);
                if (!cur || !cur->gate || !cur->up || !cur->down ||
                    !cur->gate->cuda || !cur->up->cuda || !cur->down->cuda)
                    return q35_errf(err, cap,
                        "Q35 CUDA batch expert unavailable at layer %d expert %d", layer, eid);
                if (!slot) slot = cur;
                rows[n] = r; rw[n] = wt[j];
                if (!coli_cuda_pipe_copy(dev,
                        s->dvexpert_in + (size_t)n * H,
                        input_dev + (size_t)r * H,
                        (size_t)H * sizeof(float))) return 0;
                ++n;
                break;
            }
        }
        if (!slot || n < 1) continue;
        if (!coli_tensor_matmul_device(&m->exec, s->dvgate, s->dvexpert_in, slot->gate, n) ||
            !coli_tensor_matmul_device(&m->exec, s->dvup, s->dvexpert_in, slot->up, n) ||
            !coli_cuda_pipe_silu_mul(dev, s->dvgate, s->dvup, (size_t)n * F) ||
            !coli_tensor_matmul_device(&m->exec, s->dvexpert_out, s->dvgate, slot->down, n)) return 0;
        for (int r = 0; r < n; ++r)
            if (!coli_cuda_pipe_axpy(dev, out_dev + (size_t)rows[r] * H,
                                     s->dvexpert_out + (size_t)r * H,
                                     rw[r], (size_t)H)) return 0;
    }
    return 1;
}

static int q35_forward_batch_cuda(Q35Model *m, Q35Scratch *s,
                                   const int *tokens, int S, int pos0,
                                   char *err, size_t cap) {
    if (!tokens || S < 1 || S > s->verify_cap)
        return q35_errf(err, cap, "invalid Q35 CUDA verification batch");
    if (pos0 < 0 || pos0 + S > m->context_capacity)
        return q35_errf(err, cap, "Q35 CUDA verification batch outside context");
    const int dev = m->exec.device, H = m->hidden;
    const int QD = m->n_heads * m->head_dim, QG = 2 * QD;
    for (int r = 0; r < S; ++r) {
        if (tokens[r] < 0 || tokens[r] >= m->vocab ||
            !q35_row_to_device(m, &m->token_embd, (uint64_t)tokens[r],
                s->dvx + (size_t)r * H, s->x, H, err, cap))
            return q35_errf(err, cap, "Q35 CUDA batch embedding decode failed at row %d", r);
    }

    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    const size_t conv_layer_stride = (size_t)m->conv_dim * m->conv_kernel;
    const size_t state_layer_stride = (size_t)m->n_value_heads * m->state_size * m->value_head_dim;

    for (int l = 0; l < m->n_layers; ++l) {
        Q35Layer *L = &m->layers[l];
        const float *w = q35_f32_device(m, &L->attn_norm, H, err, cap);
        if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dvnorm, s->dvx, w, S, H, m->eps)) return 0;

        if (L->recurrent) {
            if (!q35_mm_device_s(m, s->dvqkv, s->dvnorm, &L->qkv, S, err, cap) ||
                !q35_mm_device_s(m, s->dvz, s->dvnorm, &L->z, S, err, cap) ||
                !q35_mm_device_s(m, s->dvbeta, s->dvnorm, &L->beta, S, err, cap) ||
                !q35_mm_device_s(m, s->dvalpha, s->dvnorm, &L->alpha, S, err, cap)) return 0;
            const float *conv_w = (const float *)coli_tensor_device_data(&L->conv1d);
            const float *dt = q35_f32_device(m, &L->dt_bias, m->n_value_heads, err, cap);
            const float *a = q35_f32_device(m, &L->a, m->n_value_heads, err, cap);
            const float *nw = q35_f32_device(m, &L->ssm_norm, m->value_head_dim, err, cap);
            float *conv_state = m->conv_state_dev + (size_t)L->recurrent_index * conv_layer_stride;
            float *rec_state = m->recurrent_state_dev + (size_t)L->recurrent_index * state_layer_stride;
            if (!conv_w || !dt || !a || !nw) return 0;
            for (int r = 0; r < S; ++r) {
                if (!coli_cuda_pipe_gated_delta_decode_separate(dev,
                    s->dvdelta + (size_t)r * m->value_dim,
                    s->dvqkv + (size_t)r * m->conv_dim,
                    s->dvz + (size_t)r * m->value_dim,
                    s->dvbeta + (size_t)r * m->n_value_heads,
                    s->dvalpha + (size_t)r * m->n_value_heads,
                    conv_w, dt, a, nw, conv_state, rec_state,
                    m->n_key_heads, m->n_value_heads, m->state_size,
                    m->conv_kernel, m->eps)) return 0;
            }
            if (!q35_mm_device_s(m, s->dvmixer, s->dvdelta, &L->ssm_out,
                                  S, err, cap)) return 0;
        } else {
            if (!q35_mm_device_s(m, s->dvqg, s->dvnorm, &L->q, S, err, cap) ||
                !q35_mm_device_s(m, s->dvk, s->dvnorm, &L->k, S, err, cap) ||
                !q35_mm_device_s(m, s->dvv, s->dvnorm, &L->v, S, err, cap)) return 0;
            for (int r = 0; r < S; ++r)
                if (!coli_cuda_pipe_qg_split(dev,
                    s->dvattn_q + (size_t)r * QD,
                    s->dvattn_gate + (size_t)r * QD,
                    s->dvqg + (size_t)r * QG,
                    m->n_heads, m->head_dim)) return 0;
            w = q35_f32_device(m, &L->q_norm, m->head_dim, err, cap);
            if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dvattn_q, s->dvattn_q, w,
                                               S * m->n_heads, m->head_dim, m->eps)) return 0;
            w = q35_f32_device(m, &L->k_norm, m->head_dim, err, cap);
            if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dvk, s->dvk, w,
                                               S * m->n_kv_heads, m->head_dim, m->eps)) return 0;
            float *kc = m->k_cache_dev + (size_t)L->attention_index * kv_layer_stride;
            float *vc = m->v_cache_dev + (size_t)L->attention_index * kv_layer_stride;
            for (int r = 0; r < S; ++r) {
                const int pos = pos0 + r;
                float *qr = s->dvattn_q + (size_t)r * QD;
                float *kr = s->dvk + (size_t)r * m->kv_dim;
                if (!coli_cuda_pipe_rope_neox(dev, qr, pos, m->n_heads,
                        m->head_dim, m->rope_dims, m->rope_base) ||
                    !coli_cuda_pipe_rope_neox(dev, kr, pos, m->n_kv_heads,
                        m->head_dim, m->rope_dims, m->rope_base) ||
                    !coli_cuda_pipe_gqa_decode(dev,
                        s->dvattn + (size_t)r * QD, qr, kr,
                        s->dvv + (size_t)r * m->kv_dim,
                        kc, vc, s->dscores, pos, m->context_capacity,
                        m->n_heads, m->n_kv_heads, m->head_dim, m->attention_scale) ||
                    !coli_cuda_pipe_sigmoid_mul(dev,
                        s->dvattn + (size_t)r * QD,
                        s->dvattn_gate + (size_t)r * QD, (size_t)QD)) return 0;
            }
            if (!q35_mm_device_s(m, s->dvmixer, s->dvattn, &L->o, S, err, cap)) return 0;
        }

        if (!coli_cuda_pipe_add(dev, s->dvx, s->dvmixer, (size_t)S * H)) return 0;
        w = q35_f32_device(m, &L->post_norm, H, err, cap);
        if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dvpost, s->dvx, w, S, H, m->eps) ||
            !q35_moe_batch_cuda(m, s, L, l, s->dvpost, s->dvmoe, S, err, cap) ||
            !coli_cuda_pipe_add(dev, s->dvx, s->dvmoe, (size_t)S * H)) return 0;
    }

    const float *w = q35_f32_device(m, &m->output_norm, H, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dvnorm, s->dvx, w, S, H, m->eps) ||
        !coli_cuda_pipe_copy(dev, s->dhidden_norm,
            s->dvnorm + (size_t)(S - 1) * H, (size_t)H * sizeof(float))) return 0;
    ColiTensor *outw = m->tied_output ? &m->token_embd : &m->output;
    if (!q35_mm_device_s(m, s->dvlogits, s->dvnorm, outw, S, err, cap) ||
        !coli_cuda_pipe_download(dev, s->dvlogits, s->vlogits,
                                 (size_t)S * m->vocab * sizeof(float))) return 0;
    return 1;
}

static int q35_forward_cuda(Q35Model *m, Q35Scratch *s, int token, int pos,
                             char *err, size_t cap) {
    const int dev = m->exec.device;
    if (token < 0 || token >= m->vocab || pos < 0 || pos >= m->context_capacity)
        return q35_errf(err, cap, "invalid token/position");
    if (!q35_row_to_device(m, &m->token_embd, (uint64_t)token,
                           s->dx, s->x, m->hidden, err, cap))
        return q35_errf(err, cap, "Q35 CUDA embedding decode failed");

    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    const size_t conv_layer_stride = (size_t)m->conv_dim * m->conv_kernel;
    const size_t state_layer_stride = (size_t)m->n_value_heads * m->state_size * m->value_head_dim;

    for (int l = 0; l < m->n_layers; ++l) {
        Q35Layer *L = &m->layers[l];
        const float *w = q35_f32_device(m, &L->attn_norm, m->hidden, err, cap);
        if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dnorm, s->dx, w, 1, m->hidden, m->eps))
            return q35_errf(err, cap, "Q35 CUDA input norm failed at layer %d", l);

        if (L->recurrent) {
            if (!q35_mm_device(m, s->dqkv, s->dnorm, &L->qkv, err, cap) ||
                !q35_mm_device(m, s->dz, s->dnorm, &L->z, err, cap) ||
                !q35_mm_device(m, s->dbeta, s->dnorm, &L->beta, err, cap) ||
                !q35_mm_device(m, s->dalpha, s->dnorm, &L->alpha, err, cap)) return 0;
            const float *conv_w = (const float *)coli_tensor_device_data(&L->conv1d);
            const float *dt = q35_f32_device(m, &L->dt_bias, m->n_value_heads, err, cap);
            const float *a = q35_f32_device(m, &L->a, m->n_value_heads, err, cap);
            const float *nw = q35_f32_device(m, &L->ssm_norm, m->value_head_dim, err, cap);
            float *conv_state = m->conv_state_dev + (size_t)L->recurrent_index * conv_layer_stride;
            float *rec_state = m->recurrent_state_dev + (size_t)L->recurrent_index * state_layer_stride;
            if (!conv_w || !dt || !a || !nw ||
                !coli_cuda_pipe_gated_delta_decode_separate(dev, s->ddelta, s->dqkv, s->dz,
                    s->dbeta, s->dalpha, conv_w, dt, a, nw, conv_state, rec_state,
                    m->n_key_heads, m->n_value_heads, m->state_size,
                    m->conv_kernel, m->eps) ||
                !q35_mm_device(m, s->dmixer, s->ddelta, &L->ssm_out, err, cap))
                return q35_errf(err, cap, "Q35 CUDA DeltaNet failed at layer %d", l);
        } else {
            if (!q35_mm_device(m, s->dqg, s->dnorm, &L->q, err, cap) ||
                !q35_mm_device(m, s->dk, s->dnorm, &L->k, err, cap) ||
                !q35_mm_device(m, s->dv, s->dnorm, &L->v, err, cap)) return 0;
            if (!coli_cuda_pipe_qg_split(dev, s->dattn_q, s->dattn_gate, s->dqg,
                                          m->n_heads, m->head_dim)) return 0;
            w = q35_f32_device(m, &L->q_norm, m->head_dim, err, cap);
            if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dattn_q, s->dattn_q, w, m->n_heads, m->head_dim, m->eps)) return 0;
            w = q35_f32_device(m, &L->k_norm, m->head_dim, err, cap);
            if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dk, s->dk, w, m->n_kv_heads, m->head_dim, m->eps)) return 0;
            if (!coli_cuda_pipe_rope_neox(dev, s->dattn_q, pos, m->n_heads, m->head_dim, m->rope_dims, m->rope_base) ||
                !coli_cuda_pipe_rope_neox(dev, s->dk, pos, m->n_kv_heads, m->head_dim, m->rope_dims, m->rope_base)) return 0;
            float *kc = m->k_cache_dev + (size_t)L->attention_index * kv_layer_stride;
            float *vc = m->v_cache_dev + (size_t)L->attention_index * kv_layer_stride;
            if (!coli_cuda_pipe_gqa_decode(dev, s->dattn, s->dattn_q, s->dk, s->dv,
                    kc, vc, s->dscores, pos, m->context_capacity,
                    m->n_heads, m->n_kv_heads, m->head_dim, m->attention_scale) ||
                !coli_cuda_pipe_sigmoid_mul(dev, s->dattn, s->dattn_gate,
                    (size_t)m->n_heads * m->head_dim) ||
                !q35_mm_device(m, s->dmixer, s->dattn, &L->o, err, cap)) return 0;
        }

        if (!coli_cuda_pipe_add(dev, s->dx, s->dmixer, (size_t)m->hidden)) return 0;
        w = q35_f32_device(m, &L->post_norm, m->hidden, err, cap);
        if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dpost, s->dx, w, 1, m->hidden, m->eps)) return 0;

        if (!q35_moe_cuda(m, s, L, l, 1, err, cap) ||
            !coli_cuda_pipe_add(dev, s->dx, s->dmoe, (size_t)m->hidden)) return 0;
    }

    const float *w = q35_f32_device(m, &m->output_norm, m->hidden, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dnorm, s->dx, w, 1, m->hidden, m->eps) ||
        !coli_cuda_pipe_zero(dev, s->dhidden_norm, (size_t)m->hidden) ||
        !coli_cuda_pipe_axpy(dev, s->dhidden_norm, s->dnorm, 1.0f, (size_t)m->hidden)) return 0;
    ColiTensor *outw = m->tied_output ? &m->token_embd : &m->output;
    if (!q35_mm_device(m, s->dlogits, s->dnorm, outw, err, cap) ||
        !coli_cuda_pipe_download(dev, s->dlogits, s->logits, (size_t)m->vocab * sizeof(float))) return 0;
    return 1;
}

static int q35_mtp_forward_cuda(Q35Model *m, Q35Scratch *s, int token,
                                 const float *hidden_dev, int pos, int need_logits,
                                 char *err, size_t cap) {
    if (!m->n_mtp_layers || m->mtp_layer < 0)
        return q35_errf(err, cap, "Q35 MTP head is not present");
    if (token < 0 || token >= m->vocab || pos < 0 || pos >= m->context_capacity)
        return q35_errf(err, cap, "invalid Q35 MTP token/position");
    const int dev = m->exec.device;
    Q35Layer *L = &m->layers[m->mtp_layer];
    ColiTensor *embed = L->has_nextn_embed ? &L->nextn_embed_tokens : &m->token_embd;
    if (!q35_row_to_device(m, embed, (uint64_t)token,
                           s->dmtp_x, s->mtp_x, m->hidden, err, cap))
        return q35_errf(err, cap, "Q35 CUDA MTP embedding decode failed");

    const float *w = q35_f32_device(m, &L->nextn_enorm, m->hidden, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dnorm, s->dmtp_x, w, 1, m->hidden, m->eps)) return 0;
    w = q35_f32_device(m, &L->nextn_hnorm, m->hidden, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dpost, hidden_dev, w, 1, m->hidden, m->eps)) return 0;
    if (!coli_cuda_pipe_zero(dev, s->dmtp_cat, (size_t)2 * m->hidden) ||
        !coli_cuda_pipe_axpy(dev, s->dmtp_cat, s->dnorm, 1.0f, (size_t)m->hidden) ||
        !coli_cuda_pipe_axpy(dev, s->dmtp_cat + m->hidden, s->dpost, 1.0f, (size_t)m->hidden) ||
        !q35_mm_device(m, s->dmtp_x, s->dmtp_cat, &L->nextn_eh_proj, err, cap)) return 0;

    w = q35_f32_device(m, &L->attn_norm, m->hidden, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dnorm, s->dmtp_x, w, 1, m->hidden, m->eps)) return 0;

    /* State-only synchronization stops after writing the MTP K/V rows.  This
     * avoids Q/gate, attention, routed/shared experts, the 248k-token output
     * head, and the device-to-host logits transfer. */
    if (!q35_mm_device(m, s->dk, s->dnorm, &L->k, err, cap) ||
        !q35_mm_device(m, s->dv, s->dnorm, &L->v, err, cap)) return 0;
    w = q35_f32_device(m, &L->k_norm, m->head_dim, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dk, s->dk, w,
                                       m->n_kv_heads, m->head_dim, m->eps) ||
        !coli_cuda_pipe_rope_neox(dev, s->dk, pos, m->n_kv_heads,
                                  m->head_dim, m->rope_dims, m->rope_base)) return 0;
    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    float *kc = m->k_cache_dev + (size_t)L->attention_index * kv_layer_stride;
    float *vc = m->v_cache_dev + (size_t)L->attention_index * kv_layer_stride;
    if (!need_logits) {
        if (!coli_cuda_pipe_copy(dev, kc + (size_t)pos * m->kv_dim, s->dk,
                                  (size_t)m->kv_dim * sizeof(float)) ||
            !coli_cuda_pipe_copy(dev, vc + (size_t)pos * m->kv_dim, s->dv,
                                  (size_t)m->kv_dim * sizeof(float))) return 0;
        ++m->mtp_kv_updates;
        return 1;
    }

    if (!q35_mm_device(m, s->dqg, s->dnorm, &L->q, err, cap) ||
        !coli_cuda_pipe_qg_split(dev, s->dattn_q, s->dattn_gate, s->dqg,
                                  m->n_heads, m->head_dim)) return 0;
    w = q35_f32_device(m, &L->q_norm, m->head_dim, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dattn_q, s->dattn_q, w,
                                       m->n_heads, m->head_dim, m->eps) ||
        !coli_cuda_pipe_rope_neox(dev, s->dattn_q, pos, m->n_heads,
                                  m->head_dim, m->rope_dims, m->rope_base) ||
        !coli_cuda_pipe_gqa_decode(dev, s->dattn, s->dattn_q, s->dk, s->dv,
            kc, vc, s->dscores, pos, m->context_capacity,
            m->n_heads, m->n_kv_heads, m->head_dim, m->attention_scale) ||
        !coli_cuda_pipe_sigmoid_mul(dev, s->dattn, s->dattn_gate,
                                     (size_t)m->n_heads * m->head_dim) ||
        !q35_mm_device(m, s->dmixer, s->dattn, &L->o, err, cap) ||
        !coli_cuda_pipe_add(dev, s->dmtp_x, s->dmixer, (size_t)m->hidden)) return 0;

    w = q35_f32_device(m, &L->post_norm, m->hidden, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dpost, s->dmtp_x, w, 1, m->hidden, m->eps) ||
        !q35_moe_cuda(m, s, L, m->mtp_layer, 0, err, cap) ||
        !coli_cuda_pipe_add(dev, s->dmtp_x, s->dmoe, (size_t)m->hidden)) return 0;

    ColiTensor *head_norm = L->has_nextn_head_norm ? &L->nextn_head_norm : &m->output_norm;
    w = q35_f32_device(m, head_norm, m->hidden, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dmtp_hidden, s->dmtp_x, w, 1, m->hidden, m->eps)) return 0;
    ColiTensor *head = L->has_nextn_head ? &L->nextn_head :
                       (m->tied_output ? &m->token_embd : &m->output);
    if (!q35_mm_device(m, s->dmtp_logits, s->dmtp_hidden, head, err, cap) ||
        !coli_cuda_pipe_download(dev, s->dmtp_logits, s->mtp_logits,
                                 (size_t)m->vocab * sizeof(float))) return 0;
    ++m->mtp_steps;
    return 1;
}

#endif

static size_t q35_conv_state_elems(const Q35Model *m) {
    return (size_t)m->n_recurrent * m->conv_dim * m->conv_kernel;
}
static size_t q35_recurrent_state_elems(const Q35Model *m) {
    return (size_t)m->n_recurrent * m->n_value_heads * m->state_size * m->value_head_dim;
}

static int q35_checkpoint_target_state(Q35Model *m, Q35Scratch *s,
                                        char *err, size_t cap) {
    const size_t nc = q35_conv_state_elems(m), ns = q35_recurrent_state_elems(m);
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        if ((nc && !coli_cuda_pipe_copy(m->exec.device, s->dconv_backup,
                                        m->conv_state_dev, nc * sizeof(float))) ||
            (ns && !coli_cuda_pipe_copy(m->exec.device, s->dstate_backup,
                                        m->recurrent_state_dev, ns * sizeof(float))) ||
            !coli_cuda_pipe_copy(m->exec.device, s->dhidden_backup,
                                 s->dhidden_norm, (size_t)m->hidden * sizeof(float)))
            return q35_errf(err, cap, "cannot checkpoint Q35 CUDA verification state");
        return 1;
    }
#endif
    if (nc) memcpy(s->conv_backup, m->conv_state, nc * sizeof(float));
    if (ns) memcpy(s->state_backup, m->recurrent_state, ns * sizeof(float));
    memcpy(s->hidden_backup, s->hidden_norm, (size_t)m->hidden * sizeof(float));
    return 1;
}

static int q35_restore_target_state(Q35Model *m, Q35Scratch *s,
                                     char *err, size_t cap) {
    const size_t nc = q35_conv_state_elems(m), ns = q35_recurrent_state_elems(m);
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        if ((nc && !coli_cuda_pipe_copy(m->exec.device, m->conv_state_dev,
                                        s->dconv_backup, nc * sizeof(float))) ||
            (ns && !coli_cuda_pipe_copy(m->exec.device, m->recurrent_state_dev,
                                        s->dstate_backup, ns * sizeof(float))) ||
            !coli_cuda_pipe_copy(m->exec.device, s->dhidden_norm,
                                 s->dhidden_backup, (size_t)m->hidden * sizeof(float)))
            return q35_errf(err, cap, "cannot restore Q35 CUDA verification state");
        return 1;
    }
#endif
    if (nc) memcpy(m->conv_state, s->conv_backup, nc * sizeof(float));
    if (ns) memcpy(m->recurrent_state, s->state_backup, ns * sizeof(float));
    memcpy(s->hidden_norm, s->hidden_backup, (size_t)m->hidden * sizeof(float));
    return 1;
}

static int q35_forward_batch(Q35Model *m, Q35Scratch *s,
                             const int *tokens, int S, int pos0,
                             char *err, size_t cap) {
    int ok;
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA)
        ok = q35_forward_batch_cuda(m, s, tokens, S, pos0, err, cap);
    else
#endif
        ok = q35_forward_batch_cpu(m, s, tokens, S, pos0, err, cap);
    if (ok) q35_repin(m);
    return ok;
}

/* Refresh one MTP K/V row from a verified target hidden state without
 * executing the rest of the appended decoder block. */
static int q35_mtp_update_kv_hidden(Q35Model *m, Q35Scratch *s, int token,
                                    const float *hidden, int pos,
                                    char *err, size_t cap) {
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA)
        return q35_mtp_forward_cuda(m, s, token, hidden, pos, 0, err, cap);
#endif
    return q35_mtp_forward_cpu(m, s, token, hidden, pos, 0, err, cap);
}

/* Synchronize the current MTP position with the target hidden row that is
 * already held by the ordinary decoder. */
static int q35_mtp_update_kv(Q35Model *m, Q35Scratch *s, int token, int pos,
                             char *err, size_t cap) {
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA)
        return q35_mtp_update_kv_hidden(m, s, token, s->dhidden_norm,
                                        pos, err, cap);
#endif
    return q35_mtp_update_kv_hidden(m, s, token, s->hidden_norm,
                                    pos, err, cap);
}

/* Refresh the appended MTP decoder KV with verified target hidden rows.
 * Position zero of every speculative block was already written correctly by
 * the first draft call: it used the real target h[p-1], not a speculative MTP
 * hidden row.  Only rows 1..S-1 need replacement. */
static int q35_mtp_catchup_batch(Q35Model *m, Q35Scratch *s,
                                 const int *tokens, int S, int pos0,
                                 char *err, size_t cap) {
    if (!m->mtp_enabled) return 1;
    for (int r = 1; r < S; ++r) {
#ifdef COLI_CUDA
        if (m->exec.kind == COLI_BACKEND_CUDA) {
            const float *h = s->dvnorm + (size_t)(r - 1) * m->hidden;
            if (!q35_mtp_update_kv_hidden(m, s, tokens[r], h,
                                          pos0 + r, err, cap)) return 0;
        } else
#endif
        {
            const float *h = s->vnorm + (size_t)(r - 1) * m->hidden;
            if (!q35_mtp_update_kv_hidden(m, s, tokens[r], h,
                                          pos0 + r, err, cap)) return 0;
        }
    }
    return 1;
}

static int q35_argmax(const float *x, int n);
static int q35_argmax_repeat(const Q35Model *m, const float *x, int n,
                              const unsigned char *seen,
                              const int *extra, int extra_count,
                              float penalty);
static int q35_forward(Q35Model *m, Q35Scratch *s, int token, int pos,
                        char *err, size_t cap) {
    int ok;
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) ok = q35_forward_cuda(m, s, token, pos, err, cap);
    else
#endif
    ok = q35_forward_cpu(m, s, token, pos, err, cap);
    if (ok) q35_repin(m);
    return ok;
}


static int q35_mtp_forward(Q35Model *m, Q35Scratch *s, int token,
                            int pos, int chained, char *err, size_t cap) {
    int ok;
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        const float *h = chained ? s->dmtp_hidden : s->dhidden_norm;
        ok = q35_mtp_forward_cuda(m, s, token, h, pos, 1, err, cap);
    } else
#endif
    {
        const float *h = chained ? s->mtp_hidden : s->hidden_norm;
        ok = q35_mtp_forward_cpu(m, s, token, h, pos, 1, err, cap);
    }
    if (ok) q35_repin(m);
    return ok;
}

static int q35_mtp_make_drafts(Q35Model *m, Q35Scratch *s, int seed_token,
                                int base_pos, int max_drafts, int *drafts,
                                const unsigned char *repeat_seen,
                                float repeat_penalty,
                                char *err, size_t cap) {
    if (!m->mtp_enabled || max_drafts <= 0) return 0;
    int token = seed_token;
    int n = 0;
    for (; n < max_drafts; ++n) {
        if (!q35_mtp_forward(m, s, token, base_pos + n, n > 0, err, cap)) return -1;
        token = q35_argmax_repeat(m, s->mtp_logits, m->vocab,
                                  repeat_seen, drafts, n, repeat_penalty);
        drafts[n] = token;
        ++m->mtp_proposed;
        if (token == coli_gguf_tokenizer_eos(m->tokenizer)) { ++n; break; }
    }
    return n;
}

static int q35_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

static int q35_argmax_repeat(const Q35Model *m, const float *x, int n,
                              const unsigned char *seen,
                              const int *extra, int extra_count,
                              float penalty) {
    if (!x || n <= 0) return 0;
    if (!seen || penalty <= 1.0f) return q35_argmax(x, n);
    int best = 0;
    float best_score = -INFINITY;
    for (int token = 0; token < n; ++token) {
        int repeated = seen[token] != 0;
        if (!repeated && extra) {
            for (int j = 0; j < extra_count; ++j) {
                if (extra[j] == token) { repeated = 1; break; }
            }
        }
        if (repeated && m && m->tokenizer &&
            coli_gguf_tokenizer_is_control(m->tokenizer, token))
            repeated = 0;
        float score = x[token];
        if (repeated) score = score <= 0.0f ? score * penalty : score / penalty;
        if (score > best_score) { best_score = score; best = token; }
    }
    return best;
}

static void q35_repeat_record(const Q35Model *m, unsigned char *seen, int token) {
    if (!seen || !m || token < 0 || token >= m->vocab) return;
    if (!coli_gguf_tokenizer_is_control(m->tokenizer, token)) seen[token] = 1;
}

static int q35_parse_expert_zero_threshold(Q35Model *m, char *err, size_t cap) {
    const char *text = getenv("Q35_EXPERT_ZERO_THRESHOLD");
    if (!text || !*text) {
        m->expert_zero_threshold = 0.0f;
        return 1;
    }
    errno = 0;
    char *end = NULL;
    const double value = strtod(text, &end);
    if (errno || end == text || *end || !isfinite(value) || value < 0.0 || value > 1e6)
        return q35_errf(err, cap,
            "invalid Q35_EXPERT_ZERO_THRESHOLD='%s' (expected a finite number >= 0)", text);
    m->expert_zero_threshold = (float)value;
    return 1;
}

static int q35_load_model(Q35Model *m, const char *path, int context, int verbose,
                           int mtp_draft_max, ColiExec exec, char *err, size_t cap) {
    memset(m, 0, sizeof(*m));
    m->gguf.fd = -1; m->verbose = verbose; m->exec = exec;
    m->cuda_requested = exec.kind == COLI_BACKEND_CUDA;
    if (!q35_parse_expert_zero_threshold(m, err, cap)) return 0;
    if (!coli_gguf_open(&m->gguf, path))
        return q35_errf(err, cap, "cannot open model container: %s", coli_gguf_error(&m->gguf));
    coli_gguf_set_preload_backend(&m->gguf, exec.kind, exec.device);
    if (m->gguf.container_kind == COLI_MODEL_CONTAINER_SGGUF &&
        m->expert_zero_threshold > 0.0f) {
        if (verbose) fprintf(stderr,
            "[SGGUF] Q35_EXPERT_ZERO_THRESHOLD is ignored: sparse pruning is encoded in the file\n");
        m->expert_zero_threshold = 0.0f;
    }
    if (!q35_model_config(m, err, cap)) return 0;
    if (mtp_draft_max < 0 || mtp_draft_max > 8)
        return q35_errf(err, cap, "--mtp-draft must be between 0 and 8");
    if (mtp_draft_max > 0 && m->n_mtp_layers == 0)
        return q35_errf(err, cap, "MTP requested but this GGUF has no nextn_predict_layers");
    m->mtp_draft_max = mtp_draft_max;
    m->mtp_enabled = mtp_draft_max > 0 && m->n_mtp_layers > 0;
    m->n_exec_layers = m->n_layers + (m->mtp_enabled ? m->n_mtp_layers : 0);
    m->n_attention_exec = m->n_attention + (m->mtp_enabled ? m->n_mtp_layers : 0);
    if (!coli_gguf_tokenizer_load(&m->tokenizer, &m->gguf, err, cap)) return 0;
    if (coli_gguf_tokenizer_vocab_size(m->tokenizer) != m->vocab)
        return q35_errf(err, cap, "tokenizer/model vocabulary mismatch");

    const double t0 = q35_now_sec();
    if (!q35_model_load_weights(m, err, cap) || !q35_build_expert_views(m, err, cap)) return 0;
    const double mapped = q35_now_sec() - t0;
    const double t1 = q35_now_sec();

    /* Mandatory runtime state has priority over every weight cache.  If CUDA
     * cannot hold it, transparently continue with mmap-backed CPU inference. */
    if (!q35_alloc_state(m, context, err, cap)) {
        char why[512];
        snprintf(why, sizeof(why), "%s", err && *err ? err : "CUDA state allocation failed");
        if (!m->cuda_requested || !q35_downgrade_to_cpu(m, context, why, err, cap)) return 0;
    }

    /* Dense residency is also best-effort.  A failure releases partial CUDA
     * allocations and restarts on CPU rather than aborting model loading. */
    if (m->exec.kind == COLI_BACKEND_CUDA && !q35_model_reside_cuda(m, err, cap)) {
        char why[512];
        snprintf(why, sizeof(why), "%s", err && *err ? err : "CUDA dense residency failed");
        if (!q35_downgrade_to_cpu(m, context, why, err, cap)) return 0;
    }

    const int64_t history = q35_usage_load(m, path);
    if (history > 0 && verbose)
        fprintf(stderr, "[USAGE] Q35 expert history: %lld selections\n", (long long)history);
    if (!q35_scheduler_init(m, err, cap)) return 0;

    if (verbose && m->expert_zero_threshold > 0.0f) {
        fprintf(stderr,
            "[Q35-PRUNE] enabled: abs(weight) < %.9g becomes exact zero for routed experts only; set Q35_EXPERT_ZERO_THRESHOLD=0 to disable\n",
            m->expert_zero_threshold);
    }

    if (verbose) {
        const char *container_name =
            m->gguf.container_kind == COLI_MODEL_CONTAINER_SGGUF ? "SGGUF" : "GGUF";
        fprintf(stderr,
            "[%s] qwen35moe: trunk=%d (%d DeltaNet + %d attention) mtp=%d/%s hidden=%d heads=%d/%d experts=%d top=%d vocab=%d\n",
            container_name, m->n_layers, m->n_recurrent, m->n_attention, m->n_mtp_layers,
            m->mtp_enabled ? "enabled" : "off", m->hidden,
            m->n_heads, m->n_kv_heads, m->n_experts, m->n_expert_used, m->vocab);
        if (m->exec.kind == COLI_BACKEND_CUDA)
            fprintf(stderr,
                "[%s] mapped %.2fs; dense residency %.2fs, %.2f MiB before expert cache; backend=cuda (Qwen3.5-MoE + Colibri scheduler); context=%d; token_embedding=row-streamed\n",
                container_name, mapped, q35_now_sec() - t1, m->cuda_dense_bytes / (1024.0 * 1024.0), context);
        else
            fprintf(stderr, "[%s] mapped %.2fs; backend=cpu%s; context=%d; threads=%d\n",
                    container_name, mapped, m->cuda_fell_back_to_cpu ? " (automatic CUDA VRAM fallback)" : " reference",
                    context, omp_get_max_threads());
    }
    return 1;
}


static int q35_emit_token(Q35Model *m, int token) {
    if (token == coli_gguf_tokenizer_eos(m->tokenizer)) return 0;
    if (!coli_gguf_tokenizer_is_control(m->tokenizer, token)) {
        char piece[4096];
        const int n = coli_gguf_tokenizer_decode(m->tokenizer, &token, 1, piece, sizeof(piece));
        if (n > 0) { fwrite(piece, 1, (size_t)n, stdout); fflush(stdout); }
    }
    return 1;
}

static void q35_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--gguf] MODEL.gguf|MODEL.sgguf --prompt TEXT [--max-tokens N] [--device cpu|cuda[:N]] [--repeat-penalty N] [--mtp-draft 0..8] [--no-mtp] [--raw-prompt] [--verbose]\n",
        prog);
}

int coli_qwen35moe_run_cli(int argc, char **argv) {
    const char *model_path = NULL, *prompt = NULL, *device_arg = "cpu";
    int max_tokens = 24, raw = 0, verbose = 0;
    float repeat_penalty = 1.0f;
    int mtp_draft_max = 0;
    const char *mtp_env = getenv("QWEN35_MTP_DRAFT");
    if (mtp_env && *mtp_env) mtp_draft_max = atoi(mtp_env);
    int i = 1;
    if (i < argc && !strcmp(argv[i], "--gguf")) ++i;
    if (i < argc) model_path = argv[i++];
    for (; i < argc; ++i) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device_arg = argv[++i];
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) repeat_penalty = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--mtp-draft") && i + 1 < argc) mtp_draft_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-mtp")) mtp_draft_max = 0;
        else if (!strcmp(argv[i], "--raw-prompt")) raw = 1;
        else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown Q35 GGUF option: %s\n", argv[i]); q35_usage(argv[0]); return 2; }
    }
    if (!model_path || !prompt || max_tokens < 0 ||
        !isfinite(repeat_penalty) || repeat_penalty < 1.0f ||
        mtp_draft_max < 0 || mtp_draft_max > 8) {
        q35_usage(argv[0]); return 2;
    }

    ColiGgufFile probe; probe.fd = -1;
    ColiGgufTokenizer *pt = NULL;
    char err[512];
    if (!coli_gguf_open(&probe, model_path) ||
        !coli_gguf_tokenizer_load(&pt, &probe, err, sizeof(err))) {
        fprintf(stderr, "%s\n", probe.fd >= 0 ? err : coli_gguf_error(&probe));
        coli_gguf_close(&probe); return 1;
    }
    char *formatted = NULL;
    if (raw) {
        const size_t pn = strlen(prompt) + 1;
        formatted = (char *)malloc(pn);
        if (formatted) memcpy(formatted, prompt, pn);
    } else {
        formatted = coli_gguf_tokenizer_format_qwen3next_prompt(pt, prompt);
    }
    if (!formatted) {
        fprintf(stderr, "cannot format Q35 prompt\n");
        coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); return 1;
    }
    const size_t max_prompt = strlen(formatted) + 2;
    if (max_prompt > INT_MAX) { free(formatted); coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); return 1; }
    int *ids = (int *)malloc(max_prompt * sizeof(int));
    if (!ids) { free(formatted); coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); return 1; }
    const int n_prompt = coli_gguf_tokenizer_encode(pt, formatted, ids, (int)max_prompt);
    coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); free(formatted);
    if (n_prompt <= 0 || max_tokens > INT_MAX - n_prompt) {
        fprintf(stderr, "Q35 prompt tokenization/context failed\n"); free(ids); return 1;
    }

    ColiExec exec = { COLI_BACKEND_CPU, 0 };
    int cuda_started = 0; (void)cuda_started;
    if (!strcmp(device_arg, "cuda") || !strncmp(device_arg, "cuda:", 5)) {
        exec.kind = COLI_BACKEND_CUDA;
        if (device_arg[4] == ':') exec.device = atoi(device_arg + 5);
#ifdef COLI_CUDA
        if (!coli_cuda_init(&exec.device, 1)) {
            fprintf(stderr, "cannot initialize CUDA device %d\n", exec.device); free(ids); return 1;
        }
        cuda_started = 1;
#else
        fprintf(stderr, "this binary was built without CUDA support\n"); free(ids); return 1;
#endif
    } else if (strcmp(device_arg, "cpu")) {
        fprintf(stderr, "invalid --device: %s\n", device_arg); free(ids); return 2;
    }

    Q35Model m;
    Q35Scratch s;
    memset(&s, 0, sizeof(s));
    if (!q35_load_model(&m, model_path, n_prompt + max_tokens, verbose, mtp_draft_max, exec, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); q35_model_free(&m); free(ids);
#ifdef COLI_CUDA
        if (cuda_started) coli_cuda_shutdown();
#endif
        return 1;
    }
    if (!q35_scratch_alloc(&m, &s, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); q35_model_free(&m); free(ids);
#ifdef COLI_CUDA
        if (cuda_started) coli_cuda_shutdown();
#endif
        return 1;
    }
    unsigned char *repeat_seen = (unsigned char *)calloc((size_t)m.vocab, 1);
    if (!repeat_seen) {
        fprintf(stderr, "out of memory allocating Q35 repeat-penalty history\n");
        q35_scratch_free(&s); q35_model_free(&m); free(ids);
#ifdef COLI_CUDA
        if (cuda_started) coli_cuda_shutdown();
#endif
        return 1;
    }
    if (verbose && repeat_penalty > 1.0f)
        fprintf(stderr, "[Q35] repeat penalty=%.3f over generated text tokens\n", repeat_penalty);

    const double t0 = q35_now_sec();
    for (int p = 0; p < n_prompt; ++p) {
        /* Qwen MTP convention (matching llama.cpp): decoder position p gets
         * token[p] and the target post-norm hidden row h[p-1]. Position zero
         * uses an all-zero pending row. Run this before the trunk consumes the
         * current token, while hidden_norm still contains h[p-1]. */
        if (m.mtp_enabled &&
            !q35_mtp_update_kv(&m, &s, ids[p], p, err, sizeof(err))) {
            if (verbose)
                fprintf(stderr, "[MTP] prompt prefill failed at position %d (%s); disabling MTP\n",
                        p, err);
            m.mtp_enabled = 0;
        }
        if (!q35_forward(&m, &s, ids[p], p, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err); goto fail;
        }
        /* Echo the exact formatted model input one token at a time after
         * each prefill step, matching the GPT-OSS runtime.  Decode control
         * tokens too so the displayed text is the actual tokenizer input,
         * not only the user-supplied --prompt string. */
        {
            char input_piece[4096];
            const int input_n = coli_gguf_tokenizer_decode(
                m.tokenizer, &ids[p], 1, input_piece, (int)sizeof(input_piece));
            if (input_n > 0) {
                fwrite(input_piece, 1, (size_t)input_n, stdout);
                fflush(stdout);
            }
        }
    }
    int pos = n_prompt, generated = 0;
    int pending = -1;
    int stop_generation = 0;
    int drafts[8];
    while (generated < max_tokens && !stop_generation) {
        int current;
        int already_emitted = 0;
        if (pending >= 0) {
            current = pending;
            pending = -1;
            already_emitted = 1;
        } else {
            current = q35_argmax_repeat(&m, s.logits, m.vocab,
                                        repeat_seen, NULL, 0, repeat_penalty);
        }
        if (!already_emitted) {
            if (!q35_emit_token(&m, current)) break;
            q35_repeat_record(&m, repeat_seen, current);
            ++generated;
            if (generated >= max_tokens) break;
        }

        if (!m.mtp_enabled) {
            if (!q35_forward(&m, &s, current, pos++, err, sizeof(err))) {
                fprintf(stderr, "\n%s\n", err); goto fail;
            }
            continue;
        }

        int want = m.mtp_draft_max;
        if (want > max_tokens - generated) want = max_tokens - generated;
        int nd = q35_mtp_make_drafts(&m, &s, current, pos, want,
                                     drafts, repeat_seen, repeat_penalty,
                                     err, sizeof(err));
        if (nd < 0) {
            if (verbose) fprintf(stderr, "\n[MTP] draft failed (%s); disabling MTP and continuing normally\n", err);
            m.mtp_enabled = 0;
            if (!q35_forward(&m, &s, current, pos++, err, sizeof(err))) {
                fprintf(stderr, "\n%s\n", err); goto fail;
            }
            continue;
        }

        int batch_done = 0;
        if (nd >= 1 && s.verify_cap >= nd + 1) {
            int verify_inputs[9];
            const int verify_n = nd + 1;
            verify_inputs[0] = current;
            for (int r = 1; r < verify_n; ++r) verify_inputs[r] = drafts[r - 1];
            if (q35_checkpoint_target_state(&m, &s, err, sizeof(err))) {
                if (q35_forward_batch(&m, &s, verify_inputs, verify_n, pos, err, sizeof(err))) {
                    ++m.mtp_verify_batches;
                    m.mtp_verify_tokens += (uint64_t)verify_n;
                    int mismatch = -1, mismatch_actual = -1;
                    for (int j = 0; j < nd; ++j) {
                        const int actual = q35_argmax_repeat(
                            &m, s.vlogits + (size_t)j * m.vocab, m.vocab,
                            repeat_seen, NULL, 0, repeat_penalty);
                        if (actual == coli_gguf_tokenizer_eos(m.tokenizer)) {
                            stop_generation = 1;
                            break;
                        }
                        if (actual == drafts[j]) ++m.mtp_accepted;
                        else { mismatch = j; mismatch_actual = actual; }
                        if (!q35_emit_token(&m, actual)) { stop_generation = 1; break; }
                        q35_repeat_record(&m, repeat_seen, actual);
                        ++generated;
                        if (generated >= max_tokens) { stop_generation = 1; break; }
                        if (mismatch >= 0) break;
                    }
                    if (!stop_generation) {
                        if (mismatch < 0) {
                            /* Target state already contains current + drafts[0..nd-2].
                             * Refresh MTP KV with verified target hidden rows, then leave
                             * the final accepted draft emitted but pending consumption. */
                            if (!q35_mtp_catchup_batch(&m, &s, verify_inputs, verify_n, pos,
                                                       err, sizeof(err))) {
                                if (verbose) fprintf(stderr,
                                    "\n[MTP] verified-state catch-up failed (%s); disabling MTP\n", err);
                                m.mtp_enabled = 0;
                            }
                            /* Row nd predicts the bonus token after every
                             * draft was accepted.  The target has consumed
                             * current plus all nd drafts (verify_n rows). */
                            const int bonus = q35_argmax_repeat(
                                &m, s.vlogits + (size_t)nd * m.vocab, m.vocab,
                                repeat_seen, NULL, 0, repeat_penalty);
                            pos += verify_n;
                            if (bonus == coli_gguf_tokenizer_eos(m.tokenizer)) {
                                stop_generation = 1;
                            } else {
                                if (!q35_emit_token(&m, bonus)) stop_generation = 1;
                                else {
                                    q35_repeat_record(&m, repeat_seen, bonus);
                                    ++generated;
                                    if (generated >= max_tokens) stop_generation = 1;
                                    else pending = bonus;
                                }
                            }
                        } else {
                            /* The speculative target pass consumed rejected suffix rows.
                             * Restore recurrent state and replay only the exact prefix
                             * current,draft[0..mismatch-1].  Full-attention/MTP suffix
                             * cells are position-addressed and are overwritten later. */
                            if (!q35_restore_target_state(&m, &s, err, sizeof(err))) {
                                fprintf(stderr, "\n%s\n", err); goto fail;
                            }
                            for (int r = 0; r <= mismatch; ++r) {
                                /* The first speculative row used the real target
                                 * hidden state and is already synchronized. */
                                if (m.mtp_enabled && r > 0 &&
                                    !q35_mtp_update_kv(&m, &s, verify_inputs[r],
                                                       pos + r, err, sizeof(err))) {
                                    if (verbose) fprintf(stderr,
                                        "\n[MTP] replay catch-up failed (%s); disabling MTP\n", err);
                                    m.mtp_enabled = 0;
                                }
                                if (!q35_forward(&m, &s, verify_inputs[r], pos + r,
                                                 err, sizeof(err))) {
                                    fprintf(stderr, "\n%s\n", err); goto fail;
                                }
                            }
                            ++m.mtp_verify_replays;
                            pos += mismatch + 1;
                            pending = mismatch_actual;
                        }
                    }
                    batch_done = 1;
                } else {
                    char batch_err[512];
                    snprintf(batch_err, sizeof(batch_err), "%s", err);
                    if (!q35_restore_target_state(&m, &s, err, sizeof(err))) {
                        fprintf(stderr, "\n%s\n", err); goto fail;
                    }
                    if (verbose) fprintf(stderr,
                        "\n[MTP] block verification failed (%s); using sequential verification\n",
                        batch_err);
                }
            }
        }
        if (batch_done) continue;

        int verify_token = current;
        for (int j = 0; ; ++j) {
            /* Catch the MTP memory up with the verified target pair before the
             * trunk consumes this token.  This overwrites any speculative KV at
             * the same position with the target-hidden-aligned value. */
            if (m.mtp_enabled && !(j == 0 && nd > 0) &&
                !q35_mtp_update_kv(&m, &s, verify_token, pos, err, sizeof(err))) {
                if (verbose) fprintf(stderr,
                    "\n[MTP] sequential catch-up failed (%s); disabling MTP\n", err);
                m.mtp_enabled = 0;
            }
            if (!q35_forward(&m, &s, verify_token, pos++, err, sizeof(err))) {
                fprintf(stderr, "\n%s\n", err); goto fail;
            }
            const int actual = q35_argmax_repeat(&m, s.logits, m.vocab,
                                                  repeat_seen, NULL, 0,
                                                  repeat_penalty);
            if (actual == coli_gguf_tokenizer_eos(m.tokenizer)) {
                stop_generation = 1;
                break;
            }
            const int accepted = j < nd && drafts[j] == actual;
            if (accepted) ++m.mtp_accepted;
            if (!q35_emit_token(&m, actual)) { stop_generation = 1; break; }
            q35_repeat_record(&m, repeat_seen, actual);
            ++generated;
            if (generated >= max_tokens) { stop_generation = 1; break; }
            if (!accepted || j + 1 >= nd) {
                pending = actual; /* emitted, but not yet consumed by the trunk */
                break;
            }
            verify_token = actual;
        }
    }
    fputc('\n', stdout);
    if (verbose) {
        const double elapsed = q35_now_sec() - t0;
        fprintf(stderr, "[GGUF] prompt=%d generated=%d elapsed=%.2fs (%.3f tok/s)\n",
                n_prompt, generated, elapsed, (n_prompt + generated) / elapsed);
        fprintf(stderr,
            "[SCHED] hits=%llu (pin=%llu LRU=%llu) misses=%llu admissions=%llu evictions=%llu speculative=%llu/%llu\n",
            (unsigned long long)m.scheduler_stats.hits,
            (unsigned long long)m.scheduler_stats.pin_hits,
            (unsigned long long)m.scheduler_stats.cache_hits,
            (unsigned long long)m.scheduler_stats.misses,
            (unsigned long long)m.scheduler_stats.admissions,
            (unsigned long long)m.scheduler_stats.evictions,
            (unsigned long long)m.scheduler_stats.speculative_loads,
            (unsigned long long)m.scheduler_stats.speculative_drops);
        if (m.n_mtp_layers) {
            const double accept = m.mtp_proposed ? 100.0 * (double)m.mtp_accepted / (double)m.mtp_proposed : 0.0;
            fprintf(stderr,
                "[MTP] enabled=%d draft_max=%d mode=%s draft_steps=%llu kv_updates=%llu proposed=%llu accepted=%llu (%.1f%%) verify_batches=%llu verify_tokens=%llu replays=%llu\n",
                m.mtp_enabled, m.mtp_draft_max,
                m.mtp_draft_max > 0 ? "block-verify" : "off",
                (unsigned long long)m.mtp_steps,
                (unsigned long long)m.mtp_kv_updates,
                (unsigned long long)m.mtp_proposed,
                (unsigned long long)m.mtp_accepted, accept,
                (unsigned long long)m.mtp_verify_batches,
                (unsigned long long)m.mtp_verify_tokens,
                (unsigned long long)m.mtp_verify_replays);
        }
#ifdef COLI_CUDA
        if (m.exec.kind == COLI_BACKEND_CUDA)
            fprintf(stderr,
                "[Q35] router=%s router_gpu=%llu router_cpu_fallback=%llu expert_cpu_fallback=%llu residency_fail=%llu compute_fail=%llu admissions=%s pilot=%s%s group_calls=%llu group_experts=%llu group_fail=%llu\n",
                m.cuda_router_gpu_enabled ? "gpu" : "cpu",
                (unsigned long long)m.cuda_router_gpu_calls,
                (unsigned long long)m.cuda_router_cpu_fallbacks,
                (unsigned long long)m.cuda_expert_cpu_fallbacks,
                (unsigned long long)m.cuda_expert_residency_failures,
                (unsigned long long)m.cuda_expert_compute_failures,
                m.cuda_expert_admission_disabled ? "disabled" : "enabled",
                m.pilot ? (m.pilot_real ? "real" : "hint") : "off",
                m.pilot_real_downgraded ? " (auto-downgraded)" : "",
                (unsigned long long)m.focus_group_calls,
                (unsigned long long)m.focus_group_experts,
                (unsigned long long)m.focus_group_failures);
#endif
        if (m.expert_zero_threshold > 0.0f) {
            const double pct = m.expert_prune_values_seen ?
                100.0 * (double)m.expert_prune_values_zeroed /
                    (double)m.expert_prune_values_seen : 0.0;
            fprintf(stderr,
                "[Q35-PRUNE] threshold=%.9g CUDA_materializations=%llu values_zeroed=%llu/%llu (%.3f%%); CPU fallbacks apply the same threshold during matmul\n",
                m.expert_zero_threshold,
                (unsigned long long)m.expert_prune_materializations,
                (unsigned long long)m.expert_prune_values_zeroed,
                (unsigned long long)m.expert_prune_values_seen, pct);
        }
    }
    q35_usage_save(&m); free(repeat_seen); q35_scratch_free(&s); q35_model_free(&m); free(ids);
#ifdef COLI_CUDA
    if (cuda_started) coli_cuda_shutdown();
#endif
    return 0;

fail:
    q35_usage_save(&m); free(repeat_seen); q35_scratch_free(&s); q35_model_free(&m); free(ids);
#ifdef COLI_CUDA
    if (cuda_started) coli_cuda_shutdown();
#endif
    return 1;
}
