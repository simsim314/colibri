#include "gguf_qwen3next.h"
#include "tensor.h"
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
    uint64_t used;
} QwenExpertSlot;

static const ColiExpertSlotLayout g_qwen_slot_layout = {
    sizeof(QwenExpertSlot), offsetof(QwenExpertSlot, eid), offsetof(QwenExpertSlot, used)
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
    ColiTensor qkv, z, ba, conv1d, dt_bias, a, ssm_norm, ssm_out;

    /* Routed and shared experts. */
    ColiTensor router;
    ColiTensor gate_exps, up_exps, down_exps;
    ColiTensor shared_gate_inp, shared_gate, shared_up, shared_down;
    ColiTensor *gate_expert, *up_expert, *down_expert;
    int expert_views;

    QwenExpertSlot *pin, *cache;
    int npin, ncache, cache_cap;
    uint32_t *heat, *last, *usage;
} QwenLayer;

typedef struct {
    ColiGgufFile gguf;
    ColiGgufTokenizer *tokenizer;
    char *architecture;

    int n_layers, hidden, vocab;
    int n_heads, n_kv_heads, head_dim, key_length, value_length, kv_dim;
    int rope_dims, context_length, full_attention_interval;

    int n_experts, n_expert_used, expert_ff, shared_ff;

    int conv_kernel, state_size, n_key_heads, n_value_heads;
    int inner_size, value_head_dim, key_dim, value_dim, conv_dim;
    int n_recurrent, n_attention;

    float rope_base, eps, attention_scale;

    ColiTensor token_embd, output_norm, output;
    int tied_output;
    QwenLayer *layers;

    int context_capacity;
    float *k_cache, *v_cache;
    float *conv_state, *recurrent_state;
#ifdef COLI_CUDA
    float *k_cache_dev, *v_cache_dev;
    float *conv_state_dev, *recurrent_state_dev;
#endif

    size_t cuda_dense_bytes, cuda_expert_bytes, cuda_weight_bytes;
    size_t expert_bytes;

    uint64_t expert_clock;
    uint32_t expert_access_clock;
    ColiExpertSchedulerStats scheduler_stats;
    int scheduler_evict_guard;
    int repin_interval, tokens_since_repin;
    int pilot, pilot_real, pilot_k;
    int couple, couple_k, couple_d;
    ColiExpertCoupling coupling;
    char usage_path[2048];
    int64_t usage_history;

    int verbose;
    ColiExec exec;
} QwenModel;

typedef struct {
    /* Host scratch. */
    float *x, *norm, *mixer, *post;
    float *qg, *attn_q, *attn_gate, *k, *v, *attn, *scores;
    float *qkv, *z, *ba, *conv, *delta;
    float *router, *gate, *up, *expert_out, *moe;
    float *shared_gate, *shared_up, *shared_out;
    float *logits, *weight;
    int *top_idx;
    float *top_w;

#ifdef COLI_CUDA
    /* Device scratch. */
    float *dx, *dnorm, *dmixer, *dpost;
    float *dqg, *dattn_q, *dattn_gate, *dk, *dv, *dattn, *dscores;
    float *dqkv, *dz, *dba, *ddelta;
    float *drouter, *dgate, *dup, *dexpert_out, *dmoe;
    float *dshared_gate, *dshared_up, *dshared_out;
    float *dlogits;
    int cuda_device, cuda_allocated;
#endif
} QwenScratch;

static double qwen_now_sec(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int qwen_errf(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

static int qwen_kv_u64(const ColiGgufFile *g, const char *key, uint64_t *out,
                       int required, char *err, size_t cap) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    if (!kv) return required ? qwen_errf(err, cap, "missing metadata: %s", key) : 0;
    return coli_gguf_kv_read_u64(g, kv, out) ? 1 :
           qwen_errf(err, cap, "invalid integer metadata: %s", key);
}

static int qwen_kv_f32(const ColiGgufFile *g, const char *key, float *out,
                       int required, char *err, size_t cap) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    double v;
    if (!kv) return required ? qwen_errf(err, cap, "missing metadata: %s", key) : 0;
    if (!coli_gguf_kv_read_f64(g, kv, &v))
        return qwen_errf(err, cap, "invalid float metadata: %s", key);
    *out = (float)v;
    return 1;
}

static int qwen_to_int(uint64_t v, int *out, const char *name,
                       char *err, size_t cap) {
    if (v == 0 || v > INT_MAX)
        return qwen_errf(err, cap, "invalid %s: %llu", name, (unsigned long long)v);
    *out = (int)v;
    return 1;
}

static void qwen_tensor_free(ColiTensor *t) { coli_tensor_destroy(t); }

static int qwen_tensor_dims(const ColiTensor *t, int nd, const uint64_t *dims) {
    if (!t || (int)t->n_dims != nd) return 0;
    for (int i = 0; i < nd; ++i) if (t->dims[i] != dims[i]) return 0;
    return 1;
}

static int qwen_load_tensor(QwenModel *m, const char *name, ColiTensor *out,
                            int nd, const uint64_t *dims,
                            char *err, size_t cap) {
    const ColiGgufTensorInfo *ti = coli_gguf_find_tensor(&m->gguf, name);
    if (!ti) return qwen_errf(err, cap, "missing tensor: %s", name);
    if (!coli_tensor_bind_gguf(&m->gguf, ti, out, err, cap)) return 0;
    if (!qwen_tensor_dims(out, nd, dims)) {
        qwen_tensor_free(out);
        return qwen_errf(err, cap, "wrong dimensions for tensor: %s", name);
    }
    return 1;
}

static int qwen_load_layer_tensor(QwenModel *m, int layer, const char *suffix,
                                  ColiTensor *out, int nd, const uint64_t *dims,
                                  char *err, size_t cap) {
    char name[160];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return qwen_load_tensor(m, name, out, nd, dims, err, cap);
}

static void qwen_layer_free(QwenLayer *l) {
    if (!l) return;
    /* Destroy child views before their aggregate tensors. Today each expert
     * owns an independent CUDA upload, but this ordering also stays correct if
     * row views later become zero-copy views of a resident aggregate. */
    for (int e = 0; e < l->expert_views; ++e) {
        qwen_tensor_free(&l->gate_expert[e]);
        qwen_tensor_free(&l->up_expert[e]);
        qwen_tensor_free(&l->down_expert[e]);
    }
#define QFREE(name) qwen_tensor_free(&l->name)
    QFREE(attn_norm); QFREE(post_norm);
    QFREE(q); QFREE(k); QFREE(v); QFREE(q_norm); QFREE(k_norm); QFREE(o);
    QFREE(qkv); QFREE(z); QFREE(ba); QFREE(conv1d); QFREE(dt_bias); QFREE(a);
    QFREE(ssm_norm); QFREE(ssm_out);
    QFREE(router); QFREE(gate_exps); QFREE(up_exps); QFREE(down_exps);
    QFREE(shared_gate_inp); QFREE(shared_gate); QFREE(shared_up); QFREE(shared_down);
#undef QFREE
    free(l->gate_expert); free(l->up_expert); free(l->down_expert);
    free(l->pin); free(l->cache); free(l->heat); free(l->last); free(l->usage);
    memset(l, 0, sizeof(*l));
}

static void qwen_model_free(QwenModel *m) {
    if (!m) return;
    if (m->layers) for (int i = 0; i < m->n_layers; ++i) qwen_layer_free(&m->layers[i]);
    free(m->layers);
    qwen_tensor_free(&m->token_embd);
    qwen_tensor_free(&m->output_norm);
    qwen_tensor_free(&m->output);
#ifdef COLI_CUDA
    if (m->k_cache_dev) coli_cuda_pipe_free(m->exec.device, m->k_cache_dev);
    if (m->v_cache_dev) coli_cuda_pipe_free(m->exec.device, m->v_cache_dev);
    if (m->conv_state_dev) coli_cuda_pipe_free(m->exec.device, m->conv_state_dev);
    if (m->recurrent_state_dev) coli_cuda_pipe_free(m->exec.device, m->recurrent_state_dev);
#endif
    free(m->k_cache); free(m->v_cache); free(m->conv_state); free(m->recurrent_state);
    free(m->architecture);
    coli_expert_coupling_destroy(&m->coupling);
    coli_gguf_tokenizer_destroy(m->tokenizer);
    coli_gguf_close(&m->gguf);
    memset(m, 0, sizeof(*m));
}

static int qwen_model_config(QwenModel *m, char *err, size_t cap) {
    const ColiGgufKV *akv = coli_gguf_find_kv(&m->gguf, "general.architecture");
    if (!akv || !coli_gguf_kv_read_string(&m->gguf, akv, &m->architecture))
        return qwen_errf(err, cap, "general.architecture missing");
    if (strcmp(m->architecture, "qwen3next") != 0)
        return qwen_errf(err, cap, "unsupported GGUF architecture '%s' (expected qwen3next)", m->architecture);

    uint64_t u;
#define QREADI(key, field) do { \
    if (!qwen_kv_u64(&m->gguf, key, &u, 1, err, cap) || \
        !qwen_to_int(u, &m->field, key, err, cap)) return 0; \
} while (0)
    QREADI("qwen3next.block_count", n_layers);
    QREADI("qwen3next.context_length", context_length);
    QREADI("qwen3next.embedding_length", hidden);
    QREADI("qwen3next.attention.head_count", n_heads);
    QREADI("qwen3next.attention.head_count_kv", n_kv_heads);
    QREADI("qwen3next.attention.key_length", key_length);
    QREADI("qwen3next.attention.value_length", value_length);
    QREADI("qwen3next.expert_count", n_experts);
    QREADI("qwen3next.expert_used_count", n_expert_used);
    QREADI("qwen3next.expert_feed_forward_length", expert_ff);
    QREADI("qwen3next.expert_shared_feed_forward_length", shared_ff);
    QREADI("qwen3next.ssm.conv_kernel", conv_kernel);
    QREADI("qwen3next.ssm.state_size", state_size);
    QREADI("qwen3next.ssm.group_count", n_key_heads);
    QREADI("qwen3next.ssm.time_step_rank", n_value_heads);
    QREADI("qwen3next.ssm.inner_size", inner_size);
    QREADI("qwen3next.full_attention_interval", full_attention_interval);
    QREADI("qwen3next.rope.dimension_count", rope_dims);
#undef QREADI

    if (m->n_heads % m->n_kv_heads || m->key_length != m->value_length)
        return qwen_errf(err, cap, "invalid full-attention head dimensions: heads=%d/%d key=%d value=%d",
                         m->n_heads, m->n_kv_heads, m->key_length, m->value_length);
    /* Qwen3-Next projects 2048 hidden features to 16 heads of width 256.
     * The attention head width is explicit GGUF metadata and is not
     * hidden_size / head_count (which would incorrectly give 128). */
    m->head_dim = m->key_length;
    m->kv_dim = m->n_kv_heads * m->head_dim;
    if (m->rope_dims > m->head_dim || (m->rope_dims & 1))
        return qwen_errf(err, cap, "invalid partial RoPE dimension count");

    if (m->inner_size % m->n_value_heads)
        return qwen_errf(err, cap, "SSM inner size is not divisible by value heads");
    m->value_head_dim = m->inner_size / m->n_value_heads;
    if (m->state_size != m->value_head_dim)
        return qwen_errf(err, cap, "unsupported Qwen3-Next state/value head mismatch: %d/%d",
                         m->state_size, m->value_head_dim);
    if (m->n_value_heads % m->n_key_heads)
        return qwen_errf(err, cap, "value-head count is not divisible by key-head count");
    m->key_dim = m->state_size * m->n_key_heads;
    m->value_dim = m->value_head_dim * m->n_value_heads;
    m->conv_dim = 2 * m->key_dim + m->value_dim;
    if (m->conv_kernel < 1 || m->conv_kernel > 16)
        return qwen_errf(err, cap, "unsupported convolution width: %d", m->conv_kernel);
    if (m->full_attention_interval < 1)
        return qwen_errf(err, cap, "invalid full-attention interval: %d", m->full_attention_interval);
    if (m->n_expert_used > m->n_experts)
        return qwen_errf(err, cap, "expert_used_count exceeds expert_count");

    m->n_recurrent = 0;
    m->n_attention = 0;
    for (int i = 0; i < m->n_layers; ++i) {
        if ((i + 1) % m->full_attention_interval) ++m->n_recurrent;
        else ++m->n_attention;
    }

    const ColiGgufTensorInfo *emb = coli_gguf_find_tensor(&m->gguf, "token_embd.weight");
    if (!emb || emb->n_dims != 2 || emb->dims[0] != (uint64_t)m->hidden || emb->dims[1] > INT_MAX)
        return qwen_errf(err, cap, "invalid token_embd.weight dimensions");
    m->vocab = (int)emb->dims[1];

    m->rope_base = 10000000.0f;
    (void)qwen_kv_f32(&m->gguf, "qwen3next.rope.freq_base", &m->rope_base, 0, err, cap);
    m->eps = 1e-6f;
    (void)qwen_kv_f32(&m->gguf, "qwen3next.attention.layer_norm_rms_epsilon", &m->eps, 0, err, cap);
    m->attention_scale = 1.0f / sqrtf((float)m->head_dim);
    return 1;
}

static int qwen_model_load_weights(QwenModel *m, char *err, size_t cap) {
    uint64_t d[3];
    d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->vocab;
    if (!qwen_load_tensor(m, "token_embd.weight", &m->token_embd, 2, d, err, cap)) return 0;
    d[0] = (uint64_t)m->hidden;
    if (!qwen_load_tensor(m, "output_norm.weight", &m->output_norm, 1, d, err, cap)) return 0;
    const ColiGgufTensorInfo *outi = coli_gguf_find_tensor(&m->gguf, "output.weight");
    if (outi) {
        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->vocab;
        if (!qwen_load_tensor(m, "output.weight", &m->output, 2, d, err, cap)) return 0;
    } else {
        m->tied_output = 1;
    }

    m->layers = (QwenLayer *)calloc((size_t)m->n_layers, sizeof(*m->layers));
    if (!m->layers) return qwen_errf(err, cap, "out of memory allocating Qwen3-Next layers");

    int ri = 0, ai = 0;
    for (int l = 0; l < m->n_layers; ++l) {
        QwenLayer *x = &m->layers[l];
        x->recurrent = ((l + 1) % m->full_attention_interval) != 0;
        x->recurrent_index = x->recurrent ? ri++ : -1;
        x->attention_index = x->recurrent ? -1 : ai++;

        d[0] = (uint64_t)m->hidden;
        if (!qwen_load_layer_tensor(m, l, "attn_norm.weight", &x->attn_norm, 1, d, err, cap) ||
            !qwen_load_layer_tensor(m, l, "post_attention_norm.weight", &x->post_norm, 1, d, err, cap)) return 0;

        if (x->recurrent) {
            d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->conv_dim;
            if (!qwen_load_layer_tensor(m, l, "attn_qkv.weight", &x->qkv, 2, d, err, cap)) return 0;
            d[1] = (uint64_t)m->value_dim;
            if (!qwen_load_layer_tensor(m, l, "attn_gate.weight", &x->z, 2, d, err, cap)) return 0;
            d[1] = (uint64_t)(2 * m->n_value_heads);
            if (!qwen_load_layer_tensor(m, l, "ssm_ba.weight", &x->ba, 2, d, err, cap)) return 0;
            d[0] = (uint64_t)m->conv_kernel; d[1] = (uint64_t)m->conv_dim;
            if (!qwen_load_layer_tensor(m, l, "ssm_conv1d.weight", &x->conv1d, 2, d, err, cap)) return 0;
            d[0] = (uint64_t)m->n_value_heads;
            if (!qwen_load_layer_tensor(m, l, "ssm_dt.bias", &x->dt_bias, 1, d, err, cap) ||
                !qwen_load_layer_tensor(m, l, "ssm_a", &x->a, 1, d, err, cap)) return 0;
            d[0] = (uint64_t)m->value_head_dim;
            if (!qwen_load_layer_tensor(m, l, "ssm_norm.weight", &x->ssm_norm, 1, d, err, cap)) return 0;
            d[0] = (uint64_t)m->value_dim; d[1] = (uint64_t)m->hidden;
            if (!qwen_load_layer_tensor(m, l, "ssm_out.weight", &x->ssm_out, 2, d, err, cap)) return 0;
        } else {
            d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)(2 * m->n_heads * m->head_dim);
            if (!qwen_load_layer_tensor(m, l, "attn_q.weight", &x->q, 2, d, err, cap)) return 0;
            d[1] = (uint64_t)m->kv_dim;
            if (!qwen_load_layer_tensor(m, l, "attn_k.weight", &x->k, 2, d, err, cap) ||
                !qwen_load_layer_tensor(m, l, "attn_v.weight", &x->v, 2, d, err, cap)) return 0;
            d[0] = (uint64_t)m->head_dim;
            if (!qwen_load_layer_tensor(m, l, "attn_q_norm.weight", &x->q_norm, 1, d, err, cap) ||
                !qwen_load_layer_tensor(m, l, "attn_k_norm.weight", &x->k_norm, 1, d, err, cap)) return 0;
            d[0] = (uint64_t)(m->n_heads * m->head_dim); d[1] = (uint64_t)m->hidden;
            if (!qwen_load_layer_tensor(m, l, "attn_output.weight", &x->o, 2, d, err, cap)) return 0;
        }

        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->n_experts;
        if (!qwen_load_layer_tensor(m, l, "ffn_gate_inp.weight", &x->router, 2, d, err, cap)) return 0;
        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->expert_ff; d[2] = (uint64_t)m->n_experts;
        if (!qwen_load_layer_tensor(m, l, "ffn_gate_exps.weight", &x->gate_exps, 3, d, err, cap) ||
            !qwen_load_layer_tensor(m, l, "ffn_up_exps.weight", &x->up_exps, 3, d, err, cap)) return 0;
        d[0] = (uint64_t)m->expert_ff; d[1] = (uint64_t)m->hidden; d[2] = (uint64_t)m->n_experts;
        if (!qwen_load_layer_tensor(m, l, "ffn_down_exps.weight", &x->down_exps, 3, d, err, cap)) return 0;

        d[0] = (uint64_t)m->hidden;
        if (!qwen_load_layer_tensor(m, l, "ffn_gate_inp_shexp.weight", &x->shared_gate_inp, 1, d, err, cap)) return 0;
        d[0] = (uint64_t)m->hidden; d[1] = (uint64_t)m->shared_ff;
        if (!qwen_load_layer_tensor(m, l, "ffn_gate_shexp.weight", &x->shared_gate, 2, d, err, cap) ||
            !qwen_load_layer_tensor(m, l, "ffn_up_shexp.weight", &x->shared_up, 2, d, err, cap)) return 0;
        d[0] = (uint64_t)m->shared_ff; d[1] = (uint64_t)m->hidden;
        if (!qwen_load_layer_tensor(m, l, "ffn_down_shexp.weight", &x->shared_down, 2, d, err, cap)) return 0;

        if (m->verbose && ((l + 1) % 8 == 0 || l + 1 == m->n_layers))
            fprintf(stderr, "[GGUF] bound Qwen3-Next layer %d/%d\n", l + 1, m->n_layers);
    }
    return 1;
}

static int qwen_build_expert_views(QwenModel *m, char *err, size_t cap) {
    for (int l = 0; l < m->n_layers; ++l) {
        QwenLayer *x = &m->layers[l];
        x->gate_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->up_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->down_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->heat = (uint32_t *)calloc((size_t)m->n_experts, sizeof(uint32_t));
        x->last = (uint32_t *)calloc((size_t)m->n_experts, sizeof(uint32_t));
        x->usage = (uint32_t *)calloc((size_t)m->n_experts, sizeof(uint32_t));
        if (!x->gate_expert || !x->up_expert || !x->down_expert ||
            !x->heat || !x->last || !x->usage)
            return qwen_errf(err, cap, "out of memory allocating Qwen expert views");
        for (int e = 0; e < m->n_experts; ++e) {
            if (!coli_tensor_rows_view(&x->gate_exps, (uint64_t)e * m->expert_ff,
                                       (uint64_t)m->expert_ff, &x->gate_expert[e]) ||
                !coli_tensor_rows_view(&x->up_exps, (uint64_t)e * m->expert_ff,
                                       (uint64_t)m->expert_ff, &x->up_expert[e]) ||
                !coli_tensor_rows_view(&x->down_exps, (uint64_t)e * m->hidden,
                                       (uint64_t)m->hidden, &x->down_expert[e]))
                return qwen_errf(err, cap, "invalid Qwen expert view at layer %d expert %d", l, e);
            x->expert_views = e + 1;
        }
    }
    /* Quantization can vary by layer (the target GGUF mixes Q4_K and Q6_K
     * expert-down matrices). Budget every scheduler slot at the largest expert
     * triple so encoded VRAM residency can never exceed the requested tier. */
    m->expert_bytes = 0;
    for (int l = 0; l < m->n_layers; ++l) {
        QwenLayer *x = &m->layers[l];
        const size_t bytes = (size_t)x->gate_expert[0].storage_bytes +
                             (size_t)x->up_expert[0].storage_bytes +
                             (size_t)x->down_expert[0].storage_bytes;
        if (bytes > m->expert_bytes) m->expert_bytes = bytes;
    }
    return 1;
}

/* ---- CPU reference operations ------------------------------------------------ */

static float qwen_sigmoid(float x) {
    if (x >= 0.0f) { float z = expf(-x); return 1.0f / (1.0f + z); }
    float z = expf(x); return z / (1.0f + z);
}

static float qwen_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* Full-attention q_proj is laid out per head as [Q(head), gate(head)].
 * It is not family-major. Split it before Q normalization/RoPE and retain the
 * raw gate for the post-attention sigmoid. */
static void qwen_split_q_gate(float *q, float *gate, const float *qg,
                              int n_heads, int head_dim) {
    for (int h = 0; h < n_heads; ++h) {
        const float *src = qg + (size_t)h * 2 * head_dim;
        memcpy(q + (size_t)h * head_dim, src, (size_t)head_dim * sizeof(float));
        memcpy(gate + (size_t)h * head_dim, src + head_dim,
               (size_t)head_dim * sizeof(float));
    }
}

static void qwen_rope_neox(float *v, int n_heads, int head_dim,
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

static void qwen_l2norm_heads(float *x, int heads, int dim, float eps) {
    for (int h = 0; h < heads; ++h) {
        float *p = x + (size_t)h * dim;
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) sum += (double)p[i] * p[i];
        const float inv = 1.0f / sqrtf((float)sum + eps);
        for (int i = 0; i < dim; ++i) p[i] *= inv;
    }
}

static int qwen_f32_vector(const ColiTensor *t, float *out, int n,
                           char *err, size_t cap) {
    if (!t || t->row_count != 1 || t->dims[0] != (uint64_t)n ||
        !coli_tensor_read_row_f32(t, 0, out, (uint64_t)n))
        return qwen_errf(err, cap, "cannot decode vector %s", t && t->name ? t->name : "<unnamed>");
    return 1;
}

static int qwen_mm_cpu(float *y, const float *x, ColiTensor *w,
                       int I, int O, char *err, size_t cap) {
    ColiExec cpu = { COLI_BACKEND_CPU, 0 };
    if (!coli_tensor_matmul(&cpu, y, x, w, 1, I, O))
        return qwen_errf(err, cap, "CPU native matmul failed for %s", w->name ? w->name : "<unnamed>");
    return 1;
}

static int qwen_conv_update_cpu(QwenModel *m, QwenLayer *l,
                                const float *input, float *output,
                                float *state, char *err, size_t cap) {
    if (l->conv1d.dtype != COLI_DTYPE_F32 || l->conv1d.dims[0] != (uint64_t)m->conv_kernel)
        return qwen_errf(err, cap, "Qwen conv1d must be F32");
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

static int qwen_delta_decode_cpu(QwenModel *m, QwenLayer *l,
                                 float *conv, const float *z, const float *ba,
                                 float *state, float *out,
                                 char *err, size_t cap) {
    float *q = conv;
    float *k = conv + m->key_dim;
    float *v = conv + 2 * m->key_dim;
    qwen_l2norm_heads(q, m->n_key_heads, m->state_size, 1e-6f);
    qwen_l2norm_heads(k, m->n_key_heads, m->state_size, 1e-6f);

    float *dt = (float *)malloc((size_t)m->n_value_heads * sizeof(float));
    float *a = (float *)malloc((size_t)m->n_value_heads * sizeof(float));
    float *nw = (float *)malloc((size_t)m->value_head_dim * sizeof(float));
    if (!dt || !a || !nw) { free(dt); free(a); free(nw); return qwen_errf(err, cap, "DeltaNet vector scratch OOM"); }
    if (!qwen_f32_vector(&l->dt_bias, dt, m->n_value_heads, err, cap) ||
        !qwen_f32_vector(&l->a, a, m->n_value_heads, err, cap) ||
        !qwen_f32_vector(&l->ssm_norm, nw, m->value_head_dim, err, cap)) {
        free(dt); free(a); free(nw); return 0;
    }

    const int ratio = m->n_value_heads / m->n_key_heads;
    const float scale = 1.0f / sqrtf((float)m->state_size);
    for (int vh = 0; vh < m->n_value_heads; ++vh) {
        const int kh = vh / ratio;
        const int sub = vh % ratio;
        const int base = kh * 2 * ratio;
        const float beta = qwen_sigmoid(ba[base + sub]);
        const float decay_log = a[vh] * qwen_softplus(ba[base + ratio + sub] + dt[vh]);
        const float decay = expf(decay_log);
        const float *qh = q + (size_t)kh * m->state_size;
        const float *khv = k + (size_t)kh * m->state_size;
        const float *vhv = v + (size_t)vh * m->value_head_dim;
        const float *zh = z + (size_t)vh * m->value_head_dim;
        float *S = state + (size_t)vh * m->state_size * m->value_head_dim;
        float *oh = out + (size_t)vh * m->value_head_dim;

        for (int col = 0; col < m->value_head_dim; ++col) {
            float *Sc = S + (size_t)col * m->state_size; /* transposed state: column-major rows */
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

/* ---- Scheduler integration: one existing policy, Qwen storage callbacks ------- */

static ColiExpertLayerStore qwen_store(QwenModel *m, int layer) {
    QwenLayer *l = &m->layers[layer];
    ColiExpertLayerStore s;
    memset(&s, 0, sizeof(s));
    s.pin = l->pin; s.npin = l->npin;
    s.cache = l->cache; s.ncache = &l->ncache; s.cache_cap = l->cache_cap;
    s.n_experts = m->n_experts; s.layout = g_qwen_slot_layout;
    s.clock = &m->expert_clock; s.heat = l->heat; s.last = l->last; s.usage = l->usage;
    s.access_clock = &m->expert_access_clock;
    return s;
}

static void qwen_slot_bind(QwenModel *m, int layer, int eid, QwenExpertSlot *s) {
    QwenLayer *l = &m->layers[layer];
    s->eid = eid;
    s->gate = &l->gate_expert[eid];
    s->up = &l->up_expert[eid];
    s->down = &l->down_expert[eid];
}

static int qwen_storage_load(void *ctx, int layer, int eid, void *slot, int demand) {
    QwenModel *m = (QwenModel *)ctx;
    QwenExpertSlot *s = (QwenExpertSlot *)slot;
    (void)demand;
    qwen_slot_bind(m, layer, eid, s);
    coli_tensor_prefetch_host(s->gate);
    coli_tensor_prefetch_host(s->up);
    coli_tensor_prefetch_host(s->down);
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        if (!coli_tensor_reside(&m->exec, s->gate) ||
            !coli_tensor_reside(&m->exec, s->up) ||
            !coli_tensor_reside(&m->exec, s->down)) {
            coli_tensor_release_backend(&m->exec, s->gate);
            coli_tensor_release_backend(&m->exec, s->up);
            coli_tensor_release_backend(&m->exec, s->down);
            s->eid = -1; s->gate = s->up = s->down = NULL;
            return 0;
        }
        m->cuda_expert_bytes += m->expert_bytes;
        m->cuda_weight_bytes = m->cuda_dense_bytes + m->cuda_expert_bytes;
    }
#endif
    return 1;
}

static void qwen_storage_evict(void *ctx, int layer, void *slot) {
    QwenModel *m = (QwenModel *)ctx;
    QwenExpertSlot *s = (QwenExpertSlot *)slot;
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
    s->eid = -1; s->gate = s->up = s->down = NULL; s->used = 0;
}

static size_t qwen_storage_bytes(void *ctx, int layer, const void *slot) {
    (void)layer; (void)slot;
    return ((QwenModel *)ctx)->expert_bytes;
}

static const ColiExpertStorageOps g_qwen_storage = {
    qwen_storage_load, qwen_storage_evict, qwen_storage_bytes
};

static QwenExpertSlot *qwen_expert_acquire(QwenModel *m, int layer, int eid, int demand) {
    ColiExpertLayerStore st = qwen_store(m, layer);
    return (QwenExpertSlot *)coli_expert_acquire(&st, layer, eid, demand,
            m->scheduler_evict_guard, &g_qwen_storage, m, &m->scheduler_stats);
}

static void qwen_usage_path(QwenModel *m, const char *model_path) {
    const size_t n = strlen(model_path);
    if (n + sizeof(".coli_usage") > sizeof(m->usage_path)) return;
    memcpy(m->usage_path, model_path, n);
    memcpy(m->usage_path + n, ".coli_usage", sizeof(".coli_usage"));
}

static void qwen_usage_rows(QwenModel *m, uint32_t **rows) {
    for (int l = 0; l < m->n_layers; ++l) rows[l] = m->layers[l].usage;
}

static int64_t qwen_usage_load(QwenModel *m, const char *model_path) {
    qwen_usage_path(m, model_path);
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return 0;
    qwen_usage_rows(m, rows);
    const int64_t total = coli_expert_usage_load(m->usage_path, rows, m->n_layers, m->n_experts);
    free(rows);
    m->usage_history = total;
    return total;
}

static void qwen_usage_save(QwenModel *m) {
    if (!m || !m->usage_path[0]) return;
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return;
    qwen_usage_rows(m, rows);
    (void)coli_expert_usage_save(m->usage_path, rows, m->n_layers, m->n_experts);
    free(rows);
}

static int qwen_usage_top(QwenModel *m, int *ids, int cap) {
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return 0;
    qwen_usage_rows(m, rows);
    const int n = coli_expert_usage_top(rows, m->n_layers, m->n_experts, ids, cap);
    free(rows);
    return n;
}

static int qwen_stats_fallback(QwenModel *m, char *path, size_t cap) {
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

static int qwen_scheduler_init(QwenModel *m, char *err, size_t cap) {
    m->scheduler_evict_guard = getenv("PILOT_EVICT_GUARD") ? atoi(getenv("PILOT_EVICT_GUARD")) : 1;
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
                                              m->n_layers, m->n_experts, &used);
        if (m->verbose) fprintf(stderr, "[COUPLE] Qwen GGUF: %ld conditioning entries\n", used);
    }

    const int total = m->n_layers * m->n_experts;
    /* mmap-backed CPU experts need scheduler metadata, not a 48 GB eager
     * WILLNEED sweep. Keep enough LRU slots for the routed set per layer. */
    int slots = m->n_layers * m->n_expert_used;
    if (slots < m->n_layers) slots = m->n_layers;
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
        if (slots < m->n_layers)
            return qwen_errf(err, cap,
                "CUDA expert budget fits %d slots; need at least %d after dense Qwen residency",
                slots, m->n_layers);
    }
#endif

    int pin_total = slots == total ? total : 0;
    int *pinids = NULL, npinids = 0;
    const char *pinfile = getenv("PIN");
    if (slots < total) {
        const int maxpins = slots > m->n_layers ? slots - m->n_layers : 0;
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
                else if (qwen_stats_fallback(m, auto_stats, sizeof(auto_stats))) source = auto_stats;
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
            if (!pinids) return qwen_errf(err, cap, "Qwen scheduler pin ranking OOM");
            npinids = source == m->usage_path ? qwen_usage_top(m, pinids, want) :
                       coli_expert_usage_top_file(source, m->n_layers, m->n_experts,
                                                  pinids, want, NULL);
            pin_total = npinids;
            if (m->verbose) fprintf(stderr, "[PIN] Qwen GGUF: %d experts from %s\n", npinids, source);
        }
    }

    int *pc = (int *)calloc((size_t)m->n_layers, sizeof(int));
    if (!pc) { free(pinids); return qwen_errf(err, cap, "Qwen scheduler OOM"); }
    if (pin_total == total) for (int l = 0; l < m->n_layers; ++l) pc[l] = m->n_experts;
    else for (int i = 0; i < npinids; ++i) ++pc[pinids[i] / m->n_experts];

    const int cache_slots = slots - pin_total;
    const int base = cache_slots / m->n_layers;
    const int extra = cache_slots % m->n_layers;
    for (int l = 0; l < m->n_layers; ++l) {
        QwenLayer *x = &m->layers[l];
        x->npin = pc[l];
        x->cache_cap = base + (l < extra);
        if (x->npin) x->pin = (QwenExpertSlot *)calloc((size_t)x->npin, sizeof(*x->pin));
        if (x->cache_cap) x->cache = (QwenExpertSlot *)calloc((size_t)x->cache_cap, sizeof(*x->cache));
        if ((x->npin && !x->pin) || (x->cache_cap && !x->cache)) {
            free(pc); free(pinids); return qwen_errf(err, cap, "Qwen scheduler slot OOM");
        }
        for (int z = 0; z < x->npin; ++z) x->pin[z].eid = -1;
        for (int z = 0; z < x->cache_cap; ++z) x->cache[z].eid = -1;
    }

    if (pin_total == total) {
        for (int l = 0; l < m->n_layers; ++l) for (int e = 0; e < m->n_experts; ++e) {
            QwenExpertSlot *q = &m->layers[l].pin[e];
            if (!qwen_storage_load(m, l, e, q, 0)) {
                free(pc); free(pinids); return qwen_errf(err, cap, "Qwen expert pin residency failed");
            }
            q->used = ++m->expert_clock;
        }
    } else {
        int *next = (int *)calloc((size_t)m->n_layers, sizeof(int));
        if (!next) { free(pc); free(pinids); return qwen_errf(err, cap, "Qwen pin OOM"); }
        for (int i = 0; i < npinids; ++i) {
            const int l = pinids[i] / m->n_experts, e = pinids[i] % m->n_experts;
            QwenExpertSlot *q = &m->layers[l].pin[next[l]++];
            if (!qwen_storage_load(m, l, e, q, 0)) {
                free(next); free(pc); free(pinids);
                return qwen_errf(err, cap, "Qwen expert pin residency failed");
            }
            q->used = ++m->expert_clock;
        }
        free(next);
    }
    free(pc); free(pinids);

    if (m->verbose)
        fprintf(stderr, "[SCHED] one native scheduler: Qwen %d pin + %d LRU slots, %.2f MiB/slot; PILOT=%s COUPLE=%s\n",
                pin_total, cache_slots, m->expert_bytes / (1024.0 * 1024.0),
                m->pilot ? (m->pilot_real ? "real" : "hint") : "off",
                m->couple ? "on" : "off");
    return 1;
}

static void qwen_prefetch_ids(QwenModel *m, int layer, const int *ids, int n) {
    if (layer < 0 || layer >= m->n_layers) return;
    QwenLayer *l = &m->layers[layer];
    for (int i = 0; i < n; ++i) {
        const int eid = ids[i];
        if (eid < 0 || eid >= m->n_experts) continue;
        if (m->pilot_real) (void)qwen_expert_acquire(m, layer, eid, 0);
        else {
            coli_tensor_prefetch_host(&l->gate_expert[eid]);
            coli_tensor_prefetch_host(&l->up_expert[eid]);
            coli_tensor_prefetch_host(&l->down_expert[eid]);
        }
    }
}

static void qwen_couple_prefetch(QwenModel *m, int layer, const int *routed, int nrouted) {
    if (!m->couple) return;
    for (int d = 1; d <= m->couple_d; ++d) {
        const int target = layer + d;
        if (target >= m->n_layers) break;
        int pred[32];
        const int n = coli_expert_coupling_predict(&m->coupling, layer, d, routed, nrouted,
                                                   pred, m->couple_k);
        qwen_prefetch_ids(m, target, pred, n);
    }
}

static void qwen_repin(QwenModel *m) {
    if (m->repin_interval <= 0 || ++m->tokens_since_repin < m->repin_interval) return;
    m->tokens_since_repin = 0;
    for (int l = 0; l < m->n_layers; ++l) {
        ColiExpertLayerStore st = qwen_store(m, l);
        int pi, e; long gain;
        if (!coli_expert_repin_pick(&st, &pi, &e, &gain)) continue;
        QwenExpertSlot *q = &m->layers[l].pin[pi];
        const int old = q->eid;
        const int moved = coli_expert_repin_promote_cached(&st, pi, e, l,
                                                           &g_qwen_storage, m);
        if (moved > 0) {
            if (m->verbose)
                fprintf(stderr, "[REPIN] Qwen layer %d: %d <- cached %d (gain %ld)\n",
                        l, old, e, gain);
            coli_expert_decay_heat(&st);
            continue;
        }
        if (moved < 0) continue;
        qwen_storage_evict(m, l, q);
        if (qwen_storage_load(m, l, e, q, 0)) {
            q->used = ++m->expert_clock;
            if (m->verbose)
                fprintf(stderr, "[REPIN] Qwen layer %d: %d <- %d (gain %ld)\n", l, old, e, gain);
        } else if (old >= 0) {
            (void)qwen_storage_load(m, l, old, q, 0);
            q->used = ++m->expert_clock;
        }
        coli_expert_decay_heat(&st);
    }
}

/* ---- Dense residency and state ------------------------------------------------ */

#ifdef COLI_CUDA
static int qwen_reside_one(QwenModel *m, ColiTensor *t, char *err, size_t cap) {
    if (!t->data) return 1;
    if (!coli_tensor_reside(&m->exec, t))
        return qwen_errf(err, cap, "cannot make dense tensor resident: %s", t->name ? t->name : "<unnamed>");
    m->cuda_dense_bytes += (size_t)t->storage_bytes;
    return 1;
}
#endif

static int qwen_model_reside_cuda(QwenModel *m, char *err, size_t cap) {
#ifndef COLI_CUDA
    (void)m; (void)err; (void)cap;
    return 1;
#else
    if (m->exec.kind != COLI_BACKEND_CUDA) return 1;
#define QRES(t) do { if (!qwen_reside_one(m, &(t), err, cap)) return 0; } while (0)
    QRES(m->token_embd); QRES(m->output_norm); if (!m->tied_output) QRES(m->output);
    for (int l = 0; l < m->n_layers; ++l) {
        QwenLayer *x = &m->layers[l];
        QRES(x->attn_norm); QRES(x->post_norm);
        if (x->recurrent) {
            QRES(x->qkv); QRES(x->z); QRES(x->ba); QRES(x->conv1d);
            QRES(x->dt_bias); QRES(x->a); QRES(x->ssm_norm); QRES(x->ssm_out);
        } else {
            QRES(x->q); QRES(x->k); QRES(x->v); QRES(x->q_norm); QRES(x->k_norm); QRES(x->o);
        }
        QRES(x->router); QRES(x->shared_gate_inp); QRES(x->shared_gate);
        QRES(x->shared_up); QRES(x->shared_down);
        if (m->verbose && ((l + 1) % 8 == 0 || l + 1 == m->n_layers))
            fprintf(stderr, "[CUDA] resident Qwen dense layer %d/%d\n", l + 1, m->n_layers);
    }
#undef QRES
    m->cuda_weight_bytes = m->cuda_dense_bytes;
    return 1;
#endif
}

static int qwen_alloc_state(QwenModel *m, int context, char *err, size_t cap) {
    if (context <= 0 || context > m->context_length)
        return qwen_errf(err, cap, "requested context %d exceeds model context %d", context, m->context_length);

    size_t kv_elems = (size_t)m->n_attention * context * m->kv_dim;
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
        if (!m->k_cache_dev || !m->v_cache_dev || !m->conv_state_dev || !m->recurrent_state_dev)
            return qwen_errf(err, cap, "out of VRAM allocating Qwen hybrid state");
        if ((kv_elems && (!coli_cuda_pipe_zero(m->exec.device, m->k_cache_dev, kv_elems) ||
                          !coli_cuda_pipe_zero(m->exec.device, m->v_cache_dev, kv_elems))) ||
            (conv_elems && !coli_cuda_pipe_zero(m->exec.device, m->conv_state_dev, conv_elems)) ||
            (state_elems && !coli_cuda_pipe_zero(m->exec.device, m->recurrent_state_dev, state_elems)))
            return qwen_errf(err, cap, "cannot initialize Qwen CUDA state");
    } else
#endif
    {
        m->k_cache = kv_elems ? (float *)calloc(kv_elems, sizeof(float)) : NULL;
        m->v_cache = kv_elems ? (float *)calloc(kv_elems, sizeof(float)) : NULL;
        m->conv_state = conv_elems ? (float *)calloc(conv_elems, sizeof(float)) : NULL;
        m->recurrent_state = state_elems ? (float *)calloc(state_elems, sizeof(float)) : NULL;
        if ((kv_elems && (!m->k_cache || !m->v_cache)) ||
            (conv_elems && !m->conv_state) || (state_elems && !m->recurrent_state))
            return qwen_errf(err, cap, "out of memory allocating Qwen hybrid state");
    }
    m->context_capacity = context;
    return 1;
}

/* ---- Scratch ------------------------------------------------------------------ */

static void qwen_scratch_free(QwenScratch *s) {
#ifdef COLI_CUDA
    if (s->cuda_allocated) {
#define QDFREE(x) do { if (s->x) coli_cuda_pipe_free(s->cuda_device, s->x); } while (0)
        QDFREE(dx); QDFREE(dnorm); QDFREE(dmixer); QDFREE(dpost);
        QDFREE(dqg); QDFREE(dattn_q); QDFREE(dattn_gate); QDFREE(dk); QDFREE(dv); QDFREE(dattn); QDFREE(dscores);
        QDFREE(dqkv); QDFREE(dz); QDFREE(dba); QDFREE(ddelta);
        QDFREE(drouter); QDFREE(dgate); QDFREE(dup); QDFREE(dexpert_out); QDFREE(dmoe);
        QDFREE(dshared_gate); QDFREE(dshared_up); QDFREE(dshared_out); QDFREE(dlogits);
#undef QDFREE
    }
#endif
#define QHFREE(x) free(s->x)
    QHFREE(x); QHFREE(norm); QHFREE(mixer); QHFREE(post);
    QHFREE(qg); QHFREE(attn_q); QHFREE(attn_gate); QHFREE(k); QHFREE(v); QHFREE(attn); QHFREE(scores);
    QHFREE(qkv); QHFREE(z); QHFREE(ba); QHFREE(conv); QHFREE(delta);
    QHFREE(router); QHFREE(gate); QHFREE(up); QHFREE(expert_out); QHFREE(moe);
    QHFREE(shared_gate); QHFREE(shared_up); QHFREE(shared_out);
    QHFREE(logits); QHFREE(weight); QHFREE(top_idx); QHFREE(top_w);
#undef QHFREE
    memset(s, 0, sizeof(*s));
}

static int qwen_scratch_alloc(const QwenModel *m, QwenScratch *s, char *err, size_t cap) {
#define QALLOC(field, n, type) do { \
    s->field = (type *)calloc((size_t)(n), sizeof(type)); \
    if (!s->field) { qwen_scratch_free(s); return qwen_errf(err, cap, "Qwen inference scratch OOM"); } \
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
    QALLOC(ba, 2 * m->n_value_heads, float); QALLOC(conv, m->conv_dim, float);
    QALLOC(delta, m->value_dim, float);
    QALLOC(router, m->n_experts, float);
    QALLOC(gate, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff, float);
    QALLOC(up, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff, float);
    QALLOC(expert_out, m->hidden, float); QALLOC(moe, m->hidden, float);
    QALLOC(shared_gate, 1, float); QALLOC(shared_up, m->shared_ff, float);
    QALLOC(shared_out, m->hidden, float);
    QALLOC(logits, m->vocab, float); QALLOC(weight, m->hidden > m->head_dim ? m->hidden : m->head_dim, float);
    QALLOC(top_idx, m->n_expert_used, int); QALLOC(top_w, m->n_expert_used, float);
#undef QALLOC

#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        s->cuda_device = m->exec.device;
#define QDALLOC(field, n) do { \
    s->field = (float *)coli_cuda_pipe_alloc(m->exec.device, (size_t)(n) * sizeof(float)); \
    if (!s->field) { qwen_scratch_free(s); return qwen_errf(err, cap, "Qwen CUDA scratch OOM: %s", #field); } \
} while (0)
        QDALLOC(dx, m->hidden); QDALLOC(dnorm, m->hidden); QDALLOC(dmixer, m->hidden); QDALLOC(dpost, m->hidden);
        QDALLOC(dqg, 2 * m->n_heads * m->head_dim);
        QDALLOC(dattn_q, m->n_heads * m->head_dim); QDALLOC(dattn_gate, m->n_heads * m->head_dim);
        QDALLOC(dk, m->kv_dim); QDALLOC(dv, m->kv_dim);
        QDALLOC(dattn, m->n_heads * m->head_dim); QDALLOC(dscores, (size_t)m->n_heads * m->context_capacity);
        QDALLOC(dqkv, m->conv_dim); QDALLOC(dz, m->value_dim); QDALLOC(dba, 2 * m->n_value_heads);
        QDALLOC(ddelta, m->value_dim); QDALLOC(drouter, m->n_experts);
        QDALLOC(dgate, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff);
        QDALLOC(dup, m->expert_ff > m->shared_ff ? m->expert_ff : m->shared_ff);
        QDALLOC(dexpert_out, m->hidden); QDALLOC(dmoe, m->hidden);
        QDALLOC(dshared_gate, 1); QDALLOC(dshared_up, m->shared_ff); QDALLOC(dshared_out, m->hidden);
        QDALLOC(dlogits, m->vocab);
#undef QDALLOC
        s->cuda_allocated = 1;
    }
#endif
    return 1;
}

/* ---- CPU forward --------------------------------------------------------------- */

static int qwen_forward_cpu(QwenModel *m, QwenScratch *s, int token, int pos,
                            char *err, size_t cap) {
    if (token < 0 || token >= m->vocab) return qwen_errf(err, cap, "token id outside vocabulary");
    if (pos < 0 || pos >= m->context_capacity) return qwen_errf(err, cap, "position outside context");
    if (!coli_tensor_read_row_f32(&m->token_embd, (uint64_t)token, s->x, (uint64_t)m->hidden))
        return qwen_errf(err, cap, "embedding decode failed");

    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    const size_t conv_layer_stride = (size_t)m->conv_dim * m->conv_kernel;
    const size_t state_layer_stride = (size_t)m->n_value_heads * m->state_size * m->value_head_dim;

    for (int l = 0; l < m->n_layers; ++l) {
        QwenLayer *L = &m->layers[l];
        if (!qwen_f32_vector(&L->attn_norm, s->weight, m->hidden, err, cap)) return 0;
        coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);

        if (L->recurrent) {
            if (!qwen_mm_cpu(s->qkv, s->norm, &L->qkv, m->hidden, m->conv_dim, err, cap) ||
                !qwen_mm_cpu(s->z, s->norm, &L->z, m->hidden, m->value_dim, err, cap) ||
                !qwen_mm_cpu(s->ba, s->norm, &L->ba, m->hidden, 2 * m->n_value_heads, err, cap)) return 0;
            float *conv_state = m->conv_state + (size_t)L->recurrent_index * conv_layer_stride;
            float *rec_state = m->recurrent_state + (size_t)L->recurrent_index * state_layer_stride;
            if (!qwen_conv_update_cpu(m, L, s->qkv, s->conv, conv_state, err, cap) ||
                !qwen_delta_decode_cpu(m, L, s->conv, s->z, s->ba, rec_state, s->delta, err, cap) ||
                !qwen_mm_cpu(s->mixer, s->delta, &L->ssm_out, m->value_dim, m->hidden, err, cap)) return 0;
        } else {
            if (!qwen_mm_cpu(s->qg, s->norm, &L->q, m->hidden, 2 * m->n_heads * m->head_dim, err, cap) ||
                !qwen_mm_cpu(s->k, s->norm, &L->k, m->hidden, m->kv_dim, err, cap) ||
                !qwen_mm_cpu(s->v, s->norm, &L->v, m->hidden, m->kv_dim, err, cap)) return 0;
            qwen_split_q_gate(s->attn_q, s->attn_gate, s->qg, m->n_heads, m->head_dim);
            if (!qwen_f32_vector(&L->q_norm, s->weight, m->head_dim, err, cap)) return 0;
            for (int h = 0; h < m->n_heads; ++h)
                coli_f32_rmsnorm(s->attn_q + (size_t)h * m->head_dim,
                                 s->attn_q + (size_t)h * m->head_dim,
                                 s->weight, m->head_dim, m->eps);
            if (!qwen_f32_vector(&L->k_norm, s->weight, m->head_dim, err, cap)) return 0;
            for (int h = 0; h < m->n_kv_heads; ++h)
                coli_f32_rmsnorm(s->k + (size_t)h * m->head_dim,
                                 s->k + (size_t)h * m->head_dim,
                                 s->weight, m->head_dim, m->eps);
            qwen_rope_neox(s->attn_q, m->n_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
            qwen_rope_neox(s->k, m->n_kv_heads, m->head_dim, m->rope_dims, pos, m->rope_base);
            float *kc = m->k_cache + (size_t)L->attention_index * kv_layer_stride;
            float *vc = m->v_cache + (size_t)L->attention_index * kv_layer_stride;
            memcpy(kc + (size_t)pos * m->kv_dim, s->k, (size_t)m->kv_dim * sizeof(float));
            memcpy(vc + (size_t)pos * m->kv_dim, s->v, (size_t)m->kv_dim * sizeof(float));
            coli_f32_gqa_attention(s->attn, s->attn_q, kc, vc, pos, m->n_heads, m->n_kv_heads,
                                   m->head_dim, m->attention_scale, s->scores);
            for (int i = 0; i < m->n_heads * m->head_dim; ++i)
                s->attn[i] *= qwen_sigmoid(s->attn_gate[i]);
            if (!qwen_mm_cpu(s->mixer, s->attn, &L->o,
                             m->n_heads * m->head_dim, m->hidden, err, cap)) return 0;
        }

        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->mixer[i];
        if (!qwen_f32_vector(&L->post_norm, s->weight, m->hidden, err, cap)) return 0;
        coli_f32_rmsnorm(s->post, s->x, s->weight, m->hidden, m->eps);

        if (!qwen_mm_cpu(s->router, s->post, &L->router, m->hidden, m->n_experts, err, cap)) return 0;
        const int nk = coli_f32_router_topk(s->router, m->n_experts, m->n_expert_used,
                                            s->top_idx, s->top_w);
        qwen_couple_prefetch(m, l, s->top_idx, nk);
        if (m->pilot && l + 1 < m->n_layers) {
            int pidx[64]; float pw[64];
            const int keep = m->pilot_k < 64 ? m->pilot_k : 64;
            if (!qwen_mm_cpu(s->router, s->post, &m->layers[l + 1].router,
                             m->hidden, m->n_experts, err, cap)) return 0;
            const int pn = coli_f32_router_topk(s->router, m->n_experts, keep, pidx, pw);
            qwen_prefetch_ids(m, l + 1, pidx, pn);
        }

        memset(s->moe, 0, (size_t)m->hidden * sizeof(float));
        if (!qwen_mm_cpu(s->shared_gate, s->post, &L->shared_gate_inp, m->hidden, 1, err, cap) ||
            !qwen_mm_cpu(s->gate, s->post, &L->shared_gate, m->hidden, m->shared_ff, err, cap) ||
            !qwen_mm_cpu(s->shared_up, s->post, &L->shared_up, m->hidden, m->shared_ff, err, cap)) return 0;
        for (int i = 0; i < m->shared_ff; ++i) s->gate[i] = coli_f32_silu(s->gate[i]) * s->shared_up[i];
        if (!qwen_mm_cpu(s->shared_out, s->gate, &L->shared_down,
                         m->shared_ff, m->hidden, err, cap)) return 0;
        const float sw = qwen_sigmoid(s->shared_gate[0]);
        for (int i = 0; i < m->hidden; ++i) s->moe[i] = sw * s->shared_out[i];

        for (int j = 0; j < nk; ++j) {
            const int e = s->top_idx[j];
            QwenExpertSlot *slot = qwen_expert_acquire(m, l, e, 1);
            if (!slot) return qwen_errf(err, cap, "Qwen expert admission failed at layer %d expert %d", l, e);
            if (!qwen_mm_cpu(s->gate, s->post, slot->gate, m->hidden, m->expert_ff, err, cap) ||
                !qwen_mm_cpu(s->up, s->post, slot->up, m->hidden, m->expert_ff, err, cap)) return 0;
            for (int i = 0; i < m->expert_ff; ++i) s->gate[i] = coli_f32_silu(s->gate[i]) * s->up[i];
            if (!qwen_mm_cpu(s->expert_out, s->gate, slot->down,
                             m->expert_ff, m->hidden, err, cap)) return 0;
            for (int i = 0; i < m->hidden; ++i) s->moe[i] += s->top_w[j] * s->expert_out[i];
        }
        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->moe[i];
    }

    if (!qwen_f32_vector(&m->output_norm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
    ColiTensor *outw = m->tied_output ? &m->token_embd : &m->output;
    return qwen_mm_cpu(s->logits, s->norm, outw, m->hidden, m->vocab, err, cap);
}

#ifdef COLI_CUDA
static const float *qwen_f32_device(QwenModel *m, ColiTensor *t, int n,
                                    char *err, size_t cap) {
    if (t->dtype != COLI_DTYPE_F32 || t->row_count != 1 || t->dims[0] != (uint64_t)n ||
        !coli_tensor_reside(&m->exec, t)) {
        qwen_errf(err, cap, "invalid resident Qwen F32 vector: %s", t->name ? t->name : "<unnamed>");
        return NULL;
    }
    return (const float *)coli_tensor_device_data(t);
}

static int qwen_mm_device(QwenModel *m, float *y, const float *x, ColiTensor *w,
                          char *err, size_t cap) {
    if (!coli_tensor_matmul_device(&m->exec, y, x, w, 1))
        return qwen_errf(err, cap, "Qwen CUDA matmul failed for %s", w->name ? w->name : "<unnamed>");
    return 1;
}

static int qwen_forward_cuda(QwenModel *m, QwenScratch *s, int token, int pos,
                             char *err, size_t cap) {
    const int dev = m->exec.device;
    if (token < 0 || token >= m->vocab || pos < 0 || pos >= m->context_capacity)
        return qwen_errf(err, cap, "invalid token/position");
    if (!coli_tensor_read_row_device(&m->exec, &m->token_embd, (uint64_t)token, s->dx, 1.0f))
        return qwen_errf(err, cap, "Qwen CUDA embedding decode failed");

    const size_t kv_layer_stride = (size_t)m->context_capacity * m->kv_dim;
    const size_t conv_layer_stride = (size_t)m->conv_dim * m->conv_kernel;
    const size_t state_layer_stride = (size_t)m->n_value_heads * m->state_size * m->value_head_dim;

    for (int l = 0; l < m->n_layers; ++l) {
        QwenLayer *L = &m->layers[l];
        const float *w = qwen_f32_device(m, &L->attn_norm, m->hidden, err, cap);
        if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dnorm, s->dx, w, 1, m->hidden, m->eps))
            return qwen_errf(err, cap, "Qwen CUDA input norm failed at layer %d", l);

        if (L->recurrent) {
            if (!qwen_mm_device(m, s->dqkv, s->dnorm, &L->qkv, err, cap) ||
                !qwen_mm_device(m, s->dz, s->dnorm, &L->z, err, cap) ||
                !qwen_mm_device(m, s->dba, s->dnorm, &L->ba, err, cap)) return 0;
            const float *conv_w = (const float *)coli_tensor_device_data(&L->conv1d);
            const float *dt = qwen_f32_device(m, &L->dt_bias, m->n_value_heads, err, cap);
            const float *a = qwen_f32_device(m, &L->a, m->n_value_heads, err, cap);
            const float *nw = qwen_f32_device(m, &L->ssm_norm, m->value_head_dim, err, cap);
            float *conv_state = m->conv_state_dev + (size_t)L->recurrent_index * conv_layer_stride;
            float *rec_state = m->recurrent_state_dev + (size_t)L->recurrent_index * state_layer_stride;
            if (!conv_w || !dt || !a || !nw ||
                !coli_cuda_pipe_gated_delta_decode(dev, s->ddelta, s->dqkv, s->dz, s->dba,
                    conv_w, dt, a, nw, conv_state, rec_state,
                    m->n_key_heads, m->n_value_heads, m->state_size,
                    m->conv_kernel, m->eps) ||
                !qwen_mm_device(m, s->dmixer, s->ddelta, &L->ssm_out, err, cap))
                return qwen_errf(err, cap, "Qwen CUDA DeltaNet failed at layer %d", l);
        } else {
            if (!qwen_mm_device(m, s->dqg, s->dnorm, &L->q, err, cap) ||
                !qwen_mm_device(m, s->dk, s->dnorm, &L->k, err, cap) ||
                !qwen_mm_device(m, s->dv, s->dnorm, &L->v, err, cap)) return 0;
            if (!coli_cuda_pipe_qg_split(dev, s->dattn_q, s->dattn_gate, s->dqg,
                                          m->n_heads, m->head_dim)) return 0;
            w = qwen_f32_device(m, &L->q_norm, m->head_dim, err, cap);
            if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dattn_q, s->dattn_q, w, m->n_heads, m->head_dim, m->eps)) return 0;
            w = qwen_f32_device(m, &L->k_norm, m->head_dim, err, cap);
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
                !qwen_mm_device(m, s->dmixer, s->dattn, &L->o, err, cap)) return 0;
        }

        if (!coli_cuda_pipe_add(dev, s->dx, s->dmixer, (size_t)m->hidden)) return 0;
        w = qwen_f32_device(m, &L->post_norm, m->hidden, err, cap);
        if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dpost, s->dx, w, 1, m->hidden, m->eps)) return 0;

        if (!qwen_mm_device(m, s->drouter, s->dpost, &L->router, err, cap) ||
            !coli_cuda_pipe_download(dev, s->drouter, s->router,
                                     (size_t)m->n_experts * sizeof(float))) return 0;
        const int nk = coli_f32_router_topk(s->router, m->n_experts, m->n_expert_used,
                                            s->top_idx, s->top_w);
        qwen_couple_prefetch(m, l, s->top_idx, nk);
        if (m->pilot && l + 1 < m->n_layers) {
            int pidx[64]; float pw[64];
            const int keep = m->pilot_k < 64 ? m->pilot_k : 64;
            if (!qwen_mm_device(m, s->drouter, s->dpost, &m->layers[l + 1].router, err, cap) ||
                !coli_cuda_pipe_download(dev, s->drouter, s->router,
                                         (size_t)m->n_experts * sizeof(float))) return 0;
            const int pn = coli_f32_router_topk(s->router, m->n_experts, keep, pidx, pw);
            qwen_prefetch_ids(m, l + 1, pidx, pn);
        }

        if (!coli_cuda_pipe_zero(dev, s->dmoe, (size_t)m->hidden) ||
            !qwen_mm_device(m, s->dshared_gate, s->dpost, &L->shared_gate_inp, err, cap) ||
            !qwen_mm_device(m, s->dgate, s->dpost, &L->shared_gate, err, cap) ||
            !qwen_mm_device(m, s->dshared_up, s->dpost, &L->shared_up, err, cap) ||
            !coli_cuda_pipe_silu_mul(dev, s->dgate, s->dshared_up, (size_t)m->shared_ff) ||
            !qwen_mm_device(m, s->dshared_out, s->dgate, &L->shared_down, err, cap) ||
            !coli_cuda_pipe_sigmoid_scale(dev, s->dshared_out, s->dshared_gate, (size_t)m->hidden) ||
            !coli_cuda_pipe_add(dev, s->dmoe, s->dshared_out, (size_t)m->hidden)) return 0;

        for (int j = 0; j < nk; ++j) {
            const int e = s->top_idx[j];
            QwenExpertSlot *slot = qwen_expert_acquire(m, l, e, 1);
            if (!slot) return qwen_errf(err, cap, "Qwen expert admission failed at layer %d expert %d", l, e);
            if (!qwen_mm_device(m, s->dgate, s->dpost, slot->gate, err, cap) ||
                !qwen_mm_device(m, s->dup, s->dpost, slot->up, err, cap) ||
                !coli_cuda_pipe_silu_mul(dev, s->dgate, s->dup, (size_t)m->expert_ff) ||
                !qwen_mm_device(m, s->dexpert_out, s->dgate, slot->down, err, cap) ||
                !coli_cuda_pipe_axpy(dev, s->dmoe, s->dexpert_out, s->top_w[j], (size_t)m->hidden)) return 0;
        }
        if (!coli_cuda_pipe_add(dev, s->dx, s->dmoe, (size_t)m->hidden)) return 0;
    }

    const float *w = qwen_f32_device(m, &m->output_norm, m->hidden, err, cap);
    if (!w || !coli_cuda_pipe_rmsnorm(dev, s->dnorm, s->dx, w, 1, m->hidden, m->eps)) return 0;
    ColiTensor *outw = m->tied_output ? &m->token_embd : &m->output;
    if (!qwen_mm_device(m, s->dlogits, s->dnorm, outw, err, cap) ||
        !coli_cuda_pipe_download(dev, s->dlogits, s->logits, (size_t)m->vocab * sizeof(float))) return 0;
    return 1;
}
#endif

static int qwen_forward(QwenModel *m, QwenScratch *s, int token, int pos,
                        char *err, size_t cap) {
    int ok;
#ifdef COLI_CUDA
    if (m->exec.kind == COLI_BACKEND_CUDA) ok = qwen_forward_cuda(m, s, token, pos, err, cap);
    else
#endif
    ok = qwen_forward_cpu(m, s, token, pos, err, cap);
    if (ok) qwen_repin(m);
    return ok;
}

static int qwen_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

static int qwen_load_model(QwenModel *m, const char *path, int context, int verbose,
                           ColiExec exec, char *err, size_t cap) {
    memset(m, 0, sizeof(*m));
    m->gguf.fd = -1; m->verbose = verbose; m->exec = exec;
    if (!coli_gguf_open(&m->gguf, path))
        return qwen_errf(err, cap, "cannot open GGUF: %s", coli_gguf_error(&m->gguf));
    if (!qwen_model_config(m, err, cap)) return 0;
    if (!coli_gguf_tokenizer_load(&m->tokenizer, &m->gguf, err, cap)) return 0;
    if (coli_gguf_tokenizer_vocab_size(m->tokenizer) != m->vocab)
        return qwen_errf(err, cap, "tokenizer/model vocabulary mismatch");

    const double t0 = qwen_now_sec();
    if (!qwen_model_load_weights(m, err, cap) || !qwen_build_expert_views(m, err, cap)) return 0;
    const double mapped = qwen_now_sec() - t0;
    const double t1 = qwen_now_sec();
    if (!qwen_model_reside_cuda(m, err, cap)) return 0;
    const int64_t history = qwen_usage_load(m, path);
    if (history > 0 && verbose)
        fprintf(stderr, "[USAGE] Qwen expert history: %lld selections\n", (long long)history);
    /* Allocate exact KV/recurrent state before sizing the expert tier so the
     * scheduler only budgets VRAM that truly remains. */
    if (!qwen_alloc_state(m, context, err, cap) || !qwen_scheduler_init(m, err, cap)) return 0;

    if (verbose) {
        fprintf(stderr,
            "[GGUF] qwen3next: layers=%d (%d DeltaNet + %d attention) hidden=%d heads=%d/%d experts=%d top=%d vocab=%d\n",
            m->n_layers, m->n_recurrent, m->n_attention, m->hidden,
            m->n_heads, m->n_kv_heads, m->n_experts, m->n_expert_used, m->vocab);
        if (exec.kind == COLI_BACKEND_CUDA)
            fprintf(stderr,
                "[GGUF] mapped %.2fs; dense residency %.2fs, %.2f MiB before expert cache; backend=cuda (Qwen3-Next + Colibri scheduler); context=%d\n",
                mapped, qwen_now_sec() - t1, m->cuda_dense_bytes / (1024.0 * 1024.0), context);
        else
            fprintf(stderr, "[GGUF] mapped %.2fs; backend=cpu reference; context=%d; threads=%d\n",
                    mapped, context, omp_get_max_threads());
    }
    return 1;
}

static void qwen_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--gguf] MODEL.gguf --prompt TEXT [--max-tokens N] [--device cpu|cuda[:N]] [--raw-prompt] [--verbose]\n",
        prog);
}

int coli_qwen3next_run_cli(int argc, char **argv) {
    const char *model_path = NULL, *prompt = NULL, *device_arg = "cpu";
    int max_tokens = 24, raw = 0, verbose = 0;
    int i = 1;
    if (i < argc && !strcmp(argv[i], "--gguf")) ++i;
    if (i < argc) model_path = argv[i++];
    for (; i < argc; ++i) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device_arg = argv[++i];
        else if (!strcmp(argv[i], "--raw-prompt")) raw = 1;
        else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown Qwen GGUF option: %s\n", argv[i]); qwen_usage(argv[0]); return 2; }
    }
    if (!model_path || !prompt || max_tokens < 0) { qwen_usage(argv[0]); return 2; }

    ColiGgufFile probe; probe.fd = -1;
    ColiGgufTokenizer *pt = NULL;
    char err[512];
    if (!coli_gguf_open(&probe, model_path) ||
        !coli_gguf_tokenizer_load(&pt, &probe, err, sizeof(err))) {
        fprintf(stderr, "%s\n", probe.fd >= 0 ? err : coli_gguf_error(&probe));
        coli_gguf_close(&probe); return 1;
    }
    char *formatted = raw ? strdup(prompt) : coli_gguf_tokenizer_format_qwen3next_prompt(pt, prompt);
    if (!formatted) {
        fprintf(stderr, "cannot format Qwen prompt\n");
        coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); return 1;
    }
    const size_t max_prompt = strlen(formatted) + 2;
    if (max_prompt > INT_MAX) { free(formatted); coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); return 1; }
    int *ids = (int *)malloc(max_prompt * sizeof(int));
    if (!ids) { free(formatted); coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); return 1; }
    const int n_prompt = coli_gguf_tokenizer_encode(pt, formatted, ids, (int)max_prompt);
    coli_gguf_tokenizer_destroy(pt); coli_gguf_close(&probe); free(formatted);
    if (n_prompt <= 0 || max_tokens > INT_MAX - n_prompt) {
        fprintf(stderr, "Qwen prompt tokenization/context failed\n"); free(ids); return 1;
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

    QwenModel m;
    QwenScratch s;
    memset(&s, 0, sizeof(s));
    if (!qwen_load_model(&m, model_path, n_prompt + max_tokens, verbose, exec, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); qwen_model_free(&m); free(ids);
#ifdef COLI_CUDA
        if (cuda_started) coli_cuda_shutdown();
#endif
        return 1;
    }
    if (!qwen_scratch_alloc(&m, &s, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); qwen_model_free(&m); free(ids);
#ifdef COLI_CUDA
        if (cuda_started) coli_cuda_shutdown();
#endif
        return 1;
    }

    const double t0 = qwen_now_sec();
    for (int p = 0; p < n_prompt; ++p)
        if (!qwen_forward(&m, &s, ids[p], p, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err); goto fail;
        }
    int pos = n_prompt, generated = 0;
    while (generated < max_tokens) {
        const int next = qwen_argmax(s.logits, m.vocab);
        if (next == coli_gguf_tokenizer_eos(m.tokenizer)) break;
        if (!coli_gguf_tokenizer_is_control(m.tokenizer, next)) {
            char piece[4096];
            const int n = coli_gguf_tokenizer_decode(m.tokenizer, &next, 1, piece, sizeof(piece));
            if (n > 0) { fwrite(piece, 1, (size_t)n, stdout); fflush(stdout); }
        }
        ++generated;
        if (generated >= max_tokens) break;
        if (!qwen_forward(&m, &s, next, pos++, err, sizeof(err))) {
            fprintf(stderr, "\n%s\n", err); goto fail;
        }
    }
    fputc('\n', stdout);
    if (verbose) {
        const double elapsed = qwen_now_sec() - t0;
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
    }
    qwen_usage_save(&m); qwen_scratch_free(&s); qwen_model_free(&m); free(ids);
#ifdef COLI_CUDA
    if (cuda_started) coli_cuda_shutdown();
#endif
    return 0;

fail:
    qwen_usage_save(&m); qwen_scratch_free(&s); qwen_model_free(&m); free(ids);
#ifdef COLI_CUDA
    if (cuda_started) coli_cuda_shutdown();
#endif
    return 1;
}
