#include "gguf_gptoss.h"
#include "tensor.h"
#include "gguf_tokenizer.h"
#include "f32_kernels.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif

typedef struct {
    int eid;
    uint64_t age;
    int cuda_resident;
} GptOssExpertSlot;

typedef struct {
    ColiTensor attn_norm, post_attn_norm;
    ColiTensor q, k, v, q_bias, k_bias, v_bias, o, o_bias, sinks;
    ColiTensor router, router_bias;
    ColiTensor gate, up, down;
    ColiTensor gate_bias, up_bias, down_bias;
    ColiTensor *gate_expert, *up_expert, *down_expert;
    int expert_views;
    GptOssExpertSlot *cache;
    int cache_cap;
} GptOssLayer;

typedef struct {
    ColiGgufFile gguf;
    ColiGgufTokenizer *tokenizer;
    char *architecture;
    ColiExec exec;
    int verbose;

    int n_layers, hidden, vocab, context_train;
    int n_heads, n_kv_heads, head_dim, q_dim, kv_dim;
    int n_experts, top_k, expert_ff, sliding_window;
    float eps, attention_scale;
    float rope_base, rope_factor, yarn_beta_fast, yarn_beta_slow;
    int rope_original_context;
    float yarn_concentration;
    float *rope_inv_freq;

    ColiTensor token_embd, output_norm, output;
    GptOssLayer *layers;

    int context_capacity;
    float *k_cache, *v_cache;
    uint64_t cache_clock;
    int dense_streaming;
    size_t cuda_dense_bytes;
} GptOssModel;

typedef struct {
    float *x, *norm, *q, *k, *v, *attn, *proj;
    float *router, *gate, *up, *hidden, *expert_out, *moe;
    float *scores, *logits, *weight;
    float *q_bias, *k_bias, *v_bias, *attn_bias, *router_bias, *gate_bias, *up_bias, *down_bias, *sinks;
    int *top_idx;
    float *top_w;
} GptOssScratch;

static double now_sec(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int errf(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap);
    }
    return 0;
}

static int kv_u64_optional(const ColiGgufFile *g, const char *key, uint64_t *out) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    return kv && coli_gguf_kv_read_u64(g, kv, out);
}

static int kv_f32_optional(const ColiGgufFile *g, const char *key, float *out) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key); double v;
    if (!kv || !coli_gguf_kv_read_f64(g, kv, &v)) return 0;
    *out = (float)v; return 1;
}

static int get_i(const ColiGgufFile *g, const char *key, int fallback) {
    uint64_t v = 0;
    if (!kv_u64_optional(g, key, &v) || v > INT_MAX) return fallback;
    return (int)v;
}

static float get_f(const ColiGgufFile *g, const char *key, float fallback) {
    float v = fallback; (void)kv_f32_optional(g, key, &v); return v;
}

static void tensor_free(ColiTensor *t) { coli_tensor_destroy(t); }

static void layer_free(GptOssModel *m, GptOssLayer *l) {
    if (!l) return;
    if (l->cache) {
        for (int i = 0; i < l->cache_cap; ++i) {
            int e = l->cache[i].eid;
            if (e >= 0 && e < l->expert_views && l->cache[i].cuda_resident) {
                coli_tensor_release_backend(&m->exec, &l->gate_expert[e]);
                coli_tensor_release_backend(&m->exec, &l->up_expert[e]);
                coli_tensor_release_backend(&m->exec, &l->down_expert[e]);
            }
        }
    }
    if (l->gate_expert) {
        for (int e = 0; e < l->expert_views; ++e) {
            tensor_free(&l->gate_expert[e]);
            tensor_free(&l->up_expert[e]);
            tensor_free(&l->down_expert[e]);
        }
    }
    free(l->cache); free(l->gate_expert); free(l->up_expert); free(l->down_expert);
    tensor_free(&l->attn_norm); tensor_free(&l->post_attn_norm);
    tensor_free(&l->q); tensor_free(&l->k); tensor_free(&l->v);
    tensor_free(&l->q_bias); tensor_free(&l->k_bias); tensor_free(&l->v_bias);
    tensor_free(&l->o);
    tensor_free(&l->o_bias); tensor_free(&l->sinks);
    tensor_free(&l->router); tensor_free(&l->router_bias);
    tensor_free(&l->gate); tensor_free(&l->up); tensor_free(&l->down);
    tensor_free(&l->gate_bias); tensor_free(&l->up_bias); tensor_free(&l->down_bias);
}

static void model_free(GptOssModel *m) {
    if (!m) return;
    if (m->layers) for (int i = 0; i < m->n_layers; ++i) layer_free(m, &m->layers[i]);
    free(m->layers); free(m->k_cache); free(m->v_cache); free(m->rope_inv_freq);
    tensor_free(&m->token_embd); tensor_free(&m->output_norm); tensor_free(&m->output);
    free(m->architecture); coli_gguf_tokenizer_destroy(m->tokenizer); coli_gguf_close(&m->gguf);
    memset(m, 0, sizeof(*m));
}

static int tensor_dims(const ColiTensor *t, int nd, const uint64_t *dims) {
    if (!t || (int)t->n_dims != nd) return 0;
    for (int i = 0; i < nd; ++i) if (t->dims[i] != dims[i]) return 0;
    return 1;
}

static int load_named(GptOssModel *m, const char *name, ColiTensor *out,
                      int nd, const uint64_t *dims, char *err, size_t cap) {
    const ColiGgufTensorInfo *ti = coli_gguf_find_tensor(&m->gguf, name);
    if (!ti) return 0;
    if (!coli_tensor_bind_gguf(&m->gguf, ti, out, err, cap)) return 0;
    if (!tensor_dims(out, nd, dims)) {
        tensor_free(out);
        return errf(err, cap, "wrong dimensions for tensor %s", name);
    }
    return 1;
}

static int load_alias(GptOssModel *m, ColiTensor *out, int nd, const uint64_t *dims,
                      char *err, size_t cap, const char *a, const char *b, const char *c) {
    if (a && load_named(m, a, out, nd, dims, err, cap)) return 1;
    if (b && load_named(m, b, out, nd, dims, err, cap)) return 1;
    if (c && load_named(m, c, out, nd, dims, err, cap)) return 1;
    return errf(err, cap, "missing tensor (tried %s%s%s)", a ? a : "",
                b ? ", " : "", b ? b : (c ? c : ""));
}

static int load_layer_alias(GptOssModel *m, int layer, ColiTensor *out,
                            int nd, const uint64_t *dims, char *err, size_t cap,
                            const char *sa, const char *sb, const char *sc) {
    char a[160], b[160], c[160];
    const char *pa = NULL, *pb = NULL, *pc = NULL;
    if (sa) { snprintf(a, sizeof(a), "blk.%d.%s", layer, sa); pa = a; }
    if (sb) { snprintf(b, sizeof(b), "blk.%d.%s", layer, sb); pb = b; }
    if (sc) { snprintf(c, sizeof(c), "blk.%d.%s", layer, sc); pc = c; }
    return load_alias(m, out, nd, dims, err, cap, pa, pb, pc);
}

static int model_config(GptOssModel *m, char *err, size_t cap) {
    const ColiGgufKV *akv = coli_gguf_find_kv(&m->gguf, "general.architecture");
    if (!akv || !coli_gguf_kv_read_string(&m->gguf, akv, &m->architecture))
        return errf(err, cap, "general.architecture missing");
    if (strcmp(m->architecture, "gpt-oss") != 0)
        return errf(err, cap, "unsupported architecture %s", m->architecture);

    m->n_layers = get_i(&m->gguf, "gpt-oss.block_count", 36);
    m->hidden = get_i(&m->gguf, "gpt-oss.embedding_length", 2880);
    m->n_heads = get_i(&m->gguf, "gpt-oss.attention.head_count", 64);
    m->n_kv_heads = get_i(&m->gguf, "gpt-oss.attention.head_count_kv", 8);
    m->head_dim = get_i(&m->gguf, "gpt-oss.attention.key_length", 64);
    m->n_experts = get_i(&m->gguf, "gpt-oss.expert_count", 128);
    m->top_k = get_i(&m->gguf, "gpt-oss.expert_used_count", 4);
    m->expert_ff = get_i(&m->gguf, "gpt-oss.expert_feed_forward_length", 2880);
    m->context_train = get_i(&m->gguf, "gpt-oss.context_length", 131072);
    m->sliding_window = get_i(&m->gguf, "gpt-oss.attention.sliding_window", 128);
    m->eps = get_f(&m->gguf, "gpt-oss.attention.layer_norm_rms_epsilon", 1e-5f);
    m->rope_base = get_f(&m->gguf, "gpt-oss.rope.freq_base", 150000.0f);
    m->rope_factor = get_f(&m->gguf, "gpt-oss.rope.scaling.factor", 32.0f);
    m->rope_original_context = get_i(&m->gguf, "gpt-oss.rope.scaling.original_context_length", 4096);
    m->yarn_beta_fast = get_f(&m->gguf, "gpt-oss.rope.scaling.yarn_beta_fast", 32.0f);
    m->yarn_beta_slow = get_f(&m->gguf, "gpt-oss.rope.scaling.yarn_beta_slow", 1.0f);

    if (m->hidden <= 0 || m->head_dim <= 0 || m->n_heads <= 0 || m->n_kv_heads <= 0 ||
        m->n_heads % m->n_kv_heads || m->n_experts <= 0 || m->top_k <= 0 ||
        m->top_k > m->n_experts || (m->head_dim & 1))
        return errf(err, cap, "invalid GPT-OSS dimensions");
    m->q_dim = m->n_heads * m->head_dim;
    m->kv_dim = m->n_kv_heads * m->head_dim;
    m->attention_scale = 1.0f / sqrtf((float)m->head_dim);

    const ColiGgufTensorInfo *emb = coli_gguf_find_tensor(&m->gguf, "token_embd.weight");
    if (!emb || emb->n_dims != 2 || emb->dims[0] != (uint64_t)m->hidden || emb->dims[1] > INT_MAX)
        return errf(err, cap, "invalid token_embd.weight");
    m->vocab = (int)emb->dims[1];

    int half = m->head_dim / 2;
    m->rope_inv_freq = (float *)malloc((size_t)half * sizeof(float));
    if (!m->rope_inv_freq) return errf(err, cap, "out of memory allocating YaRN frequencies");
    m->yarn_concentration = m->rope_factor > 1.0f ? 0.1f * logf(m->rope_factor) + 1.0f : 1.0f;
    float low = 0.0f, high = 0.0f;
    if (m->rope_factor > 1.0f) {
        low = half * logf(m->rope_original_context / (m->yarn_beta_fast * 2.0f * (float)M_PI)) / logf(m->rope_base);
        high = half * logf(m->rope_original_context / (m->yarn_beta_slow * 2.0f * (float)M_PI)) / logf(m->rope_base);
    }
    for (int i = 0; i < half; ++i) {
        float freq = powf(m->rope_base, (float)(2 * i) / (float)m->head_dim);
        if (m->rope_factor > 1.0f) {
            float ramp = (i - low) / (high - low);
            if (ramp < 0.0f) ramp = 0.0f; if (ramp > 1.0f) ramp = 1.0f;
            float mask = 1.0f - ramp;
            m->rope_inv_freq[i] = (1.0f / (m->rope_factor * freq)) * (1.0f - mask) + (1.0f / freq) * mask;
        } else m->rope_inv_freq[i] = 1.0f / freq;
    }
    return 1;
}

static int model_load_weights(GptOssModel *m, char *err, size_t cap) {
    uint64_t d[3];
    d[0] = m->hidden; d[1] = m->vocab;
    if (!load_named(m, "token_embd.weight", &m->token_embd, 2, d, err, cap)) return 0;
    d[0] = m->hidden;
    if (!load_named(m, "output_norm.weight", &m->output_norm, 1, d, err, cap)) return 0;
    d[0] = m->hidden; d[1] = m->vocab;
    if (!load_named(m, "output.weight", &m->output, 2, d, err, cap)) return 0;

    m->layers = (GptOssLayer *)calloc((size_t)m->n_layers, sizeof(*m->layers));
    if (!m->layers) return errf(err, cap, "out of memory allocating GPT-OSS layers");
    for (int l = 0; l < m->n_layers; ++l) {
        GptOssLayer *x = &m->layers[l];
        d[0] = m->hidden;
        if (!load_layer_alias(m, l, &x->attn_norm, 1, d, err, cap,
                              "attn_norm.weight", "input_layernorm.weight", NULL)) return 0;
        if (!load_layer_alias(m, l, &x->post_attn_norm, 1, d, err, cap,
                              "post_attention_norm.weight", "attn_post_norm.weight", "ffn_norm.weight")) return 0;
        d[0] = m->hidden; d[1] = m->q_dim;
        if (!load_layer_alias(m, l, &x->q, 2, d, err, cap, "attn_q.weight", "attn_q_proj.weight", NULL)) return 0;
        d[0] = m->q_dim;
        if (!load_layer_alias(m, l, &x->q_bias, 1, d, err, cap, "attn_q.bias", "attn_q_proj.bias", NULL)) return 0;
        d[0] = m->hidden; d[1] = m->kv_dim;
        if (!load_layer_alias(m, l, &x->k, 2, d, err, cap, "attn_k.weight", "attn_k_proj.weight", NULL)) return 0;
        if (!load_layer_alias(m, l, &x->v, 2, d, err, cap, "attn_v.weight", "attn_v_proj.weight", NULL)) return 0;
        d[0] = m->kv_dim;
        if (!load_layer_alias(m, l, &x->k_bias, 1, d, err, cap, "attn_k.bias", "attn_k_proj.bias", NULL)) return 0;
        if (!load_layer_alias(m, l, &x->v_bias, 1, d, err, cap, "attn_v.bias", "attn_v_proj.bias", NULL)) return 0;
        d[0] = m->q_dim; d[1] = m->hidden;
        if (!load_layer_alias(m, l, &x->o, 2, d, err, cap, "attn_output.weight", "attn_o.weight", NULL)) return 0;
        d[0] = m->hidden;
        if (!load_layer_alias(m, l, &x->o_bias, 1, d, err, cap, "attn_output.bias", "attn_o.bias", NULL)) return 0;
        d[0] = m->n_heads;
        if (!load_layer_alias(m, l, &x->sinks, 1, d, err, cap, "attn_sinks.weight", "attn_sinks", NULL)) return 0;
        d[0] = m->hidden; d[1] = m->n_experts;
        if (!load_layer_alias(m, l, &x->router, 2, d, err, cap, "ffn_gate_inp.weight", "router.weight", NULL)) return 0;
        d[0] = m->n_experts;
        if (!load_layer_alias(m, l, &x->router_bias, 1, d, err, cap, "ffn_gate_inp.bias", "router.bias", NULL)) return 0;
        d[0] = m->hidden; d[1] = m->expert_ff; d[2] = m->n_experts;
        if (!load_layer_alias(m, l, &x->gate, 3, d, err, cap, "ffn_gate_exps.weight", NULL, NULL)) return 0;
        if (!load_layer_alias(m, l, &x->up, 3, d, err, cap, "ffn_up_exps.weight", NULL, NULL)) return 0;
        d[0] = m->expert_ff; d[1] = m->hidden; d[2] = m->n_experts;
        if (!load_layer_alias(m, l, &x->down, 3, d, err, cap, "ffn_down_exps.weight", NULL, NULL)) return 0;
        d[0] = m->expert_ff; d[1] = m->n_experts;
        if (!load_layer_alias(m, l, &x->gate_bias, 2, d, err, cap, "ffn_gate_exps.bias", NULL, NULL)) return 0;
        if (!load_layer_alias(m, l, &x->up_bias, 2, d, err, cap, "ffn_up_exps.bias", NULL, NULL)) return 0;
        d[0] = m->hidden; d[1] = m->n_experts;
        if (!load_layer_alias(m, l, &x->down_bias, 2, d, err, cap, "ffn_down_exps.bias", NULL, NULL)) return 0;
        if (m->verbose) fprintf(stderr, "[GPT-OSS] bind layer %d/%d\r", l + 1, m->n_layers);
    }
    if (m->verbose) fputc('\n', stderr);
    return 1;
}

static int model_build_expert_views(GptOssModel *m, char *err, size_t cap) {
    int cache_cap = 4;
    const char *env = getenv("GPTOSS_EXPERT_CACHE_PER_LAYER");
    if (env && atoi(env) > 0) cache_cap = atoi(env);
    if (cache_cap < m->top_k) cache_cap = m->top_k;
    if (cache_cap > m->n_experts) cache_cap = m->n_experts;
    for (int l = 0; l < m->n_layers; ++l) {
        GptOssLayer *x = &m->layers[l];
        x->gate_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->up_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->down_expert = (ColiTensor *)calloc((size_t)m->n_experts, sizeof(ColiTensor));
        x->cache = (GptOssExpertSlot *)calloc((size_t)cache_cap, sizeof(*x->cache));
        if (!x->gate_expert || !x->up_expert || !x->down_expert || !x->cache)
            return errf(err, cap, "out of memory allocating GPT-OSS expert views");
        x->cache_cap = cache_cap;
        for (int i = 0; i < cache_cap; ++i) x->cache[i].eid = -1;
        for (int e = 0; e < m->n_experts; ++e) {
            if (!coli_tensor_rows_view(&x->gate, (uint64_t)e * m->expert_ff, m->expert_ff, &x->gate_expert[e]) ||
                !coli_tensor_rows_view(&x->up, (uint64_t)e * m->expert_ff, m->expert_ff, &x->up_expert[e]) ||
                !coli_tensor_rows_view(&x->down, (uint64_t)e * m->hidden, m->hidden, &x->down_expert[e]))
                return errf(err, cap, "cannot create expert view layer=%d expert=%d", l, e);
            x->expert_views = e + 1;
        }
    }
    return 1;
}

static void release_dense_cuda(GptOssModel *m) {
    if (m->exec.kind != COLI_BACKEND_CUDA || !m->layers) return;
    for (int l = 0; l < m->n_layers; ++l) {
        GptOssLayer *x = &m->layers[l];
        coli_tensor_release_backend(&m->exec, &x->q); coli_tensor_release_backend(&m->exec, &x->k);
        coli_tensor_release_backend(&m->exec, &x->v); coli_tensor_release_backend(&m->exec, &x->o);
        coli_tensor_release_backend(&m->exec, &x->router);
    }
    m->cuda_dense_bytes = 0;
}

static void try_reside_dense_cuda(GptOssModel *m) {
    if (m->exec.kind != COLI_BACKEND_CUDA) return;
    size_t total = 0; int ok = 1;
    for (int l = 0; l < m->n_layers && ok; ++l) {
        GptOssLayer *x = &m->layers[l]; ColiTensor *a[] = {&x->q,&x->k,&x->v,&x->o,&x->router};
        for (size_t i = 0; i < sizeof(a)/sizeof(a[0]); ++i) {
            if (!coli_tensor_reside(&m->exec, a[i])) { ok = 0; break; }
            total += (size_t)a[i]->storage_bytes;
        }
    }
    if (!ok) {
        release_dense_cuda(m); m->dense_streaming = 1;
        if (m->verbose) fprintf(stderr, "[GPT-OSS] dense CUDA residency exceeds budget; streaming dense tensors\n");
    } else {
        m->cuda_dense_bytes = total;
        if (m->verbose) fprintf(stderr, "[GPT-OSS] dense attention/router resident: %.2f MiB\n", total/(1024.0*1024.0));
    }
}

static GptOssExpertSlot *expert_acquire(GptOssModel *m, int layer, int eid) {
    GptOssLayer *l = &m->layers[layer];
    for (int i = 0; i < l->cache_cap; ++i) if (l->cache[i].eid == eid) {
        GptOssExpertSlot *slot = &l->cache[i];
        slot->age = ++m->cache_clock;
        if (m->exec.kind == COLI_BACKEND_CUDA && !slot->cuda_resident) {
            if (coli_tensor_reside(&m->exec, &l->gate_expert[eid]) &&
                coli_tensor_reside(&m->exec, &l->up_expert[eid]) &&
                coli_tensor_reside(&m->exec, &l->down_expert[eid])) {
                slot->cuda_resident = 1;
            } else {
                coli_tensor_release_backend(&m->exec, &l->gate_expert[eid]);
                coli_tensor_release_backend(&m->exec, &l->up_expert[eid]);
                coli_tensor_release_backend(&m->exec, &l->down_expert[eid]);
            }
        }
        return slot;
    }
    int pick = 0;
    for (int i = 0; i < l->cache_cap; ++i) {
        if (l->cache[i].eid < 0) { pick = i; break; }
        if (l->cache[i].age < l->cache[pick].age) pick = i;
    }
    GptOssExpertSlot *slot = &l->cache[pick];
    if (slot->eid >= 0 && slot->cuda_resident) {
        int old = slot->eid;
        coli_tensor_release_backend(&m->exec, &l->gate_expert[old]);
        coli_tensor_release_backend(&m->exec, &l->up_expert[old]);
        coli_tensor_release_backend(&m->exec, &l->down_expert[old]);
    }
    slot->eid = eid; slot->age = ++m->cache_clock; slot->cuda_resident = 0;
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        if (coli_tensor_reside(&m->exec, &l->gate_expert[eid]) &&
            coli_tensor_reside(&m->exec, &l->up_expert[eid]) &&
            coli_tensor_reside(&m->exec, &l->down_expert[eid])) slot->cuda_resident = 1;
        else {
            coli_tensor_release_backend(&m->exec, &l->gate_expert[eid]);
            coli_tensor_release_backend(&m->exec, &l->up_expert[eid]);
            coli_tensor_release_backend(&m->exec, &l->down_expert[eid]);
        }
    }
    return slot;
}

static int alloc_cache(GptOssModel *m, int context, char *err, size_t cap) {
    if (context < 1 || context > m->context_train) return errf(err, cap, "invalid context %d", context);
    size_t n = (size_t)m->n_layers * (size_t)context;
    if (n > SIZE_MAX / (size_t)m->kv_dim || n * (size_t)m->kv_dim > SIZE_MAX / sizeof(float))
        return errf(err, cap, "KV cache overflow");
    n *= (size_t)m->kv_dim;
    m->k_cache = (float *)calloc(n, sizeof(float));
    m->v_cache = (float *)calloc(n, sizeof(float));
    if (!m->k_cache || !m->v_cache)
        return errf(err, cap, "out of memory allocating %.2f MiB KV cache", 2.0*n*sizeof(float)/(1024.0*1024.0));
    m->context_capacity = context; return 1;
}

static void scratch_free(GptOssScratch *s) {
    if (!s) return;
    free(s->x); free(s->norm); free(s->q); free(s->k); free(s->v); free(s->attn); free(s->proj);
    free(s->router); free(s->gate); free(s->up); free(s->hidden); free(s->expert_out); free(s->moe);
    free(s->scores); free(s->logits); free(s->weight);
    free(s->q_bias); free(s->k_bias); free(s->v_bias); free(s->attn_bias); free(s->router_bias);
    free(s->gate_bias); free(s->up_bias); free(s->down_bias); free(s->sinks); free(s->top_idx); free(s->top_w);
    memset(s, 0, sizeof(*s));
}

static int scratch_alloc(const GptOssModel *m, GptOssScratch *s, char *err, size_t cap) {
#define ALLOC(f,n,t) do { s->f=(t*)calloc((size_t)(n),sizeof(t)); if(!s->f){scratch_free(s);return errf(err,cap,"out of memory allocating GPT-OSS scratch");} } while(0)
    ALLOC(x,m->hidden,float); ALLOC(norm,m->hidden,float); ALLOC(q,m->q_dim,float);
    ALLOC(k,m->kv_dim,float); ALLOC(v,m->kv_dim,float); ALLOC(attn,m->q_dim,float); ALLOC(proj,m->hidden,float);
    ALLOC(router,m->n_experts,float); ALLOC(gate,m->expert_ff,float); ALLOC(up,m->expert_ff,float);
    ALLOC(hidden,m->expert_ff,float); ALLOC(expert_out,m->hidden,float); ALLOC(moe,m->hidden,float);
    ALLOC(scores,(size_t)m->n_heads*m->context_capacity,float); ALLOC(logits,m->vocab,float); ALLOC(weight,m->hidden,float);
    ALLOC(q_bias,m->q_dim,float); ALLOC(k_bias,m->kv_dim,float); ALLOC(v_bias,m->kv_dim,float);
    ALLOC(attn_bias,m->hidden,float); ALLOC(router_bias,m->n_experts,float);
    ALLOC(gate_bias,m->expert_ff,float); ALLOC(up_bias,m->expert_ff,float); ALLOC(down_bias,m->hidden,float);
    ALLOC(sinks,m->n_heads,float); ALLOC(top_idx,m->top_k,int); ALLOC(top_w,m->top_k,float);
#undef ALLOC
    return 1;
}

static int tensor_vector(const ColiTensor *t, float *dst, int n, char *err, size_t cap) {
    if (!t || t->row_count != 1 || t->dims[0] != (uint64_t)n ||
        !coli_tensor_read_row_f32(t, 0, dst, (uint64_t)n))
        return errf(err, cap, "failed reading vector %s", t && t->name ? t->name : "<unnamed>");
    return 1;
}

static int tensor_mm_auto(GptOssModel *m, float *y, const float *x, ColiTensor *t,
                          int I, int O, int transient, char *err, size_t cap) {
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        if (coli_tensor_matmul(&m->exec, y, x, t, 1, I, O)) {
            if (transient && m->dense_streaming) coli_tensor_release_backend(&m->exec, t);
            return 1;
        }
        coli_tensor_release_backend(&m->exec, t);
        ColiExec cpu = {COLI_BACKEND_CPU, 0};
        if (coli_tensor_matmul(&cpu, y, x, t, 1, I, O)) return 1;
    } else if (coli_tensor_matmul(&m->exec, y, x, t, 1, I, O)) return 1;
    return errf(err, cap, "matmul failed for %s", t->name ? t->name : "<unnamed>");
}

static int tensor_mm_cpu(float *y, const float *x, ColiTensor *t,
                         int I, int O, char *err, size_t cap) {
    ColiExec cpu = {COLI_BACKEND_CPU, 0};
    if (coli_tensor_matmul(&cpu, y, x, t, 1, I, O)) return 1;
    return errf(err, cap, "CPU matmul failed for %s", t->name ? t->name : "<unnamed>");
}

float coli_gptoss_oai_swiglu(float gate, float linear) {
    if (gate > 7.0f) gate = 7.0f;
    if (linear > 7.0f) linear = 7.0f;
    if (linear < -7.0f) linear = -7.0f;
    return (gate / (1.0f + expf(-1.702f * gate))) * (linear + 1.0f);
}

void coli_gptoss_yarn_rotate(float *vector, int heads, int head_dim, int position,
                             const float *inv_freq, float concentration) {
    if (!vector || !inv_freq || heads < 1 || head_dim < 2 || (head_dim & 1)) return;
    int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        float *v = vector + (size_t)h * head_dim;
        for (int i = 0; i < half; ++i) {
            float a = position * inv_freq[i];
            float c = cosf(a) * concentration, sn = sinf(a) * concentration;
            float x1 = v[i], x2 = v[i + half];
            v[i] = x1 * c - x2 * sn;
            v[i + half] = x2 * c + x1 * sn;
        }
    }
}

void coli_gptoss_attention_sink_reference(float *out, const float *q,
                                           const float *k_cache, const float *v_cache,
                                           const float *sinks, float *scores,
                                           int pos, int window, int n_heads,
                                           int n_kv_heads, int head_dim, float scale) {
    int qmul = n_heads / n_kv_heads;
#pragma omp parallel for schedule(static)
    for (int qh = 0; qh < n_heads; ++qh) {
        int kvh = qh / qmul;
        int first = window > 0 && pos + 1 > window ? pos + 1 - window : 0;
        const float *qq = q + (size_t)qh * head_dim;
        float mx = sinks[qh];
        float *local_scores = scores + (size_t)qh * (pos + 1);
        for (int t = first; t <= pos; ++t) {
            const float *kk = k_cache + ((size_t)t * n_kv_heads + kvh) * head_dim;
            double sum = 0.0; for (int d = 0; d < head_dim; ++d) sum += (double)qq[d] * kk[d];
            float sc = (float)sum * scale; local_scores[t] = sc; if (sc > mx) mx = sc;
        }
        double den = exp((double)sinks[qh] - mx);
        for (int t = first; t <= pos; ++t) den += exp((double)local_scores[t] - mx);
        float *oo = out + (size_t)qh * head_dim;
        memset(oo, 0, (size_t)head_dim * sizeof(float));
        for (int t = first; t <= pos; ++t) {
            float a = (float)(exp((double)local_scores[t] - mx) / den);
            const float *vv = v_cache + ((size_t)t * n_kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; ++d) oo[d] += a * vv[d];
        }
    }
}

int coli_gptoss_topk_softmax(const float *logits, int n, int k, int *idx, float *weights) {
    if (!logits || !idx || !weights || n < 1 || k < 1 || k > n) return 0;
    for (int j = 0; j < k; ++j) { idx[j] = -1; weights[j] = -INFINITY; }
    for (int e = 0; e < n; ++e) {
        float v = logits[e]; int p = k;
        while (p > 0 && (idx[p-1] < 0 || v > weights[p-1])) --p;
        if (p < k) {
            for (int j = k - 1; j > p; --j) { weights[j] = weights[j-1]; idx[j] = idx[j-1]; }
            weights[p] = v; idx[p] = e;
        }
    }
    float mx = weights[0], sum = 0.0f;
    for (int j = 0; j < k; ++j) { weights[j] = expf(weights[j] - mx); sum += weights[j]; }
    for (int j = 0; j < k; ++j) weights[j] /= sum;
    return k;
}

static int model_forward(GptOssModel *m, GptOssScratch *s, int token, int pos,
                         char *err, size_t cap) {
    if (token < 0 || token >= m->vocab || pos < 0 || pos >= m->context_capacity)
        return errf(err, cap, "token/position outside model bounds");
    if (!coli_tensor_read_row_f32(&m->token_embd, (uint64_t)token, s->x, (uint64_t)m->hidden))
        return errf(err, cap, "embedding row decode failed");

    for (int l = 0; l < m->n_layers; ++l) {
        GptOssLayer *L = &m->layers[l];
        if (!tensor_vector(&L->attn_norm, s->weight, m->hidden, err, cap)) return 0;
        coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
        if (!tensor_mm_auto(m, s->q, s->norm, &L->q, m->hidden, m->q_dim, 1, err, cap) ||
            !tensor_mm_auto(m, s->k, s->norm, &L->k, m->hidden, m->kv_dim, 1, err, cap) ||
            !tensor_mm_auto(m, s->v, s->norm, &L->v, m->hidden, m->kv_dim, 1, err, cap) ||
            !tensor_vector(&L->q_bias, s->q_bias, m->q_dim, err, cap) ||
            !tensor_vector(&L->k_bias, s->k_bias, m->kv_dim, err, cap) ||
            !tensor_vector(&L->v_bias, s->v_bias, m->kv_dim, err, cap)) return 0;
        for (int i = 0; i < m->q_dim; ++i) s->q[i] += s->q_bias[i];
        for (int i = 0; i < m->kv_dim; ++i) { s->k[i] += s->k_bias[i]; s->v[i] += s->v_bias[i]; }
        coli_gptoss_yarn_rotate(s->q, m->n_heads, m->head_dim, pos, m->rope_inv_freq, m->yarn_concentration);
        coli_gptoss_yarn_rotate(s->k, m->n_kv_heads, m->head_dim, pos, m->rope_inv_freq, m->yarn_concentration);
        size_t stride = (size_t)m->context_capacity * m->kv_dim;
        float *kc = m->k_cache + (size_t)l * stride, *vc = m->v_cache + (size_t)l * stride;
        memcpy(kc + (size_t)pos * m->kv_dim, s->k, (size_t)m->kv_dim * sizeof(float));
        memcpy(vc + (size_t)pos * m->kv_dim, s->v, (size_t)m->kv_dim * sizeof(float));
        if (!tensor_vector(&L->sinks, s->sinks, m->n_heads, err, cap)) return 0;
        int window = (l % 2 == 0) ? m->sliding_window : 0;
        coli_gptoss_attention_sink_reference(
            s->attn, s->q, kc, vc, s->sinks, s->scores, pos, window,
            m->n_heads, m->n_kv_heads, m->head_dim, m->attention_scale);
        if (!tensor_mm_auto(m, s->proj, s->attn, &L->o, m->q_dim, m->hidden, 1, err, cap) ||
            !tensor_vector(&L->o_bias, s->attn_bias, m->hidden, err, cap)) return 0;
        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->proj[i] + s->attn_bias[i];

        if (!tensor_vector(&L->post_attn_norm, s->weight, m->hidden, err, cap)) return 0;
        coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
        if (!tensor_mm_auto(m, s->router, s->norm, &L->router, m->hidden, m->n_experts, 1, err, cap) ||
            !tensor_vector(&L->router_bias, s->router_bias, m->n_experts, err, cap)) return 0;
        for (int e = 0; e < m->n_experts; ++e) s->router[e] += s->router_bias[e];
        int nk = coli_gptoss_topk_softmax(
            s->router, m->n_experts, m->top_k, s->top_idx, s->top_w);
        if (nk != m->top_k) return errf(err, cap, "invalid GPT-OSS router selection");
        memset(s->moe, 0, (size_t)m->hidden * sizeof(float));
        for (int j = 0; j < nk; ++j) {
            int e = s->top_idx[j];
            GptOssExpertSlot *slot = expert_acquire(m, l, e);
            const int use_cuda = m->exec.kind == COLI_BACKEND_CUDA && slot && slot->cuda_resident;
            int gate_ok = use_cuda
                ? tensor_mm_auto(m, s->gate, s->norm, &L->gate_expert[e], m->hidden, m->expert_ff, 0, err, cap)
                : tensor_mm_cpu(s->gate, s->norm, &L->gate_expert[e], m->hidden, m->expert_ff, err, cap);
            int up_ok = use_cuda
                ? tensor_mm_auto(m, s->up, s->norm, &L->up_expert[e], m->hidden, m->expert_ff, 0, err, cap)
                : tensor_mm_cpu(s->up, s->norm, &L->up_expert[e], m->hidden, m->expert_ff, err, cap);
            if (!gate_ok || !up_ok ||
                !coli_tensor_read_row_f32(&L->gate_bias, (uint64_t)e, s->gate_bias, m->expert_ff) ||
                !coli_tensor_read_row_f32(&L->up_bias, (uint64_t)e, s->up_bias, m->expert_ff))
                return errf(err, cap, "expert input projection/bias failed layer=%d expert=%d", l, e);
            for (int i = 0; i < m->expert_ff; ++i)
                s->hidden[i] = coli_gptoss_oai_swiglu(s->gate[i] + s->gate_bias[i], s->up[i] + s->up_bias[i]);
            int down_ok = use_cuda
                ? tensor_mm_auto(m, s->expert_out, s->hidden, &L->down_expert[e], m->expert_ff, m->hidden, 0, err, cap)
                : tensor_mm_cpu(s->expert_out, s->hidden, &L->down_expert[e], m->expert_ff, m->hidden, err, cap);
            if (!down_ok ||
                !coli_tensor_read_row_f32(&L->down_bias, (uint64_t)e, s->down_bias, m->hidden))
                return errf(err, cap, "expert down projection/bias failed layer=%d expert=%d", l, e);
            float rw = s->top_w[j];
            for (int i = 0; i < m->hidden; ++i) s->moe[i] += rw * (s->expert_out[i] + s->down_bias[i]);
        }
        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->moe[i];
    }
    if (!tensor_vector(&m->output_norm, s->weight, m->hidden, err, cap)) return 0;
    coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
    ColiExec cpu = {COLI_BACKEND_CPU, 0};
    if (!coli_tensor_matmul(&cpu, s->logits, s->norm, &m->output, 1, m->hidden, m->vocab))
        return errf(err, cap, "output projection failed");
    return 1;
}

static int load_model(GptOssModel *m, const char *path, int context, int verbose,
                      ColiExec exec, char *err, size_t cap) {
    memset(m, 0, sizeof(*m)); m->gguf.fd = -1; m->verbose = verbose; m->exec = exec;
    if (!coli_gguf_open(&m->gguf, path)) return errf(err, cap, "cannot open model: %s", coli_gguf_error(&m->gguf));
    if (!model_config(m, err, cap) || !coli_gguf_tokenizer_load(&m->tokenizer, &m->gguf, err, cap) ||
        !model_load_weights(m, err, cap) || !model_build_expert_views(m, err, cap) ||
        !alloc_cache(m, context, err, cap)) return 0;
    if (coli_gguf_tokenizer_vocab_size(m->tokenizer) != m->vocab)
        return errf(err, cap, "tokenizer/model vocabulary mismatch");
    try_reside_dense_cuda(m);
    if (verbose) fprintf(stderr,
        "[GPT-OSS] layers=%d hidden=%d q=%d kv=%d heads=%d/%d experts=%d top=%d ff=%d vocab=%d context=%d backend=%s\n",
        m->n_layers,m->hidden,m->q_dim,m->kv_dim,m->n_heads,m->n_kv_heads,m->n_experts,m->top_k,
        m->expert_ff,m->vocab,context,exec.kind==COLI_BACKEND_CUDA?"cuda-hybrid":"cpu");
    return 1;
}

static char *format_harmony(const char *user) {
    static const char prefix[] = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n<|end|><|start|>user<|message|>";
    static const char suffix[] = "<|end|><|start|>assistant<|channel|>analysis<|message|>";
    size_t a = sizeof(prefix)-1, b = strlen(user), c = sizeof(suffix)-1;
    if (a > SIZE_MAX-b || a+b > SIZE_MAX-c-1) return NULL;
    char *out = (char *)malloc(a+b+c+1); if (!out) return NULL;
    memcpy(out,prefix,a); memcpy(out+a,user,b); memcpy(out+a+b,suffix,c); out[a+b+c]=0; return out;
}

static int argmax(const float *x, int n) { int b=0; for(int i=1;i<n;++i) if(x[i]>x[b]) b=i; return b; }

static void usage(const char *p) {
    fprintf(stderr,"Usage: %s [--gguf] MODEL --prompt TEXT [--max-tokens N] [--context N] [--device cpu|cuda[:N]] [--raw-prompt] [--verbose]\n",p);
}

int coli_gptoss_run_cli(int argc, char **argv) {
    const char *model_path=NULL,*prompt=NULL,*device_arg="cpu"; int max_tokens=24,context=0,raw=0,verbose=0;
    int i=1; if(i<argc&&!strcmp(argv[i],"--gguf"))++i; if(i<argc)model_path=argv[i++];
    for(;i<argc;++i){
        if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"--max-tokens")&&i+1<argc)max_tokens=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--context")&&i+1<argc)context=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--device")&&i+1<argc)device_arg=argv[++i];
        else if(!strcmp(argv[i],"--raw-prompt"))raw=1;
        else if(!strcmp(argv[i],"--verbose")||!strcmp(argv[i],"-v"))verbose=1;
        else if(!strcmp(argv[i],"--mtp-draft")&&i+1<argc)++i;
        else if(!strcmp(argv[i],"--no-mtp")){}
        else {fprintf(stderr,"unknown GPT-OSS option: %s\n",argv[i]);usage(argv[0]);return 2;}
    }
    if(!model_path||!prompt||max_tokens<0){usage(argv[0]);return 2;}
    ColiGgufFile probe;probe.fd=-1;ColiGgufTokenizer*tok=NULL;char err[512];
    if(!coli_gguf_open(&probe,model_path)||!coli_gguf_tokenizer_load(&tok,&probe,err,sizeof(err))){
        fprintf(stderr,"%s\n",probe.fd>=0?err:coli_gguf_error(&probe));coli_gguf_close(&probe);return 1;}
    char *formatted=raw?strdup(prompt):format_harmony(prompt);if(!formatted){fprintf(stderr,"cannot format prompt\n");return 1;}
    size_t capids=strlen(formatted)+4;int *ids=(int*)malloc(capids*sizeof(int));
    int n_prompt=ids?coli_gguf_tokenizer_encode(tok,formatted,ids,(int)capids):0;
    free(formatted);coli_gguf_tokenizer_destroy(tok);coli_gguf_close(&probe);
    if(n_prompt<=0){free(ids);fprintf(stderr,"prompt tokenization failed\n");return 1;}
    if(!context)context=n_prompt+max_tokens+1;if(context<n_prompt+max_tokens)context=n_prompt+max_tokens+1;

    ColiExec exec={COLI_BACKEND_CPU,0};int cuda_started=0; (void)cuda_started;
    if(!strcmp(device_arg,"cuda")||!strncmp(device_arg,"cuda:",5)){
        exec.kind=COLI_BACKEND_CUDA;if(device_arg[4]==':')exec.device=atoi(device_arg+5);
#ifdef COLI_CUDA
        if(!coli_cuda_init(&exec.device,1)){fprintf(stderr,"cannot initialize CUDA\n");free(ids);return 1;}cuda_started=1;
#else
        fprintf(stderr,"binary built without CUDA\n");free(ids);return 1;
#endif
    }else if(strcmp(device_arg,"cpu")){fprintf(stderr,"invalid device %s\n",device_arg);free(ids);return 2;}

    GptOssModel m;GptOssScratch s={0};double t0=now_sec();
    if(!load_model(&m,model_path,context,verbose,exec,err,sizeof(err))||!scratch_alloc(&m,&s,err,sizeof(err))){
        fprintf(stderr,"%s\n",err);scratch_free(&s);model_free(&m);free(ids);
#ifdef COLI_CUDA
        if(cuda_started)coli_cuda_shutdown();
#endif
        return 1;
    }
    for(int p=0;p<n_prompt;++p)if(!model_forward(&m,&s,ids[p],p,err,sizeof(err))){fprintf(stderr,"%s\n",err);goto fail;}
    int pos=n_prompt,generated=0;int stop_return=coli_gguf_tokenizer_id(m.tokenizer,"<|return|>");
    int stop_end=coli_gguf_tokenizer_id(m.tokenizer,"<|end|>");
    while(generated<max_tokens){
        int next=argmax(s.logits,m.vocab);
        if(next==coli_gguf_tokenizer_eos(m.tokenizer)||next==stop_return||next==stop_end)break;
        if(!coli_gguf_tokenizer_is_control(m.tokenizer,next)){char piece[4096];int n=coli_gguf_tokenizer_decode(m.tokenizer,&next,1,piece,sizeof(piece));if(n>0){fwrite(piece,1,(size_t)n,stdout);fflush(stdout);}}
        ++generated;if(generated>=max_tokens)break;
        if(!model_forward(&m,&s,next,pos++,err,sizeof(err))){fprintf(stderr,"\n%s\n",err);goto fail;}
    }
    fputc('\n',stdout);if(verbose)fprintf(stderr,"[GPT-OSS] prompt=%d generated=%d elapsed=%.2fs %.3f tok/s total\n",n_prompt,generated,now_sec()-t0,(n_prompt+generated)/(now_sec()-t0));
    scratch_free(&s);model_free(&m);free(ids);
#ifdef COLI_CUDA
    if(cuda_started)coli_cuda_shutdown();
#endif
    return 0;
fail:
    scratch_free(&s);model_free(&m);free(ids);
#ifdef COLI_CUDA
    if(cuda_started)coli_cuda_shutdown();
#endif
    return 1;
}
