#ifndef COLIBRI_F32_KERNELS_H
#define COLIBRI_F32_KERNELS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Shared CPU F32 kernels. Matrix storage follows Colibri/GGML:
 * W[O,I], x[S,I], y[S,O]. GGUF dimensions are [I,O,...]. */
static inline void coli_f32_matmul(float *y, const float *x, const float *W,
                                   int S, int I, int O) {
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; ++s) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.0f;
            for (int i = 0; i < I; ++i) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

static inline void coli_f32_rmsnorm(float *out, const float *x, const float *w,
                                    int n, float eps) {
    double sum2 = 0.0;
    for (int i = 0; i < n; ++i) sum2 += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(sum2 / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

static inline void coli_f32_softmax(float *x, int n) {
    if (n <= 0) return;
    float maxv = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > maxv) maxv = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    const float inv = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

static inline float coli_f32_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* GGML normal RoPE: adjacent pairs inside each head. */
static inline void coli_f32_rope(float *v, int n_heads, int head_dim,
                                 int rope_dims, int position, float freq_base) {
    if (rope_dims > head_dim) rope_dims = head_dim;
    rope_dims &= ~1;
    for (int h = 0; h < n_heads; ++h) {
        float *head = v + (int64_t)h * head_dim;
        for (int i = 0; i < rope_dims; i += 2) {
            const float theta = (float)position * powf(freq_base, -(float)i / rope_dims);
            const float c = cosf(theta), s = sinf(theta);
            const float a = head[i], b = head[i + 1];
            head[i]     = a * c - b * s;
            head[i + 1] = a * s + b * c;
        }
    }
}

/* Softmax across all router logits, select top-k, then renormalize selected mass.
 * Ties are stable: lower expert index wins. */
static inline int coli_f32_router_topk(const float *logits, int n, int k,
                                       int *indices, float *weights) {
    if (n <= 0 || k <= 0) return 0;
    if (k > n) k = n;
    float maxv = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > maxv) maxv = logits[i];

    for (int j = 0; j < k; ++j) {
        int best = -1;
        for (int i = 0; i < n; ++i) {
            int used = 0;
            for (int p = 0; p < j; ++p) if (indices[p] == i) { used = 1; break; }
            if (used) continue;
            if (best < 0 || logits[i] > logits[best] ||
                (logits[i] == logits[best] && i < best)) best = i;
        }
        indices[j] = best;
        weights[j] = expf(logits[best] - maxv);
    }
    double sum = 0.0;
    for (int j = 0; j < k; ++j) sum += weights[j];
    const float inv = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
    for (int j = 0; j < k; ++j) weights[j] *= inv;
    return k;
}

/* One-token causal GQA attention. K/V cache layout:
 * [position][kv_head][head_dim]. */
static inline void coli_f32_gqa_attention(float *out,
                                          const float *q,
                                          const float *k_cache,
                                          const float *v_cache,
                                          int position,
                                          int n_heads,
                                          int n_kv_heads,
                                          int head_dim,
                                          float score_scale,
                                          float *score_scratch) {
    const int n_pos = position + 1;
    for (int qh = 0; qh < n_heads; ++qh) {
        const int kvh = (int)((int64_t)qh * n_kv_heads / n_heads);
        const float *qv = q + (int64_t)qh * head_dim;
        float maxv = -INFINITY;
        for (int t = 0; t < n_pos; ++t) {
            const float *kv = k_cache + ((int64_t)t * n_kv_heads + kvh) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
            const float score = dot * score_scale;
            score_scratch[t] = score;
            if (score > maxv) maxv = score;
        }
        double sum = 0.0;
        for (int t = 0; t < n_pos; ++t) {
            score_scratch[t] = expf(score_scratch[t] - maxv);
            sum += score_scratch[t];
        }
        const float inv = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
        float *dst = out + (int64_t)qh * head_dim;
        memset(dst, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t < n_pos; ++t) {
            const float a = score_scratch[t] * inv;
            const float *vv = v_cache + ((int64_t)t * n_kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; ++d) dst[d] += a * vv[d];
        }
    }
}

#endif
