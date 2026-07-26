#include "gguf_gptoss.h"
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* GPTOSS_DEBUG_LIGHT_V1 */
/* GPTOSS_DEFAULT_PROGRESS_V2 */
/* GPTOSS_PROGRESS_FIX_V3 */
/* GPTOSS_PLAIN_PROGRESS_V4 */
#include <string.h>
#include <time.h>
#include <strings.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <direct.h>
#include <process.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif

typedef struct GptOssDebugState GptOssDebugState;

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
    const ColiGgufTensorInfo *gate_info, *up_info, *down_info;
    ColiTensor gate_bias, up_bias, down_bias;
    ColiTensor *gate_expert, *up_expert, *down_expert;
    int expert_views;
    uint32_t *usage;
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
    char usage_path[PATH_MAX];
    int64_t usage_history;
    GptOssDebugState *debug;
} GptOssModel;

typedef struct {
    float *x, *norm, *q, *k, *v, *attn, *proj;
    float *router, *gate, *up, *hidden, *expert_out, *moe;
    float *scores, *logits, *weight;
    float *q_bias, *k_bias, *v_bias, *attn_bias, *router_bias, *gate_bias, *up_bias, *down_bias, *sinks;
    int *top_idx;
    float *top_w;
} GptOssScratch;

typedef struct {
    int eid;
    float weight;
    int cache_hit;
    int cache_slot;
    int evicted_eid;
    int resident_before;
    int resident_after;
    int ran_cuda;
    size_t copied_bytes;
    double copy_ms;
    double run_ms;
    double free_ms;
    double alloc_ms;
} GptOssDebugExpertRecord;

typedef struct {
    int layer;
    int selected_count;
    int experts_ran;
    int cuda_ran;
    int cpu_ran;
    int cache_hits;
    int cache_misses;
    int evictions;
    int cuda_resident_before;
    int cuda_resident_after;
    double copy_ms;
    double run_ms;
    double free_ms;
    double alloc_ms;
    double total_ms;
    GptOssDebugExpertRecord *experts;
} GptOssDebugLayerRecord;

typedef struct {
    uint64_t seq;
    int pos;
    int token;
    int is_prompt;
    int ok;
    double begin_wall;
    double total_ms;
    double copy_ms;
    double run_ms;
    double free_ms;
    double alloc_ms;
    GptOssDebugLayerRecord *layers;
    GptOssDebugExpertRecord *experts;
} GptOssDebugTokenRecord;

struct GptOssDebugState {
    int enabled;
    char dir[PATH_MAX];
    FILE *run_log;
    FILE *token_log;
    char token_log_path[PATH_MAX];
    double start_wall;
    uint64_t event_seq;
    GptOssDebugTokenRecord *tokens;
    size_t token_count;
    size_t token_cap;
    GptOssDebugTokenRecord *current;
    double total_copy_ms;
    double total_run_ms;
    double total_free_ms;
    double total_alloc_ms;
    double total_log_ms;
    uint64_t copy_ops;
    uint64_t run_ops;
    uint64_t free_ops;
    uint64_t alloc_ops;
    int exit_status;
};

static int errf(char *err, size_t cap, const char *fmt, ...);

static double now_sec(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int gptoss_debug_mkdir_one(const char *path) {
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0755) == 0 || errno == EEXIST;
#endif
}

static int gptoss_debug_mkdir_p(const char *path) {
    if (!path || !*path) return 0;
    char tmp[PATH_MAX];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return 0;
    memcpy(tmp, path, n + 1);
    for (char *q = tmp + 1; *q; ++q) {
        if (*q != '/' && *q != '\\') continue;
        char save = *q;
        *q = 0;
        if (*tmp && !gptoss_debug_mkdir_one(tmp)) return 0;
        *q = save;
    }
    return gptoss_debug_mkdir_one(tmp);
}

static long gptoss_debug_pid(void) {
#ifdef _WIN32
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

static void gptoss_debug_default_dir(char *out, size_t cap) {
    time_t now = time(NULL);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    snprintf(out, cap, "colibri-debug-%04d%02d%02d-%02d%02d%02d-%ld",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec, gptoss_debug_pid());
}

static const char *gptoss_debug_phase(const GptOssDebugTokenRecord *t) {
    return t && t->is_prompt ? "prompt" : "generated";
}

static void gptoss_debug_emit(GptOssModel *m, const char *fmt, ...) {
    if (!m || !m->debug || !m->debug->enabled) return;
    GptOssDebugState *d = m->debug;
    char body[8192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);
    double t0 = now_sec();
    double rel_ms = (t0 - d->start_wall) * 1000.0;
    uint64_t seq = ++d->event_seq;
    fprintf(stderr, "DBG|seq=%llu|t_ms=%.3f|%s\n",
            (unsigned long long)seq, rel_ms, body);
    fflush(stderr);
    if (d->run_log) {
        fprintf(d->run_log, "DBG|seq=%llu|t_ms=%.3f|%s\n",
                (unsigned long long)seq, rel_ms, body);
        fflush(d->run_log);
    }
    if (d->token_log) {
        fprintf(d->token_log, "DBG|seq=%llu|t_ms=%.3f|%s\n",
                (unsigned long long)seq, rel_ms, body);
        fflush(d->token_log);
    }
    d->total_log_ms += (now_sec() - t0) * 1000.0;
}

static int gptoss_debug_layer_resident(const GptOssLayer *l) {
    if (!l || !l->cache) return 0;
    int n = 0;
    for (int i = 0; i < l->cache_cap; ++i)
        if (l->cache[i].eid >= 0 && l->cache[i].cuda_resident) ++n;
    return n;
}

static GptOssDebugLayerRecord *gptoss_debug_layer(GptOssModel *m, int layer) {
    if (!m || !m->debug || !m->debug->current || layer < 0 || layer >= m->n_layers) return NULL;
    return &m->debug->current->layers[layer];
}

static GptOssDebugExpertRecord *gptoss_debug_expert(GptOssModel *m, int layer, int rank) {
    GptOssDebugLayerRecord *lr = gptoss_debug_layer(m, layer);
    if (!lr || rank < 0 || rank >= m->top_k) return NULL;
    return &lr->experts[rank];
}

static double gptoss_debug_clock(GptOssModel *m) {
    return m && m->debug && m->debug->enabled ? now_sec() : 0.0;
}

static void gptoss_debug_op(GptOssModel *m, int layer, int expert,
                            const char *klass, const char *name,
                            const char *backend, size_t bytes, double ms) {
    if (!m || !m->debug || !m->debug->enabled) return;
    GptOssDebugState *d = m->debug;
    GptOssDebugTokenRecord *tr = d->current;
    GptOssDebugLayerRecord *lr = gptoss_debug_layer(m, layer);
    GptOssDebugExpertRecord *er = NULL;
    if (lr && expert >= 0) {
        for (int i = 0; i < lr->selected_count && i < m->top_k; ++i)
            if (lr->experts[i].eid == expert) { er = &lr->experts[i]; break; }
    }
    double *dt = NULL, *tt = NULL, *lt = NULL, *et = NULL;
    uint64_t *count = NULL;
    if (!strcmp(klass, "copy")) {
        dt = &d->total_copy_ms; count = &d->copy_ops;
        if (tr) tt = &tr->copy_ms; if (lr) lt = &lr->copy_ms; if (er) et = &er->copy_ms;
        if (er) er->copied_bytes += bytes;
    } else if (!strcmp(klass, "run")) {
        dt = &d->total_run_ms; count = &d->run_ops;
        if (tr) tt = &tr->run_ms; if (lr) lt = &lr->run_ms; if (er) et = &er->run_ms;
    } else if (!strcmp(klass, "free")) {
        dt = &d->total_free_ms; count = &d->free_ops;
        if (tr) tt = &tr->free_ms; if (lr) lt = &lr->free_ms; if (er) et = &er->free_ms;
    } else if (!strcmp(klass, "alloc")) {
        dt = &d->total_alloc_ms; count = &d->alloc_ops;
        if (tr) tt = &tr->alloc_ms; if (lr) lt = &lr->alloc_ms; if (er) et = &er->alloc_ms;
    }
    if (dt) *dt += ms;
    if (tt) *tt += ms;
    if (lt) *lt += ms;
    if (et) *et += ms;
    if (count) ++*count;
    gptoss_debug_emit(m,
        "event=op|phase=%s|token_seq=%lld|pos=%d|token=%d|layer=%d|expert=%d|class=%s|name=%s|backend=%s|bytes=%zu|ms=%.3f",
        tr ? gptoss_debug_phase(tr) : (d->token_count ? "between_tokens" : "startup"),
        tr ? (long long)tr->seq : -1LL,
        tr ? tr->pos : -1, tr ? tr->token : -1,
        layer, expert, klass, name, backend ? backend : "none", bytes, ms);
}

static int gptoss_debug_init(GptOssModel *m, int enabled, const char *requested_dir,
                             const char *model_path, char *err, size_t cap) {
    if (!enabled) return 1;
    GptOssDebugState *d = (GptOssDebugState *)calloc(1, sizeof(*d));
    if (!d) return errf(err, cap, "out of memory allocating GPT-OSS debug state");
    d->enabled = 1;
    d->exit_status = 1;
    d->start_wall = now_sec();
    if (requested_dir && *requested_dir) {
        if (strlen(requested_dir) >= sizeof(d->dir)) { free(d); return errf(err, cap, "debug directory path too long"); }
        strcpy(d->dir, requested_dir);
    } else {
        gptoss_debug_default_dir(d->dir, sizeof(d->dir));
    }
    if (!gptoss_debug_mkdir_p(d->dir)) {
        int saved = errno;
        free(d);
        return errf(err, cap, "cannot create debug directory: %s", strerror(saved));
    }
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/run.log", d->dir) >= (int)sizeof(path)) {
        free(d); return errf(err, cap, "debug run.log path too long");
    }
    d->run_log = fopen(path, "w");
    if (!d->run_log) {
        int saved = errno; free(d); return errf(err, cap, "cannot open debug run.log: %s", strerror(saved));
    }
    setvbuf(d->run_log, NULL, _IOLBF, 0);
    m->debug = d;
    gptoss_debug_emit(m, "event=debug_start|dir=%s|model=%s|pid=%ld", d->dir,
                      model_path ? model_path : "", gptoss_debug_pid());
    fprintf(stderr, "[DEBUG] GPT-OSS logs: %s\n", d->dir);
    fflush(stderr);
    return 1;
}

static int gptoss_debug_begin_token(GptOssModel *m, int token, int pos, int is_prompt) {
    if (!m || !m->debug || !m->debug->enabled) return 1;
    GptOssDebugState *d = m->debug;
    double alloc_begin = now_sec();
    if (d->token_log) { fclose(d->token_log); d->token_log = NULL; }
    if (d->token_count == d->token_cap) {
        size_t nc = d->token_cap ? d->token_cap * 2 : 32;
        GptOssDebugTokenRecord *nt = (GptOssDebugTokenRecord *)realloc(d->tokens, nc * sizeof(*nt));
        if (!nt) return 0;
        memset(nt + d->token_cap, 0, (nc - d->token_cap) * sizeof(*nt));
        d->tokens = nt; d->token_cap = nc;
    }
    GptOssDebugTokenRecord *tr = &d->tokens[d->token_count];
    memset(tr, 0, sizeof(*tr));
    tr->seq = d->token_count;
    tr->pos = pos; tr->token = token; tr->is_prompt = is_prompt; tr->begin_wall = now_sec();
    tr->layers = (GptOssDebugLayerRecord *)calloc((size_t)m->n_layers, sizeof(*tr->layers));
    tr->experts = (GptOssDebugExpertRecord *)calloc((size_t)m->n_layers * (size_t)m->top_k, sizeof(*tr->experts));
    if (!tr->layers || !tr->experts) { free(tr->layers); free(tr->experts); memset(tr,0,sizeof(*tr)); return 0; }
    for (int l = 0; l < m->n_layers; ++l) {
        tr->layers[l].layer = l;
        tr->layers[l].experts = tr->experts + (size_t)l * m->top_k;
        for (int j = 0; j < m->top_k; ++j) {
            tr->layers[l].experts[j].eid = -1;
            tr->layers[l].experts[j].cache_slot = -1;
            tr->layers[l].experts[j].evicted_eid = -1;
        }
    }
    d->current = tr;
    ++d->token_count;
    const char *phase = is_prompt ? "prompt" : "generated";
    if (snprintf(d->token_log_path, sizeof(d->token_log_path),
                 "%s/token-%06llu-pos-%06d-%s-id-%d.log", d->dir,
                 (unsigned long long)tr->seq, pos, phase, token) < (int)sizeof(d->token_log_path)) {
        d->token_log = fopen(d->token_log_path, "w");
        if (d->token_log) setvbuf(d->token_log, NULL, _IOLBF, 0);
    }
    gptoss_debug_op(m, -1, -1, "alloc", "debug_token_record_and_log", "host+disk", 0,
                    (now_sec() - alloc_begin) * 1000.0);
    gptoss_debug_emit(m,
        "event=token_begin|phase=%s|token_seq=%llu|pos=%d|token=%d|layers=%d|top_k=%d|token_log=%s",
        phase, (unsigned long long)tr->seq, pos, token, m->n_layers, m->top_k,
        d->token_log ? d->token_log_path : "OPEN_FAILED");
    return 1;
}

static void gptoss_debug_end_token(GptOssModel *m, int ok) {
    if (!m || !m->debug || !m->debug->current) return;
    GptOssDebugState *d = m->debug;
    GptOssDebugTokenRecord *tr = d->current;
    tr->ok = ok;
    tr->total_ms = (now_sec() - tr->begin_wall) * 1000.0;
    gptoss_debug_emit(m,
        "event=token_end|phase=%s|token_seq=%llu|pos=%d|token=%d|ok=%d|copy_ms=%.3f|run_ms=%.3f|free_ms=%.3f|alloc_ms=%.3f|total_ms=%.3f",
        gptoss_debug_phase(tr), (unsigned long long)tr->seq, tr->pos, tr->token, ok,
        tr->copy_ms, tr->run_ms, tr->free_ms, tr->alloc_ms, tr->total_ms);
    if (d->token_log) { fclose(d->token_log); d->token_log = NULL; }
    d->current = NULL;
}

static void gptoss_debug_log_output(GptOssModel *m, int generated_index, int token,
                                    int control, int stop, const char *piece, int n) {
    if (!m || !m->debug || !m->debug->enabled) return;
    char hex[8192];
    size_t p = 0;
    if (piece && n > 0) {
        static const char h[] = "0123456789abcdef";
        for (int i = 0; i < n && p + 2 < sizeof(hex); ++i) {
            unsigned char c = (unsigned char)piece[i];
            hex[p++] = h[c >> 4]; hex[p++] = h[c & 15];
        }
    }
    hex[p] = 0;
    gptoss_debug_emit(m,
        "event=output|generated_index=%d|token=%d|control=%d|stop=%d|bytes=%d|hex=%s",
        generated_index, token, control, stop, n > 0 ? n : 0, hex);
}

static void gptoss_debug_write_summary(GptOssModel *m, int exit_status) {
    if (!m || !m->debug || !m->debug->enabled) return;
    GptOssDebugState *d = m->debug;
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/summary.log", d->dir) >= (int)sizeof(path)) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f,
        "DBG_SUMMARY|event=run|exit_status=%d|tokens=%zu|copy_ops=%llu|copy_ms=%.3f|run_ops=%llu|run_ms=%.3f|free_ops=%llu|free_ms=%.3f|alloc_ops=%llu|alloc_ms=%.3f|log_ms=%.3f\n",
        exit_status, d->token_count,
        (unsigned long long)d->copy_ops, d->total_copy_ms,
        (unsigned long long)d->run_ops, d->total_run_ms,
        (unsigned long long)d->free_ops, d->total_free_ms,
        (unsigned long long)d->alloc_ops, d->total_alloc_ms,
        d->total_log_ms);
    for (size_t ti = 0; ti < d->token_count; ++ti) {
        GptOssDebugTokenRecord *tr = &d->tokens[ti];
        fprintf(f,
            "DBG_SUMMARY|event=token|phase=%s|token_seq=%llu|pos=%d|token=%d|ok=%d|copy_ms=%.3f|run_ms=%.3f|free_ms=%.3f|alloc_ms=%.3f|total_ms=%.3f\n",
            gptoss_debug_phase(tr), (unsigned long long)tr->seq, tr->pos, tr->token, tr->ok,
            tr->copy_ms, tr->run_ms, tr->free_ms, tr->alloc_ms, tr->total_ms);
        for (int l = 0; l < m->n_layers; ++l) {
            GptOssDebugLayerRecord *lr = &tr->layers[l];
            fprintf(f,
                "DBG_SUMMARY|event=layer|phase=%s|token_seq=%llu|pos=%d|token=%d|layer=%d|selected=%d|ran=%d|cuda=%d|cpu=%d|cache_hits=%d|cache_misses=%d|evictions=%d|cuda_resident_before=%d|cuda_resident_after=%d|copy_ms=%.3f|run_ms=%.3f|free_ms=%.3f|alloc_ms=%.3f|total_ms=%.3f",
                gptoss_debug_phase(tr), (unsigned long long)tr->seq, tr->pos, tr->token, l,
                lr->selected_count, lr->experts_ran, lr->cuda_ran, lr->cpu_ran,
                lr->cache_hits, lr->cache_misses, lr->evictions,
                lr->cuda_resident_before, lr->cuda_resident_after,
                lr->copy_ms, lr->run_ms, lr->free_ms, lr->alloc_ms, lr->total_ms);
            for (int j = 0; j < lr->selected_count && j < m->top_k; ++j) {
                GptOssDebugExpertRecord *er = &lr->experts[j];
                fprintf(f, "|e%d=id:%d,w:%.6f,hit:%d,slot:%d,evicted:%d,cuda:%d,bytes:%zu,copy_ms:%.3f,run_ms:%.3f,free_ms:%.3f",
                        j, er->eid, er->weight, er->cache_hit, er->cache_slot, er->evicted_eid,
                        er->ran_cuda, er->copied_bytes, er->copy_ms, er->run_ms, er->free_ms);
            }
            fputc('\n', f);
        }
    }
    fclose(f);
    gptoss_debug_emit(m,
        "event=debug_summary|path=%s|tokens=%zu|copy_ms=%.3f|run_ms=%.3f|free_ms=%.3f|alloc_ms=%.3f|log_ms=%.3f",
        path, d->token_count, d->total_copy_ms, d->total_run_ms,
        d->total_free_ms, d->total_alloc_ms, d->total_log_ms);
}

static void gptoss_debug_destroy(GptOssModel *m) {
    if (!m || !m->debug) return;
    GptOssDebugState *d = m->debug;
    if (d->token_log) fclose(d->token_log);
    if (d->run_log) fclose(d->run_log);
    for (size_t i = 0; i < d->token_count; ++i) {
        free(d->tokens[i].layers);
        free(d->tokens[i].experts);
    }
    free(d->tokens);
    free(d);
    m->debug = NULL;
}

/* GPTOSS_STARTUP_TRACE_V1
 * Diagnostic-only first-forward tracing. Enable with:
 *   GPTOSS_TRACE_STARTUP=1
 * Add per-tensor mmap residency details with:
 *   GPTOSS_TRACE_TENSORS=1
 */
typedef struct {
    double wall;
    struct rusage usage;
} GptOssTracePoint;

static int gptoss_trace_env(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0 && strcasecmp(v, "false") != 0 && strcasecmp(v, "no") != 0;
}

static int gptoss_trace_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = gptoss_trace_env("GPTOSS_TRACE_STARTUP");
    return cached;
}

static int gptoss_trace_tensors(void) {
    static int cached = -1;
    if (cached < 0) cached = gptoss_trace_env("GPTOSS_TRACE_TENSORS");
    return cached;
}

static GptOssTracePoint gptoss_trace_point(void) {
    GptOssTracePoint p;
    memset(&p, 0, sizeof(p));
    if (!gptoss_trace_enabled()) return p;
    p.wall = now_sec();
    (void)getrusage(RUSAGE_SELF, &p.usage);
    return p;
}

static double gptoss_tv_sec(struct timeval v) {
    return (double)v.tv_sec + 1e-6 * (double)v.tv_usec;
}

static void gptoss_trace_delta(const char *label, GptOssTracePoint p) {
    if (!gptoss_trace_enabled()) return;
    struct rusage now;
    memset(&now, 0, sizeof(now));
    (void)getrusage(RUSAGE_SELF, &now);
    fprintf(stderr,
            "[GPT-OSS-TRACE] %-34s wall=%8.3fs cpu=%8.3fs minflt=%ld majflt=%ld inblock=%ld rss=%ld KiB\n",
            label,
            now_sec() - p.wall,
            (gptoss_tv_sec(now.ru_utime) + gptoss_tv_sec(now.ru_stime)) -
                (gptoss_tv_sec(p.usage.ru_utime) + gptoss_tv_sec(p.usage.ru_stime)),
            now.ru_minflt - p.usage.ru_minflt,
            now.ru_majflt - p.usage.ru_majflt,
            now.ru_inblock - p.usage.ru_inblock,
            now.ru_maxrss);
    fflush(stderr);
}

static int gptoss_tensor_pages(const ColiTensor *t, size_t *resident, size_t *total) {
    if (resident) *resident = 0;
    if (total) *total = 0;
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
    if (!t || !t->mmap_backed || !t->data || !t->storage_bytes) return 0;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    uintptr_t lo = (uintptr_t)t->data & ~((uintptr_t)page - 1u);
    uintptr_t hi = ((uintptr_t)t->data + (uintptr_t)t->storage_bytes +
                    (uintptr_t)page - 1u) & ~((uintptr_t)page - 1u);
    if (hi <= lo) return 0;
    size_t pages = (size_t)((hi - lo) / (uintptr_t)page);
    unsigned char *vec = (unsigned char *)malloc(pages);
    if (!vec) return 0;
    int ok = mincore((void *)lo, (size_t)(hi - lo), vec) == 0;
    size_t present = 0;
    if (ok) for (size_t i = 0; i < pages; ++i) present += (vec[i] & 1u) != 0;
    free(vec);
    if (!ok) return 0;
    if (resident) *resident = present;
    if (total) *total = pages;
    return 1;
#else
    (void)t;
    return 0;
#endif
}

static int gptoss_trace_reside(GptOssModel *m, ColiTensor *t,
                               int layer, int expert, const char *kind) {
    const int trace = gptoss_trace_enabled();
    const int already = coli_tensor_is_resident(&m->exec, t);
    size_t before = 0, pages = 0, after = 0, after_pages = 0;
    int have_before = 0, have_after = 0;
    if (trace && gptoss_trace_tensors())
        have_before = gptoss_tensor_pages(t, &before, &pages);
    GptOssTracePoint p = gptoss_trace_point();
    int ok = already ? 1 : coli_tensor_reside(&m->exec, t);
    if (trace && gptoss_trace_tensors())
        have_after = gptoss_tensor_pages(t, &after, &after_pages);
    if (trace) {
        struct rusage now;
        memset(&now, 0, sizeof(now));
        (void)getrusage(RUSAGE_SELF, &now);
        fprintf(stderr,
                "[GPT-OSS-TRACE] L%02d E%03d %-4s reside=%s prior=%s bytes=%7.2f MiB wall=%7.3fs majflt=%ld inblock=%ld",
                layer, expert, kind, ok ? "ok" : "FAIL", already ? "vram" : "host",
                (double)t->storage_bytes / (1024.0 * 1024.0), now_sec() - p.wall,
                now.ru_majflt - p.usage.ru_majflt,
                now.ru_inblock - p.usage.ru_inblock);
        if (have_before && pages)
            fprintf(stderr, " pages=%zu/%zu(%.1f%%)", before, pages, 100.0 * (double)before / (double)pages);
        if (have_after && after_pages)
            fprintf(stderr, " -> %zu/%zu(%.1f%%)", after, after_pages, 100.0 * (double)after / (double)after_pages);
        fputc('\n', stderr);
        fflush(stderr);
    }
    return ok;
}

static int gptoss_cache_has(const GptOssLayer *l, int eid) {
    if (!l || !l->cache) return 0;
    for (int i = 0; i < l->cache_cap; ++i)
        if (l->cache[i].eid == eid) return 1;
    return 0;
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
    free(l->usage); free(l->cache); free(l->gate_expert); free(l->up_expert); free(l->down_expert);
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
    double all0 = gptoss_debug_clock(m);
    if (m->layers) {
        for (int i = 0; i < m->n_layers; ++i) {
            double t0 = gptoss_debug_clock(m);
            layer_free(m, &m->layers[i]);
            if (t0) gptoss_debug_op(m, i, -1, "free", "layer_cleanup", "host+cuda", 0,
                                    (now_sec() - t0) * 1000.0);
        }
    }
    double t0 = gptoss_debug_clock(m);
    free(m->layers); free(m->k_cache); free(m->v_cache); free(m->rope_inv_freq);
    if (t0) gptoss_debug_op(m, -1, -1, "free", "model_host_arrays", "host", 0,
                            (now_sec() - t0) * 1000.0);
    t0 = gptoss_debug_clock(m);
    tensor_free(&m->token_embd); tensor_free(&m->output_norm); tensor_free(&m->output);
    free(m->architecture); coli_gguf_tokenizer_destroy(m->tokenizer); coli_gguf_close(&m->gguf);
    if (t0) gptoss_debug_op(m, -1, -1, "free", "model_global_tensors_and_gguf", "host+cuda", 0,
                            (now_sec() - t0) * 1000.0);
    if (all0) gptoss_debug_emit(m, "event=model_cleanup_end|total_ms=%.3f", (now_sec() - all0) * 1000.0);
    if (m->debug) gptoss_debug_write_summary(m, m->debug->exit_status);
    gptoss_debug_destroy(m);
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

static int info_dims(const ColiGgufTensorInfo *t, int nd, const uint64_t *dims) {
    if (!t || (int)t->n_dims != nd) return 0;
    for (int i = 0; i < nd; ++i) if (t->dims[i] != dims[i]) return 0;
    return 1;
}

static const ColiGgufTensorInfo *find_layer_alias_info(
    GptOssModel *m, int layer, int nd, const uint64_t *dims,
    char *err, size_t cap, const char *sa, const char *sb, const char *sc) {
    char names[3][160];
    const char *suffixes[3] = {sa, sb, sc};
    for (int i = 0; i < 3; ++i) {
        if (!suffixes[i]) continue;
        snprintf(names[i], sizeof(names[i]), "blk.%d.%s", layer, suffixes[i]);
        const ColiGgufTensorInfo *t = coli_gguf_find_tensor(&m->gguf, names[i]);
        if (!t) continue;
        if (!info_dims(t, nd, dims)) {
            errf(err, cap, "wrong dimensions for tensor %s", names[i]);
            return NULL;
        }
        return t;
    }
    errf(err, cap, "missing expert tensor in layer %d", layer);
    return NULL;
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
        x->gate_info = find_layer_alias_info(m, l, 3, d, err, cap,
                                             "ffn_gate_exps.weight", NULL, NULL);
        x->up_info = find_layer_alias_info(m, l, 3, d, err, cap,
                                           "ffn_up_exps.weight", NULL, NULL);
        if (!x->gate_info || !x->up_info) return 0;
        if (x->gate_info->storage_kind != COLI_TENSOR_STORAGE_DENSE &&
            !coli_tensor_bind_gguf(&m->gguf, x->gate_info, &x->gate, err, cap)) return 0;
        if (x->up_info->storage_kind != COLI_TENSOR_STORAGE_DENSE &&
            !coli_tensor_bind_gguf(&m->gguf, x->up_info, &x->up, err, cap)) return 0;
        d[0] = m->expert_ff; d[1] = m->hidden; d[2] = m->n_experts;
        x->down_info = find_layer_alias_info(m, l, 3, d, err, cap,
                                             "ffn_down_exps.weight", NULL, NULL);
        if (!x->down_info) return 0;
        if (x->down_info->storage_kind != COLI_TENSOR_STORAGE_DENSE &&
            !coli_tensor_bind_gguf(&m->gguf, x->down_info, &x->down, err, cap)) return 0;
        d[0] = m->expert_ff; d[1] = m->n_experts;
        if (!load_layer_alias(m, l, &x->gate_bias, 2, d, err, cap, "ffn_gate_exps.bias", NULL, NULL)) return 0;
        if (!load_layer_alias(m, l, &x->up_bias, 2, d, err, cap, "ffn_up_exps.bias", NULL, NULL)) return 0;
        d[0] = m->hidden; d[1] = m->n_experts;
        if (!load_layer_alias(m, l, &x->down_bias, 2, d, err, cap, "ffn_down_exps.bias", NULL, NULL)) return 0;
        fprintf(stderr, "[GPT-OSS] bind layer %d/%d\n", l + 1, m->n_layers);
        fflush(stderr);
    }
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
        x->usage = (uint32_t *)calloc((size_t)m->n_experts, sizeof(*x->usage));
        x->cache = (GptOssExpertSlot *)calloc((size_t)cache_cap, sizeof(*x->cache));
        if (!x->gate_expert || !x->up_expert || !x->down_expert || !x->usage || !x->cache)
            return errf(err, cap, "out of memory allocating GPT-OSS expert views");
        x->cache_cap = cache_cap;
        for (int i = 0; i < cache_cap; ++i) x->cache[i].eid = -1;
        for (int e = 0; e < m->n_experts; ++e) {
            const int gate_ok = x->gate_info->storage_kind == COLI_TENSOR_STORAGE_DENSE
                ? coli_tensor_bind_gguf_rows(&m->gguf, x->gate_info,
                    (uint64_t)e * m->expert_ff, m->expert_ff,
                    &x->gate_expert[e], err, cap)
                : coli_tensor_rows_view(&x->gate, (uint64_t)e * m->expert_ff,
                    m->expert_ff, &x->gate_expert[e]);
            const int up_ok = x->up_info->storage_kind == COLI_TENSOR_STORAGE_DENSE
                ? coli_tensor_bind_gguf_rows(&m->gguf, x->up_info,
                    (uint64_t)e * m->expert_ff, m->expert_ff,
                    &x->up_expert[e], err, cap)
                : coli_tensor_rows_view(&x->up, (uint64_t)e * m->expert_ff,
                    m->expert_ff, &x->up_expert[e]);
            const int down_ok = x->down_info->storage_kind == COLI_TENSOR_STORAGE_DENSE
                ? coli_tensor_bind_gguf_rows(&m->gguf, x->down_info,
                    (uint64_t)e * m->hidden, m->hidden,
                    &x->down_expert[e], err, cap)
                : coli_tensor_rows_view(&x->down, (uint64_t)e * m->hidden,
                    m->hidden, &x->down_expert[e]);
            if (!gate_ok || !up_ok || !down_ok)
                return errf(err, cap, "cannot bind expert fragment layer=%d expert=%d: %s",
                            l, e, err && *err ? err : "unknown error");
            x->expert_views = e + 1;
        }
    }
    return 1;
}

static int gptoss_usage_configure(GptOssModel *m, const char *model_path,
                                   const char *cli_path, char *err, size_t cap) {
    const char *path = cli_path && *cli_path ? cli_path : getenv("COLI_USAGE_FILE");
    if (path && *path) {
        const size_t n = strlen(path);
        if (n >= sizeof(m->usage_path))
            return errf(err, cap, "GPT-OSS usage path is too long");
        memcpy(m->usage_path, path, n + 1);
    } else {
        const size_t n = strlen(model_path);
        if (n + sizeof(".coli_usage") > sizeof(m->usage_path))
            return errf(err, cap, "GPT-OSS model path is too long for .coli_usage");
        memcpy(m->usage_path, model_path, n);
        memcpy(m->usage_path + n, ".coli_usage", sizeof(".coli_usage"));
    }
    FILE *f = fopen(m->usage_path, "a");
    if (!f) return errf(err, cap, "cannot create/open GPT-OSS usage file %s: %s",
                        m->usage_path, strerror(errno));
    if (fclose(f) != 0)
        return errf(err, cap, "cannot close GPT-OSS usage file %s: %s",
                    m->usage_path, strerror(errno));
    return 1;
}

static int64_t gptoss_usage_load(GptOssModel *m) {
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return 0;
    for (int l = 0; l < m->n_layers; ++l) rows[l] = m->layers[l].usage;
    const int64_t total = coli_expert_usage_load(
        m->usage_path, rows, m->n_layers, m->n_experts);
    free(rows);
    m->usage_history = total;
    return total;
}

static int gptoss_usage_save(GptOssModel *m) {
    if (!m || !m->usage_path[0] || !m->layers) return 0;
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return 0;
    for (int l = 0; l < m->n_layers; ++l) rows[l] = m->layers[l].usage;
    const int ok = coli_expert_usage_save(
        m->usage_path, rows, m->n_layers, m->n_experts);
    free(rows);
    return ok;
}

static void gptoss_usage_record(GptOssLayer *layer, int eid) {
    if (!layer || !layer->usage || eid < 0) return;
    if (layer->usage[eid] != UINT32_MAX) layer->usage[eid]++;
}

static int gptoss_reside_profile(GptOssModel *m, ColiTensor *t,
                                  int layer, int expert, const char *name,
                                  int *was_resident) {
    int already = coli_tensor_is_resident(&m->exec, t);
    if (was_resident) *was_resident = already;
    if (already) {
        if (m->debug && m->debug->enabled)
            gptoss_debug_emit(m,
                "event=residency|phase=%s|token_seq=%lld|pos=%d|layer=%d|expert=%d|tensor=%s|before=cuda+host|after=cuda+host|status=hit|bytes=0|ms=0.000",
                m->debug->current ? gptoss_debug_phase(m->debug->current) : "startup",
                m->debug->current ? (long long)m->debug->current->seq : -1LL,
                m->debug->current ? m->debug->current->pos : -1,
                layer, expert, name);
        return 1;
    }
    double t0 = gptoss_debug_clock(m);
    int ok;
    if (gptoss_trace_enabled() && !(m->debug && m->debug->enabled))
        ok = gptoss_trace_reside(m, t, layer, expert, name);
    else
        ok = coli_tensor_reside(&m->exec, t);
    double ms = t0 ? (now_sec() - t0) * 1000.0 : 0.0;
    if (m->debug && m->debug->enabled) {
        gptoss_debug_op(m, layer, expert, "copy", name, "host_to_cuda",
                        ok ? (size_t)t->storage_bytes : 0, ms);
        gptoss_debug_emit(m,
            "event=residency|phase=%s|token_seq=%lld|pos=%d|layer=%d|expert=%d|tensor=%s|before=host|after=%s|status=%s|bytes=%llu|ms=%.3f",
            m->debug->current ? gptoss_debug_phase(m->debug->current) : "startup",
            m->debug->current ? (long long)m->debug->current->seq : -1LL,
            m->debug->current ? m->debug->current->pos : -1,
            layer, expert, name, ok ? "cuda+host" : "host",
            ok ? "ok" : "failed", (unsigned long long)(ok ? t->storage_bytes : 0), ms);
    }
    return ok;
}

static void gptoss_release_profile(GptOssModel *m, ColiTensor *t,
                                   int layer, int expert, const char *name) {
    if (!m || !t || !coli_tensor_is_resident(&m->exec, t)) return;
    size_t bytes = (size_t)t->storage_bytes;
    double t0 = gptoss_debug_clock(m);
    coli_tensor_release_backend(&m->exec, t);
    double ms = t0 ? (now_sec() - t0) * 1000.0 : 0.0;
    if (m->debug && m->debug->enabled) {
        gptoss_debug_op(m, layer, expert, "free", name, "cuda", bytes, ms);
        gptoss_debug_emit(m,
            "event=residency|phase=%s|token_seq=%lld|pos=%d|layer=%d|expert=%d|tensor=%s|before=cuda+host|after=host|status=released|bytes=%zu|ms=%.3f",
            m->debug->current ? gptoss_debug_phase(m->debug->current) : "cleanup",
            m->debug->current ? (long long)m->debug->current->seq : -1LL,
            m->debug->current ? m->debug->current->pos : -1,
            layer, expert, name, bytes, ms);
    }
}

static void release_dense_cuda(GptOssModel *m) {
    if (m->exec.kind != COLI_BACKEND_CUDA || !m->layers) return;
    for (int l = 0; l < m->n_layers; ++l) {
        GptOssLayer *x = &m->layers[l];
        gptoss_release_profile(m, &x->q, l, -1, "dense_q_release");
        gptoss_release_profile(m, &x->k, l, -1, "dense_k_release");
        gptoss_release_profile(m, &x->v, l, -1, "dense_v_release");
        gptoss_release_profile(m, &x->o, l, -1, "dense_o_release");
        gptoss_release_profile(m, &x->router, l, -1, "dense_router_release");
    }
    m->cuda_dense_bytes = 0;
}

static void try_reside_dense_cuda(GptOssModel *m) {
    if (m->exec.kind != COLI_BACKEND_CUDA) return;
    size_t total = 0; int ok = 1;
    for (int l = 0; l < m->n_layers && ok; ++l) {
        GptOssLayer *x = &m->layers[l];
        ColiTensor *a[] = {&x->q,&x->k,&x->v,&x->o,&x->router};
        const char *names[] = {"dense_q_upload","dense_k_upload","dense_v_upload","dense_o_upload","dense_router_upload"};
        for (size_t i = 0; i < sizeof(a)/sizeof(a[0]); ++i) {
            int prior = 0;
            if (!gptoss_reside_profile(m, a[i], l, -1, names[i], &prior)) { ok = 0; break; }
            total += (size_t)a[i]->storage_bytes;
        }
    }
    if (!ok) {
        release_dense_cuda(m); m->dense_streaming = 1;
        if (m->verbose) fprintf(stderr, "[GPT-OSS] dense CUDA residency exceeds budget; streaming dense tensors\n");
        if (m->debug && m->debug->enabled)
            gptoss_debug_emit(m, "event=dense_residency|status=streaming|bytes=0");
    } else {
        m->cuda_dense_bytes = total;
        if (m->verbose) fprintf(stderr, "[GPT-OSS] dense attention/router resident: %.2f MiB\n", total/(1024.0*1024.0));
        if (m->debug && m->debug->enabled)
            gptoss_debug_emit(m, "event=dense_residency|status=resident|bytes=%zu|mib=%.3f", total, total/(1024.0*1024.0));
    }
}

static GptOssExpertSlot *expert_acquire(GptOssModel *m, int layer, int eid,
                                        GptOssDebugExpertRecord *er) {
    GptOssLayer *l = &m->layers[layer];
    int resident_before = gptoss_debug_layer_resident(l);
    if (er) { er->resident_before = resident_before; er->eid = eid; }
    for (int i = 0; i < l->cache_cap; ++i) if (l->cache[i].eid == eid) {
        GptOssExpertSlot *slot = &l->cache[i];
        int hit_was_cuda = slot->cuda_resident;
        if (er) { er->cache_hit = 1; er->cache_slot = i; }
        slot->age = ++m->cache_clock;
        if (m->exec.kind == COLI_BACKEND_CUDA && !slot->cuda_resident) {
            int a=0,b=0,c=0;
            if (gptoss_reside_profile(m, &l->gate_expert[eid], layer, eid, "expert_gate_upload", &a) &&
                gptoss_reside_profile(m, &l->up_expert[eid], layer, eid, "expert_up_upload", &b) &&
                gptoss_reside_profile(m, &l->down_expert[eid], layer, eid, "expert_down_upload", &c)) {
                slot->cuda_resident = 1;
            } else {
                gptoss_release_profile(m, &l->gate_expert[eid], layer, eid, "expert_gate_failed_release");
                gptoss_release_profile(m, &l->up_expert[eid], layer, eid, "expert_up_failed_release");
                gptoss_release_profile(m, &l->down_expert[eid], layer, eid, "expert_down_failed_release");
            }
        }
        if (er) er->resident_after = gptoss_debug_layer_resident(l);
        if (m->debug && m->debug->enabled)
            gptoss_debug_emit(m,
                "event=expert_acquire|phase=%s|token_seq=%lld|pos=%d|layer=%d|expert=%d|cache=hit|slot=%d|evicted=-1|before=%s|after=%s|layer_cuda_before=%d|layer_cuda_after=%d",
                m->debug->current ? gptoss_debug_phase(m->debug->current) : "startup",
                m->debug->current ? (long long)m->debug->current->seq : -1LL,
                m->debug->current ? m->debug->current->pos : -1,
                layer, eid, i, hit_was_cuda ? "cuda+host" : "host",
                slot->cuda_resident ? "cuda+host" : "host", resident_before,
                gptoss_debug_layer_resident(l));
        return slot;
    }
    int pick = 0;
    for (int i = 0; i < l->cache_cap; ++i) {
        if (l->cache[i].eid < 0) { pick = i; break; }
        if (l->cache[i].age < l->cache[pick].age) pick = i;
    }
    GptOssExpertSlot *slot = &l->cache[pick];
    int old = slot->eid;
    if (er) { er->cache_hit = 0; er->cache_slot = pick; er->evicted_eid = old; }
    if (old >= 0 && slot->cuda_resident) {
        gptoss_release_profile(m, &l->gate_expert[old], layer, old, "evict_gate_release");
        gptoss_release_profile(m, &l->up_expert[old], layer, old, "evict_up_release");
        gptoss_release_profile(m, &l->down_expert[old], layer, old, "evict_down_release");
    }
    slot->eid = eid; slot->age = ++m->cache_clock; slot->cuda_resident = 0;
    if (m->exec.kind == COLI_BACKEND_CUDA) {
        int a=0,b=0,c=0;
        if (gptoss_reside_profile(m, &l->gate_expert[eid], layer, eid, "expert_gate_upload", &a) &&
            gptoss_reside_profile(m, &l->up_expert[eid], layer, eid, "expert_up_upload", &b) &&
            gptoss_reside_profile(m, &l->down_expert[eid], layer, eid, "expert_down_upload", &c)) slot->cuda_resident = 1;
        else {
            gptoss_release_profile(m, &l->gate_expert[eid], layer, eid, "expert_gate_failed_release");
            gptoss_release_profile(m, &l->up_expert[eid], layer, eid, "expert_up_failed_release");
            gptoss_release_profile(m, &l->down_expert[eid], layer, eid, "expert_down_failed_release");
        }
    }
    if (er) er->resident_after = gptoss_debug_layer_resident(l);
    if (m->debug && m->debug->enabled)
        gptoss_debug_emit(m,
            "event=expert_acquire|phase=%s|token_seq=%lld|pos=%d|layer=%d|expert=%d|cache=miss|slot=%d|evicted=%d|before=host|after=%s|layer_cuda_before=%d|layer_cuda_after=%d",
            m->debug->current ? gptoss_debug_phase(m->debug->current) : "startup",
            m->debug->current ? (long long)m->debug->current->seq : -1LL,
            m->debug->current ? m->debug->current->pos : -1,
            layer, eid, pick, old, slot->cuda_resident ? "cuda+host" : "host",
            resident_before, gptoss_debug_layer_resident(l));
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

static int tensor_vector_profile(GptOssModel *m, const ColiTensor *t, float *dst, int n,
                                 int layer, int expert, const char *name,
                                 char *err, size_t cap) {
    double t0 = gptoss_debug_clock(m);
    int ok = t && t->row_count == 1 && t->dims[0] == (uint64_t)n &&
             coli_tensor_read_row_f32(t, 0, dst, (uint64_t)n);
    if (t0) gptoss_debug_op(m, layer, expert, "copy", name, "host_decode",
                            ok ? (size_t)n * sizeof(float) : 0,
                            (now_sec() - t0) * 1000.0);
    if (!ok) return errf(err, cap, "failed reading vector %s", t && t->name ? t->name : "<unnamed>");
    return 1;
}

static int tensor_row_profile(GptOssModel *m, const ColiTensor *t, uint64_t row,
                              float *dst, int n, int layer, int expert,
                              const char *name, char *err, size_t cap) {
    double t0 = gptoss_debug_clock(m);
    int ok = t && coli_tensor_read_row_f32(t, row, dst, (uint64_t)n);
    if (t0) gptoss_debug_op(m, layer, expert, "copy", name, "host_decode",
                            ok ? (size_t)n * sizeof(float) : 0,
                            (now_sec() - t0) * 1000.0);
    if (!ok) return errf(err, cap, "failed reading row %llu from %s",
                         (unsigned long long)row, t && t->name ? t->name : "<unnamed>");
    return 1;
}

static int tensor_vector(const ColiTensor *t, float *dst, int n, char *err, size_t cap) {
    if (!t || t->row_count != 1 || t->dims[0] != (uint64_t)n ||
        !coli_tensor_read_row_f32(t, 0, dst, (uint64_t)n))
        return errf(err, cap, "failed reading vector %s", t && t->name ? t->name : "<unnamed>");
    return 1;
}

static int tensor_mm_profile(GptOssModel *m, float *y, const float *x, ColiTensor *t,
                             int I, int O, int transient, int layer, int expert,
                             const char *name, int force_cpu, int *used_cuda,
                             char *err, size_t cap) {
    if (used_cuda) *used_cuda = 0;
    if (!force_cpu && m->exec.kind == COLI_BACKEND_CUDA) {
        int prior = 0;
        if (gptoss_reside_profile(m, t, layer, expert, name, &prior)) {
            double t0 = gptoss_debug_clock(m);
            int ok = coli_tensor_matmul(&m->exec, y, x, t, 1, I, O);
            double ms = t0 ? (now_sec() - t0) * 1000.0 : 0.0;
            if (t0) gptoss_debug_op(m, layer, expert, "run", name, "cuda", 0, ms);
            if (transient && m->dense_streaming)
                gptoss_release_profile(m, t, layer, expert, "dense_transient_release");
            if (ok) { if (used_cuda) *used_cuda = 1; return 1; }
            gptoss_release_profile(m, t, layer, expert, "cuda_failure_release");
            if (m->debug && m->debug->enabled)
                gptoss_debug_emit(m, "event=fallback|layer=%d|expert=%d|name=%s|from=cuda|to=cpu", layer, expert, name);
        }
    }
    ColiExec cpu = {COLI_BACKEND_CPU, 0};
    double t0 = gptoss_debug_clock(m);
    int ok = coli_tensor_matmul(&cpu, y, x, t, 1, I, O);
    double ms = t0 ? (now_sec() - t0) * 1000.0 : 0.0;
    if (t0) gptoss_debug_op(m, layer, expert, "run", name, "cpu", 0, ms);
    if (ok) return 1;
    return errf(err, cap, "%s matmul failed for %s", force_cpu ? "CPU" : "", t->name ? t->name : "<unnamed>");
}

static int tensor_mm_auto(GptOssModel *m, float *y, const float *x, ColiTensor *t,
                          int I, int O, int transient, char *err, size_t cap) {
    return tensor_mm_profile(m, y, x, t, I, O, transient, -1, -1,
                             t && t->name ? t->name : "matmul", 0, NULL, err, cap);
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
    const int debug = m->debug && m->debug->enabled;
    const int trace_forward = gptoss_trace_enabled() && pos == 0 && !debug;
    GptOssTracePoint trace_forward_point = gptoss_trace_point();
    if (trace_forward) {
        fprintf(stderr, "[GPT-OSS-TRACE] forward begin pos=%d token=%d\n", pos, token);
        fflush(stderr);
    }
    if (debug)
        gptoss_debug_emit(m, "event=forward_begin|phase=%s|token_seq=%llu|pos=%d|token=%d",
                          gptoss_debug_phase(m->debug->current),
                          (unsigned long long)m->debug->current->seq, pos, token);

    if (token < 0 || token >= m->vocab || pos < 0 || pos >= m->context_capacity)
        return errf(err, cap, "token/position outside model bounds");
    double op0 = gptoss_debug_clock(m);
    if (!coli_tensor_read_row_f32(&m->token_embd, (uint64_t)token, s->x, (uint64_t)m->hidden))
        return errf(err, cap, "embedding row decode failed");
    if (op0) gptoss_debug_op(m, -1, -1, "copy", "token_embedding_decode", "host_decode",
                             (size_t)m->hidden * sizeof(float), (now_sec() - op0) * 1000.0);
    if (trace_forward) gptoss_trace_delta("embedding decode", trace_forward_point);

    for (int l = 0; l < m->n_layers; ++l) {
        GptOssLayer *L = &m->layers[l];
        GptOssDebugLayerRecord *lr = gptoss_debug_layer(m, l);
        double layer0 = gptoss_debug_clock(m);
        int resident_before = gptoss_debug_layer_resident(L);
        if (lr) lr->cuda_resident_before = resident_before;
        if (debug) {
            char slots[1024]; size_t used = 0;
            slots[0] = 0;
            for (int si = 0; si < L->cache_cap && used + 32 < sizeof(slots); ++si) {
                int n = snprintf(slots + used, sizeof(slots) - used, "%ss%d:E%d:%s",
                                 si ? "," : "", si, L->cache[si].eid,
                                 L->cache[si].cuda_resident ? "cuda" : "host");
                if (n < 0) break; used += (size_t)n;
            }
            gptoss_debug_emit(m,
                "event=layer_begin|phase=%s|token_seq=%llu|pos=%d|token=%d|layer=%d|experts_total=%d|host_mmap=%d|cuda_resident=%d|host_only=%d|cache_cap=%d|cache=%s",
                gptoss_debug_phase(m->debug->current),
                (unsigned long long)m->debug->current->seq, pos, token, l,
                m->n_experts, m->n_experts, resident_before,
                m->n_experts - resident_before, L->cache_cap, slots);
        }
        if (trace_forward) { fprintf(stderr, "[GPT-OSS-TRACE] L%02d begin\n", l); fflush(stderr); }

        if (!tensor_vector_profile(m, &L->attn_norm, s->weight, m->hidden, l, -1,
                                   "attn_norm_read", err, cap)) return 0;
        op0 = gptoss_debug_clock(m);
        coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
        if (op0) gptoss_debug_op(m, l, -1, "run", "attn_rmsnorm", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);

        int used_cuda = 0;
        if (!tensor_mm_profile(m, s->q, s->norm, &L->q, m->hidden, m->q_dim, 1,
                               l, -1, "attn_q_matmul", 0, &used_cuda, err, cap) ||
            !tensor_mm_profile(m, s->k, s->norm, &L->k, m->hidden, m->kv_dim, 1,
                               l, -1, "attn_k_matmul", 0, &used_cuda, err, cap) ||
            !tensor_mm_profile(m, s->v, s->norm, &L->v, m->hidden, m->kv_dim, 1,
                               l, -1, "attn_v_matmul", 0, &used_cuda, err, cap) ||
            !tensor_vector_profile(m, &L->q_bias, s->q_bias, m->q_dim, l, -1,
                                   "attn_q_bias_read", err, cap) ||
            !tensor_vector_profile(m, &L->k_bias, s->k_bias, m->kv_dim, l, -1,
                                   "attn_k_bias_read", err, cap) ||
            !tensor_vector_profile(m, &L->v_bias, s->v_bias, m->kv_dim, l, -1,
                                   "attn_v_bias_read", err, cap)) return 0;
        op0 = gptoss_debug_clock(m);
        for (int i = 0; i < m->q_dim; ++i) s->q[i] += s->q_bias[i];
        for (int i = 0; i < m->kv_dim; ++i) { s->k[i] += s->k_bias[i]; s->v[i] += s->v_bias[i]; }
        if (op0) gptoss_debug_op(m, l, -1, "run", "qkv_bias_add", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);
        op0 = gptoss_debug_clock(m);
        coli_gptoss_yarn_rotate(s->q, m->n_heads, m->head_dim, pos, m->rope_inv_freq, m->yarn_concentration);
        if (op0) gptoss_debug_op(m, l, -1, "run", "rope_q", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);
        op0 = gptoss_debug_clock(m);
        coli_gptoss_yarn_rotate(s->k, m->n_kv_heads, m->head_dim, pos, m->rope_inv_freq, m->yarn_concentration);
        if (op0) gptoss_debug_op(m, l, -1, "run", "rope_k", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);
        size_t stride = (size_t)m->context_capacity * m->kv_dim;
        float *kc = m->k_cache + (size_t)l * stride, *vc = m->v_cache + (size_t)l * stride;
        size_t kv_bytes = (size_t)m->kv_dim * sizeof(float);
        op0 = gptoss_debug_clock(m);
        memcpy(kc + (size_t)pos * m->kv_dim, s->k, kv_bytes);
        if (op0) gptoss_debug_op(m, l, -1, "copy", "kv_cache_k_copy", "host_to_host", kv_bytes,
                                 (now_sec() - op0) * 1000.0);
        op0 = gptoss_debug_clock(m);
        memcpy(vc + (size_t)pos * m->kv_dim, s->v, kv_bytes);
        if (op0) gptoss_debug_op(m, l, -1, "copy", "kv_cache_v_copy", "host_to_host", kv_bytes,
                                 (now_sec() - op0) * 1000.0);
        if (!tensor_vector_profile(m, &L->sinks, s->sinks, m->n_heads, l, -1,
                                   "attention_sinks_read", err, cap)) return 0;
        int window = (l % 2 == 0) ? m->sliding_window : 0;
        op0 = gptoss_debug_clock(m);
        coli_gptoss_attention_sink_reference(
            s->attn, s->q, kc, vc, s->sinks, s->scores, pos, window,
            m->n_heads, m->n_kv_heads, m->head_dim, m->attention_scale);
        if (op0) gptoss_debug_op(m, l, -1, "run", "attention_sink", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);
        if (!tensor_mm_profile(m, s->proj, s->attn, &L->o, m->q_dim, m->hidden, 1,
                               l, -1, "attn_output_matmul", 0, &used_cuda, err, cap) ||
            !tensor_vector_profile(m, &L->o_bias, s->attn_bias, m->hidden, l, -1,
                                   "attn_output_bias_read", err, cap)) return 0;
        op0 = gptoss_debug_clock(m);
        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->proj[i] + s->attn_bias[i];
        if (op0) gptoss_debug_op(m, l, -1, "run", "attention_residual_add", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);

        if (!tensor_vector_profile(m, &L->post_attn_norm, s->weight, m->hidden, l, -1,
                                   "post_attn_norm_read", err, cap)) return 0;
        op0 = gptoss_debug_clock(m);
        coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
        if (op0) gptoss_debug_op(m, l, -1, "run", "post_attn_rmsnorm", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);
        if (!tensor_mm_profile(m, s->router, s->norm, &L->router, m->hidden, m->n_experts, 1,
                               l, -1, "router_matmul", 0, &used_cuda, err, cap) ||
            !tensor_vector_profile(m, &L->router_bias, s->router_bias, m->n_experts, l, -1,
                                   "router_bias_read", err, cap)) return 0;
        op0 = gptoss_debug_clock(m);
        for (int e = 0; e < m->n_experts; ++e) s->router[e] += s->router_bias[e];
        if (op0) gptoss_debug_op(m, l, -1, "run", "router_bias_add", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);
        op0 = gptoss_debug_clock(m);
        int nk = coli_gptoss_topk_softmax(s->router, m->n_experts, m->top_k, s->top_idx, s->top_w);
        if (op0) gptoss_debug_op(m, l, -1, "run", "router_topk_softmax", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);
        if (nk != m->top_k) return errf(err, cap, "invalid GPT-OSS router selection");
        if (lr) {
            lr->selected_count = nk;
            for (int j = 0; j < nk; ++j) {
                lr->experts[j].eid = s->top_idx[j];
                lr->experts[j].weight = s->top_w[j];
            }
        }
        if (debug) {
            char route[1024]; size_t used = 0; route[0] = 0;
            for (int j = 0; j < nk && used + 64 < sizeof(route); ++j) {
                int e = s->top_idx[j];
                int cached = gptoss_cache_has(L, e);
                int cuda = 0;
                for (int si = 0; si < L->cache_cap; ++si)
                    if (L->cache[si].eid == e) cuda = L->cache[si].cuda_resident;
                int n = snprintf(route + used, sizeof(route) - used,
                                 "%sE%d:w=%.6f:before=%s:cache=%s",
                                 j ? "," : "", e, s->top_w[j],
                                 cuda ? "cuda+host" : "host", cached ? "hit" : "miss");
                if (n < 0) break; used += (size_t)n;
            }
            gptoss_debug_emit(m,
                "event=router|phase=%s|token_seq=%llu|pos=%d|token=%d|layer=%d|selected=%d|experts=%s",
                gptoss_debug_phase(m->debug->current),
                (unsigned long long)m->debug->current->seq, pos, token, l, nk, route);
        }
        if (trace_forward) {
            fprintf(stderr, "[GPT-OSS-TRACE] L%02d router", l);
            for (int tj = 0; tj < nk; ++tj) fprintf(stderr, " E%03d=%.5f", s->top_idx[tj], s->top_w[tj]);
            fputc('\n', stderr);
        }

        op0 = gptoss_debug_clock(m);
        memset(s->moe, 0, (size_t)m->hidden * sizeof(float));
        if (op0) gptoss_debug_op(m, l, -1, "copy", "moe_zero", "host", (size_t)m->hidden * sizeof(float),
                                 (now_sec() - op0) * 1000.0);
        for (int j = 0; j < nk; ++j) {
            int e = s->top_idx[j];
            gptoss_usage_record(L, e);
            GptOssDebugExpertRecord *er = gptoss_debug_expert(m, l, j);
            GptOssExpertSlot *slot = expert_acquire(m, l, e, er);
            const int use_cuda = m->exec.kind == COLI_BACKEND_CUDA && slot && slot->cuda_resident;
            if (er) er->ran_cuda = use_cuda;
            if (lr) {
                ++lr->experts_ran;
                if (use_cuda) ++lr->cuda_ran; else ++lr->cpu_ran;
                if (er && er->cache_hit) ++lr->cache_hits; else ++lr->cache_misses;
                if (er && er->evicted_eid >= 0) ++lr->evictions;
            }
            int gate_cuda = 0, up_cuda = 0, down_cuda = 0;
            if (!tensor_mm_profile(m, s->gate, s->norm, &L->gate_expert[e], m->hidden, m->expert_ff, 0,
                                   l, e, "expert_gate_matmul", !use_cuda, &gate_cuda, err, cap) ||
                !tensor_mm_profile(m, s->up, s->norm, &L->up_expert[e], m->hidden, m->expert_ff, 0,
                                   l, e, "expert_up_matmul", !use_cuda, &up_cuda, err, cap) ||
                !tensor_row_profile(m, &L->gate_bias, (uint64_t)e, s->gate_bias, m->expert_ff,
                                    l, e, "expert_gate_bias_read", err, cap) ||
                !tensor_row_profile(m, &L->up_bias, (uint64_t)e, s->up_bias, m->expert_ff,
                                    l, e, "expert_up_bias_read", err, cap))
                return 0;
            op0 = gptoss_debug_clock(m);
            for (int i = 0; i < m->expert_ff; ++i)
                s->hidden[i] = coli_gptoss_oai_swiglu(s->gate[i] + s->gate_bias[i], s->up[i] + s->up_bias[i]);
            if (op0) gptoss_debug_op(m, l, e, "run", "expert_swiglu", "cpu", 0,
                                     (now_sec() - op0) * 1000.0);
            if (!tensor_mm_profile(m, s->expert_out, s->hidden, &L->down_expert[e], m->expert_ff, m->hidden, 0,
                                   l, e, "expert_down_matmul", !use_cuda, &down_cuda, err, cap) ||
                !tensor_row_profile(m, &L->down_bias, (uint64_t)e, s->down_bias, m->hidden,
                                    l, e, "expert_down_bias_read", err, cap))
                return 0;
            float rw = s->top_w[j];
            op0 = gptoss_debug_clock(m);
            for (int i = 0; i < m->hidden; ++i) s->moe[i] += rw * (s->expert_out[i] + s->down_bias[i]);
            if (op0) gptoss_debug_op(m, l, e, "run", "expert_weighted_accumulate", "cpu", 0,
                                     (now_sec() - op0) * 1000.0);
            if (debug && er)
                gptoss_debug_emit(m,
                    "event=expert_end|phase=%s|token_seq=%llu|pos=%d|token=%d|layer=%d|rank=%d|expert=%d|weight=%.6f|cache=%s|slot=%d|evicted=%d|backend=%s|copied_bytes=%zu|copy_ms=%.3f|run_ms=%.3f|free_ms=%.3f",
                    gptoss_debug_phase(m->debug->current),
                    (unsigned long long)m->debug->current->seq, pos, token, l, j, e, rw,
                    er->cache_hit ? "hit" : "miss", er->cache_slot, er->evicted_eid,
                    use_cuda ? "cuda" : "cpu", er->copied_bytes,
                    er->copy_ms, er->run_ms, er->free_ms);
        }
        op0 = gptoss_debug_clock(m);
        for (int i = 0; i < m->hidden; ++i) s->x[i] += s->moe[i];
        if (op0) gptoss_debug_op(m, l, -1, "run", "moe_residual_add", "cpu", 0,
                                 (now_sec() - op0) * 1000.0);

        int resident_after = gptoss_debug_layer_resident(L);
        if (lr) {
            lr->cuda_resident_after = resident_after;
            lr->total_ms = layer0 ? (now_sec() - layer0) * 1000.0 : 0.0;
        }
        if (debug) {
            char slots[1024]; size_t used = 0; slots[0] = 0;
            for (int si = 0; si < L->cache_cap && used + 32 < sizeof(slots); ++si) {
                int n = snprintf(slots + used, sizeof(slots) - used, "%ss%d:E%d:%s",
                                 si ? "," : "", si, L->cache[si].eid,
                                 L->cache[si].cuda_resident ? "cuda" : "host");
                if (n < 0) break; used += (size_t)n;
            }
            gptoss_debug_emit(m,
                "event=layer_end|phase=%s|token_seq=%llu|pos=%d|token=%d|layer=%d|selected=%d|ran=%d|cuda_ran=%d|cpu_ran=%d|cache_hits=%d|cache_misses=%d|evictions=%d|experts_total=%d|host_mmap=%d|cuda_resident=%d|host_only=%d|copy_ms=%.3f|run_ms=%.3f|free_ms=%.3f|alloc_ms=%.3f|total_ms=%.3f|cache=%s",
                gptoss_debug_phase(m->debug->current),
                (unsigned long long)m->debug->current->seq, pos, token, l,
                lr ? lr->selected_count : nk, lr ? lr->experts_ran : nk,
                lr ? lr->cuda_ran : 0, lr ? lr->cpu_ran : 0,
                lr ? lr->cache_hits : 0, lr ? lr->cache_misses : 0,
                lr ? lr->evictions : 0, m->n_experts, m->n_experts,
                resident_after, m->n_experts - resident_after,
                lr ? lr->copy_ms : 0.0, lr ? lr->run_ms : 0.0,
                lr ? lr->free_ms : 0.0, lr ? lr->alloc_ms : 0.0,
                lr ? lr->total_ms : 0.0, slots);
        }
    }

    if (!tensor_vector_profile(m, &m->output_norm, s->weight, m->hidden, -1, -1,
                               "output_norm_read", err, cap)) return 0;
    op0 = gptoss_debug_clock(m);
    coli_f32_rmsnorm(s->norm, s->x, s->weight, m->hidden, m->eps);
    if (op0) gptoss_debug_op(m, -1, -1, "run", "output_rmsnorm", "cpu", 0,
                             (now_sec() - op0) * 1000.0);
    int output_cuda = 0;
    if (!tensor_mm_profile(m, s->logits, s->norm, &m->output, m->hidden, m->vocab, 0,
                           -1, -1, "output_vocab_projection", 1, &output_cuda, err, cap))
        return errf(err, cap, "output projection failed");
    if (debug)
        gptoss_debug_emit(m, "event=forward_end|phase=%s|token_seq=%llu|pos=%d|token=%d|logits_ready=1",
                          gptoss_debug_phase(m->debug->current),
                          (unsigned long long)m->debug->current->seq, pos, token);
    if (trace_forward) {
        gptoss_trace_delta("first forward total", trace_forward_point);
        fprintf(stderr, "[GPT-OSS-TRACE] forward complete; logits are ready\n");
        fflush(stderr);
    }
    return 1;
}

static int load_model_with_usage_debug(GptOssModel *m, const char *path, const char *usage_file,
                                       int context, int verbose, ColiExec exec,
                                       int debug, const char *debug_dir,
                                       char *err, size_t cap) {
    memset(m, 0, sizeof(*m)); m->gguf.fd = -1; m->verbose = verbose; m->exec = exec;
    if (!gptoss_debug_init(m, debug, debug_dir, path, err, cap)) return 0;
    double t0 = gptoss_debug_clock(m);
    if (!coli_gguf_open(&m->gguf, path))
        return errf(err, cap, "cannot open model: %s", coli_gguf_error(&m->gguf));
    if (t0) gptoss_debug_op(m, -1, -1, "alloc", "gguf_open_mmap", "host", 0,
                            (now_sec() - t0) * 1000.0);

    /* Preserve split/preload placement from the current runtime. */
    coli_gguf_set_preload_backend(&m->gguf, exec.kind, exec.device);

    t0 = gptoss_debug_clock(m);
    if (!model_config(m, err, cap)) return 0;
    if (t0) gptoss_debug_op(m, -1, -1, "alloc", "model_config", "host", 0,
                            (now_sec() - t0) * 1000.0);

    t0 = gptoss_debug_clock(m);
    if (!coli_gguf_tokenizer_load(&m->tokenizer, &m->gguf, err, cap)) return 0;
    if (t0) gptoss_debug_op(m, -1, -1, "alloc", "tokenizer_load", "host", 0,
                            (now_sec() - t0) * 1000.0);

    t0 = gptoss_debug_clock(m);
    if (!model_load_weights(m, err, cap)) return 0;
    if (t0) gptoss_debug_op(m, -1, -1, "alloc", "bind_model_weights", "host_mmap", 0,
                            (now_sec() - t0) * 1000.0);

    t0 = gptoss_debug_clock(m);
    if (!model_build_expert_views(m, err, cap)) return 0;
    if (t0) gptoss_debug_op(m, -1, -1, "alloc", "build_expert_views_and_cache", "host", 0,
                            (now_sec() - t0) * 1000.0);

    if (!gptoss_usage_configure(m, path, usage_file, err, cap)) return 0;

    t0 = gptoss_debug_clock(m);
    if (!alloc_cache(m, context, err, cap)) return 0;
    if (t0) gptoss_debug_op(m, -1, -1, "alloc", "kv_cache_alloc", "host",
                            2u * (size_t)m->n_layers * (size_t)context *
                            (size_t)m->kv_dim * sizeof(float),
                            (now_sec() - t0) * 1000.0);

    if (coli_gguf_tokenizer_vocab_size(m->tokenizer) != m->vocab)
        return errf(err, cap, "tokenizer/model vocabulary mismatch");

    try_reside_dense_cuda(m);

    t0 = gptoss_debug_clock(m);
    const int64_t history = gptoss_usage_load(m);
    if (t0) gptoss_debug_op(m, -1, -1, "copy", "usage_history_load", "disk_to_host", 0,
                            (now_sec() - t0) * 1000.0);

    fprintf(stderr,
        "[USAGE] GPT-OSS path=%s loaded=%lld selections\n",
        m->usage_path, (long long)history);
    if (verbose) fprintf(stderr,
        "[GPT-OSS] layers=%d hidden=%d q=%d kv=%d heads=%d/%d experts=%d top=%d ff=%d vocab=%d context=%d backend=%s\n",
        m->n_layers,m->hidden,m->q_dim,m->kv_dim,m->n_heads,m->n_kv_heads,m->n_experts,m->top_k,
        m->expert_ff,m->vocab,context,exec.kind==COLI_BACKEND_CUDA?"cuda-hybrid":"cpu");
    if (m->debug && m->debug->enabled)
        gptoss_debug_emit(m,
            "event=model_ready|layers=%d|experts_per_layer=%d|top_k=%d|cache_per_layer=%d|context=%d|dense_cuda_bytes=%zu|usage_history=%lld",
            m->n_layers, m->n_experts, m->top_k,
            m->n_layers ? m->layers[0].cache_cap : 0, context,
            m->cuda_dense_bytes, (long long)history);
    return 1;
}

static int load_model_with_usage(GptOssModel *m, const char *path, const char *usage_file,
                                 int context, int verbose, ColiExec exec, char *err, size_t cap) {
    return load_model_with_usage_debug(m, path, usage_file, context, verbose, exec,
                                       0, NULL, err, cap);
}

/* Keep the original internal API used by tests and embedders. */
static int load_model(GptOssModel *m, const char *path, int context, int verbose,
                      ColiExec exec, char *err, size_t cap) {
    return load_model_with_usage(m, path, NULL, context, verbose, exec, err, cap);
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


static void gptoss_light_print_escaped(FILE *out, const char *piece, int n) {
    for (int i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)piece[i];
        if (c == '\n') fputs("\\n", out);
        else if (c == '\r') fputs("\\r", out);
        else if (c == '\t') fputs("\\t", out);
        else if (c == '\\') fputs("\\\\", out);
        else if (c == '"') fputs("\\\"", out);
        else if (c >= 32 && c != 127) fputc((int)c, out);
        else fprintf(out, "\\x%02x", (unsigned)c);
    }
}

static void gptoss_light_token_generated(int index, int maximum, int token,
                                         int control, int stop,
                                         const char *piece, int n) {
    fprintf(stderr, "\n[LIGHT] token %d/%d generated id=%d", index, maximum, token);
    if (stop) fputs(" stop=yes", stderr);
    else if (control) fputs(" control=yes", stderr);
    else {
        fputs(" text=\"", stderr);
        if (n > 0) gptoss_light_print_escaped(stderr, piece, n);
        fputc('"', stderr);
    }
    fputc('\n', stderr);
    fflush(stderr);
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s [--gguf] MODEL --prompt TEXT [--max-tokens N] [--context N] "
        "[--device cpu|cuda[:N]] [--usage-file PATH] [--raw-prompt] [--verbose] "
        "[--debug] [--debug-dir DIR] [--debug-light]\n", p);
}

int coli_gptoss_run_cli(int argc, char **argv) {
    const char *model_path=NULL,*prompt=NULL,*device_arg="cpu",*usage_file=NULL,*debug_dir=NULL;
    int max_tokens=24,context=0,raw=0,verbose=0,debug=0,debug_light=0;
    int i=1; if(i<argc&&!strcmp(argv[i],"--gguf"))++i; if(i<argc)model_path=argv[i++];
    for(;i<argc;++i){
        if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"--max-tokens")&&i+1<argc)max_tokens=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--context")&&i+1<argc)context=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--device")&&i+1<argc)device_arg=argv[++i];
        else if(!strcmp(argv[i],"--usage-file")&&i+1<argc)usage_file=argv[++i];
        else if(!strcmp(argv[i],"--raw-prompt"))raw=1;
        else if(!strcmp(argv[i],"--verbose")||!strcmp(argv[i],"-v"))verbose=1;
        else if(!strcmp(argv[i],"--debug-light"))debug_light=1;
        else if(!strcmp(argv[i],"--debug"))debug=1;
        else if(!strcmp(argv[i],"--debug-dir")&&i+1<argc){debug=1;debug_dir=argv[++i];}
        else if(!strcmp(argv[i],"--mtp-draft")&&i+1<argc)++i;
        else if(!strcmp(argv[i],"--no-mtp")){}
        else {fprintf(stderr,"unknown GPT-OSS option: %s\n",argv[i]);usage(argv[0]);return 2;}
    }
    if(!model_path||!prompt||max_tokens<0){usage(argv[0]);return 2;}
    /* RAM preload dots are always on; [LIGHT] token diagnostics are opt-in. */
#ifdef _WIN32
    _putenv_s("COLI_RAM_PROGRESS", "1");
    if(debug_light) _putenv_s("COLI_DEBUG_LIGHT", "1");
    else _putenv_s("COLI_DEBUG_LIGHT", "");
#else
    setenv("COLI_RAM_PROGRESS", "1", 1);
    if(debug_light) setenv("COLI_DEBUG_LIGHT", "1", 1);
    else unsetenv("COLI_DEBUG_LIGHT");
#endif
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

    GptOssModel m; memset(&m,0,sizeof(m)); m.gguf.fd=-1;
    GptOssScratch s={0};double t0=now_sec();
    if(!load_model_with_usage_debug(&m,model_path,usage_file,context,verbose,exec,
                                    debug,debug_dir,err,sizeof(err))){
        fprintf(stderr,"%s\n",err);model_free(&m);free(ids);
#ifdef COLI_CUDA
        if(cuda_started)coli_cuda_shutdown();
#endif
        return 1;
    }
    double alloc0=gptoss_debug_clock(&m);
    if(!scratch_alloc(&m,&s,err,sizeof(err))){
        fprintf(stderr,"%s\n",err);if(m.debug)m.debug->exit_status=1;model_free(&m);free(ids);
#ifdef COLI_CUDA
        if(cuda_started)coli_cuda_shutdown();
#endif
        return 1;
    }
    if(alloc0)gptoss_debug_op(&m,-1,-1,"alloc","scratch_alloc","host",0,(now_sec()-alloc0)*1000.0);
    if (gptoss_trace_enabled()) {
        fprintf(stderr, "[GPT-OSS-TRACE] model and scratch ready; prompt_tokens=%d max_tokens=%d\n",
                n_prompt, max_tokens);
        fflush(stderr);
    }
    if(m.debug&&m.debug->enabled)
        gptoss_debug_emit(&m,"event=inference_start|prompt_tokens=%d|max_tokens=%d|raw_prompt=%d|context=%d",
                          n_prompt,max_tokens,raw,context);

    for (int p = 0; p < n_prompt; ++p) {
        GptOssTracePoint trace_prompt_point = gptoss_trace_point();
        if (gptoss_trace_enabled()) {
            fprintf(stderr, "[GPT-OSS-TRACE] prompt forward %d/%d begin token=%d\n",
                    p + 1, n_prompt, ids[p]);
            fflush(stderr);
        }
        if(!gptoss_debug_begin_token(&m,ids[p],p,1)){
            fprintf(stderr,"cannot allocate debug token record\n");goto fail;
        }
        if (!model_forward(&m, &s, ids[p], p, err, sizeof(err))) {
            gptoss_debug_end_token(&m,0);
            fprintf(stderr, "%s\n", err);
            goto fail;
        }
        gptoss_debug_end_token(&m,1);
        {
            char input_piece[4096];
            int input_n = coli_gguf_tokenizer_decode(
                m.tokenizer, &ids[p], 1, input_piece, (int)sizeof(input_piece));
            if (input_n > 0) {
                fwrite(input_piece, 1, (size_t)input_n, stdout);
                fflush(stdout);
            }
        }
        if (gptoss_trace_enabled()) {
            char trace_label[96];
            snprintf(trace_label, sizeof(trace_label), "prompt forward %d/%d", p + 1, n_prompt);
            gptoss_trace_delta(trace_label, trace_prompt_point);
        }
    }
    if (gptoss_trace_enabled()) {
        fprintf(stderr, "[GPT-OSS-TRACE] GENERATION START: prompt complete, logits ready\n");
        fflush(stderr);
    }
    if(m.debug&&m.debug->enabled)
        gptoss_debug_emit(&m,"event=generation_start|prompt_tokens=%d|next_pos=%d",n_prompt,n_prompt);

    int pos=n_prompt,generated=0;
    int stop_eos=coli_gguf_tokenizer_eos(m.tokenizer);
    int stop_return=coli_gguf_tokenizer_id(m.tokenizer,"<|return|>");
    int stop_call=coli_gguf_tokenizer_id(m.tokenizer,"<|call|>");
    if(debug_light&&max_tokens>0){
        fprintf(stderr,"\n[LIGHT] now generating token 1/%d\n",max_tokens);
        fflush(stderr);
    }
    while(generated<max_tokens){
        double op0=gptoss_debug_clock(&m);
        int next=argmax(s.logits,m.vocab);
        if(op0)gptoss_debug_op(&m,-1,-1,"run","argmax","cpu",0,(now_sec()-op0)*1000.0);
        /* Harmony can emit several assistant messages. <|end|> separates them;
         * only EOS, <|return|>, and <|call|> terminate this sampling pass. */
        int stop=next==stop_eos||next==stop_return||next==stop_call;
        int control=coli_gguf_tokenizer_is_control(m.tokenizer,next);
        char piece[4096];int n=0;
        if(!stop&&!control){
            op0=gptoss_debug_clock(&m);
            n=coli_gguf_tokenizer_decode(m.tokenizer,&next,1,piece,sizeof(piece));
            if(op0)gptoss_debug_op(&m,-1,-1,"run","token_decode","cpu",0,(now_sec()-op0)*1000.0);
            if(n<0)n=0;
        }
        gptoss_debug_log_output(&m,generated,next,control,stop,piece,n);
        if(debug_light)gptoss_light_token_generated(generated+1,max_tokens,next,control,stop,piece,n);
        if(stop){
            if(m.debug&&m.debug->enabled)
                gptoss_debug_emit(&m,"event=generation_stop|reason=stop_token|generated=%d|token=%d",generated,next);
            break;
        }
        if(!control&&n>0){fwrite(piece,1,(size_t)n,stdout);fflush(stdout);}
        ++generated;
        if(generated>=max_tokens){
            if(m.debug&&m.debug->enabled)
                gptoss_debug_emit(&m,"event=generation_stop|reason=max_tokens|generated=%d",generated);
            break;
        }
        if(debug_light){
            fprintf(stderr,"\n[LIGHT] now generating token %d/%d\n",generated+1,max_tokens);
            fflush(stderr);
        }
        if(!gptoss_debug_begin_token(&m,next,pos,0)){
            fprintf(stderr,"cannot allocate debug token record\n");goto fail;
        }
        if(!model_forward(&m,&s,next,pos++,err,sizeof(err))){
            gptoss_debug_end_token(&m,0);
            fprintf(stderr,"\n%s\n",err);goto fail;
        }
        gptoss_debug_end_token(&m,1);
    }
    fputc('\n',stdout);
    if(verbose)fprintf(stderr,"[GPT-OSS] prompt=%d generated=%d elapsed=%.2fs %.3f tok/s total\n",
                       n_prompt,generated,now_sec()-t0,(n_prompt+generated)/(now_sec()-t0));
    if(m.debug&&m.debug->enabled)
        gptoss_debug_emit(&m,"event=inference_end|status=ok|prompt=%d|generated=%d|elapsed_s=%.3f",
                          n_prompt,generated,now_sec()-t0);
    {
        double save0=gptoss_debug_clock(&m);
        int saved=gptoss_usage_save(&m);
        if(save0)gptoss_debug_op(&m,-1,-1,"copy","usage_history_save","host_to_disk",0,(now_sec()-save0)*1000.0);
        if(!saved)fprintf(stderr,"[USAGE] failed to save GPT-OSS usage: %s\n",m.usage_path);
        else fprintf(stderr,"[USAGE] saved GPT-OSS usage: %s\n",m.usage_path);
    }
    if(m.debug)m.debug->exit_status=0;
    {
        double free0=gptoss_debug_clock(&m);scratch_free(&s);
        if(free0)gptoss_debug_op(&m,-1,-1,"free","scratch_free","host",0,(now_sec()-free0)*1000.0);
    }
    model_free(&m);free(ids);
#ifdef COLI_CUDA
    if(cuda_started)coli_cuda_shutdown();
#endif
    return 0;
fail:
    if(m.debug&&m.debug->current)gptoss_debug_end_token(&m,0);
    if(m.debug&&m.debug->enabled)
        gptoss_debug_emit(&m,"event=inference_end|status=error|message=%s",err);
    {
        double save0=gptoss_debug_clock(&m);int saved=gptoss_usage_save(&m);
        if(save0)gptoss_debug_op(&m,-1,-1,"copy","usage_history_save","host_to_disk",0,(now_sec()-save0)*1000.0);
        if(!saved)fprintf(stderr,"[USAGE] failed to save GPT-OSS usage: %s\n",m.usage_path);
    }
    if(m.debug)m.debug->exit_status=1;
    {
        double free0=gptoss_debug_clock(&m);scratch_free(&s);
        if(free0)gptoss_debug_op(&m,-1,-1,"free","scratch_free","host",0,(now_sec()-free0)*1000.0);
    }
    model_free(&m);free(ids);
#ifdef COLI_CUDA
    if(cuda_started)coli_cuda_shutdown();
#endif
    return 1;
}
