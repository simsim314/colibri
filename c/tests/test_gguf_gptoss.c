#include "../gguf_gptoss.c"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#define unlink _unlink
#define getpid _getpid
#else
#include <unistd.h>
#endif

static int closef(float a, float b, float e) { return fabsf(a - b) <= e; }

static void put_u32(FILE *f, uint32_t v) {
    uint8_t p[4];
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
    assert(fwrite(p, 1, 4, f) == 4);
}
static void put_u64(FILE *f, uint64_t v) {
    uint8_t p[8];
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
    assert(fwrite(p, 1, 8, f) == 8);
}
static void put_f32(FILE *f, float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    put_u32(f, u);
}
static void put_str(FILE *f, const char *s) {
    size_t n = strlen(s);
    put_u64(f, n);
    assert(fwrite(s, 1, n, f) == n);
}
static void kv_str(FILE *f, const char *k, const char *v) {
    put_str(f, k); put_u32(f, COLI_GGUF_TYPE_STRING); put_str(f, v);
}
static void kv_u32(FILE *f, const char *k, uint32_t v) {
    put_str(f, k); put_u32(f, COLI_GGUF_TYPE_UINT32); put_u32(f, v);
}
static void kv_f32(FILE *f, const char *k, float v) {
    put_str(f, k); put_u32(f, COLI_GGUF_TYPE_FLOAT32); put_f32(f, v);
}
static void kv_bool(FILE *f, const char *k, int v) {
    put_str(f, k); put_u32(f, COLI_GGUF_TYPE_BOOL); fputc(v ? 1 : 0, f);
}
static void kv_str_array(FILE *f, const char *k, const char **v, int n) {
    put_str(f, k); put_u32(f, COLI_GGUF_TYPE_ARRAY); put_u32(f, COLI_GGUF_TYPE_STRING); put_u64(f, (uint64_t)n);
    for (int i = 0; i < n; ++i) put_str(f, v[i]);
}
static void kv_u32_array(FILE *f, const char *k, const uint32_t *v, int n) {
    put_str(f, k); put_u32(f, COLI_GGUF_TYPE_ARRAY); put_u32(f, COLI_GGUF_TYPE_UINT32); put_u64(f, (uint64_t)n);
    for (int i = 0; i < n; ++i) put_u32(f, v[i]);
}
static uint64_t align32(uint64_t x) { return (x + 31u) & ~31u; }

typedef struct {
    const char *name;
    int nd;
    uint64_t d[3];
    uint64_t off;
    const float *data;
    size_t count;
} TD;

static void tensor_desc(FILE *f, const TD *t) {
    put_str(f, t->name); put_u32(f, (uint32_t)t->nd);
    for (int i = 0; i < t->nd; ++i) put_u64(f, t->d[i]);
    put_u32(f, COLI_DTYPE_F32); put_u64(f, t->off);
}
static void pad32(FILE *f) { while (ftell(f) % 32) fputc(0, f); }
static void pad_to(FILE *f, long base, uint64_t off) {
    while ((uint64_t)(ftell(f) - base) < off) fputc(0, f);
    assert((uint64_t)(ftell(f) - base) == off);
}

static void write_fixture(const char *path) {
    static const float eye4[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1,
    };
    static const float ones4[4] = {1,1,1,1};
    static const float zero4[4] = {0,0,0,0};
    static const float zero2[2] = {0,0};
    static const float router_bias[2] = {1,-1};
    static const float zero8[8] = {0};
    static const float zero16[16] = {0};
    static const float zero32[32] = {0};

    TD t[] = {
        {"token_embd.weight", 2, {4,4}, 0, eye4, 16},
        {"output_norm.weight", 1, {4}, 0, ones4, 4},
        {"output.weight", 2, {4,4}, 0, eye4, 16},
        {"blk.0.attn_norm.weight", 1, {4}, 0, ones4, 4},
        {"blk.0.post_attention_norm.weight", 1, {4}, 0, ones4, 4},
        {"blk.0.attn_q.weight", 2, {4,4}, 0, zero16, 16},
        {"blk.0.attn_q.bias", 1, {4}, 0, zero4, 4},
        {"blk.0.attn_k.weight", 2, {4,2}, 0, zero8, 8},
        {"blk.0.attn_v.weight", 2, {4,2}, 0, zero8, 8},
        {"blk.0.attn_k.bias", 1, {2}, 0, zero2, 2},
        {"blk.0.attn_v.bias", 1, {2}, 0, zero2, 2},
        {"blk.0.attn_output.weight", 2, {4,4}, 0, zero16, 16},
        {"blk.0.attn_output.bias", 1, {4}, 0, zero4, 4},
        {"blk.0.attn_sinks.weight", 1, {2}, 0, zero2, 2},
        {"blk.0.ffn_gate_inp.weight", 2, {4,2}, 0, zero8, 8},
        {"blk.0.ffn_gate_inp.bias", 1, {2}, 0, router_bias, 2},
        {"blk.0.ffn_gate_exps.weight", 3, {4,4,2}, 0, zero32, 32},
        {"blk.0.ffn_up_exps.weight", 3, {4,4,2}, 0, zero32, 32},
        {"blk.0.ffn_down_exps.weight", 3, {4,4,2}, 0, zero32, 32},
        {"blk.0.ffn_gate_exps.bias", 2, {4,2}, 0, zero8, 8},
        {"blk.0.ffn_up_exps.bias", 2, {4,2}, 0, zero8, 8},
        {"blk.0.ffn_down_exps.bias", 2, {4,2}, 0, zero8, 8},
    };
    const int nt = (int)(sizeof(t) / sizeof(t[0]));
    uint64_t off = 0;
    for (int i = 0; i < nt; ++i) {
        t[i].off = off;
        off = align32(off + t[i].count * sizeof(float));
    }

    FILE *f = fopen(path, "wb"); assert(f);
    fwrite("GGUF", 1, 4, f); put_u32(f, 3); put_u64(f, (uint64_t)nt); put_u64(f, 28);
    kv_str(f, "general.architecture", "gpt-oss");
    kv_u32(f, "gpt-oss.block_count", 1);
    kv_u32(f, "gpt-oss.context_length", 8);
    kv_u32(f, "gpt-oss.embedding_length", 4);
    kv_u32(f, "gpt-oss.feed_forward_length", 4);
    kv_u32(f, "gpt-oss.attention.head_count", 2);
    kv_u32(f, "gpt-oss.attention.head_count_kv", 1);
    kv_u32(f, "gpt-oss.attention.key_length", 2);
    kv_u32(f, "gpt-oss.attention.value_length", 2);
    kv_u32(f, "gpt-oss.attention.sliding_window", 2);
    kv_u32(f, "gpt-oss.expert_count", 2);
    kv_u32(f, "gpt-oss.expert_used_count", 1);
    kv_u32(f, "gpt-oss.expert_feed_forward_length", 4);
    kv_f32(f, "gpt-oss.attention.layer_norm_rms_epsilon", 1e-5f);
    kv_f32(f, "gpt-oss.rope.freq_base", 10000.0f);
    kv_str(f, "gpt-oss.rope.scaling.type", "yarn");
    kv_f32(f, "gpt-oss.rope.scaling.factor", 1.0f);
    kv_u32(f, "gpt-oss.rope.scaling.original_context_length", 8);
    kv_f32(f, "gpt-oss.rope.scaling.yarn_beta_fast", 32.0f);
    kv_f32(f, "gpt-oss.rope.scaling.yarn_beta_slow", 1.0f);
    kv_str(f, "tokenizer.ggml.model", "gpt2");
    kv_str(f, "tokenizer.ggml.pre", "refact");
    const char *tokens[] = {"<eos>", "A", "B", "C"};
    const uint32_t types[] = {3,1,1,1};
    kv_str_array(f, "tokenizer.ggml.tokens", tokens, 4);
    kv_u32_array(f, "tokenizer.ggml.token_type", types, 4);
    kv_str_array(f, "tokenizer.ggml.merges", NULL, 0);
    kv_u32(f, "tokenizer.ggml.eos_token_id", 0);
    kv_u32(f, "tokenizer.ggml.bos_token_id", 0);
    kv_bool(f, "tokenizer.ggml.add_bos_token", 0);

    for (int i = 0; i < nt; ++i) tensor_desc(f, &t[i]);
    pad32(f); long base = ftell(f);
    for (int i = 0; i < nt; ++i) {
        pad_to(f, base, t[i].off);
        for (size_t j = 0; j < t[i].count; ++j) put_f32(f, t[i].data[j]);
    }
    assert(fclose(f) == 0);
}


static int candidate_selected(const GptOssCacheCandidate *items, int count,
                              int layer, int eid) {
    for (int i = 0; i < count; ++i)
        if (items[i].layer == layer && items[i].eid == eid)
            return items[i].selected;
    return 0;
}

static void test_stats_cache_greedy(void) {
    GptOssCacheCandidate items[] = {
        {0, 0, 100, 100, 0, 0},
        {1, 0,  90,  30, 0, 0},
        {1, 1,  80,  30, 0, 0},
        {2, 0,   0,   0, 1, 0},
    };
    size_t bytes = 0;
    int selected = gptoss_cache_select_greedy(
        items, (int)(sizeof(items) / sizeof(items[0])), 60, &bytes);
    assert(selected == 3);
    assert(bytes == 60);
    assert(!candidate_selected(items, 4, 0, 0));
    assert(candidate_selected(items, 4, 1, 0));
    assert(candidate_selected(items, 4, 1, 1));
    assert(candidate_selected(items, 4, 2, 0));
}

static void test_cache_mode_flags(void) {
    GptOssModel m;
    char err[256];

    memset(&m, 0, sizeof(m));
    m.top_k = 4; m.n_experts = 128;
    assert(gptoss_cache_configure(&m, "stats", -1, err, sizeof(err)));
    assert(m.expert_cache_mode == GPTOSS_EXPERT_CACHE_STATS);

    memset(&m, 0, sizeof(m));
    m.top_k = 4; m.n_experts = 128;
    assert(gptoss_cache_configure(&m, "fixed", 2, err, sizeof(err)));
    assert(m.expert_cache_mode == GPTOSS_EXPERT_CACHE_FIXED);
    assert(m.expert_cache_fixed_per_layer == 2);

    memset(&m, 0, sizeof(m));
    m.top_k = 4; m.n_experts = 128;
    assert(gptoss_cache_configure(&m, NULL, 3, err, sizeof(err)));
    assert(m.expert_cache_mode == GPTOSS_EXPERT_CACHE_FIXED);
    assert(m.expert_cache_fixed_per_layer == 3);

    memset(&m, 0, sizeof(m));
    m.top_k = 4; m.n_experts = 128;
    assert(!gptoss_cache_configure(&m, "stats", 2, err, sizeof(err)));
}

static void test_repeat_penalty(void) {
    float logits[4] = {8.0f, -2.0f, 3.0f, 1.0f};
    const int history[4] = {0, 1, 0, 9};
    apply_repeat_penalty(logits, 4, history, 4, 2.0f);
    assert(closef(logits[0], 4.0f, 1e-7f));
    assert(closef(logits[1], -4.0f, 1e-7f));
    assert(closef(logits[2], 3.0f, 1e-7f));
    assert(closef(logits[3], 1.0f, 1e-7f));
}

static void test_mmap_read_pool(void) {
    uint8_t source[4][64];
    uint8_t output[4][64];
    GptOssMmapReadTask tasks[4];
    memset(tasks, 0, sizeof(tasks));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 64; ++j) source[i][j] = (uint8_t)(i * 64 + j);
        memset(output[i], 0, sizeof(output[i]));
        tasks[i].source = source[i];
        tasks[i].buffer = output[i];
        tasks[i].bytes = sizeof(output[i]);
    }
    GptOssMmapReadPool pool;
    assert(gptoss_mmap_read_pool_init(&pool, 3) == 3);
    gptoss_mmap_read_pool_run(&pool, tasks, 4);
    for (int i = 0; i < 4; ++i) {
        assert(tasks[i].ok);
        assert(memcmp(source[i], output[i], sizeof(output[i])) == 0);
        tasks[i].ok = 0;
        memset(output[i], 0, sizeof(output[i]));
    }
    gptoss_mmap_read_pool_run(&pool, tasks, 4);
    for (int i = 0; i < 4; ++i)
        assert(memcmp(source[i], output[i], sizeof(output[i])) == 0);
    gptoss_mmap_read_pool_destroy(&pool);
}

static void test_math_helpers(void) {
    assert(closef(coli_gptoss_oai_swiglu(0.0f, 0.0f), 0.0f, 1e-7f));
    float a = coli_gptoss_oai_swiglu(100.0f, 100.0f);
    float b = (7.0f / (1.0f + expf(-1.702f * 7.0f))) * 8.0f;
    assert(closef(a, b, 1e-5f));

    float inv[2] = {1.0f, 0.5f}; float v[4] = {1,2,3,4};
    coli_gptoss_yarn_rotate(v, 1, 4, 0, inv, 1.25f);
    assert(closef(v[0], 1.25f, 1e-6f) && closef(v[1], 2.5f, 1e-6f));
    assert(closef(v[2], 3.75f, 1e-6f) && closef(v[3], 5.0f, 1e-6f));

    float q[1] = {1}, k[2] = {0, 0.6931471805599453f}, val[2] = {10,20};
    float sink[1] = {0}, scores[2] = {0}, out[1] = {0};
    coli_gptoss_attention_sink_reference(out, q, k, val, sink, scores, 1, 0, 1, 1, 1, 1.0f);
    assert(closef(out[0], 12.5f, 1e-5f));
    coli_gptoss_attention_sink_reference(out, q, k, val, sink, scores, 1, 1, 1, 1, 1, 1.0f);
    assert(closef(out[0], 40.0f / 3.0f, 1e-5f));

    float logits[4] = {1,3,2,4}, w[2]; int idx[2];
    assert(coli_gptoss_topk_softmax(logits, 4, 2, idx, w) == 2);
    assert(idx[0] == 3 && idx[1] == 1);
    float e = expf(-1.0f);
    assert(closef(w[0], 1.0f / (1.0f + e), 1e-6f));
    assert(closef(w[1], e / (1.0f + e), 1e-6f));
}

static void test_model_load_and_forward(void) {
    char path[256];
#ifdef _WIN32
    snprintf(path, sizeof(path), "test_gptoss_%ld.gguf", (long)getpid());
#else
    snprintf(path, sizeof(path), "/tmp/test_gptoss_%ld.gguf", (long)getpid());
#endif
    write_fixture(path);
    char usage_path[320];
    snprintf(usage_path, sizeof(usage_path), "%s.coli_usage", path);
    unlink(usage_path);
    GptOssModel m;
    GptOssScratch s = {0};
    ColiExec exec = {COLI_BACKEND_CPU, 0};
    char err[512];
    assert(load_model(&m, path, 8, 0, exec, err, sizeof(err)));
    assert(m.n_layers == 1 && m.hidden == 4 && m.vocab == 4);
    assert(m.n_experts == 2 && m.top_k == 1 && m.expert_ff == 4);
    assert(scratch_alloc(&m, &s, err, sizeof(err)));
    assert(model_forward(&m, &s, 1, 0, err, sizeof(err)));
    assert(argmax(s.logits, m.vocab) == 1);
    assert(model_forward(&m, &s, 2, 1, err, sizeof(err)));
    assert(argmax(s.logits, m.vocab) == 2);
    assert(gptoss_usage_save(&m));
    FILE *usage = fopen(usage_path, "r");
    assert(usage);
    int layer = -1, expert = -1;
    unsigned count = 0, total = 0;
    while (fscanf(usage, "%d %d %u", &layer, &expert, &count) == 3) {
        assert(layer == 0);
        assert(expert >= 0 && expert < m.n_experts);
        total += count;
    }
    fclose(usage);
    assert(total == 2);
    scratch_free(&s);
    model_free(&m);
    unlink(usage_path);
    unlink(path);
}

int main(void) {
    test_stats_cache_greedy();
    test_cache_mode_flags();
    test_repeat_penalty();
    test_mmap_read_pool();
    test_math_helpers();
    test_model_load_and_forward();
    puts("test_gguf_gptoss: ok");
    return 0;
}
