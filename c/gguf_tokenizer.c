#include "gguf_tokenizer.h"
#include "tok.h"

#include <stdarg.h>
#include <time.h>

struct ColiGgufTokenizer {
    Tok tok;
    char **tokens;
    uint64_t token_count;
    int bos_id;
    int eos_id;
    int add_bos;
    char *chat_template;
};

static int fail(char *error, size_t cap, const char *fmt, ...) {
    if (error && cap) {
        va_list ap; va_start(ap, fmt); vsnprintf(error, cap, fmt, ap); va_end(ap);
    }
    return 0;
}

static int next_pow2_capacity(uint64_t n, int *out) {
    uint64_t cap = 1;
    while (cap < n * 2 + 1) {
        if (cap > (uint64_t)INT_MAX / 2) return 0;
        cap <<= 1;
    }
    *out = (int)cap;
    return 1;
}

static int read_optional_id(const ColiGgufFile *g, const char *key, int fallback) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    uint64_t u;
    return kv && coli_gguf_kv_read_u64(g, kv, &u) && u <= INT_MAX ? (int)u : fallback;
}

int coli_gguf_tokenizer_load(ColiGgufTokenizer **out, const ColiGgufFile *gguf,
                             char *error, size_t error_size) {
    if (!out || !gguf) return fail(error, error_size, "invalid tokenizer arguments");
    *out = NULL;
    const ColiGgufKV *tokens_kv = coli_gguf_find_kv(gguf, "tokenizer.ggml.tokens");
    const ColiGgufKV *merges_kv = coli_gguf_find_kv(gguf, "tokenizer.ggml.merges");
    if (!tokens_kv || !merges_kv) return fail(error, error_size, "GGUF tokenizer tokens/merges missing");

    char **tokens = NULL, **merges = NULL;
    uint64_t nt = 0, nm = 0;
    if (!coli_gguf_kv_read_string_array(gguf, tokens_kv, &tokens, &nt) || nt == 0 || nt > INT_MAX)
        return fail(error, error_size, "cannot read tokenizer.ggml.tokens");
    if (!coli_gguf_kv_read_string_array(gguf, merges_kv, &merges, &nm)) {
        coli_gguf_free_string_array(tokens, nt);
        return fail(error, error_size, "cannot read tokenizer.ggml.merges");
    }

    uint32_t *types = NULL; uint64_t ntypes = 0;
    const ColiGgufKV *types_kv = coli_gguf_find_kv(gguf, "tokenizer.ggml.token_type");
    if (types_kv && (!coli_gguf_kv_read_u32_array(gguf, types_kv, &types, &ntypes) || ntypes != nt)) {
        coli_gguf_free_string_array(tokens, nt); coli_gguf_free_string_array(merges, nm); free(types);
        return fail(error, error_size, "invalid tokenizer.ggml.token_type");
    }

    ColiGgufTokenizer *gt = (ColiGgufTokenizer *)calloc(1, sizeof(*gt));
    if (!gt) goto oom;
    Tok *T = &gt->tok;
    tk_build_bytemap(T);
    T->n_ids = (int)nt;
    T->id2str = (char **)calloc(nt, sizeof(char *));
    T->id_added = (int *)calloc(nt, sizeof(int));
    T->id_special = (int *)calloc(nt, sizeof(int));
    int vcap, mcap;
    if (!T->id2str || !T->id_added || !T->id_special ||
        !next_pow2_capacity(nt, &vcap) || !next_pow2_capacity(nm, &mcap)) goto oom;
    hm_init(&T->vocab, vcap);
    hm_init(&T->merges, mcap);
    if (!T->vocab.e || !T->merges.e) goto oom;

    int nsp = 0;
    for (uint64_t i = 0; i < nt; ++i) {
        T->id2str[i] = tokens[i];
        hm_put(&T->vocab, tokens[i], (int)strlen(tokens[i]), (int)i);
        const uint32_t type = types ? types[i] : 1;
        if (type == 3 || type == 4) ++nsp; /* CONTROL or USER_DEFINED */
    }
    T->sp = nsp ? (Special *)calloc((size_t)nsp, sizeof(Special)) : NULL;
    if (nsp && !T->sp) goto oom;
    T->nsp = nsp;
    for (uint64_t i = 0, p = 0; i < nt; ++i) {
        const uint32_t type = types ? types[i] : 1;
        if (type == 3 || type == 4) {
            T->sp[p].str = tokens[i]; T->sp[p].len = (int)strlen(tokens[i]); T->sp[p].id = (int)i; ++p;
            T->id_added[i] = 1;
            if (type == 3) T->id_special[i] = 1;
        }
    }
    if (nsp) qsort(T->sp, (size_t)nsp, sizeof(Special), cmp_sp_len);

    for (uint64_t i = 0; i < nm; ++i) {
        char *space = strchr(merges[i], ' ');
        if (!space || space == merges[i] || !space[1]) {
            coli_gguf_tokenizer_destroy(gt); gt = NULL;
            coli_gguf_free_string_array(merges, nm); free(types);
            return fail(error, error_size, "malformed GGUF merge at index %llu", (unsigned long long)i);
        }
        const int ll = (int)(space - merges[i]);
        const int rl = (int)strlen(space + 1);
        char *key = (char *)malloc((size_t)ll + 1 + (size_t)rl);
        if (!key) goto oom;
        memcpy(key, merges[i], (size_t)ll); key[ll] = '\0'; memcpy(key + ll + 1, space + 1, (size_t)rl);
        hm_put(&T->merges, key, ll + 1 + rl, (int)i);
    }

    char *pre = NULL;
    const ColiGgufKV *pre_kv = coli_gguf_find_kv(gguf, "tokenizer.ggml.pre");
    if (pre_kv) coli_gguf_kv_read_string(gguf, pre_kv, &pre);
    if (pre && strcmp(pre, "refact") == 0) T->refact = 1;
    else if (pre && (strstr(pre, "o200k") || strstr(pre, "gpt-4o"))) T->o200k = 1;
    free(pre);

    const ColiGgufKV *chat_kv = coli_gguf_find_kv(gguf, "tokenizer.chat_template");
    if (chat_kv && !coli_gguf_kv_read_string(gguf, chat_kv, &gt->chat_template)) {
        goto oom;
    }

    gt->tokens = tokens; gt->token_count = nt;
    gt->bos_id = read_optional_id(gguf, "tokenizer.ggml.bos_token_id", -1);
    gt->eos_id = read_optional_id(gguf, "tokenizer.ggml.eos_token_id", -1);
    const ColiGgufKV *add_bos_kv = coli_gguf_find_kv(gguf, "tokenizer.ggml.add_bos_token");
    if (add_bos_kv) coli_gguf_kv_read_bool(gguf, add_bos_kv, &gt->add_bos);

    coli_gguf_free_string_array(merges, nm); free(types);
    *out = gt;
    if (error && error_size) error[0] = '\0';
    return 1;

oom:
    if (gt) coli_gguf_tokenizer_destroy(gt); else coli_gguf_free_string_array(tokens, nt);
    coli_gguf_free_string_array(merges, nm); free(types);
    return fail(error, error_size, "out of memory loading GGUF tokenizer");
}

void coli_gguf_tokenizer_destroy(ColiGgufTokenizer *gt) {
    if (!gt) return;
    Tok *T = &gt->tok;
    if (T->merges.e) {
        for (int i = 0; i < T->merges.cap; ++i) if (T->merges.e[i].used) free((void *)T->merges.e[i].k);
    }
    free(T->vocab.e); free(T->merges.e); free(T->id2str); free(T->id_added); free(T->id_special); free(T->sp);
    coli_gguf_free_string_array(gt->tokens, gt->token_count); free(gt->chat_template);
    free(gt);
}

int coli_gguf_tokenizer_encode(ColiGgufTokenizer *gt, const char *text, int *ids, int capacity) {
    if (!gt || !text || !ids || capacity <= 0) return -1;
    int off = 0;
    if (gt->add_bos && gt->bos_id >= 0 && off < capacity) ids[off++] = gt->bos_id;
    return off + tok_encode(&gt->tok, text, (int)strlen(text), ids + off, capacity - off);
}
int coli_gguf_tokenizer_decode(ColiGgufTokenizer *gt, const int *ids, int count, char *out, int capacity) {
    if (!gt || !ids || count < 0 || !out || capacity <= 0) return -1;
    return tok_decode(&gt->tok, ids, count, out, capacity - 1);
}
int coli_gguf_tokenizer_id(const ColiGgufTokenizer *gt, const char *text) {
    if (!gt || !text) return -1;
    return hm_get((hmap *)&gt->tok.vocab, text, (int)strlen(text));
}
int coli_gguf_tokenizer_eos(const ColiGgufTokenizer *gt) { return gt ? gt->eos_id : -1; }
int coli_gguf_tokenizer_bos(const ColiGgufTokenizer *gt) { return gt ? gt->bos_id : -1; }
int coli_gguf_tokenizer_is_control(const ColiGgufTokenizer *gt, int id) {
    return gt && id >= 0 && id < gt->tok.n_ids ? gt->tok.id_special[id] : 0;
}
int coli_gguf_tokenizer_vocab_size(const ColiGgufTokenizer *gt) { return gt ? gt->tok.n_ids : 0; }

static char *format_granite_prompt(const char *user_prompt, int include_date) {
    static const char *months[] = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    };
    static const char role_system[] = "<|start_of_role|>system<|end_of_role|>";
    static const char role_user[] = "<|start_of_role|>user<|end_of_role|>";
    static const char role_assistant[] = "<|start_of_role|>assistant<|end_of_role|>";
    static const char eot[] = "<|end_of_text|>\n";
    static const char sys_cutoff[] = "Knowledge Cutoff Date: April 2024.\n";
    static const char sys_date[] = "Today's Date: ";
    static const char sys_identity[] = "You are Granite, developed by IBM. You are a helpful AI assistant.";
    char date[64] = {0};
    if (!user_prompt) return NULL;
    if (include_date) {
        time_t now = time(NULL);
        struct tm tm_now;
        if (now == (time_t)-1 || !localtime_r(&now, &tm_now) || tm_now.tm_mon < 0 || tm_now.tm_mon > 11) {
            return NULL;
        }
        if (snprintf(date, sizeof(date), "%s %02d, %04d.\n", months[tm_now.tm_mon],
                     tm_now.tm_mday, tm_now.tm_year + 1900) >= (int)sizeof(date)) return NULL;
    }

    const size_t lengths[] = {
        sizeof(role_system)-1, sizeof(sys_cutoff)-1,
        include_date ? sizeof(sys_date)-1 : 0, include_date ? strlen(date) : 0,
        sizeof(sys_identity)-1, sizeof(eot)-1,
        sizeof(role_user)-1, strlen(user_prompt), sizeof(eot)-1,
        sizeof(role_assistant)-1
    };
    size_t total = 0;
    for (size_t i = 0; i < sizeof(lengths)/sizeof(lengths[0]); ++i) {
        if (total > SIZE_MAX - lengths[i]) return NULL;
        total += lengths[i];
    }
    if (total == SIZE_MAX) return NULL;
    char *out = (char *)malloc(total + 1);
    if (!out) return NULL;
    char *w = out;
#define APPEND_TEXT(text, len) do { memcpy(w, (text), (len)); w += (len); } while (0)
    APPEND_TEXT(role_system, sizeof(role_system)-1);
    APPEND_TEXT(sys_cutoff, sizeof(sys_cutoff)-1);
    if (include_date) { APPEND_TEXT(sys_date, sizeof(sys_date)-1); APPEND_TEXT(date, strlen(date)); }
    APPEND_TEXT(sys_identity, sizeof(sys_identity)-1);
    APPEND_TEXT(eot, sizeof(eot)-1);
    APPEND_TEXT(role_user, sizeof(role_user)-1);
    APPEND_TEXT(user_prompt, strlen(user_prompt));
    APPEND_TEXT(eot, sizeof(eot)-1);
    APPEND_TEXT(role_assistant, sizeof(role_assistant)-1);
#undef APPEND_TEXT
    *w = '\0';
    return out;
}

char *coli_gguf_tokenizer_format_granite_prompt(const ColiGgufTokenizer *gt,
                                                 const char *user_prompt) {
    /* Granite 3.1 GGUFs exist with both the original static template and the
     * later template that calls strftime_now(). Follow the embedded template. */
    const int include_date = !gt || !gt->chat_template ||
        strstr(gt->chat_template, "strftime_now") || strstr(gt->chat_template, "Today's Date:");
    return format_granite_prompt(user_prompt, include_date);
}

char *coli_granite_format_prompt(const char *user_prompt) {
    return format_granite_prompt(user_prompt, 1);
}
