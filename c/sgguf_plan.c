#include "gguf_reader.h"
#include "split_plan.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    const char *model;
    const char *fast_dir;
    const char *slow_dir;
    const char *prefix;
    const char *placements_output;
    const char *usage_file;
    uint64_t fast_reserve;
    uint64_t ram_breathing;
    uint64_t vram_breathing;
    uint64_t ram_runtime_override;
    uint64_t vram_workspace;
    uint64_t vram_runtime_override;
    int have_ram_runtime_override;
    int have_vram_runtime_override;
    int context;
    int device;
    int command_only;
} Options;

typedef struct {
    char architecture[64];
    uint64_t layers;
    uint64_t hidden;
    uint64_t heads;
    uint64_t kv_heads;
    uint64_t head_dim;
    uint64_t kv_dim;
    uint64_t q_dim;
    uint64_t experts;
    uint64_t top_k;
    uint64_t expert_ff;
    uint64_t vocab;
    uint64_t dense_vram_candidate_bytes;
    uint64_t runtime_ram_bytes;
    uint64_t runtime_vram_bytes;
    int dense_all_or_none;
} ModelProfile;

static uint64_t mib(double x) { return (uint64_t)(x * 1024.0 * 1024.0); }
static uint64_t gib(double x) { return (uint64_t)(x * 1024.0 * 1024.0 * 1024.0); }

static uint64_t sat_add(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static uint64_t sat_mul(uint64_t a, uint64_t b) {
    return a && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static const char *human(uint64_t value, char out[64]) {
    static const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double x = (double)value;
    int u = 0;
    while (x >= 1024.0 && u < 4) { x /= 1024.0; ++u; }
    snprintf(out, 64, "%.2f %s", x, units[u]);
    return out;
}

static int parse_double_arg(int argc, char **argv, int *i, double *out) {
    if (*i + 1 >= argc) return 0;
    char *end = NULL;
    errno = 0;
    double v = strtod(argv[++*i], &end);
    if (errno || end == argv[*i] || *end || v < 0.0) return 0;
    *out = v;
    return 1;
}

static int parse_int_arg(int argc, char **argv, int *i, int *out) {
    if (*i + 1 >= argc) return 0;
    char *end = NULL;
    errno = 0;
    long v = strtol(argv[++*i], &end, 10);
    if (errno || end == argv[*i] || *end || v < 0 || v > INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s MODEL --fast-dir DIR --slow-dir DIR [options]\n\n"
        "The planner uses current free disk, MemAvailable and free VRAM.\n"
        "MiB/GiB are binary units: 1 MiB = 1,048,576 bytes.\n\n"
        "Options:\n"
        "  --prefix NAME                    output basename\n"
        "  --usage-file PATH                cumulative layer/expert counts (required)\n"
        "  --fast-reserve-mib N             free fast-disk breathing room; default 384\n"
        "  --ram-breathing-mib N            free RAM after runtime allocations; default 384\n"
        "  --ram-reserve-mib N              compatibility alias for --ram-breathing-mib\n"
        "  --vram-breathing-mib N           free VRAM after runtime allocations; default 128\n"
        "  --vram-reserve-mib N             compatibility alias for --vram-breathing-mib\n"
        "  --context N                      context used for runtime estimates; default 4096\n"
        "  --vram-workspace-mib N           CUDA context/scratch reserve; default 256\n"
        "  --runtime-ram-mib N              override automatic runtime RAM estimate\n"
        "  --runtime-vram-mib N             override cache+workspace VRAM estimate\n"
        "  --device N                       NVIDIA GPU index; default 0\n"
        "  --placements-output FILE         optional TSV tensor placement report\n"
        "  --command-only                   print only the recommended split command\n",
        p);
}

static int parse_args(int argc, char **argv, Options *o) {
    if (argc < 2) return 0;
    memset(o, 0, sizeof(*o));
    o->model = argv[1];
    o->device = 0;
    o->context = 4096;
    o->fast_reserve = mib(384);
    o->ram_breathing = mib(384);
    o->vram_breathing = mib(128);
    o->vram_workspace = mib(256);

    for (int i = 2; i < argc; ++i) {
        double v;
        int iv;
        if (!strcmp(argv[i], "--fast-dir") && i + 1 < argc) o->fast_dir = argv[++i];
        else if (!strcmp(argv[i], "--slow-dir") && i + 1 < argc) o->slow_dir = argv[++i];
        else if (!strcmp(argv[i], "--prefix") && i + 1 < argc) o->prefix = argv[++i];
        else if (!strcmp(argv[i], "--placements-output") && i + 1 < argc) o->placements_output = argv[++i];
        else if (!strcmp(argv[i], "--usage-file") && i + 1 < argc) o->usage_file = argv[++i];
        else if (!strcmp(argv[i], "--fast-reserve-mib") && parse_double_arg(argc, argv, &i, &v)) o->fast_reserve = mib(v);
        else if ((!strcmp(argv[i], "--ram-breathing-mib") || !strcmp(argv[i], "--ram-reserve-mib")) && parse_double_arg(argc, argv, &i, &v)) o->ram_breathing = mib(v);
        else if ((!strcmp(argv[i], "--vram-breathing-mib") || !strcmp(argv[i], "--vram-reserve-mib")) && parse_double_arg(argc, argv, &i, &v)) o->vram_breathing = mib(v);
        else if (!strcmp(argv[i], "--vram-workspace-mib") && parse_double_arg(argc, argv, &i, &v)) o->vram_workspace = mib(v);
        else if (!strcmp(argv[i], "--runtime-ram-mib") && parse_double_arg(argc, argv, &i, &v)) {
            o->ram_runtime_override = mib(v); o->have_ram_runtime_override = 1;
        }
        else if (!strcmp(argv[i], "--runtime-vram-mib") && parse_double_arg(argc, argv, &i, &v)) {
            o->vram_runtime_override = mib(v); o->have_vram_runtime_override = 1;
        }
        else if (!strcmp(argv[i], "--context") && parse_int_arg(argc, argv, &i, &iv) && iv > 0) o->context = iv;
        else if (!strcmp(argv[i], "--device") && parse_int_arg(argc, argv, &i, &iv)) o->device = iv;
        else if (!strcmp(argv[i], "--command-only")) o->command_only = 1;
        else return 0;
    }
    return o->fast_dir && o->slow_dir && o->usage_file;
}

static int disk_free(const char *path, uint64_t *out) {
    struct statvfs v;
    if (statvfs(path, &v) != 0) return 0;
    *out = (uint64_t)v.f_bavail * (uint64_t)v.f_frsize;
    return 1;
}

static int mem_available(uint64_t *out) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[64], unit[32];
    unsigned long long value;
    int ok = 0;
    while (fscanf(f, "%63s %llu %31s", key, &value, unit) == 3) {
        if (!strcmp(key, "MemAvailable:")) {
            *out = (uint64_t)value * 1024u;
            ok = 1;
            break;
        }
    }
    fclose(f);
    return ok;
}

static int vram_info(int device, uint64_t *free_out, uint64_t *total_out) {
    char command[512];
    snprintf(command, sizeof(command),
        "nvidia-smi -i %d --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null",
        device);
    FILE *p = popen(command, "r");
    if (!p) return 0;
    unsigned long long free_mib = 0, total_mib = 0;
    int ok = fscanf(p, "%llu, %llu", &free_mib, &total_mib) == 2;
    pclose(p);
    if (!ok) return 0;
    *free_out = free_mib * 1024u * 1024u;
    *total_out = total_mib * 1024u * 1024u;
    return 1;
}

static void basename_prefix(const char *path, char out[256]) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    snprintf(out, 256, "%s", base);
    char *p = strstr(out, ".gguf");
    if (!p) p = strstr(out, ".sgguf");
    if (p) *p = 0;
    if (!*out) strcpy(out, "model");
}

static uint64_t kv_u64(const ColiGgufFile *g, const char *key, uint64_t fallback) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    uint64_t u;
    int64_t i;
    if (!kv) return fallback;
    if (coli_gguf_kv_read_u64(g, kv, &u)) return u;
    if (coli_gguf_kv_read_i64(g, kv, &i) && i >= 0) return (uint64_t)i;
    return fallback;
}

static void architecture_name(const ColiGgufFile *g, char out[64]) {
    snprintf(out, 64, "unknown");
    const ColiGgufKV *kv = coli_gguf_find_kv(g, "general.architecture");
    char *value = NULL;
    if (kv && coli_gguf_kv_read_string(g, kv, &value) && value) {
        snprintf(out, 64, "%s", value);
        free(value);
    }
}

static int contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle || !*needle) return 0;
    size_t n = strlen(needle);
    for (const char *p = haystack; *p; ++p) {
        size_t i = 0;
        while (i < n && p[i] &&
               (unsigned char)tolower((unsigned char)p[i]) ==
               (unsigned char)tolower((unsigned char)needle[i])) ++i;
        if (i == n) return 1;
    }
    return 0;
}

static void profile_model(const ColiGgufFile *model, const Options *o, ModelProfile *p) {
    memset(p, 0, sizeof(*p));
    architecture_name(model, p->architecture);

    if (!strcmp(p->architecture, "gpt-oss")) {
        p->layers = kv_u64(model, "gpt-oss.block_count", 36);
        p->hidden = kv_u64(model, "gpt-oss.embedding_length", 2880);
        p->heads = kv_u64(model, "gpt-oss.attention.head_count", 64);
        p->kv_heads = kv_u64(model, "gpt-oss.attention.head_count_kv", 8);
        p->head_dim = kv_u64(model, "gpt-oss.attention.key_length", 64);
        p->experts = kv_u64(model, "gpt-oss.expert_count", 128);
        p->top_k = kv_u64(model, "gpt-oss.expert_used_count", 4);
        p->expert_ff = kv_u64(model, "gpt-oss.expert_feed_forward_length", 2880);
        p->dense_all_or_none = 1;
    }

    const ColiGgufTensorInfo *emb = coli_gguf_find_tensor(model, "token_embd.weight");
    if (emb && emb->n_dims >= 2) {
        if (!p->hidden) p->hidden = emb->dims[0];
        p->vocab = emb->dims[1];
    }
    p->kv_dim = sat_mul(p->kv_heads, p->head_dim);
    p->q_dim = sat_mul(p->heads, p->head_dim);

    for (uint64_t i = 0; i < model->tensor_count; ++i) {
        const ColiGgufTensorInfo *t = &model->tensors[i];
        if (coli_split_is_vram_candidate(t))
            p->dense_vram_candidate_bytes = sat_add(p->dense_vram_candidate_bytes, t->payload_size);
    }

    if (!strcmp(p->architecture, "gpt-oss")) {
        uint64_t context = (uint64_t)o->context;
        uint64_t kv = sat_mul(2u, sat_mul(p->layers, sat_mul(context, sat_mul(p->kv_dim, 4u))));
        uint64_t scores = sat_mul(p->heads, sat_mul(context, 4u));
        uint64_t scratch_floats = 0;
        scratch_floats = sat_add(scratch_floats, sat_mul(8u, p->hidden));
        scratch_floats = sat_add(scratch_floats, sat_mul(3u, p->q_dim));
        scratch_floats = sat_add(scratch_floats, sat_mul(4u, p->kv_dim));
        scratch_floats = sat_add(scratch_floats, sat_mul(5u, p->expert_ff));
        scratch_floats = sat_add(scratch_floats, sat_mul(2u, p->experts));
        scratch_floats = sat_add(scratch_floats, p->vocab);
        uint64_t scratch = sat_add(sat_mul(scratch_floats, 4u), scores);
        p->runtime_ram_bytes = sat_add(sat_add(kv, scratch), mib(128));
    } else {
        p->runtime_ram_bytes = mib(256);
    }

    /* The runtime stats cache greedily consumes VRAM left after static
     * placement. Only CUDA context/scratch must be reserved here. */
    p->runtime_vram_bytes = o->vram_workspace;
    if (o->have_ram_runtime_override) p->runtime_ram_bytes = o->ram_runtime_override;
    if (o->have_vram_runtime_override) p->runtime_vram_bytes = o->vram_runtime_override;
}

static int write_placements(const char *path,
                            const ColiGgufFile *model,
                            const ColiSplitPlan *plan) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "location\tbytes\tMiB\tlayer\texpert\tusage\ttensor\n");
    for (uint64_t i = 0; i < plan->entry_count; ++i) {
        const ColiSplitDecision *d = &plan->entries[i];
        fprintf(f, "%s\t%" PRIu64 "\t%.3f\t%d\t%d\t%u\t%s\n",
                coli_split_location_name(d->location), d->bytes,
                d->bytes / (1024.0 * 1024.0), d->moe_layer,
                d->expert_id, d->usage_count,
                model->tensors[d->tensor_index].name);
    }
    return fclose(f) == 0;
}

static void print_command(const Options *o,
                          const char *fast_path,
                          const char *preload_path,
                          const char *tail_path,
                          uint64_t fast_budget,
                          uint64_t ram_budget,
                          uint64_t vram_budget) {
    printf("./c/sgguf-split '%s' \\\n", o->model);
    printf("  --usage-file '%s' \\\n", o->usage_file);
    printf("  --fast-output '%s' \\\n", fast_path);
    printf("  --preload-output '%s' \\\n", preload_path);
    printf("  --tail-output '%s' \\\n", tail_path);
    printf("  --fast-bytes %" PRIu64 " \\\n", fast_budget);
    printf("  --ram-bytes %" PRIu64 " \\\n", ram_budget);
    printf("  --vram-bytes %" PRIu64 " \\\n", vram_budget);
    printf("  --verify\n");
}

int main(int argc, char **argv) {
    Options o;
    if (!parse_args(argc, argv, &o)) { usage(argv[0]); return 2; }

    struct stat fast_st, slow_st;
    if (stat(o.fast_dir, &fast_st) || !S_ISDIR(fast_st.st_mode) ||
        stat(o.slow_dir, &slow_st) || !S_ISDIR(slow_st.st_mode)) {
        fprintf(stderr, "error: fast and slow directories must exist\n");
        return 1;
    }

    uint64_t free_disk = 0, slow_free = 0, available_ram = 0;
    uint64_t free_vram = 0, total_vram = 0;
    if (!disk_free(o.fast_dir, &free_disk)) { perror("statvfs fast-dir"); return 1; }
    if (!disk_free(o.slow_dir, &slow_free)) { perror("statvfs slow-dir"); return 1; }
    if (!mem_available(&available_ram)) {
        fprintf(stderr, "error: cannot read MemAvailable\n");
        return 1;
    }
    int have_vram = vram_info(o.device, &free_vram, &total_vram);

    ColiGgufFile model;
    memset(&model, 0, sizeof(model));
    model.fd = -1;
    if (!coli_gguf_open(&model, o.model)) {
        fprintf(stderr, "error: %s\n", coli_gguf_error(&model));
        return 1;
    }
    if (coli_gguf_is_split(&model)) {
        fprintf(stderr, "error: model is already split\n");
        coli_gguf_close(&model);
        return 1;
    }

    ModelProfile profile;
    profile_model(&model, &o, &profile);

    uint64_t fast_budget = free_disk > o.fast_reserve ? free_disk - o.fast_reserve : 0;
    uint64_t ram_required = sat_add(o.ram_breathing, profile.runtime_ram_bytes);
    uint64_t ram_budget = available_ram > ram_required ? available_ram - ram_required : 0;

    uint64_t vram_required = sat_add(o.vram_breathing, profile.runtime_vram_bytes);
    uint64_t vram_static_capacity = have_vram && free_vram > vram_required
        ? free_vram - vram_required : 0;
    uint64_t vram_budget = vram_static_capacity;

    /* GPT-OSS dense residency is all-or-none: try_reside_dense_cuda() releases
     * every q/k/v/o/router tensor if even one cannot fit. Do not recommend a
     * partial permanent set that the runtime will immediately discard. */
    if (profile.dense_all_or_none) {
        vram_budget = vram_static_capacity >= profile.dense_vram_candidate_bytes
            ? profile.dense_vram_candidate_bytes : 0;
    }

    ColiSplitPlan plan;
    ColiSplitPlanSummary s;
    char err[256];
    if (!coli_split_make_plan(&model, o.usage_file,
                              fast_budget, ram_budget, vram_budget,
                              &plan, &s, err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        coli_gguf_close(&model);
        return 1;
    }

    char pfx[256];
    if (o.prefix) snprintf(pfx, sizeof(pfx), "%s", o.prefix);
    else basename_prefix(o.model, pfx);
    char fast_path[PATH_MAX], preload_path[PATH_MAX], tail_path[PATH_MAX];
    snprintf(fast_path, sizeof(fast_path), "%s/%s.fast.split.gguf", o.fast_dir, pfx);
    snprintf(preload_path, sizeof(preload_path), "%s/%s.preload.shard", o.slow_dir, pfx);
    snprintf(tail_path, sizeof(tail_path), "%s/%s.tail.shard", o.slow_dir, pfx);

    if (o.placements_output && !write_placements(o.placements_output, &model, &plan)) {
        fprintf(stderr, "error: cannot write placements report: %s\n", o.placements_output);
        coli_split_plan_destroy(&plan);
        coli_gguf_close(&model);
        return 1;
    }

    if (o.command_only) {
        print_command(&o, fast_path, preload_path, tail_path,
                      fast_budget, ram_budget, vram_budget);
        coli_split_plan_destroy(&plan);
        coli_gguf_close(&model);
        return 0;
    }

    char a[64], b[64], c[64], e[64], f[64], g[64], h[64], i[64];
    printf("Units: MiB/GiB are binary (1 MiB = 1,048,576 bytes).\n");
    printf("Current processes are already excluded by MemAvailable and nvidia-smi free VRAM.\n\n");

    printf("Fast disk:\n");
    printf("  free now:                 %s\n", human(free_disk, a));
    printf("  breathing reserve:        %s\n", human(o.fast_reserve, b));
    printf("  safe fast-shard budget:   %s\n", human(fast_budget, c));

    printf("\nRAM:\n");
    printf("  MemAvailable now:         %s\n", human(available_ram, a));
    printf("  estimated runtime RAM:    %s (context=%d)\n", human(profile.runtime_ram_bytes, b), o.context);
    printf("  breathing reserve:        %s\n", human(o.ram_breathing, c));
    printf("  safe preload-RAM budget:  %s\n", human(ram_budget, e));

    printf("\nVRAM (device %d):\n", o.device);
    if (have_vram) {
        uint64_t used_vram = total_vram > free_vram ? total_vram - free_vram : 0;
        uint64_t workspace = profile.runtime_vram_bytes;
        uint64_t after_plan = free_vram;
        after_plan = after_plan > o.vram_breathing ? after_plan - o.vram_breathing : 0;
        after_plan = after_plan > profile.runtime_vram_bytes ? after_plan - profile.runtime_vram_bytes : 0;
        after_plan = after_plan > s.preload_vram_bytes ? after_plan - s.preload_vram_bytes : 0;

        printf("  total / free now:         %s / %s\n", human(total_vram, a), human(free_vram, b));
        printf("  currently occupied:       %s\n", human(used_vram, c));
        printf("  dynamic stats cache:      remaining VRAM after static placement\n");
        printf("  CUDA context/workspace:   %s\n", human(workspace, f));
        printf("  breathing reserve:        %s\n", human(o.vram_breathing, g));
        printf("  safe static capacity:     %s\n", human(vram_static_capacity, h));
        printf("  dense resident set:       %s\n", human(profile.dense_vram_candidate_bytes, i));
        printf("  planned preload-VRAM:     %s\n", human(s.preload_vram_bytes, a));
        printf("  uncommitted headroom:     %s\n", human(after_plan, b));
        if (profile.dense_all_or_none && !vram_budget && profile.dense_vram_candidate_bytes)
            printf("  decision:                 dense set does not safely fit; use dense streaming\n");
        else if (profile.dense_all_or_none)
            printf("  decision:                 full dense q/k/v/o/router set fits and stays resident\n");
    } else {
        printf("  unavailable; planned preload-VRAM is 0\n");
    }

    printf("\nHot expert-fragment placement:\n");
    printf("  usage history:            %" PRIu64 " selections (%" PRIu64 " nonzero layer/experts)\n",
           plan.usage_total, plan.usage_nonzero);
    printf("  expert bundles:           %" PRIu64 " total; RAM hot=%" PRIu64
           " fast=%" PRIu64 " tail=%" PRIu64 "\n",
           plan.expert_bundle_count, s.hot_ram_bundle_count,
           s.fast_bundle_count, s.tail_bundle_count);
    printf("  fragmented tensors:       %" PRIu64 "\n", s.fragmented_tensor_count);
    printf("  fast mmap:                %s (%" PRIu64 " fragments)\n",
           human(s.fast_allocated, a), s.fast_fragment_count);
    printf("  preload RAM:              %s (%" PRIu64 " fragments)\n",
           human(s.preload_ram_bytes, b), s.ram_fragment_count);
    printf("  preload VRAM:             %s (%" PRIu64 " fragments)\n",
           human(s.preload_vram_bytes, c), s.vram_fragment_count);
    printf("  slow tail mmap:           %s (%" PRIu64 " fragments)\n",
           human(s.tail_bytes, e), s.tail_fragment_count);
    printf("  slow-disk free now:       %s\n", human(slow_free, f));

    if (o.placements_output)
        printf("  placement TSV:            %s\n", o.placements_output);

    const char *legacy = getenv("CUDA_RESERVE_GB");
    if (legacy && *legacy)
        printf("\nNote: CUDA_RESERVE_GB=%s was not used by this planner.\n"
               "The runtime cache, CUDA workspace and breathing room are estimated separately above.\n",
               legacy);

    printf("\nRecommended command:\n");
    print_command(&o, fast_path, preload_path, tail_path,
                  fast_budget, ram_budget, vram_budget);
    printf("\nRecommended runtime cache setting:\n");
    printf("GPTOSS_EXPERT_CACHE_RESERVE_MIB=384 ./c/colibri --gguf '%s' --usage-file '%s' ...\n",
           fast_path, o.usage_file);

    coli_split_plan_destroy(&plan);
    coli_gguf_close(&model);
    return 0;
}
