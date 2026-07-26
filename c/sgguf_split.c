#include "gguf_reader.h"
#include "split_plan.h"
#include "split_storage.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define COPY_CHUNK (16u * 1024u * 1024u)

typedef struct {
    const char *input;
    const char *fast_output;
    const char *preload_output;
    const char *tail_output;
    const char *usage_file;
    uint64_t fast_bytes;
    uint64_t ram_bytes;
    uint64_t vram_bytes;
    int overwrite;
    int dry_run;
    int verify;
    int verbose;
} Options;

static int failf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("error: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    return 0;
}

static const char *human(uint64_t value, char out[64]) {
    static const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double x = (double)value;
    int u = 0;
    while (x >= 1024.0 && u < 4) { x /= 1024.0; ++u; }
    snprintf(out, 64, "%.2f %s", x, units[u]);
    return out;
}

static int parse_size(const char *text, uint64_t *out) {
    if (!text || !*text || !out) return 0;
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || end == text || value < 0.0) return 0;
    while (*end == ' ' || *end == '\t') ++end;
    double scale = 1.0;
    if (*end) {
        char c = (char)(*end >= 'a' && *end <= 'z' ? *end - 32 : *end);
        if (c == 'K') scale = 1024.0;
        else if (c == 'M') scale = 1024.0 * 1024.0;
        else if (c == 'G') scale = 1024.0 * 1024.0 * 1024.0;
        else if (c == 'T') scale = 1024.0 * 1024.0 * 1024.0 * 1024.0;
        else return 0;
        ++end;
        if ((end[0] == 'i' || end[0] == 'I') && (end[1] == 'b' || end[1] == 'B')) end += 2;
        else if (end[0] == 'b' || end[0] == 'B') ++end;
        while (*end == ' ' || *end == '\t') ++end;
        if (*end) return 0;
    }
    long double bytes = (long double)value * (long double)scale;
    if (bytes > UINT64_MAX) return 0;
    *out = (uint64_t)bytes;
    return 1;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s INPUT --usage-file PATH --fast-output PATH --preload-output PATH --tail-output PATH\n"
        "  --fast-bytes SIZE --ram-bytes SIZE --vram-bytes SIZE [options]\n\n"
        "The usage file contains: layer expert cumulative_selection_count.\n"
        "Complete gate/up/down expert-weight bundles are ranked together.\n\n"
        "Options:\n"
        "  --dry-run       print the exact hot placement plan without copying\n"
        "  --verify        sample-check every tensor after writing\n"
        "  --overwrite     replace existing outputs\n"
        "  --verbose       print every fragment placement\n",
        prog);
}

static int parse_args(int argc, char **argv, Options *o) {
    memset(o, 0, sizeof(*o));
    if (argc < 2) return 0;
    o->input = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--usage-file") && i + 1 < argc) o->usage_file = argv[++i];
        else if (!strcmp(argv[i], "--fast-output") && i + 1 < argc) o->fast_output = argv[++i];
        else if (!strcmp(argv[i], "--preload-output") && i + 1 < argc) o->preload_output = argv[++i];
        else if (!strcmp(argv[i], "--tail-output") && i + 1 < argc) o->tail_output = argv[++i];
        else if (!strcmp(argv[i], "--fast-bytes") && i + 1 < argc) {
            if (!parse_size(argv[++i], &o->fast_bytes)) return 0;
        } else if (!strcmp(argv[i], "--ram-bytes") && i + 1 < argc) {
            if (!parse_size(argv[++i], &o->ram_bytes)) return 0;
        } else if (!strcmp(argv[i], "--vram-bytes") && i + 1 < argc) {
            if (!parse_size(argv[++i], &o->vram_bytes)) return 0;
        } else if (!strcmp(argv[i], "--overwrite")) o->overwrite = 1;
        else if (!strcmp(argv[i], "--dry-run")) o->dry_run = 1;
        else if (!strcmp(argv[i], "--verify")) o->verify = 1;
        else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) o->verbose = 1;
        else return 0;
    }
    return o->input && o->usage_file && o->fast_output && o->preload_output &&
           o->tail_output && o->fast_bytes;
}

static int pwrite_full(int fd, uint64_t offset, const void *src, size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pwrite(fd, (const uint8_t *)src + done, bytes - done,
                           (off_t)(offset + (uint64_t)done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static int copy_range(const ColiGgufFile *src, uint64_t src_offset,
                      int out_fd, uint64_t out_offset, uint64_t bytes,
                      uint8_t *buffer) {
    uint64_t done = 0;
    while (done < bytes) {
        size_t take = (size_t)((bytes - done) > COPY_CHUNK ? COPY_CHUNK : (bytes - done));
        if (!coli_gguf_read_at(src, src_offset + done, buffer, take) ||
            !pwrite_full(out_fd, out_offset + done, buffer, take)) return 0;
        done += take;
    }
    return 1;
}

static int make_absolute(const char *path, char out[PATH_MAX]) {
    if (!path || !*path) return 0;
    if (path[0] == '/') {
        if (strlen(path) >= PATH_MAX) return 0;
        strcpy(out, path);
        return 1;
    }
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd))) return 0;
    return snprintf(out, PATH_MAX, "%s/%s", cwd, path) < PATH_MAX;
}

static int path_parent_exists(const char *path) {
    char temp[PATH_MAX];
    if (strlen(path) >= sizeof(temp)) return 0;
    strcpy(temp, path);
    char *slash = strrchr(temp, '/');
    if (!slash) strcpy(temp, ".");
    else if (slash == temp) slash[1] = '\0';
    else *slash = '\0';
    struct stat st;
    return stat(temp, &st) == 0 && S_ISDIR(st.st_mode);
}

static int output_allowed(const char *path, int overwrite) {
    struct stat st;
    if (stat(path, &st) != 0) return errno == ENOENT;
    return overwrite && S_ISREG(st.st_mode);
}

static int open_temp_for(const char *final_path, char temp[PATH_MAX]) {
    if (snprintf(temp, PATH_MAX, "%s.tmp.%ld", final_path, (long)getpid()) >= PATH_MAX) return -1;
    unlink(temp);
    return open(temp, O_CREAT | O_EXCL | O_RDWR, 0644);
}

static int write_shard_header(int fd, uint32_t kind, uint64_t source_size,
                              uint64_t payload_bytes, uint64_t entry_count) {
    uint8_t page[COLI_SPLIT_SHARD_HEADER_BYTES];
    memset(page, 0, sizeof(page));
    memcpy(page, COLI_SPLIT_SHARD_MAGIC, 8);
    coli_split_store_u32(page + 8, COLI_SPLIT_VERSION);
    coli_split_store_u32(page + 12, kind);
    coli_split_store_u64(page + 16, source_size);
    coli_split_store_u64(page + 24, payload_bytes);
    coli_split_store_u64(page + 32, entry_count);
    coli_split_store_u64(page + 40, COLI_SPLIT_SHARD_HEADER_BYTES);
    return pwrite_full(fd, 0, page, sizeof(page));
}

static uint8_t *build_manifest(const ColiGgufFile *src,
                               const ColiSplitPlan *plan,
                               const char *preload_path,
                               const char *tail_path,
                               uint64_t *size_out) {
    uint64_t pl = strlen(preload_path), tl = strlen(tail_path);
    uint64_t size = COLI_SPLIT_MANIFEST_HEADER_BYTES + pl + tl;
    if (plan->entry_count > (UINT64_MAX - size) / COLI_SPLIT_ENTRY_BYTES) return NULL;
    size += plan->entry_count * COLI_SPLIT_ENTRY_BYTES;
    if (size > SIZE_MAX) return NULL;
    uint8_t *m = (uint8_t *)calloc(1, (size_t)size);
    if (!m) return NULL;
    memcpy(m, COLI_SPLIT_MANIFEST_MAGIC, 8);
    coli_split_store_u32(m + 8, COLI_SPLIT_VERSION);
    coli_split_store_u32(m + 12, COLI_SPLIT_MANIFEST_HEADER_BYTES);
    coli_split_store_u64(m + 16, plan->entry_count);
    coli_split_store_u64(m + 24, pl);
    coli_split_store_u64(m + 32, tl);
    coli_split_store_u64(m + 40, src->file_size);
    coli_split_store_u64(m + 48, src->tensor_count);
    coli_split_store_u64(m + 56, plan->usage_total);
    memcpy(m + COLI_SPLIT_MANIFEST_HEADER_BYTES, preload_path, (size_t)pl);
    memcpy(m + COLI_SPLIT_MANIFEST_HEADER_BYTES + pl, tail_path, (size_t)tl);
    uint8_t *ep = m + COLI_SPLIT_MANIFEST_HEADER_BYTES + pl + tl;
    for (uint64_t i = 0; i < plan->entry_count; ++i, ep += COLI_SPLIT_ENTRY_BYTES) {
        const ColiSplitDecision *d = &plan->entries[i];
        coli_split_store_u64(ep + 0, d->original_offset);
        coli_split_store_u64(ep + 8, d->bytes);
        coli_split_store_u64(ep + 16, d->shard_offset);
        coli_split_store_u32(ep + 24, d->location);
        coli_split_store_u32(ep + 28, d->tensor_index);
        uint64_t tag = ((uint64_t)(uint32_t)(d->moe_layer + 1) << 32) |
                       (uint32_t)(d->expert_id + 1);
        coli_split_store_u64(ep + 32, tag);
    }
    *size_out = size;
    return m;
}

static int write_footer(int fd, uint64_t offset, uint64_t logical_size,
                        uint64_t manifest_offset, uint64_t manifest_size,
                        uint64_t entry_count, uint64_t checksum) {
    uint8_t f[COLI_SPLIT_FOOTER_BYTES];
    memset(f, 0, sizeof(f));
    memcpy(f, COLI_SPLIT_FOOTER_MAGIC, 8);
    coli_split_store_u32(f + 8, COLI_SPLIT_VERSION);
    coli_split_store_u64(f + 16, logical_size);
    coli_split_store_u64(f + 24, manifest_offset);
    coli_split_store_u64(f + 32, manifest_size);
    coli_split_store_u64(f + 40, entry_count);
    coli_split_store_u64(f + 48, checksum);
    return pwrite_full(fd, offset, f, sizeof(f));
}

static void print_plan(const ColiGgufFile *src, const ColiSplitPlan *plan,
                       const ColiSplitPlanSummary *s, int verbose) {
    char a[64], b[64], c[64], d[64], e[64], f[64];
    printf("source logical size: %s, tensors=%" PRIu64 ", fragments=%" PRIu64 "\n",
           human(src->file_size, a), src->tensor_count, plan->entry_count);
    printf("usage history:       %" PRIu64 " selections, %" PRIu64 " nonzero experts\n",
           plan->usage_total, plan->usage_nonzero);
    printf("expert bundles:      %" PRIu64 " total; RAM hot=%" PRIu64
           " fast=%" PRIu64 " tail=%" PRIu64 "\n",
           plan->expert_bundle_count, s->hot_ram_bundle_count,
           s->fast_bundle_count, s->tail_bundle_count);
    printf("fast mmap:           %s / %s, fragments=%" PRIu64 "\n",
           human(s->fast_allocated, b), human(s->fast_budget, c), s->fast_fragment_count);
    printf("slow preload RAM:    %s / %s, fragments=%" PRIu64 "\n",
           human(s->preload_ram_bytes, d), human(s->ram_budget, e), s->ram_fragment_count);
    printf("slow preload VRAM:   %s / %s, fragments=%" PRIu64 "\n",
           human(s->preload_vram_bytes, f), human(s->vram_budget, a), s->vram_fragment_count);
    printf("slow tail mmap:      %s, fragments=%" PRIu64 "\n",
           human(s->tail_bytes, b), s->tail_fragment_count);
    if (verbose) {
        for (uint64_t i = 0; i < plan->entry_count; ++i) {
            const ColiSplitDecision *x = &plan->entries[i];
            const char *name = src->tensors[x->tensor_index].name;
            printf("%-18s %9.2f MiB layer=%d expert=%d usage=%u %s\n",
                   coli_split_location_name(x->location),
                   x->bytes / (1024.0 * 1024.0), x->moe_layer,
                   x->expert_id, x->usage_count, name);
        }
    }
}

static int verify_samples(const char *source_path, const char *split_path) {
    ColiGgufFile a, b;
    a.fd = b.fd = -1;
    if (!coli_gguf_open(&a, source_path)) return failf("verify source: %s", coli_gguf_error(&a));
    if (!coli_gguf_open(&b, split_path)) {
        coli_gguf_close(&a);
        return failf("verify split: %s", coli_gguf_error(&b));
    }
    uint8_t *x = (uint8_t *)malloc(4096), *y = (uint8_t *)malloc(4096);
    int ok = x && y && a.tensor_count == b.tensor_count;
    for (uint64_t i = 0; ok && i < a.tensor_count; ++i) {
        const ColiGgufTensorInfo *ta = &a.tensors[i], *tb = &b.tensors[i];
        if (strcmp(ta->name, tb->name) || ta->payload_size != tb->payload_size) { ok = 0; break; }
        uint64_t points[5] = {0, ta->payload_size / 4, ta->payload_size / 2,
                              (ta->payload_size * 3) / 4,
                              ta->payload_size > 4096 ? ta->payload_size - 4096 : 0};
        for (int p = 0; p < 5; ++p) {
            uint64_t off = points[p];
            size_t n = (size_t)((ta->payload_size - off) > 4096 ? 4096 : (ta->payload_size - off));
            if (!n) continue;
            if (!coli_gguf_read_tensor_bytes(&a, ta, off, x, n) ||
                !coli_gguf_read_tensor_bytes(&b, tb, off, y, n) || memcmp(x, y, n)) { ok = 0; break; }
        }
    }
    free(x); free(y); coli_gguf_close(&b); coli_gguf_close(&a);
    if (!ok) return failf("split sample verification failed");
    puts("verification: all tensor samples match");
    return 1;
}

int main(int argc, char **argv) {
    Options o;
    if (!parse_args(argc, argv, &o)) { usage(argv[0]); return 2; }
    char fast_abs[PATH_MAX], preload_abs[PATH_MAX], tail_abs[PATH_MAX];
    if (!make_absolute(o.fast_output, fast_abs) || !make_absolute(o.preload_output, preload_abs) ||
        !make_absolute(o.tail_output, tail_abs)) return failf("output path too long") ? 0 : 1;
    if (!path_parent_exists(fast_abs) || !path_parent_exists(preload_abs) || !path_parent_exists(tail_abs))
        return failf("all output parent directories must already exist") ? 0 : 1;
    if (!output_allowed(fast_abs, o.overwrite) || !output_allowed(preload_abs, o.overwrite) ||
        !output_allowed(tail_abs, o.overwrite))
        return failf("output exists; use --overwrite") ? 0 : 1;

    ColiGgufFile src; src.fd = -1;
    if (!coli_gguf_open(&src, o.input)) return failf("cannot open input: %s", coli_gguf_error(&src)) ? 0 : 1;
    if (coli_gguf_is_split(&src)) { coli_gguf_close(&src); return failf("input is already split") ? 0 : 1; }
    if (src.tensor_count > UINT32_MAX) { coli_gguf_close(&src); return failf("too many tensors") ? 0 : 1; }

    ColiSplitPlan plan;
    ColiSplitPlanSummary summary;
    char err[256];
    if (!coli_split_make_plan(&src, o.usage_file, o.fast_bytes, o.ram_bytes, o.vram_bytes,
                              &plan, &summary, err, sizeof(err))) {
        coli_gguf_close(&src);
        return failf("%s", err) ? 0 : 1;
    }
    print_plan(&src, &plan, &summary, o.verbose);
    if (o.dry_run) { coli_split_plan_destroy(&plan); coli_gguf_close(&src); return 0; }

    char fast_tmp[PATH_MAX], preload_tmp[PATH_MAX], tail_tmp[PATH_MAX];
    int ffd = open_temp_for(fast_abs, fast_tmp);
    int pfd = open_temp_for(preload_abs, preload_tmp);
    int tfd = open_temp_for(tail_abs, tail_tmp);
    if (ffd < 0 || pfd < 0 || tfd < 0) {
        if (ffd >= 0) close(ffd); if (pfd >= 0) close(pfd); if (tfd >= 0) close(tfd);
        unlink(fast_tmp); unlink(preload_tmp); unlink(tail_tmp);
        coli_split_plan_destroy(&plan); coli_gguf_close(&src);
        return failf("cannot create temporary outputs: %s", strerror(errno)) ? 0 : 1;
    }

    uint8_t *buffer = (uint8_t *)malloc(COPY_CHUNK);
    int ok = buffer != NULL;
    uint64_t preload_pos = COLI_SPLIT_SHARD_HEADER_BYTES;
    uint64_t tail_pos = COLI_SPLIT_SHARD_HEADER_BYTES;
    uint64_t preload_count = 0, tail_count = 0, copied = 0;
    if (ok) ok = write_shard_header(pfd, 1, src.file_size, 0, 0) &&
                 write_shard_header(tfd, 2, src.file_size, 0, 0);
    if (ok) ok = ftruncate(ffd, (off_t)src.file_size) == 0;
    if (ok) ok = copy_range(&src, 0, ffd, 0, src.data_offset, buffer);

    for (uint64_t i = 0; ok && i < plan.entry_count; ++i) {
        ColiSplitDecision *x = &plan.entries[i];
        int out_fd = ffd;
        uint64_t out_off = x->original_offset;
        if (x->location == COLI_SPLIT_PRELOAD_RAM || x->location == COLI_SPLIT_PRELOAD_VRAM) {
            preload_pos = coli_split_round_up(preload_pos, 64);
            x->shard_offset = preload_pos; out_fd = pfd; out_off = preload_pos;
            preload_pos += x->bytes; ++preload_count;
        } else if (x->location == COLI_SPLIT_SLOW_TAIL_MMAP) {
            tail_pos = coli_split_round_up(tail_pos, 64);
            x->shard_offset = tail_pos; out_fd = tfd; out_off = tail_pos;
            tail_pos += x->bytes; ++tail_count;
        } else {
            x->shard_offset = x->original_offset;
        }
        ok = copy_range(&src, x->original_offset, out_fd, out_off, x->bytes, buffer);
        copied += x->bytes;
        if (ok && (o.verbose || i + 1 == plan.entry_count || (i + 1) % 256 == 0)) {
            const char *name = src.tensors[x->tensor_index].name;
            fprintf(stderr,
                    "[HOT-SPLIT %5" PRIu64 "/%" PRIu64 "] %6.2f%% %-18s layer=%d expert=%d %s\n",
                    i + 1, plan.entry_count,
                    summary.source_payload_bytes ? 100.0 * copied / summary.source_payload_bytes : 100.0,
                    coli_split_location_name(x->location), x->moe_layer, x->expert_id, name);
        }
    }

    if (ok) ok = ftruncate(pfd, (off_t)preload_pos) == 0 &&
                 ftruncate(tfd, (off_t)tail_pos) == 0 &&
                 write_shard_header(pfd, 1, src.file_size,
                                    preload_pos - COLI_SPLIT_SHARD_HEADER_BYTES, preload_count) &&
                 write_shard_header(tfd, 2, src.file_size,
                                    tail_pos - COLI_SPLIT_SHARD_HEADER_BYTES, tail_count);

    uint64_t manifest_size = 0;
    uint8_t *manifest = ok ? build_manifest(&src, &plan, preload_abs, tail_abs, &manifest_size) : NULL;
    if (ok && !manifest) ok = 0;
    uint64_t manifest_offset = coli_split_round_up(src.file_size, 4096);
    uint64_t footer_offset = manifest_offset + manifest_size;
    if (ok) ok = pwrite_full(ffd, manifest_offset, manifest, (size_t)manifest_size) &&
                 write_footer(ffd, footer_offset, src.file_size, manifest_offset, manifest_size,
                              plan.entry_count, coli_split_fnv1a64(manifest, (size_t)manifest_size)) &&
                 ftruncate(ffd, (off_t)(footer_offset + COLI_SPLIT_FOOTER_BYTES)) == 0;
    if (ok) ok = fsync(ffd) == 0 && fsync(pfd) == 0 && fsync(tfd) == 0;

    free(manifest); free(buffer);
    close(ffd); close(pfd); close(tfd);
    coli_gguf_close(&src);
    coli_split_plan_destroy(&plan);

    if (!ok) {
        unlink(fast_tmp); unlink(preload_tmp); unlink(tail_tmp);
        return failf("hot split copy failed: %s", strerror(errno)) ? 0 : 1;
    }
    if (o.overwrite) { unlink(fast_abs); unlink(preload_abs); unlink(tail_abs); }
    if (rename(preload_tmp, preload_abs) != 0 || rename(tail_tmp, tail_abs) != 0 ||
        rename(fast_tmp, fast_abs) != 0) {
        unlink(fast_tmp); unlink(preload_tmp); unlink(tail_tmp);
        return failf("cannot finalize outputs: %s", strerror(errno)) ? 0 : 1;
    }

    puts("hot split complete");
    printf("primary: %s\npreload: %s\ntail: %s\n", fast_abs, preload_abs, tail_abs);
    puts("The cumulative usage file is not copied or modified by the splitter.");
    puts("The primary is sparse; use 'du -h', not only 'ls -lh', for real disk use.");
    if (o.verify && !verify_samples(o.input, fast_abs)) return 1;
    return 0;
}
