#include "split_storage.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* GPTOSS_DEBUG_LIGHT_V1 */
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

struct ColiSplitState {
    char *primary_path;
    char *preload_path;
    char *tail_path;
    int preload_fd;
    int tail_fd;
    uint64_t preload_size;
    uint64_t tail_size;
    void *preload_mapping;
    void *tail_mapping;
    ColiSplitEntry *entries;
    ColiSplitEntry **by_tensor;
    uint64_t entry_count;
    uint64_t tensor_count;
    uint64_t source_size;
    uint64_t ram_loaded;
    uint64_t ram_total;
    unsigned ram_next_percent;
    int ram_progress_started;
    int ram_progress_done;
    int verbose;
    int light_debug;
    int preload_backend_kind;
};

static int split_fail(char *error, size_t cap, const char *fmt, ...) {
    if (error && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(error, cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

static char *split_strdup_n(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1u);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

uint16_t coli_split_load_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
uint32_t coli_split_load_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t coli_split_load_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
    return v;
}
void coli_split_store_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
void coli_split_store_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
void coli_split_store_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

uint64_t coli_split_fnv1a64(const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int pread_full(int fd, uint64_t offset, void *dst, size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, (uint8_t *)dst + done, bytes - done,
                          (off_t)(offset + (uint64_t)done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static int open_shard(const char *path, uint32_t expected_kind,
                      int *fd_out, uint64_t *size_out, void **mapping_out,
                      uint64_t source_size, char *error, size_t error_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return split_fail(error, error_size,
        "cannot open split shard '%s': %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)COLI_SPLIT_SHARD_HEADER_BYTES) {
        int saved = errno; close(fd);
        return split_fail(error, error_size, "invalid split shard '%s': %s",
                          path, saved ? strerror(saved) : "too small");
    }
    uint8_t header[64];
    if (!pread_full(fd, 0, header, sizeof(header)) ||
        memcmp(header, COLI_SPLIT_SHARD_MAGIC, 8) != 0 ||
        coli_split_load_u32(header + 8) != COLI_SPLIT_VERSION ||
        coli_split_load_u32(header + 12) != expected_kind ||
        coli_split_load_u64(header + 16) != source_size ||
        coli_split_load_u64(header + 40) != COLI_SPLIT_SHARD_HEADER_BYTES) {
        close(fd);
        return split_fail(error, error_size, "split shard header mismatch: %s", path);
    }
    uint64_t file_size = (uint64_t)st.st_size;
    void *mapping = NULL;
#ifndef _WIN32
    if (file_size <= SIZE_MAX) {
        void *p = mmap(NULL, (size_t)file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p != MAP_FAILED) mapping = p;
    }
#endif
    *fd_out = fd;
    *size_out = file_size;
    *mapping_out = mapping;
    return 1;
}

static int entry_compare_offset(const void *a, const void *b) {
    const ColiSplitEntry *x = (const ColiSplitEntry *)a;
    const ColiSplitEntry *y = (const ColiSplitEntry *)b;
    if (x->original_offset < y->original_offset) return -1;
    if (x->original_offset > y->original_offset) return 1;
    return 0;
}

int coli_split_open_primary(int primary_fd,
                            const char *primary_path,
                            uint64_t physical_size,
                            ColiSplitState **state_out,
                            uint64_t *logical_size_out,
                            char *error,
                            size_t error_size) {
    if (!state_out || !logical_size_out) return 0;
    *state_out = NULL;
    *logical_size_out = physical_size;
    if (physical_size < COLI_SPLIT_FOOTER_BYTES) return 1;

    uint8_t footer[COLI_SPLIT_FOOTER_BYTES];
    if (!pread_full(primary_fd, physical_size - sizeof(footer), footer, sizeof(footer)))
        return split_fail(error, error_size, "cannot read split footer");
    if (memcmp(footer, COLI_SPLIT_FOOTER_MAGIC, 8) != 0) return 1;
    if (coli_split_load_u32(footer + 8) != COLI_SPLIT_VERSION)
        return split_fail(error, error_size, "unsupported split format version");

    uint64_t logical_size = coli_split_load_u64(footer + 16);
    uint64_t manifest_offset = coli_split_load_u64(footer + 24);
    uint64_t manifest_size = coli_split_load_u64(footer + 32);
    uint64_t footer_entries = coli_split_load_u64(footer + 40);
    uint64_t checksum = coli_split_load_u64(footer + 48);
    if (!logical_size || manifest_size < COLI_SPLIT_MANIFEST_HEADER_BYTES ||
        manifest_offset < logical_size ||
        manifest_offset > physical_size ||
        manifest_size > physical_size - manifest_offset - COLI_SPLIT_FOOTER_BYTES ||
        manifest_size > SIZE_MAX) {
        return split_fail(error, error_size, "corrupt split footer");
    }

    uint8_t *manifest = (uint8_t *)malloc((size_t)manifest_size);
    if (!manifest) return split_fail(error, error_size, "out of memory reading split manifest");
    if (!pread_full(primary_fd, manifest_offset, manifest, (size_t)manifest_size)) {
        free(manifest);
        return split_fail(error, error_size, "cannot read split manifest");
    }
    if (coli_split_fnv1a64(manifest, (size_t)manifest_size) != checksum ||
        memcmp(manifest, COLI_SPLIT_MANIFEST_MAGIC, 8) != 0 ||
        coli_split_load_u32(manifest + 8) != COLI_SPLIT_VERSION ||
        coli_split_load_u32(manifest + 12) != COLI_SPLIT_MANIFEST_HEADER_BYTES) {
        free(manifest);
        return split_fail(error, error_size, "split manifest checksum/header mismatch");
    }

    uint64_t entry_count = coli_split_load_u64(manifest + 16);
    uint64_t preload_len = coli_split_load_u64(manifest + 24);
    uint64_t tail_len = coli_split_load_u64(manifest + 32);
    uint64_t source_size = coli_split_load_u64(manifest + 40);
    uint64_t tensor_count = coli_split_load_u64(manifest + 48);
    if (entry_count != footer_entries || source_size != logical_size ||
        preload_len > 65535 || tail_len > 65535 || !tensor_count ||
        entry_count > SIZE_MAX / sizeof(ColiSplitEntry) ||
        tensor_count > SIZE_MAX / sizeof(ColiSplitEntry *)) {
        free(manifest);
        return split_fail(error, error_size, "invalid split manifest counts");
    }
    uint64_t need = COLI_SPLIT_MANIFEST_HEADER_BYTES + preload_len + tail_len;
    if (entry_count > (UINT64_MAX - need) / COLI_SPLIT_ENTRY_BYTES) {
        free(manifest);
        return split_fail(error, error_size, "split manifest size overflow");
    }
    need += entry_count * COLI_SPLIT_ENTRY_BYTES;
    if (need != manifest_size) {
        free(manifest);
        return split_fail(error, error_size, "split manifest length mismatch");
    }

    ColiSplitState *s = (ColiSplitState *)calloc(1, sizeof(*s));
    if (!s) { free(manifest); return split_fail(error, error_size, "out of memory"); }
    s->preload_fd = -1;
    s->tail_fd = -1;
    s->preload_backend_kind = -1;
    s->source_size = source_size;
    s->entry_count = entry_count;
    s->tensor_count = tensor_count;
    s->verbose = getenv("COLI_SPLIT_VERBOSE") && atoi(getenv("COLI_SPLIT_VERBOSE")) != 0;
    {
        const char *ram_progress = getenv("COLI_RAM_PROGRESS");
        const char *light_debug = getenv("COLI_DEBUG_LIGHT");
        s->light_debug =
            (ram_progress && atoi(ram_progress) != 0) ||
            (light_debug && atoi(light_debug) != 0);
    }
    s->ram_next_percent = 2;
    s->primary_path = split_strdup_n(primary_path, strlen(primary_path));
    s->preload_path = split_strdup_n((const char *)manifest + COLI_SPLIT_MANIFEST_HEADER_BYTES,
                                     (size_t)preload_len);
    s->tail_path = split_strdup_n((const char *)manifest + COLI_SPLIT_MANIFEST_HEADER_BYTES + preload_len,
                                  (size_t)tail_len);
    s->entries = entry_count ? (ColiSplitEntry *)calloc((size_t)entry_count, sizeof(*s->entries)) : NULL;
    s->by_tensor = tensor_count ? (ColiSplitEntry **)calloc((size_t)tensor_count, sizeof(*s->by_tensor)) : NULL;
    if (!s->primary_path || !s->preload_path || !s->tail_path ||
        (entry_count && !s->entries) || !s->by_tensor) {
        free(manifest); coli_split_close(s);
        return split_fail(error, error_size, "out of memory allocating split state");
    }

    const uint8_t *ep = manifest + COLI_SPLIT_MANIFEST_HEADER_BYTES + preload_len + tail_len;
    for (uint64_t i = 0; i < entry_count; ++i, ep += COLI_SPLIT_ENTRY_BYTES) {
        ColiSplitEntry *e = &s->entries[i];
        e->original_offset = coli_split_load_u64(ep + 0);
        e->bytes = coli_split_load_u64(ep + 8);
        e->shard_offset = coli_split_load_u64(ep + 16);
        e->location = coli_split_load_u32(ep + 24);
        e->tensor_index = coli_split_load_u32(ep + 28);
        e->reserved = coli_split_load_u64(ep + 32);
        if (!e->bytes || e->original_offset > source_size ||
            e->bytes > source_size - e->original_offset ||
            e->location > COLI_SPLIT_SLOW_TAIL_MMAP ||
            e->tensor_index >= tensor_count) {
            free(manifest); coli_split_close(s);
            return split_fail(error, error_size, "invalid split entry %llu",
                              (unsigned long long)i);
        }
        if (!s->by_tensor[e->tensor_index]) s->by_tensor[e->tensor_index] = e;
    }
    qsort(s->entries, (size_t)entry_count, sizeof(*s->entries), entry_compare_offset);
    memset(s->by_tensor, 0, (size_t)tensor_count * sizeof(*s->by_tensor));
    for (uint64_t i = 0; i < entry_count; ++i)
        if (!s->by_tensor[s->entries[i].tensor_index])
            s->by_tensor[s->entries[i].tensor_index] = &s->entries[i];
    for (uint64_t i = 1; i < entry_count; ++i) {
        const ColiSplitEntry *a = &s->entries[i - 1];
        const ColiSplitEntry *b = &s->entries[i];
        if (a->original_offset + a->bytes > b->original_offset) {
            free(manifest); coli_split_close(s);
            return split_fail(error, error_size, "overlapping split entries");
        }
    }

    if (!open_shard(s->preload_path, 1, &s->preload_fd, &s->preload_size,
                    &s->preload_mapping, source_size, error, error_size) ||
        !open_shard(s->tail_path, 2, &s->tail_fd, &s->tail_size,
                    &s->tail_mapping, source_size, error, error_size)) {
        free(manifest); coli_split_close(s); return 0;
    }
    for (uint64_t i = 0; i < entry_count; ++i) {
        const ColiSplitEntry *e = &s->entries[i];
        uint64_t shard_size = e->location == COLI_SPLIT_SLOW_TAIL_MMAP
            ? s->tail_size : s->preload_size;
        if (e->location != COLI_SPLIT_FAST_MMAP &&
            (e->shard_offset > shard_size || e->bytes > shard_size - e->shard_offset)) {
            free(manifest); coli_split_close(s);
            return split_fail(error, error_size, "split entry exceeds shard");
        }
    }

    if (s->verbose) {
        fprintf(stderr, "[SPLIT] primary=%s preload=%s tail=%s entries=%llu\n",
                primary_path, s->preload_path, s->tail_path,
                (unsigned long long)entry_count);
    }
    free(manifest);
    *state_out = s;
    *logical_size_out = logical_size;
    return 1;
}

void coli_split_set_preload_backend(ColiSplitState *s, int backend_kind) {
    if (!s) return;
    s->preload_backend_kind = backend_kind;
    s->ram_total = 0;
    s->ram_next_percent = 2;
    s->ram_progress_started = 0;
    s->ram_progress_done = 0;
    for (uint64_t i = 0; i < s->entry_count; ++i) {
        const ColiSplitEntry *e = &s->entries[i];
        if (e->location == COLI_SPLIT_PRELOAD_RAM ||
            (e->location == COLI_SPLIT_PRELOAD_VRAM && backend_kind == 0)) {
            if (UINT64_MAX - s->ram_total >= e->bytes) s->ram_total += e->bytes;
        }
    }
}

void coli_split_close(ColiSplitState *s) {
    if (!s) return;
    if (s->light_debug && s->ram_progress_started && !s->ram_progress_done) {
        fputc('\n', stderr);
        fflush(stderr);
    }
    if (s->entries) {
        for (uint64_t i = 0; i < s->entry_count; ++i) free(s->entries[i].ram_data);
    }
#ifndef _WIN32
    if (s->preload_mapping) munmap(s->preload_mapping, (size_t)s->preload_size);
    if (s->tail_mapping) munmap(s->tail_mapping, (size_t)s->tail_size);
#endif
    if (s->preload_fd >= 0) close(s->preload_fd);
    if (s->tail_fd >= 0) close(s->tail_fd);
    free(s->entries);
    free(s->by_tensor);
    free(s->primary_path);
    free(s->preload_path);
    free(s->tail_path);
    free(s);
}

static ColiSplitEntry *find_entry(ColiSplitState *s, uint64_t offset, uint64_t bytes) {
    if (!s || !s->entry_count) return NULL;
    uint64_t lo = 0, hi = s->entry_count;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        ColiSplitEntry *e = &s->entries[mid];
        if (offset < e->original_offset) hi = mid;
        else if (offset >= e->original_offset + e->bytes) lo = mid + 1;
        else {
            if (bytes <= e->bytes - (offset - e->original_offset)) return e;
            return NULL;
        }
    }
    return NULL;
}

static const uint8_t *shard_pointer(ColiSplitState *s, const ColiSplitEntry *e) {
    const uint8_t *base = NULL;
    uint64_t size = 0;
    if (e->location == COLI_SPLIT_SLOW_TAIL_MMAP) {
        base = (const uint8_t *)s->tail_mapping; size = s->tail_size;
    } else {
        base = (const uint8_t *)s->preload_mapping; size = s->preload_size;
    }
    if (!base || e->shard_offset > size || e->bytes > size - e->shard_offset) return NULL;
    return base + e->shard_offset;
}

static void split_light_ram_progress(ColiSplitState *s) {
    if (!s || !s->light_debug || !s->ram_total || s->ram_progress_done) return;
    if (!s->ram_progress_started) {
        fputs("loading RAM ", stderr);
        s->ram_progress_started = 1;
    }
    unsigned percent = s->ram_loaded >= s->ram_total
        ? 100u : (unsigned)((s->ram_loaded * 100u) / s->ram_total);
    while (s->ram_next_percent <= 100u && percent >= s->ram_next_percent) {
        fputc('.', stderr);
        s->ram_next_percent += 2u;
    }
    if (s->ram_loaded >= s->ram_total) {
        fputs(" 100%\n", stderr);
        s->ram_progress_done = 1;
    }
    fflush(stderr);
}

static int load_ram_entry(ColiSplitState *s, ColiSplitEntry *e,
                          char *error, size_t error_size) {
    if (e->ram_data) return 1;
    void *p = NULL;
#ifndef _WIN32
    if (posix_memalign(&p, 64, (size_t)e->bytes) != 0) p = NULL;
#else
    p = malloc((size_t)e->bytes);
#endif
    if (!p) return split_fail(error, error_size,
        "cannot allocate %.2f MiB for split RAM preload",
        (double)e->bytes / (1024.0 * 1024.0));
    const uint8_t *mapped = shard_pointer(s, e);
    int ok = mapped ? (memcpy(p, mapped, (size_t)e->bytes), 1)
                    : pread_full(s->preload_fd, e->shard_offset, p, (size_t)e->bytes);
    if (!ok) { free(p); return split_fail(error, error_size, "cannot read RAM preload tensor"); }
    e->ram_data = p;
    s->ram_loaded += e->bytes;
#ifndef _WIN32
    if (mapped) (void)madvise((void *)mapped, (size_t)e->bytes, MADV_DONTNEED);
#endif
    split_light_ram_progress(s);
    if (s->verbose && !s->light_debug) fprintf(stderr,
        "[SPLIT] RAM preload tensor=%u bytes=%.2f MiB total=%.2f MiB\n",
        e->tensor_index, (double)e->bytes / (1024.0 * 1024.0),
        (double)s->ram_loaded / (1024.0 * 1024.0));
    return 1;
}

const void *coli_split_mapped_at(ColiSplitState *s,
                                 const void *primary_mapping,
                                 uint64_t primary_mapping_size,
                                 uint64_t offset,
                                 uint64_t bytes,
                                 int *handled,
                                 int *mmap_backed,
                                 char *error,
                                 size_t error_size) {
    if (handled) *handled = 0;
    if (mmap_backed) *mmap_backed = 1;
    if (!s) return NULL;
    ColiSplitEntry *e = find_entry(s, offset, bytes);
    if (!e) {
        /* The range may begin in a valid fragment but cross a fragment
         * boundary. Mark it handled so callers do not fall back to sparse
         * holes in the primary file. */
        if (find_entry(s, offset, 1)) {
            if (handled) *handled = 1;
        }
        return NULL;
    }
    if (handled) *handled = 1;
    uint64_t delta = offset - e->original_offset;
    if (e->location == COLI_SPLIT_FAST_MMAP) {
        if (!primary_mapping || offset > primary_mapping_size || bytes > primary_mapping_size - offset)
            return NULL;
        return (const uint8_t *)primary_mapping + offset;
    }
    if (e->location == COLI_SPLIT_PRELOAD_RAM ||
        (e->location == COLI_SPLIT_PRELOAD_VRAM && s->preload_backend_kind == 0)) {
        if (!load_ram_entry(s, e, error, error_size)) return NULL;
        if (mmap_backed) *mmap_backed = 0;
        return (const uint8_t *)e->ram_data + delta;
    }
    const uint8_t *p = shard_pointer(s, e);
    return p ? p + delta : NULL;
}

static int read_source_fragment(ColiSplitState *s, ColiSplitEntry *e,
                                int primary_fd, const void *primary_mapping,
                                uint64_t primary_mapping_size,
                                uint64_t offset, void *dst, size_t bytes) {
    uint64_t delta = offset - e->original_offset;
    if (e->location == COLI_SPLIT_FAST_MMAP) {
        if (primary_mapping && offset <= primary_mapping_size && bytes <= primary_mapping_size - offset) {
            memcpy(dst, (const uint8_t *)primary_mapping + offset, bytes); return 1;
        }
        return pread_full(primary_fd, offset, dst, bytes);
    }
    if (e->location == COLI_SPLIT_PRELOAD_RAM && e->ram_data) {
        memcpy(dst, (const uint8_t *)e->ram_data + delta, bytes); return 1;
    }
    int fd = e->location == COLI_SPLIT_SLOW_TAIL_MMAP ? s->tail_fd : s->preload_fd;
    const uint8_t *mapped = shard_pointer(s, e);
    if (mapped) { memcpy(dst, mapped + delta, bytes); return 1; }
    return pread_full(fd, e->shard_offset + delta, dst, bytes);
}

int coli_split_read_at(ColiSplitState *s,
                       int primary_fd,
                       const void *primary_mapping,
                       uint64_t primary_mapping_size,
                       uint64_t logical_size,
                       uint64_t offset,
                       void *dst,
                       size_t bytes) {
    if (!s || offset > logical_size || (uint64_t)bytes > logical_size - offset) return 0;
    size_t done = 0;
    while (done < bytes) {
        uint64_t pos = offset + done;
        ColiSplitEntry *e = find_entry(s, pos, 1);
        size_t take;
        if (e) {
            uint64_t avail = e->original_offset + e->bytes - pos;
            take = bytes - done;
            if ((uint64_t)take > avail) take = (size_t)avail;
            if (!read_source_fragment(s, e, primary_fd, primary_mapping,
                                      primary_mapping_size, pos,
                                      (uint8_t *)dst + done, take)) return 0;
        } else {
            uint64_t next = logical_size;
            for (uint64_t i = 0; i < s->entry_count; ++i) {
                if (s->entries[i].original_offset > pos) { next = s->entries[i].original_offset; break; }
            }
            uint64_t avail = next - pos;
            take = bytes - done;
            if ((uint64_t)take > avail) take = (size_t)avail;
            if (primary_mapping && pos <= primary_mapping_size && take <= primary_mapping_size - pos)
                memcpy((uint8_t *)dst + done, (const uint8_t *)primary_mapping + pos, take);
            else if (!pread_full(primary_fd, pos, (uint8_t *)dst + done, take)) return 0;
        }
        done += take;
    }
    return 1;
}

const ColiSplitEntry *coli_split_entry_for_tensor(const ColiSplitState *s,
                                                   uint64_t tensor_index) {
    if (!s || tensor_index >= s->tensor_count) return NULL;
    return s->by_tensor[tensor_index];
}

int coli_split_validate_tensor(const ColiSplitState *s,
                               uint64_t tensor_index,
                               uint64_t original_offset,
                               uint64_t bytes,
                               uint32_t *location_out,
                               uint64_t *shard_offset_out) {
    if (!s || tensor_index >= s->tensor_count || !bytes) return 0;
    uint64_t pos = original_offset, end = original_offset + bytes;
    if (end < original_offset || end > s->source_size) return 0;
    uint32_t uniform = UINT32_MAX;
    uint64_t first_shard = 0;
    int pieces = 0;
    while (pos < end) {
        ColiSplitEntry *e = find_entry((ColiSplitState *)s, pos, 1);
        if (!e || e->tensor_index != tensor_index || e->original_offset != pos ||
            e->bytes > end - pos) return 0;
        if (!pieces) { uniform = e->location; first_shard = e->shard_offset; }
        else if (uniform != e->location) uniform = COLI_SPLIT_FRAGMENTED;
        pos += e->bytes;
        ++pieces;
    }
    if (!pieces || pos != end) return 0;
    if (location_out) *location_out = pieces == 1 ? uniform : COLI_SPLIT_FRAGMENTED;
    if (shard_offset_out) *shard_offset_out = first_shard;
    return 1;
}

uint32_t coli_split_location_at(const ColiSplitState *s,
                                uint64_t offset, uint64_t bytes) {
    ColiSplitEntry *e = find_entry((ColiSplitState *)s, offset, bytes);
    return e ? e->location : COLI_SPLIT_FRAGMENTED;
}

const char *coli_split_location_name(uint32_t location) {
    switch (location) {
        case COLI_SPLIT_FAST_MMAP: return "fast-mmap";
        case COLI_SPLIT_PRELOAD_RAM: return "preload-ram";
        case COLI_SPLIT_PRELOAD_VRAM: return "preload-vram";
        case COLI_SPLIT_SLOW_TAIL_MMAP: return "slow-tail-mmap";
        case COLI_SPLIT_FRAGMENTED: return "fragmented";
        default: return "unknown";
    }
}
const char *coli_split_preload_path(const ColiSplitState *s) { return s ? s->preload_path : NULL; }
const char *coli_split_tail_path(const ColiSplitState *s) { return s ? s->tail_path : NULL; }
uint64_t coli_split_ram_loaded_bytes(const ColiSplitState *s) { return s ? s->ram_loaded : 0; }
uint64_t coli_split_entry_count(const ColiSplitState *s) { return s ? s->entry_count : 0; }

void coli_split_drop_source_pages(ColiSplitState *s, uint64_t offset, uint64_t bytes) {
#ifndef _WIN32
    ColiSplitEntry *e = find_entry(s, offset, bytes);
    if (!e || e->location == COLI_SPLIT_FAST_MMAP || e->location == COLI_SPLIT_PRELOAD_RAM) return;
    const uint8_t *p = shard_pointer(s, e);
    if (p) (void)madvise((void *)(p + (offset - e->original_offset)), (size_t)bytes, MADV_DONTNEED);
#else
    (void)s; (void)offset; (void)bytes;
#endif
}
