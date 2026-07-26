#include "gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--metadata] [--tensors] [--find-tensor NAME] MODEL.gguf\n",
            argv0);
}

static void print_dims(const ColiGgufTensorInfo *t) {
    putchar('[');
    for (uint32_t d = 0; d < t->n_dims; ++d) {
        if (d) fputs(", ", stdout);
        printf("%llu", (unsigned long long)t->dims[d]);
    }
    putchar(']');
}

static void print_kv_value(const ColiGgufFile *g, const ColiGgufKV *kv) {
    char *s = NULL;
    uint64_t u;
    int64_t i;
    double f;
    int b;

    if (kv->type == COLI_GGUF_TYPE_ARRAY) {
        printf("array<%s>[%llu]", coli_gguf_value_type_name(kv->array_type),
               (unsigned long long)kv->array_count);
    } else if (coli_gguf_kv_read_string(g, kv, &s)) {
        printf("\"%s\"", s);
        free(s);
    } else if (coli_gguf_kv_read_bool(g, kv, &b)) {
        fputs(b ? "true" : "false", stdout);
    } else if (coli_gguf_kv_read_u64(g, kv, &u)) {
        printf("%llu", (unsigned long long)u);
    } else if (coli_gguf_kv_read_i64(g, kv, &i)) {
        printf("%lld", (long long)i);
    } else if (coli_gguf_kv_read_f64(g, kv, &f)) {
        printf("%.17g", f);
    } else {
        printf("<%s>", coli_gguf_value_type_name(kv->type));
    }
}

int main(int argc, char **argv) {
    int show_metadata = 0;
    int show_tensors = 0;
    const char *find_tensor = NULL;
    const char *path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--metadata") == 0) {
            show_metadata = 1;
        } else if (strcmp(argv[i], "--tensors") == 0) {
            show_tensors = 1;
        } else if (strcmp(argv[i], "--find-tensor") == 0 && i + 1 < argc) {
            find_tensor = argv[++i];
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else if (!path) {
            path = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!path) {
        usage(argv[0]);
        return 2;
    }

    ColiGgufFile g;
    if (!coli_gguf_open(&g, path)) {
        fprintf(stderr, "gguf-inspect: %s\n", coli_gguf_error(&g));
        return 1;
    }

    printf("file: %s\n", path);
    printf("version: %u\n", g.version);
    printf("file_size: %llu\n", (unsigned long long)g.file_size);
    printf("metadata_count: %llu\n", (unsigned long long)g.metadata_count);
    printf("tensor_count: %llu\n", (unsigned long long)g.tensor_count);
    printf("alignment: %u\n", g.alignment);
    printf("data_offset: %llu\n", (unsigned long long)g.data_offset);
    if (coli_gguf_is_split(&g)) {
        printf("split: yes\n");
        printf("preload_shard: %s\n", coli_split_preload_path(g.split));
        printf("tail_shard: %s\n", coli_split_tail_path(g.split));
    }

    const ColiGgufKV *arch = coli_gguf_find_kv(&g, "general.architecture");
    char *arch_value = NULL;
    if (arch && coli_gguf_kv_read_string(&g, arch, &arch_value)) {
        printf("architecture: %s\n", arch_value);
        free(arch_value);
    }

    typedef struct { uint32_t type; uint64_t count; } TypeCount;
    TypeCount *hist = g.tensor_count ? (TypeCount *)calloc((size_t)g.tensor_count, sizeof(*hist)) : NULL;
    size_t hist_n = 0;
    if (g.tensor_count && !hist) {
        fprintf(stderr, "out of memory building type histogram\n");
        coli_gguf_close(&g);
        return 1;
    }
    for (uint64_t ti = 0; ti < g.tensor_count; ++ti) {
        uint32_t type = g.tensors[ti].type;
        size_t h;
        for (h = 0; h < hist_n; ++h) if (hist[h].type == type) break;
        if (h == hist_n) hist_n++;
        hist[h].type = type;
        hist[h].count++;
    }
    puts("tensor_types:");
    for (size_t h = 0; h < hist_n; ++h) {
        printf("  %s (%u): %llu\n", coli_ggml_type_name(hist[h].type), hist[h].type,
               (unsigned long long)hist[h].count);
    }
    free(hist);

    if (find_tensor) {
        const ColiGgufTensorInfo *t = coli_gguf_find_tensor(&g, find_tensor);
        if (!t) {
            fprintf(stderr, "tensor not found: %s\n", find_tensor);
            coli_gguf_close(&g);
            return 3;
        }
        printf("tensor: %s type=%s(%u) dims=", t->name, coli_ggml_type_name(t->type), t->type);
        print_dims(t);
        printf(" relative_offset=%llu absolute_offset=%llu location=%s\n",
               (unsigned long long)t->offset,
               (unsigned long long)t->absolute_offset,
               coli_split_location_name(t->split_location));
    }

    if (show_metadata) {
        puts("metadata:");
        for (uint64_t i = 0; i < g.metadata_count; ++i) {
            const ColiGgufKV *kv = &g.metadata[i];
            printf("  %s (%s) = ", kv->key, coli_gguf_value_type_name(kv->type));
            print_kv_value(&g, kv);
            putchar('\n');
        }
    }

    if (show_tensors) {
        puts("tensors:");
        for (uint64_t i = 0; i < g.tensor_count; ++i) {
            const ColiGgufTensorInfo *t = &g.tensors[i];
            printf("  %s type=%s(%u) dims=", t->name, coli_ggml_type_name(t->type), t->type);
            print_dims(t);
            printf(" relative_offset=%llu absolute_offset=%llu location=%s\n",
                   (unsigned long long)t->offset,
                   (unsigned long long)t->absolute_offset,
                   coli_split_location_name(t->split_location));
        }
    }

    coli_gguf_close(&g);
    return 0;
}
