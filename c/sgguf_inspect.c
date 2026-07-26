#include "gguf_reader.h"
#include "ggml_types.h"
#include "sgguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t type;
    uint64_t tensors;
    uint64_t bytes;
    uint64_t moe_tensors;
    uint64_t moe_bytes;
} TypeStat;

static int is_routed_expert(const char *name) {
    int layer = -1, used = 0;
    if (!name) return 0;
    if (sscanf(name, "blk.%d.ffn_gate_exps.weight%n", &layer, &used) == 1 && name[used] == '\0') return 1;
    used = 0;
    if (sscanf(name, "blk.%d.ffn_up_exps.weight%n", &layer, &used) == 1 && name[used] == '\0') return 1;
    used = 0;
    if (sscanf(name, "blk.%d.ffn_down_exps.weight%n", &layer, &used) == 1 && name[used] == '\0') return 1;
    return 0;
}

static TypeStat *find_type(TypeStat *stats, size_t *count, uint32_t type) {
    for (size_t i = 0; i < *count; ++i) if (stats[i].type == type) return &stats[i];
    if (*count >= 128) return NULL;
    TypeStat *s = &stats[(*count)++];
    memset(s, 0, sizeof(*s));
    s->type = type;
    return s;
}

static const char *layout_name(uint32_t layout) {
    if (layout == COLI_SGGUF_LAYOUT_BITMAP_V3) return "bitmap256-v3";
    if (layout == COLI_SGGUF_LAYOUT_BITMAP_V2) return "bitmap256-v2";
    if (layout == COLI_SGGUF_LAYOUT_TREE_V1) return "tree-v1";
    return "unknown";
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s MODEL.gguf|MODEL.sgguf [--tensors] [--no-sparse-details]\n", argv0);
}

int main(int argc, char **argv) {
    int list_tensors = 0, sparse_details = 1;
    if (argc < 2) { usage(argv[0]); return 2; }
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--tensors") == 0) list_tensors = 1;
        else if (strcmp(argv[i], "--no-sparse-details") == 0) sparse_details = 0;
        else { usage(argv[0]); return 2; }
    }

    ColiGgufFile f;
    if (!coli_gguf_open(&f, argv[1])) {
        fprintf(stderr, "cannot open: %s\n", coli_gguf_error(&f));
        return 1;
    }

    char *arch = NULL;
    const ColiGgufKV *arch_kv = coli_gguf_find_kv(&f, "general.architecture");
    if (arch_kv) (void)coli_gguf_kv_read_string(&f, arch_kv, &arch);

    printf("container: %s\n", f.container_kind == COLI_MODEL_CONTAINER_SGGUF ? "SGGUF" : "GGUF");
    printf("version: %u\n", f.version);
    printf("architecture: %s\n", arch ? arch : "<unknown>");
    printf("tensors: %llu\n", (unsigned long long)f.tensor_count);
    printf("metadata: %llu\n", (unsigned long long)f.metadata_count);
    printf("alignment: %u\n", f.alignment);
    if (coli_gguf_is_split(&f)) {
        printf("split: yes\n");
        printf("preload_shard: %s\n", coli_split_preload_path(f.split));
        printf("tail_shard: %s\n", coli_split_tail_path(f.split));
    }

    TypeStat stats[128];
    size_t stat_count = 0;
    uint64_t dense_bytes = 0, sparse_bytes = 0, sparse_tensors = 0;
    uint64_t routed_tensors = 0, routed_bytes = 0;
    int unsupported_types = 0;

    for (uint64_t i = 0; i < f.tensor_count; ++i) {
        const ColiGgufTensorInfo *t = &f.tensors[i];
        TypeStat *stt = find_type(stats, &stat_count, t->type);
        if (stt) {
            stt->tensors++;
            stt->bytes += t->payload_size;
            if (is_routed_expert(t->name)) {
                stt->moe_tensors++;
                stt->moe_bytes += t->payload_size;
            }
        }
        if (!coli_dtype_traits((ColiDType)t->type)) unsupported_types = 1;
        if (is_routed_expert(t->name)) { routed_tensors++; routed_bytes += t->payload_size; }

        if (list_tensors) {
            printf("TENSOR %-64s type=%s(%u) storage=%s dims=", t->name,
                   coli_ggml_type_name(t->type), t->type,
                   t->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE ? "sparse" : "dense");
            for (uint32_t d = 0; d < t->n_dims; ++d)
                printf("%s%llu", d ? "x" : "", (unsigned long long)t->dims[d]);
            printf(" payload=%.2f MiB%s location=%s\n", t->payload_size / (1024.0 * 1024.0),
                   is_routed_expert(t->name) ? " routed-moe" : "",
                   coli_split_location_name(t->split_location));
        }

        if (t->storage_kind != COLI_TENSOR_STORAGE_SPARSE_TREE) {
            dense_bytes += t->payload_size;
            continue;
        }

        ++sparse_tensors;
        sparse_bytes += t->payload_size;
        if (!sparse_details) continue;

        const uint8_t *p = (const uint8_t *)coli_gguf_mapped_at(&f, t->absolute_offset, t->payload_size);
        ColiSggufSparseTensor st;
        char err[256];
        if (!p || !coli_sgguf_sparse_tensor_parse(p, t->payload_size, &st, err, sizeof(err))) {
            printf("SPARSE %-60s CORRUPT: %s\n", t->name, p ? err : "payload is not mapped");
            continue;
        }
        uint64_t retained = 0, occupancy_bytes = 0, values = 0, aux = 0;
        int corrupt = 0;
        for (uint64_t b = 0; b < st.total_blocks; ++b) {
            ColiSggufSparseBlock block;
            if (!coli_sgguf_sparse_block_get(&st, b, &block, err, sizeof(err))) {
                printf("SPARSE %-60s block %llu CORRUPT: %s\n", t->name,
                       (unsigned long long)b, err);
                corrupt = 1;
                break;
            }
            retained += block.retained_count;
            occupancy_bytes += (block.layout == COLI_SGGUF_LAYOUT_BITMAP_V2 ||
                                block.layout == COLI_SGGUF_LAYOUT_BITMAP_V3)
                ? COLI_SGGUF_BITMAP_BYTES : (block.tree_bits + 7u) / 8u;
            values += block.retained_payload_bytes;
            aux += block.auxiliary_bytes;
        }
        if (corrupt) continue;
        const uint64_t logical_values = st.total_rows * (uint64_t)st.cols;
        const double retained_ratio = logical_values ? (double)retained / (double)logical_values : 0.0;
        const uint64_t index_bytes = (st.total_blocks + 1u) * st.offset_width;
        const uint64_t headers = st.layout == COLI_SGGUF_LAYOUT_TREE_V1
            ? st.total_blocks * COLI_SGGUF_SPARSE_BLOCK_HEADER_BYTES : 0;
        const uint64_t accounted = COLI_SGGUF_SPARSE_TENSOR_HEADER_BYTES + index_bytes +
                                   occupancy_bytes + values + aux + headers;
        const uint64_t padding = t->payload_size > accounted ? t->payload_size - accounted : 0;
        printf("SPARSE %-60s type=%s codec=%s layout=%s experts=%u rows/expert=%u "
               "blocks=%llu retained=%.2f%% payload=%.2f MiB occupancy=%.2f values=%.2f "
               "aux=%.2f index=%.2f headers=%.2f padding=%.2f\n",
               t->name, coli_ggml_type_name(t->type), coli_sgguf_codec_name(st.codec_id),
               layout_name(st.layout), st.expert_count, st.rows_per_expert,
               (unsigned long long)st.total_blocks, retained_ratio * 100.0,
               t->payload_size / (1024.0 * 1024.0), occupancy_bytes / (1024.0 * 1024.0),
               values / (1024.0 * 1024.0), aux / (1024.0 * 1024.0),
               index_bytes / (1024.0 * 1024.0), headers / (1024.0 * 1024.0),
               padding / (1024.0 * 1024.0));
    }

    puts("type summary:");
    for (size_t i = 0; i < stat_count; ++i) {
        printf("  %-12s id=%-3u tensors=%-5llu payload=%9.2f MiB routed=%-4llu routed_payload=%9.2f MiB%s\n",
               coli_ggml_type_name(stats[i].type), stats[i].type,
               (unsigned long long)stats[i].tensors, stats[i].bytes / (1024.0 * 1024.0),
               (unsigned long long)stats[i].moe_tensors, stats[i].moe_bytes / (1024.0 * 1024.0),
               coli_dtype_traits((ColiDType)stats[i].type) ? "" : " UNSUPPORTED");
    }
    printf("routed MoE payload: %.2f MiB in %llu tensors\n", routed_bytes / (1024.0 * 1024.0),
           (unsigned long long)routed_tensors);
    printf("dense payload: %.2f MiB\n", dense_bytes / (1024.0 * 1024.0));
    printf("sparse payload: %.2f MiB in %llu tensors\n", sparse_bytes / (1024.0 * 1024.0),
           (unsigned long long)sparse_tensors);

    free(arch);
    coli_gguf_close(&f);
    return unsupported_types ? 3 : 0;
}
