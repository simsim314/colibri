#include "gguf_reader.h"
#include "sgguf.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.sgguf\n", argv[0]);
        return 2;
    }
    ColiGgufFile f;
    if (!coli_gguf_open(&f, argv[1])) {
        fprintf(stderr, "cannot open: %s\n", coli_gguf_error(&f));
        return 1;
    }
    printf("container: %s\n", f.container_kind == COLI_MODEL_CONTAINER_SGGUF ? "SGGUF" : "GGUF");
    printf("version: %u\n", f.version);
    printf("tensors: %llu\n", (unsigned long long)f.tensor_count);
    printf("metadata: %llu\n", (unsigned long long)f.metadata_count);
    printf("alignment: %u\n", f.alignment);

    uint64_t dense_bytes = 0, sparse_bytes = 0, sparse_tensors = 0;
    for (uint64_t i = 0; i < f.tensor_count; ++i) {
        const ColiGgufTensorInfo *t = &f.tensors[i];
        if (t->storage_kind == COLI_TENSOR_STORAGE_SPARSE_TREE) {
            ++sparse_tensors; sparse_bytes += t->payload_size;
            const uint8_t *p = (const uint8_t *)coli_gguf_mapped_at(&f, t->absolute_offset, t->payload_size);
            ColiSggufSparseTensor st;
            char err[256];
            if (!coli_sgguf_sparse_tensor_parse(p, t->payload_size, &st, err, sizeof(err))) {
                printf("SPARSE %-60s CORRUPT: %s\n", t->name, err);
                continue;
            }
            uint64_t retained = 0, occupancy_bytes = 0, values = 0, aux = 0;
            for (uint64_t b = 0; b < st.total_blocks; ++b) {
                ColiSggufSparseBlock block;
                if (!coli_sgguf_sparse_block_get(&st, b, &block, err, sizeof(err))) {
                    printf("SPARSE %-60s block %llu CORRUPT: %s\n", t->name,
                           (unsigned long long)b, err);
                    break;
                }
                retained += block.retained_count;
                occupancy_bytes += block.layout == COLI_SGGUF_LAYOUT_BITMAP_V2
                    ? COLI_SGGUF_BITMAP_BYTES
                    : (block.tree_bits + 7u) / 8u;
                values += block.retained_payload_bytes;
                aux += block.auxiliary_bytes;
            }
            double ratio = st.total_blocks ? (double)retained / (double)(st.total_blocks * 256u) : 0.0;
            uint64_t index_bytes = (st.total_blocks + 1u) * st.offset_width;
            uint64_t headers = st.layout == COLI_SGGUF_LAYOUT_TREE_V1
                ? st.total_blocks * COLI_SGGUF_SPARSE_BLOCK_HEADER_BYTES : 0;
            uint64_t accounted = COLI_SGGUF_SPARSE_TENSOR_HEADER_BYTES + index_bytes +
                                 occupancy_bytes + values + aux + headers;
            uint64_t padding = t->payload_size > accounted ? t->payload_size - accounted : 0;
            const char *layout = st.layout == COLI_SGGUF_LAYOUT_BITMAP_V2 ? "bitmap256-v2" : "tree-v1";
            printf("SPARSE %-60s type=%s codec=%s layout=%s experts=%u rows/expert=%u blocks=%llu retained=%.2f%% payload=%.2f MiB occupancy=%.2f values=%.2f aux=%.2f index=%.2f headers=%.2f padding=%.2f\n",
                   t->name, coli_ggml_type_name(t->type), coli_sgguf_codec_name(st.codec_id), layout,
                   st.expert_count, st.rows_per_expert, (unsigned long long)st.total_blocks,
                   ratio * 100.0, t->payload_size / (1024.0 * 1024.0),
                   occupancy_bytes / (1024.0 * 1024.0), values / (1024.0 * 1024.0),
                   aux / (1024.0 * 1024.0), index_bytes / (1024.0 * 1024.0),
                   headers / (1024.0 * 1024.0), padding / (1024.0 * 1024.0));
        } else {
            dense_bytes += t->payload_size;
        }
    }
    printf("dense payload: %.2f MiB\n", dense_bytes / (1024.0 * 1024.0));
    printf("sparse payload: %.2f MiB in %llu tensors\n", sparse_bytes / (1024.0 * 1024.0),
           (unsigned long long)sparse_tensors);
    coli_gguf_close(&f);
    return 0;
}
