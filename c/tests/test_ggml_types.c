#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "../ggml_types.h"

int main(void) {
    const ColiGgmlTypeTraits *q5 = coli_ggml_type_traits(13);
    assert(q5 && q5->block_values == 256 && q5->block_bytes == 176);
    const ColiGgmlTypeTraits *q81 = coli_ggml_type_traits(9);
    assert(q81 && q81->block_bytes == 36);
    const ColiGgmlTypeTraits *bf16 = coli_ggml_type_traits(30);
    assert(bf16 && !bf16->quantized && bf16->block_values == 1 && bf16->block_bytes == 2);
    uint64_t n;
    assert(coli_ggml_row_size(12, 512, &n) && n == 288);
    assert(coli_ggml_row_size(30, 2048, &n) && n == 4096);
    assert(!coli_ggml_row_size(12, 257, &n));
    uint64_t dims[] = { 1024, 512, 32 };
    uint64_t elems;
    assert(coli_ggml_tensor_size(13, dims, 3, &elems, &n));
    assert(elems == 1024ull * 512 * 32);
    assert(n == (elems / 256) * 176);
    uint64_t bad[] = { UINT64_MAX, 2 };
    assert(!coli_ggml_tensor_size(0, bad, 2, &elems, &n));
    uint64_t bad_row[] = { 128, 2 };
    assert(!coli_ggml_tensor_size(12, bad_row, 2, &elems, &n));
    puts("test_ggml_types: ok");
    return 0;
}
