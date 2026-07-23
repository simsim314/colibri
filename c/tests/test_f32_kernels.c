#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../f32_kernels.h"

static int closef(float a, float b) { return fabsf(a - b) < 1e-5f; }

static void test_matmul_orientation(void) {
    const float x[] = { 1, 2, 3 };
    const float w[] = { 1, 0, -1, 2, 3, 4 }; /* [O=2,I=3] */
    float y[2];
    coli_f32_matmul(y, x, w, 1, 3, 2);
    assert(closef(y[0], -2));
    assert(closef(y[1], 20));
}

static void test_rmsnorm(void) {
    const float x[] = { 3, 4 };
    const float w[] = { 2, 0.5f };
    float y[2];
    coli_f32_rmsnorm(y, x, w, 2, 0.0f);
    const float r = 1.0f / sqrtf(12.5f);
    assert(closef(y[0], 6*r));
    assert(closef(y[1], 2*r));
}

static void test_rope_position_zero(void) {
    float x[] = { 1, 2, 3, 4 };
    coli_f32_rope(x, 1, 4, 4, 0, 10000.0f);
    assert(closef(x[0], 1) && closef(x[1], 2));
    assert(closef(x[2], 3) && closef(x[3], 4));
}

static void test_router(void) {
    const float l[] = { 0, 2, 2, -1 };
    int idx[2]; float w[2];
    assert(coli_f32_router_topk(l, 4, 2, idx, w) == 2);
    assert(idx[0] == 1 && idx[1] == 2);
    assert(closef(w[0], 0.5f) && closef(w[1], 0.5f));
}


static void test_expert_slice(void) {
    /* GGUF [I=2,O=2,E=2]: each expert slice is O*I contiguous values. */
    const float experts[] = {
        1,0, 0,1,   /* expert 0: identity */
        2,0, 0,3    /* expert 1: diagonal 2,3 */
    };
    const float x[] = { 4, 5 };
    float y[2];
    coli_f32_matmul(y, x, experts + 1*4, 1, 2, 2);
    assert(closef(y[0],8) && closef(y[1],15));
}

static void test_gqa_mapping(void) {
    enum { H=4, KVH=2, D=2 };
    const float q[H*D] = { 1,0, 1,0, 0,1, 0,1 };
    const float k[KVH*D] = { 1,0, 0,1 };
    const float v[KVH*D] = { 10,11, 20,21 };
    float out[H*D], scores[1];
    coli_f32_gqa_attention(out, q, k, v, 0, H, KVH, D, 1.0f, scores);
    assert(closef(out[0],10) && closef(out[1],11));
    assert(closef(out[2],10) && closef(out[3],11));
    assert(closef(out[4],20) && closef(out[5],21));
    assert(closef(out[6],20) && closef(out[7],21));
}

int main(void) {
    test_matmul_orientation();
    test_rmsnorm();
    test_rope_position_zero();
    test_router();
    test_expert_slice();
    test_gqa_mapping();
    puts("test_f32_kernels: ok");
    return 0;
}
