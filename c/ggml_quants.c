#include "ggml_quants.h"
#include "ggml_types.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static uint16_t load_u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}
static uint32_t load_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static float load_f32(const uint8_t *p) {
    uint32_t u = load_u32(p);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* OCP MX E2M1 values are stored doubled in this table. GGML's E8M0
 * conversion for MXFP4 returns half of the power-of-two block scale. */
static const int8_t k_mxfp4_values_x2[16] = {
     0,  1,  2,  3,  4,  6,  8, 12,
     0, -1, -2, -3, -4, -6, -8,-12,
};

static const int8_t k_iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

float coli_e8m0_to_fp32_half(uint8_t e) {
    if (e == 0xffu) return NAN;
    return ldexpf(0.5f, (int)e - 127);
}

float coli_mxfp4_code_to_fp32(uint8_t e, uint8_t code) {
    return coli_e8m0_to_fp32_half(e) * (float)k_mxfp4_values_x2[code & 15u];
}

float coli_bf16_to_fp32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

float coli_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03ffu;
            out = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void deq_q4_0(const uint8_t *p, float *y) {
    float d = coli_fp16_to_fp32(load_u16(p));
    const uint8_t *q = p + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = d * ((int)(q[j] & 15) - 8);
        y[j + 16] = d * ((int)(q[j] >> 4) - 8);
    }
}
static void deq_q4_1(const uint8_t *p, float *y) {
    float d = coli_fp16_to_fp32(load_u16(p));
    float m = coli_fp16_to_fp32(load_u16(p + 2));
    const uint8_t *q = p + 4;
    for (int j = 0; j < 16; ++j) {
        y[j] = d * (q[j] & 15) + m;
        y[j + 16] = d * (q[j] >> 4) + m;
    }
}
static void deq_q5_0(const uint8_t *p, float *y) {
    float d = coli_fp16_to_fp32(load_u16(p));
    uint32_t qh = load_u32(p + 2);
    const uint8_t *q = p + 6;
    for (int j = 0; j < 16; ++j) {
        uint8_t h0 = (uint8_t)(((qh >> j) << 4) & 0x10u);
        uint8_t h1 = (uint8_t)((qh >> (j + 12)) & 0x10u);
        y[j] = d * (((q[j] & 15) | h0) - 16);
        y[j + 16] = d * (((q[j] >> 4) | h1) - 16);
    }
}
static void deq_q5_1(const uint8_t *p, float *y) {
    float d = coli_fp16_to_fp32(load_u16(p));
    float m = coli_fp16_to_fp32(load_u16(p + 2));
    uint32_t qh = load_u32(p + 4);
    const uint8_t *q = p + 8;
    for (int j = 0; j < 16; ++j) {
        uint8_t h0 = (uint8_t)(((qh >> j) << 4) & 0x10u);
        uint8_t h1 = (uint8_t)((qh >> (j + 12)) & 0x10u);
        y[j] = d * ((q[j] & 15) | h0) + m;
        y[j + 16] = d * ((q[j] >> 4) | h1) + m;
    }
}
static void deq_q8(const uint8_t *p, int header, float *y) {
    float d = coli_fp16_to_fp32(load_u16(p));
    const int8_t *q = (const int8_t *)(p + header);
    for (int j = 0; j < 32; ++j) y[j] = d * q[j];
}
static void deq_q3_k(const uint8_t *p, float *y) {
    const uint8_t *hmask = p;
    const uint8_t *q = p + 32;
    const uint8_t *packed_scales = p + 96;
    float d_all = coli_fp16_to_fp32(load_u16(p + 108));
    uint32_t aux[4] = {0,0,0,0};
    memcpy(aux, packed_scales, 12);
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *scales = (const int8_t *)aux;
    int is = 0;
    uint8_t mask = 1;
    int out = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float d0 = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                y[out++] = d0 * ((int)((q[l] >> shift) & 3) - ((hmask[l] & mask) ? 0 : 4));
            float d1 = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                y[out++] = d1 * ((int)((q[l + 16] >> shift) & 3) - ((hmask[l + 16] & mask) ? 0 : 4));
            shift += 2;
            mask <<= 1;
        }
        q += 32;
    }
}
static void deq_q4_k(const uint8_t *p, float *y) {
    float d = coli_fp16_to_fp32(load_u16(p));
    float dmin = coli_fp16_to_fp32(load_u16(p + 2));
    const uint8_t *scales = p + 4;
    const uint8_t *q = p + 16;
    int is = 0, out = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is++, scales, &sc, &m);
        float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4(is++, scales, &sc, &m);
        float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) y[out++] = d1 * (q[l] & 15) - m1;
        for (int l = 0; l < 32; ++l) y[out++] = d2 * (q[l] >> 4) - m2;
        q += 32;
    }
}
static void deq_q5_k(const uint8_t *p, float *y) {
    float d = coli_fp16_to_fp32(load_u16(p));
    float dmin = coli_fp16_to_fp32(load_u16(p + 2));
    const uint8_t *scales = p + 4;
    const uint8_t *qh = p + 16;
    const uint8_t *ql = p + 48;
    int is = 0, out = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is++, scales, &sc, &m);
        float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4(is++, scales, &sc, &m);
        float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) y[out++] = d1 * ((ql[l] & 15) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) y[out++] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        u1 <<= 2; u2 <<= 2;
    }
}
static void deq_q6_k(const uint8_t *p, float *y) {
    const uint8_t *ql = p;
    const uint8_t *qh = p + 128;
    const int8_t *sc = (const int8_t *)(p + 192);
    float d = coli_fp16_to_fp32(load_u16(p + 208));
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = ((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l] = d * sc[is + 0] * q1;
            y[n + l + 32] = d * sc[is + 2] * q2;
            y[n + l + 64] = d * sc[is + 4] * q3;
            y[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64; qh += 32; sc += 8;
    }
}
static void deq_q8_k(const uint8_t *p, float *y) {
    float d = load_f32(p);
    const int8_t *q = (const int8_t *)(p + 4);
    for (int j = 0; j < 256; ++j) y[j] = d * q[j];
}

static void deq_iq4_nl(const uint8_t *p, float *y) {
    const float d = coli_fp16_to_fp32(load_u16(p));
    const uint8_t *q = p + 2;
    for (uint32_t j = 0; j < 16; ++j) {
        y[j] = d * (float)k_iq4nl_values[q[j] & 15u];
        y[j + 16u] = d * (float)k_iq4nl_values[q[j] >> 4];
    }
}

static void deq_iq4_xs(const uint8_t *p, float *y) {
    const float d = coli_fp16_to_fp32(load_u16(p));
    const uint16_t scales_h = load_u16(p + 2);
    const uint8_t *scales_l = p + 4;
    const uint8_t *q = p + 8;
    for (uint32_t ib = 0; ib < 8; ++ib) {
        const uint32_t low = (scales_l[ib >> 1] >> (4u * (ib & 1u))) & 0x0fu;
        const uint32_t high = (scales_h >> (2u * ib)) & 0x03u;
        const float dl = d * (float)((int)(low | (high << 4)) - 32);
        const uint8_t *qb = q + ib * 16u;
        float *yb = y + ib * 32u;
        for (uint32_t j = 0; j < 16; ++j) {
            yb[j] = dl * (float)k_iq4nl_values[qb[j] & 15u];
            yb[j + 16u] = dl * (float)k_iq4nl_values[qb[j] >> 4];
        }
    }
}

static void deq_mxfp4(const uint8_t *p, float *y) {
    const uint8_t e = p[0];
    const uint8_t *q = p + 1;
    for (uint32_t j = 0; j < 16; ++j) {
        y[j] = coli_mxfp4_code_to_fp32(e, q[j] & 15u);
        y[j + 16u] = coli_mxfp4_code_to_fp32(e, q[j] >> 4);
    }
}

int coli_dtype_dequantize_row(ColiDType type, const void *encoded,
                              uint64_t element_count, float *output) {
    const ColiDTypeTraits *t = coli_dtype_traits(type);
    if (!t || !encoded || !output || element_count % t->block_values) return 0;
    const uint8_t *p = (const uint8_t *)encoded;
    uint64_t blocks = element_count / t->block_values;
    for (uint64_t b = 0; b < blocks; ++b) {
        float *y = output + b * t->block_values;
        switch (type) {
            case 0:
                for (uint32_t i = 0; i < t->block_values; ++i) y[i] = load_f32(p + 4u*i);
                break;
            case 1:
                y[0] = coli_fp16_to_fp32(load_u16(p));
                break;
            case 2: deq_q4_0(p, y); break;
            case 3: deq_q4_1(p, y); break;
            case 6: deq_q5_0(p, y); break;
            case 7: deq_q5_1(p, y); break;
            case 8: deq_q8(p, 2, y); break;
            case 9: deq_q8(p, 4, y); break;
            case 11: deq_q3_k(p, y); break;
            case 12: deq_q4_k(p, y); break;
            case 13: deq_q5_k(p, y); break;
            case 14: deq_q6_k(p, y); break;
            case 15: deq_q8_k(p, y); break;
            case 20: deq_iq4_nl(p, y); break;
            case 23: deq_iq4_xs(p, y); break;
            case 30:
                y[0] = coli_bf16_to_fp32(load_u16(p));
                break;
            case 39: deq_mxfp4(p, y); break;
            default: return 0;
        }
        p += t->block_bytes;
    }
    return 1;
}

float coli_dtype_dot_f32(ColiDType type, const void *encoded,
                         const float *x, uint64_t element_count) {
    const ColiDTypeTraits *t = coli_dtype_traits(type);
    if (!t || !encoded || !x || element_count % t->block_values) return 0.0f;
    const uint8_t *p = (const uint8_t *)encoded;
    float decoded[256];
    double sum = 0.0;
    uint64_t blocks = element_count / t->block_values;
    for (uint64_t b = 0; b < blocks; ++b) {
        if (!coli_dtype_dequantize_row(type, p, t->block_values, decoded)) return 0.0f;
        const float *xb = x + b * t->block_values;
        for (uint32_t i = 0; i < t->block_values; ++i) sum += (double)xb[i] * decoded[i];
        p += t->block_bytes;
    }
    return (float)sum;
}



static void q6_k_set_exact_zero(uint8_t *p, int idx) {
    const int half = idx >= 128;
    const int r = idx - half * 128;
    const int group = r / 32;
    const int l = r % 32;
    uint8_t *ql = p + half * 64;
    uint8_t *qh = p + 128 + half * 32;
    if (group == 0) {
        ql[l] &= 0xf0u;
        qh[l] = (uint8_t)((qh[l] & ~0x03u) | 0x02u);
    } else if (group == 1) {
        ql[l + 32] &= 0xf0u;
        qh[l] = (uint8_t)((qh[l] & ~0x0cu) | 0x08u);
    } else if (group == 2) {
        ql[l] &= 0x0fu;
        qh[l] = (uint8_t)((qh[l] & ~0x30u) | 0x20u);
    } else {
        ql[l + 32] &= 0x0fu;
        qh[l] = (uint8_t)((qh[l] & ~0xc0u) | 0x80u);
    }
}

int coli_dtype_zero_below_inplace(ColiDType type, void *encoded,
                                  uint64_t element_count, float threshold,
                                  uint64_t *changed_out) {
    if (changed_out) *changed_out = 0;
    if (!encoded || !isfinite(threshold) || threshold < 0.0f) return 0;
    if (threshold == 0.0f || element_count == 0) return 1;
    const ColiDTypeTraits *t = coli_dtype_traits(type);
    if (!t || element_count % t->block_values) return 0;

    uint8_t *p = (uint8_t *)encoded;
    uint64_t changed = 0;
    if (type == COLI_DTYPE_F32) {
        for (uint64_t i = 0; i < element_count; ++i) {
            float v = load_f32(p + 4u * i);
            if (v != 0.0f && fabsf(v) < threshold) {
                memset(p + 4u * i, 0, 4);
                ++changed;
            }
        }
    } else if (type == COLI_DTYPE_F16 || type == COLI_DTYPE_BF16) {
        for (uint64_t i = 0; i < element_count; ++i) {
            uint16_t h = load_u16(p + 2u * i);
            float v = type == COLI_DTYPE_F16 ? coli_fp16_to_fp32(h) : coli_bf16_to_fp32(h);
            if (v != 0.0f && fabsf(v) < threshold) {
                p[2u * i] = 0;
                p[2u * i + 1] = 0;
                ++changed;
            }
        }
    } else if (type == COLI_DTYPE_Q6_K) {
        float decoded[256];
        const uint64_t blocks = element_count / 256u;
        for (uint64_t b = 0; b < blocks; ++b) {
            uint8_t *block = p + b * 210u;
            if (!coli_dtype_dequantize_row(type, block, 256, decoded)) return 0;
            for (int i = 0; i < 256; ++i) {
                if (decoded[i] != 0.0f && fabsf(decoded[i]) < threshold) {
                    q6_k_set_exact_zero(block, i);
                    ++changed;
                }
            }
        }
    } else if (type == COLI_DTYPE_Q8_0 || type == COLI_DTYPE_Q8_1) {
        const uint32_t header = type == COLI_DTYPE_Q8_0 ? 2u : 4u;
        const uint64_t blocks = element_count / 32u;
        for (uint64_t b = 0; b < blocks; ++b) {
            uint8_t *block = p + b * t->block_bytes;
            const float d = coli_fp16_to_fp32(load_u16(block));
            int8_t *q = (int8_t *)(block + header);
            for (int i = 0; i < 32; ++i) {
                const float v = d * q[i];
                if (q[i] != 0 && fabsf(v) < threshold) {
                    q[i] = 0;
                    ++changed;
                }
            }
        }
    } else if (type == COLI_DTYPE_Q8_K) {
        const uint64_t blocks = element_count / 256u;
        for (uint64_t b = 0; b < blocks; ++b) {
            uint8_t *block = p + b * 292u;
            const float d = load_f32(block);
            int8_t *q = (int8_t *)(block + 4);
            for (int i = 0; i < 256; ++i) {
                const float v = d * q[i];
                if (q[i] != 0 && fabsf(v) < threshold) {
                    q[i] = 0;
                    ++changed;
                }
            }
            int16_t *bsums = (int16_t *)(block + 260);
            for (int g = 0; g < 16; ++g) {
                int sum = 0;
                for (int i = 0; i < 16; ++i) sum += q[g * 16 + i];
                bsums[g] = (int16_t)sum;
            }
        }
    } else {
        return 0;
    }
    if (changed_out) *changed_out = changed;
    return 1;
}

int coli_ggml_dequantize_row(uint32_t type, const void *encoded,
                             uint64_t element_count, float *output) {
    return coli_dtype_dequantize_row((ColiDType)type, encoded, element_count, output);
}
