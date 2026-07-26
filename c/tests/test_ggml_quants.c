#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../ggml_quants.h"

static void put_u16(uint8_t *p, uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void put_f32(uint8_t *p, float f) { uint32_t u; memcpy(&u,&f,4); p[0]=u; p[1]=u>>8; p[2]=u>>16; p[3]=u>>24; }
static void near(float a, float b) { assert(fabsf(a-b) < 1e-6f); }

int main(void) {
    near(coli_fp16_to_fp32(0x3c00), 1.0f);
    near(coli_fp16_to_fp32(0xc000), -2.0f);
    assert(fabsf(coli_fp16_to_fp32(0x0001) - 5.96046448e-8f) < 1e-12f);
    near(coli_bf16_to_fp32(0x3f80), 1.0f);
    near(coli_bf16_to_fp32(0xc000), -2.0f);


    float y[256];
    uint8_t f32b[8]; put_f32(f32b, 1.25f); put_f32(f32b+4, -3.5f);
    assert(coli_ggml_dequantize_row(0,f32b,2,y)); near(y[0],1.25f); near(y[1],-3.5f);
    uint8_t f16b[4]; put_u16(f16b,0x3c00); put_u16(f16b+2,0xc000);
    assert(coli_ggml_dequantize_row(1,f16b,2,y)); near(y[0],1); near(y[1],-2);
    uint8_t bf16b[4]; put_u16(bf16b,0x3f80); put_u16(bf16b+2,0xc000);
    assert(coli_ggml_dequantize_row(30,bf16b,2,y)); near(y[0],1); near(y[1],-2);

    uint8_t q40[18] = {0}; put_u16(q40, 0x3c00);
    for (int i=0;i<16;i++) q40[2+i]=(uint8_t)(i | ((15-i)<<4));
    assert(coli_ggml_dequantize_row(2,q40,32,y));
    near(y[0],-8); near(y[15],7); near(y[16],7); near(y[31],-8);

    uint8_t q41[20] = {0}; put_u16(q41,0x4000); put_u16(q41+2,0x3c00);
    q41[4]=0x21;
    assert(coli_ggml_dequantize_row(3,q41,32,y));
    near(y[0],3); near(y[16],5);


    uint8_t q50[22] = {0}; put_u16(q50,0x3c00); q50[6]=0xf0;
    assert(coli_ggml_dequantize_row(6,q50,32,y)); near(y[0],-16); near(y[16],-1);

    uint8_t q51[24] = {0}; put_u16(q51,0x3c00); put_u16(q51+2,0x4000); q51[8]=0x21;
    assert(coli_ggml_dequantize_row(7,q51,32,y)); near(y[0],3); near(y[16],4);

    uint8_t q80[34] = {0}; put_u16(q80,0x3800);
    for (int i=0;i<32;i++) q80[2+i]=(uint8_t)(int8_t)(i-16);
    assert(coli_ggml_dequantize_row(8,q80,32,y));
    near(y[0],-8); near(y[31],7.5f);


    uint8_t q81[36] = {0}; put_u16(q81,0x3800); q81[4]=(uint8_t)(int8_t)-4; q81[35]=6;
    assert(coli_ggml_dequantize_row(9,q81,32,y)); near(y[0],-2); near(y[31],3);

    uint8_t q3k[110] = {0}; put_u16(q3k+108,0x3c00);
    assert(coli_ggml_dequantize_row(11,q3k,256,y)); near(y[0],128); near(y[255],128);

    uint8_t q4k[144] = {0}; put_u16(q4k,0x3c00); q4k[4]=1; q4k[5]=2;
    for(int i=0;i<128;i++) q4k[16+i]=0x31;
    assert(coli_ggml_dequantize_row(12,q4k,256,y)); near(y[0],1); near(y[32],6);

    uint8_t q5k[176] = {0}; put_u16(q5k,0x3c00); q5k[4]=1; q5k[5]=2;
    for(int i=0;i<128;i++) q5k[48+i]=0x31;
    assert(coli_ggml_dequantize_row(13,q5k,256,y)); near(y[0],1); near(y[32],6);

    uint8_t q6k[210] = {0}; put_u16(q6k+208,0x3c00);
    for(int i=0;i<16;i++) q6k[192+i]=1;
    assert(coli_ggml_dequantize_row(14,q6k,256,y)); near(y[0],-32); near(y[255],-32);

    uint8_t q8k[292] = {0}; put_f32(q8k,0.25f);
    for (int i=0;i<256;i++) q8k[4+i]=(uint8_t)(int8_t)((i%17)-8);
    assert(coli_ggml_dequantize_row(15,q8k,256,y));
    near(y[0],-2); near(y[16],2); near(y[17],-2);

    uint8_t iq4nl[18] = {0};
    put_u16(iq4nl,0x3c00); /* d = 1 */
    for(int i=0;i<16;i++) iq4nl[2+i]=0xf8;
    assert(coli_ggml_dequantize_row(20,iq4nl,32,y));
    near(y[0],1); near(y[15],1); near(y[16],113); near(y[31],113);
    near(coli_dtype_dot_f32(COLI_DTYPE_IQ4_NL,iq4nl,y,32),
         16.0f + 16.0f*113.0f*113.0f);

    uint8_t iq4xs[136] = {0};
    put_u16(iq4xs,0x3c00); /* d = 1 */
    put_u16(iq4xs+2,0xaaaa); /* high two scale bits = 2 */
    for(int i=0;i<4;i++) iq4xs[4+i]=0x11; /* low scale nibble = 1 => 33-32 = 1 */
    for(int i=0;i<128;i++) iq4xs[8+i]=0xf8;
    assert(coli_ggml_dequantize_row(23,iq4xs,256,y));
    near(y[0],1); near(y[15],1); near(y[16],113); near(y[31],113);

    uint8_t mxfp4[17] = {0}; mxfp4[0]=127;
    for(int i=0;i<16;i++) mxfp4[1+i]=(uint8_t)(7u | (15u<<4));
    assert(coli_ggml_dequantize_row(39,mxfp4,32,y));
    near(y[0],6); near(y[15],6); near(y[16],-6); near(y[31],-6);
    near(coli_dtype_dot_f32(COLI_DTYPE_MXFP4,mxfp4,y,32),1152.0f);

    uint64_t changed = 0;
    assert(coli_dtype_zero_below_inplace(COLI_DTYPE_Q6_K, q6k, 256, 33.0f, &changed));
    assert(changed == 256);
    assert(coli_ggml_dequantize_row(14,q6k,256,y));
    for (int i=0;i<256;i++) near(y[i],0);

    assert(coli_dtype_zero_below_inplace(COLI_DTYPE_Q8_0, q80, 32, 1.0f, &changed));
    assert(changed == 2);
    assert(coli_ggml_dequantize_row(8,q80,32,y));
    near(y[15],0); near(y[17],0); near(y[14],-1); near(y[18],1);
    assert(!coli_dtype_zero_below_inplace(COLI_DTYPE_Q4_K, q4k, 256, 0.1f, &changed));

    assert(!coli_ggml_dequantize_row(12,q8k,255,y));
    puts("test_ggml_quants: ok");
    return 0;
}
