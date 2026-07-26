#include "../tensor.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#define unlink _unlink
#define getpid _getpid
#else
#include <unistd.h>
#endif

static void put_u16(uint8_t *p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);}
static void put_u32f(FILE*f,uint32_t v){uint8_t p[4];for(int i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));assert(fwrite(p,1,4,f)==4);}
static void put_u64f(FILE*f,uint64_t v){uint8_t p[8];for(int i=0;i<8;i++)p[i]=(uint8_t)(v>>(8*i));assert(fwrite(p,1,8,f)==8);}
static void put_str(FILE*f,const char*s){size_t n=strlen(s);put_u64f(f,n);assert(fwrite(s,1,n,f)==n);}
static void pad32(FILE*f){long p=ftell(f);assert(p>=0);while(p&31){fputc(0,f);p++;}}
static void near(float a,float b){assert(fabsf(a-b)<1e-4f);}

static void fill_scales(uint8_t *s){
    memset(s,0,12);for(int i=0;i<4;i++)s[i]=1;for(int i=8;i<12;i++)s[i]=1;
}
static void fill_q4k(uint8_t *p,uint16_t d){
    memset(p,0,144);put_u16(p,d);fill_scales(p+4);memset(p+16,0x11,128);
}
static void fill_q5k(uint8_t *p,uint16_t d){
    memset(p,0,176);put_u16(p,d);fill_scales(p+4);memset(p+48,0x11,128);
}
static void fill_q6k(uint8_t *p,uint16_t d){
    memset(p,0,210);memset(p,0x11,128);memset(p+128,0xaa,64);
    memset(p+192,1,16);put_u16(p+208,d);
}

static void test_direct_n(ColiDType dtype,uint8_t *storage,uint64_t bytes,int n,float expected){
    ColiTensor t={0};t.dtype=dtype;t.n_dims=2;t.dims[0]=(uint64_t)n;t.dims[1]=1;
    t.element_count=(uint64_t)n;t.row_count=1;t.storage_bytes=bytes;t.data=storage;
    assert(coli_dtype_row_size(dtype,(uint64_t)n,&t.row_bytes));
    float x[256],y[1];assert(n<=(int)(sizeof(x)/sizeof(x[0])));for(int i=0;i<n;i++)x[i]=1.f;
    ColiExec cpu={COLI_BACKEND_CPU,0};
    assert(coli_tensor_matmul(&cpu,y,x,&t,1,n,1));near(y[0],expected);
}
static void test_direct(ColiDType dtype,uint8_t *storage,uint64_t bytes,float expected){
    test_direct_n(dtype,storage,bytes,256,expected);
}

static void write_fixture(const char *path){
    FILE*f=fopen(path,"wb");assert(f);
    fwrite("GGUF",1,4,f);put_u32f(f,3);put_u64f(f,1);put_u64f(f,0);
    put_str(f,"w");put_u32f(f,2);put_u64f(f,256);put_u64f(f,2);put_u32f(f,COLI_DTYPE_Q4_K);put_u64f(f,0);
    pad32(f);uint8_t row[144];fill_q4k(row,0x3c00);assert(fwrite(row,1,144,f)==144);
    fill_q4k(row,0x4000);assert(fwrite(row,1,144,f)==144);assert(fclose(f)==0);
}

int main(void){
    uint8_t q4[144],q5[176],q6[210],bf16[512],iq4xs[136]={0},mxfp4[17]={0};
    fill_q4k(q4,0x3c00);fill_q5k(q5,0x3c00);fill_q6k(q6,0x3c00);
    for(int i=0;i<256;i++)put_u16(bf16+2*i,0x3f80);
    put_u16(iq4xs,0x3c00);put_u16(iq4xs+2,0xaaaa);for(int i=0;i<4;i++)iq4xs[4+i]=0x11;
    memset(iq4xs+8,0x88,128);
    mxfp4[0]=127;memset(mxfp4+1,0x11,16);
    test_direct(COLI_DTYPE_Q4_K,q4,sizeof(q4),256.f);
    test_direct(COLI_DTYPE_Q5_K,q5,sizeof(q5),256.f);
    test_direct(COLI_DTYPE_Q6_K,q6,sizeof(q6),256.f);
    test_direct(COLI_DTYPE_BF16,bf16,sizeof(bf16),256.f);
    test_direct(COLI_DTYPE_IQ4_XS,iq4xs,sizeof(iq4xs),256.f);
    test_direct_n(COLI_DTYPE_MXFP4,mxfp4,sizeof(mxfp4),32,16.f);

    char path[256];
#ifdef _WIN32
    snprintf(path,sizeof(path),"test_tensor_%ld.gguf",(long)getpid());
#else
    snprintf(path,sizeof(path),"/tmp/test_tensor_%ld.gguf",(long)getpid());
#endif
    write_fixture(path);
    ColiGgufFile g;g.fd=-1;assert(coli_gguf_open(&g,path));assert(g.mapping);
    const ColiGgufTensorInfo *ti=coli_gguf_find_tensor(&g,"w");assert(ti);
    ColiTensor t;char err[256];assert(coli_tensor_bind_gguf(&g,ti,&t,err,sizeof(err)));
    assert(t.mmap_backed&&!t.owns_data&&t.storage_bytes==288);
    assert(t.data==coli_gguf_mapped_at(&g,ti->absolute_offset,288));
    float x[256],y[2],row[256];for(int i=0;i<256;i++)x[i]=1.f;
    ColiExec cpu={COLI_BACKEND_CPU,0};assert(coli_tensor_matmul(&cpu,y,x,&t,1,256,2));
    near(y[0],256.f);near(y[1],512.f);
    assert(coli_tensor_read_row_f32(&t,1,row,256));near(row[0],2.f);near(row[255],2.f);
    ColiTensor v;assert(coli_tensor_rows_view(&t,1,1,&v));
    assert(v.data==t.data+144&&v.storage_bytes==144);assert(coli_tensor_matmul(&cpu,y,x,&v,1,256,1));near(y[0],512.f);
    coli_tensor_destroy(&v);coli_tensor_destroy(&t);coli_gguf_close(&g);unlink(path);
    puts("test_tensor: mmap/native Q4_K/Q5_K/Q6_K/BF16/IQ4_XS/MXFP4 CPU ok");return 0;
}
