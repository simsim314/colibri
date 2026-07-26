#include "../backend_cuda.h"
#include "../ggml_types.h"
#include "../sgguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Bits { unsigned char data[128]; unsigned count; };
static int put(Bits *b,unsigned bit){
    if(b->count>=sizeof(b->data)*8u)return 0;
    unsigned n=b->count++;if(bit)b->data[n>>3]|=(unsigned char)(1u<<(n&7));return 1;
}
static int enc_node(Bits *b,const unsigned char keep[256],int l,int r){
    int first=keep[l]!=0,uniform=1;
    for(int i=l+1;i<r;i++)if((keep[i]!=0)!=first){uniform=0;break;}
    if(uniform)return put(b,1)&&put(b,(unsigned)first);
    int m=(l+r)>>1;return put(b,0)&&enc_node(b,keep,l,m)&&enc_node(b,keep,m,r);
}
static void u16(unsigned char *p,unsigned v){p[0]=(unsigned char)v;p[1]=(unsigned char)(v>>8);}
static void u32(unsigned char *p,unsigned v){for(int i=0;i<4;i++)p[i]=(unsigned char)(v>>(8*i));}
static void u64(unsigned char *p,unsigned long long v){for(int i=0;i<8;i++)p[i]=(unsigned char)(v>>(8*i));}

struct SparseHost {
    std::vector<unsigned char> offsets;
    std::vector<unsigned char> blocks;
};

static int make_sparse_diagonal(float diagonal,SparseHost *out){
    if(!out)return 0;
    out->offsets.assign((256u+1u)*4u,0);
    out->blocks.clear();
    for(unsigned row=0;row<256u;row++){
        u32(out->offsets.data()+row*4u,(unsigned)out->blocks.size());
        size_t old=out->blocks.size();out->blocks.resize(old+36u,0);
        unsigned char *block=out->blocks.data()+old;
        block[row>>3]|=(unsigned char)(1u<<(row&7u));
        unsigned bits;std::memcpy(&bits,&diagonal,4);u32(block+32,bits);
    }
    u32(out->offsets.data()+256u*4u,(unsigned)out->blocks.size());
    return 1;
}

static float silu(float x){return x/(1.0f+std::exp(-x));}

static int close_rows(const float *got,const float *x,int rows,const char *what){
    for(int s=0;s<rows;s++)for(int i=0;i<256;i++){
        float xv=x[(size_t)s*256u+(unsigned)i];
        float expected=silu(xv)*xv;
        float diff=std::fabs(got[(size_t)s*256u+(unsigned)i]-expected);
        if(diff>3e-5f){
            std::fprintf(stderr,"%s mismatch row=%d col=%d expected %.9g got %.9g\n",
                         what,s,i,expected,got[(size_t)s*256u+(unsigned)i]);
            return 0;
        }
    }
    return 1;
}

static int test_sparse_single_matmul(int device){
    unsigned char keep[256];float x[256],values[256];unsigned retained=0;double expected=0;
    for(int i=0;i<256;i++){
        keep[i]=(unsigned char)((i%4)!=0 && (i<96 || i>=160));
        x[i]=(float)((i%13)-6)/7.f;values[i]=(float)(i-100)/33.f;
        if(keep[i]){retained++;expected+=(double)x[i]*values[i];}
    }
    unsigned total=32u+retained*4u;
    unsigned char *block=(unsigned char*)std::calloc(1,total);if(!block)return 0;
    unsigned char *vp=block+32;unsigned k=0;
    for(int i=0;i<256;i++)if(keep[i]){
        block[i>>3]|=(unsigned char)(1u<<(i&7));
        unsigned u;std::memcpy(&u,&values[i],4);u32(vp+4*k++,u);
    }
    unsigned char offsets[8];u32(offsets,0);u32(offsets+4,total);
    ColiCudaTensor *t=nullptr;float got=0;
    int ok=coli_cuda_tensor_upload_sgguf(&t,offsets,block,0,1,
                COLI_SGGUF_CODEC_RETAINED_F32,COLI_SGGUF_LAYOUT_BITMAP_V2,4,0,32,
                256,1,device)&&
           coli_cuda_tensor_matmul_host(t,&got,x,1);
    if(t)coli_cuda_tensor_free(t);
    std::free(block);
    if(!ok||std::fabs((double)got-expected)>2e-4){
        std::fprintf(stderr,"SGGUF bitmap CUDA mismatch expected %.9g got %.9g\n",expected,got);return 0;
    }
    return 1;
}

static int test_sparse_expert_paths(int device){
    SparseHost mh;if(!make_sparse_diagonal(1.0f,&mh))return 0;
    ColiCudaTensor *g=nullptr,*u=nullptr,*d=nullptr;
    const uint64_t blocks=256;
    int ok=coli_cuda_tensor_upload_sgguf(&g,mh.offsets.data(),mh.blocks.data(),0,blocks,
                COLI_SGGUF_CODEC_RETAINED_F32,COLI_SGGUF_LAYOUT_BITMAP_V2,4,0,32,256,256,device)&&
           coli_cuda_tensor_upload_sgguf(&u,mh.offsets.data(),mh.blocks.data(),0,blocks,
                COLI_SGGUF_CODEC_RETAINED_F32,COLI_SGGUF_LAYOUT_BITMAP_V2,4,0,32,256,256,device)&&
           coli_cuda_tensor_upload_sgguf(&d,mh.offsets.data(),mh.blocks.data(),0,blocks,
                COLI_SGGUF_CODEC_RETAINED_F32,COLI_SGGUF_LAYOUT_BITMAP_V2,4,0,32,256,256,device);
    std::vector<float> x(2u*256u),y(2u*256u);
    for(size_t i=0;i<x.size();i++)x[i]=(float)((int)(i%31u)-15)/32.f;
    if(ok)ok=coli_cuda_expert_mlp(g,u,d,y.data(),x.data(),2)&&close_rows(y.data(),x.data(),2,"sparse expert MLP");
    ColiCudaTensor *ga[1]={g},*ua[1]={u},*da[1]={d};int rows[1]={2};
    if(ok)ok=coli_cuda_expert_group(ga,ua,da,rows,1,y.data(),x.data())&&close_rows(y.data(),x.data(),2,"sparse expert group");
    if(ok)ok=coli_cuda_expert_group_issue(ga,ua,da,rows,1,x.data());
    if(ok){const float *async=coli_cuda_expert_group_take(device);ok=async&&close_rows(async,x.data(),2,"sparse expert async group");}
    if(g)coli_cuda_tensor_free(g);
    if(u)coli_cuda_tensor_free(u);
    if(d)coli_cuda_tensor_free(d);
    return ok;
}


static int test_new_dense_types(int device){
    std::vector<float> x256(256,1.0f);
    float got=0.0f;

    unsigned char iq4nl[18]={0};
    u16(iq4nl,0x3c00u);                  /* fp16 d = 1 */
    std::memset(iq4nl+2,0x88,16);        /* code 8 => +1 */
    std::vector<float> x32(32,1.0f);
    ColiCudaTensor *inl=nullptr;
    int ok=coli_cuda_tensor_upload_ggml(&inl,iq4nl,COLI_DTYPE_IQ4_NL,sizeof(iq4nl),32,1,device)&&
           coli_cuda_tensor_matmul_host(inl,&got,x32.data(),1)&&std::fabs(got-32.0f)<1e-4f;
    if(!ok)std::fprintf(stderr,"dense IQ4_NL CUDA mismatch got %.9g\n",got);
    if(inl)coli_cuda_tensor_free(inl);
    if(!ok)return 0;

    unsigned char iq4xs[136]={0};
    u16(iq4xs,0x3c00u);          /* fp16 d = 1 */
    u16(iq4xs+2,0xaaaau);        /* high scale bits = 2 */
    for(int i=0;i<4;i++)iq4xs[4+i]=0x11u; /* low bits = 1 => scale 1 */
    std::memset(iq4xs+8,0x88,128);         /* IQ4_NL code 8 => +1 */
    ColiCudaTensor *iq=nullptr;
    ok=coli_cuda_tensor_upload_ggml(&iq,iq4xs,COLI_DTYPE_IQ4_XS,sizeof(iq4xs),256,1,device)&&
           coli_cuda_tensor_matmul_host(iq,&got,x256.data(),1)&&std::fabs(got-256.0f)<1e-3f;
    if(!ok)std::fprintf(stderr,"dense IQ4_XS CUDA mismatch got %.9g\n",got);
    if(iq)coli_cuda_tensor_free(iq);
    if(!ok)return 0;

    unsigned char mx[17]={0};mx[0]=127;
    std::memset(mx+1,0x11,16);             /* code 1 => +0.5 */
    got=0.0f;
    ColiCudaTensor *mt=nullptr;
    ok=coli_cuda_tensor_upload_ggml(&mt,mx,COLI_DTYPE_MXFP4,sizeof(mx),32,1,device)&&
       coli_cuda_tensor_matmul_host(mt,&got,x32.data(),1)&&std::fabs(got-16.0f)<1e-4f;
    if(!ok)std::fprintf(stderr,"dense MXFP4 CUDA mismatch got %.9g\n",got);
    if(mt)coli_cuda_tensor_free(mt);
    return ok;
}

static int test_sparse_mxfp4_tail(int device){
    unsigned char offsets[8]={0};
    const unsigned block_bytes=32u+8u+32u;
    std::vector<unsigned char> block(block_bytes,0);
    for(unsigned i=0;i<64;i++)block[i>>3]|=(unsigned char)(1u<<(i&7u));
    block[32]=127;block[33]=127;
    std::memset(block.data()+40,0x11,32);   /* 64 retained code-1 values */
    u32(offsets,0);u32(offsets+4,block_bytes);
    std::vector<float> x(64,1.0f);float got=0.0f;
    ColiCudaTensor *t=nullptr;
    int ok=coli_cuda_tensor_upload_sgguf(&t,offsets,block.data(),0,1,
                COLI_SGGUF_CODEC_MXFP4_EXACT,COLI_SGGUF_LAYOUT_BITMAP_V3,4,8,4,
                64,1,device)&&
           coli_cuda_tensor_matmul_host(t,&got,x.data(),1)&&std::fabs(got-32.0f)<1e-4f;
    if(!ok)std::fprintf(stderr,"SPB3 MXFP4 tail CUDA mismatch got %.9g\n",got);
    if(t)coli_cuda_tensor_free(t);
    return ok;
}

static int test_sparse_iq4_nl_tail(int device){
    unsigned char offsets[8]={0};
    const unsigned block_bytes=32u+16u+32u;
    std::vector<unsigned char> block(block_bytes,0);
    for(unsigned i=0;i<64;i++)block[i>>3]|=(unsigned char)(1u<<(i&7u));
    u16(block.data()+32,0x3c00u);
    u16(block.data()+34,0x4000u);          /* second 32-value scale = 2 */
    std::memset(block.data()+48,0x88,32);  /* all retained code 8 => +1 */
    u32(offsets,0);u32(offsets+4,block_bytes);
    std::vector<float> x(64,1.0f);float got=0.0f;
    ColiCudaTensor *t=nullptr;
    int ok=coli_cuda_tensor_upload_sgguf(&t,offsets,block.data(),0,1,
                COLI_SGGUF_CODEC_IQ4_NL_EXACT,COLI_SGGUF_LAYOUT_BITMAP_V3,4,16,4,
                64,1,device)&&
           coli_cuda_tensor_matmul_host(t,&got,x.data(),1)&&std::fabs(got-96.0f)<1e-4f;
    if(!ok)std::fprintf(stderr,"SPB3 IQ4_NL tail CUDA mismatch got %.9g\n",got);
    if(t)coli_cuda_tensor_free(t);
    return ok;
}

static int test_sparse_iq4_xs(int device){
    unsigned char offsets[8]={0};
    const unsigned block_bytes=32u+8u+128u;
    std::vector<unsigned char> block(block_bytes,0);
    std::memset(block.data(),0xff,32);        /* retain all 256 values */
    u16(block.data()+32,0x3c00u);             /* fp16 d = 1 */
    u16(block.data()+34,0xaaaau);             /* scale high bits = 2 */
    std::memset(block.data()+36,0x11,4);       /* scale low bits = 1 => dl = 1 */
    std::memset(block.data()+40,0x88,128);     /* nonlinear code 8 => +1 */
    u32(offsets,0);u32(offsets+4,block_bytes);
    std::vector<float> x(256,1.0f);float got=0.0f;
    ColiCudaTensor *t=nullptr;
    int ok=coli_cuda_tensor_upload_sgguf(&t,offsets,block.data(),0,1,
                COLI_SGGUF_CODEC_IQ4_XS_EXACT,COLI_SGGUF_LAYOUT_BITMAP_V3,4,8,4,
                256,1,device)&&
           coli_cuda_tensor_matmul_host(t,&got,x.data(),1)&&std::fabs(got-256.0f)<1e-3f;
    if(!ok)std::fprintf(stderr,"SPB3 IQ4_XS CUDA mismatch got %.9g\n",got);
    if(t)coli_cuda_tensor_free(t);
    return ok;
}

static int test_native_group_rows(int device){
    std::vector<float> identity(256u*256u,0.f);
    for(unsigned i=0;i<256u;i++)identity[(size_t)i*256u+i]=1.f;
    ColiCudaTensor *g=nullptr,*u=nullptr,*d=nullptr;
    uint64_t bytes=(uint64_t)identity.size()*sizeof(float);
    int ok=coli_cuda_tensor_upload_ggml(&g,identity.data(),COLI_DTYPE_F32,bytes,256,256,device)&&
           coli_cuda_tensor_upload_ggml(&u,identity.data(),COLI_DTYPE_F32,bytes,256,256,device)&&
           coli_cuda_tensor_upload_ggml(&d,identity.data(),COLI_DTYPE_F32,bytes,256,256,device);
    std::vector<float> x(2u*256u),y(2u*256u);
    for(size_t i=0;i<x.size();i++)x[i]=(float)((int)(i%29u)-14)/30.f;
    ColiCudaTensor *ga[1]={g},*ua[1]={u},*da[1]={d};int rows[1]={2};
    if(ok)ok=coli_cuda_expert_group(ga,ua,da,rows,1,y.data(),x.data())&&close_rows(y.data(),x.data(),2,"native GGUF expert group");
    if(g)coli_cuda_tensor_free(g);
    if(u)coli_cuda_tensor_free(u);
    if(d)coli_cuda_tensor_free(d);
    return ok;
}

int main(){
    int device=0;if(!coli_cuda_init(&device,1)){std::puts("SKIP: CUDA unavailable");return 0;}
    int ok=test_sparse_single_matmul(device)&&
           test_sparse_expert_paths(device)&&
           test_new_dense_types(device)&&
           test_sparse_mxfp4_tail(device)&&
           test_sparse_iq4_nl_tail(device)&&
           test_sparse_iq4_xs(device)&&
           test_native_group_rows(device);
    coli_cuda_shutdown();
    if(!ok)return 1;
    std::puts("test_sgguf_cuda: ok");return 0;
}
