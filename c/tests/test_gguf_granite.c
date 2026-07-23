/* Tiny end-to-end Granite-MoE GGUF: metadata -> mmap-native tensors
 * -> CPU GQA/MoE forward -> tied output logits. */
#include "../gguf_granite.c"

#include <assert.h>
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

static void put_u32(FILE *f, uint32_t v) {
    unsigned char p[4]; for (int i=0;i<4;i++) p[i]=(unsigned char)(v>>(8*i)); assert(fwrite(p,1,4,f)==4);
}
static void put_u64(FILE *f, uint64_t v) {
    unsigned char p[8]; for (int i=0;i<8;i++) p[i]=(unsigned char)(v>>(8*i)); assert(fwrite(p,1,8,f)==8);
}
static void put_f32(FILE *f, float v) { uint32_t u; memcpy(&u,&v,4); put_u32(f,u); }
static void put_str(FILE *f, const char *s) { size_t n=strlen(s); put_u64(f,n); assert(fwrite(s,1,n,f)==n); }
static void wkv_str(FILE*f,const char*k,const char*v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_STRING);put_str(f,v);}
static void wkv_u32(FILE*f,const char*k,uint32_t v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_UINT32);put_u32(f,v);}
static void wkv_f32(FILE*f,const char*k,float v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_FLOAT32);put_f32(f,v);}
static void wkv_bool(FILE*f,const char*k,int v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_BOOL);fputc(v?1:0,f);}
static void wwkv_str_array(FILE*f,const char*k,const char **v,int n){
    put_str(f,k);put_u32(f,COLI_GGUF_TYPE_ARRAY);put_u32(f,COLI_GGUF_TYPE_STRING);put_u64(f,(uint64_t)n);
    for(int i=0;i<n;i++)put_str(f,v[i]);
}
static void wwkv_u32_array(FILE*f,const char*k,const uint32_t *v,int n){
    put_str(f,k);put_u32(f,COLI_GGUF_TYPE_ARRAY);put_u32(f,COLI_GGUF_TYPE_UINT32);put_u64(f,(uint64_t)n);
    for(int i=0;i<n;i++)put_u32(f,v[i]);
}
static void tensor_desc(FILE*f,const char*name,const uint64_t*d,int nd,uint64_t off){
    put_str(f,name);put_u32(f,(uint32_t)nd);for(int i=0;i<nd;i++)put_u64(f,d[i]);put_u32(f,0);put_u64(f,off);
}
static void pad_abs(FILE*f,long a){long p=ftell(f);assert(p>=0);while(p%a){fputc(0,f);p++;}}
static void pad_rel(FILE*f,long base,uint64_t off){long p=ftell(f);assert(p>=base);while((uint64_t)(p-base)<off){fputc(0,f);p++;}assert((uint64_t)(p-base)==off);}
static void write_floats(FILE*f,const float*v,int n){for(int i=0;i<n;i++)put_f32(f,v[i]);}

static void write_fixture(const char *path) {
    FILE *f=fopen(path,"wb");assert(f);
    fwrite("GGUF",1,4,f);put_u32(f,3);put_u64(f,12);put_u64(f,21);
    wkv_str(f,"general.architecture","granitemoe");
    wkv_u32(f,"granitemoe.block_count",1);wkv_u32(f,"granitemoe.embedding_length",2);
    wkv_u32(f,"granitemoe.attention.head_count",1);wkv_u32(f,"granitemoe.attention.head_count_kv",1);
    wkv_u32(f,"granitemoe.expert_count",1);wkv_u32(f,"granitemoe.expert_used_count",1);
    wkv_u32(f,"granitemoe.expert_feed_forward_length",2);wkv_u32(f,"granitemoe.rope.dimension_count",2);
    wkv_f32(f,"granitemoe.rope.freq_base",10000.f);wkv_f32(f,"granitemoe.attention.layer_norm_rms_epsilon",1e-6f);
    wkv_f32(f,"granitemoe.attention.scale",1.f);wkv_f32(f,"granitemoe.embedding_scale",1.f);
    wkv_f32(f,"granitemoe.residual_scale",1.f);wkv_f32(f,"granitemoe.logit_scale",1.f);
    wkv_str(f,"tokenizer.ggml.pre","refact");
    const char *tokens[]={"<eos>","A","B","C"};const uint32_t types[]={3,1,1,1};
    wwkv_str_array(f,"tokenizer.ggml.tokens",tokens,4);wwkv_u32_array(f,"tokenizer.ggml.token_type",types,4);
    wwkv_str_array(f,"tokenizer.ggml.merges",NULL,0);wkv_u32(f,"tokenizer.ggml.eos_token_id",0);
    wkv_bool(f,"tokenizer.ggml.add_bos_token",0);

    const uint64_t d24[]={2,4},d2[]={2},d22[]={2,2},d21[]={2,1},d221[]={2,2,1};
    tensor_desc(f,"token_embd.weight",d24,2,0);tensor_desc(f,"output_norm.weight",d2,1,32);
    tensor_desc(f,"blk.0.attn_norm.weight",d2,1,64);tensor_desc(f,"blk.0.attn_q.weight",d22,2,96);
    tensor_desc(f,"blk.0.attn_k.weight",d22,2,128);tensor_desc(f,"blk.0.attn_v.weight",d22,2,160);
    tensor_desc(f,"blk.0.attn_output.weight",d22,2,192);tensor_desc(f,"blk.0.ffn_norm.weight",d2,1,224);
    tensor_desc(f,"blk.0.ffn_gate_inp.weight",d21,2,256);tensor_desc(f,"blk.0.ffn_gate_exps.weight",d221,3,288);
    tensor_desc(f,"blk.0.ffn_up_exps.weight",d221,3,320);tensor_desc(f,"blk.0.ffn_down_exps.weight",d221,3,352);
    pad_abs(f,32);long base=ftell(f);
    const float emb[]={0,0, 1,0, 0,2, -1,0},ones[]={1,1},zero4[]={0,0,0,0},zero2[]={0,0};
    pad_rel(f,base,0);write_floats(f,emb,8);pad_rel(f,base,32);write_floats(f,ones,2);
    pad_rel(f,base,64);write_floats(f,ones,2);pad_rel(f,base,96);write_floats(f,zero4,4);
    pad_rel(f,base,128);write_floats(f,zero4,4);pad_rel(f,base,160);write_floats(f,zero4,4);
    pad_rel(f,base,192);write_floats(f,zero4,4);pad_rel(f,base,224);write_floats(f,ones,2);
    pad_rel(f,base,256);write_floats(f,zero2,2);pad_rel(f,base,288);write_floats(f,zero4,4);
    pad_rel(f,base,320);write_floats(f,zero4,4);pad_rel(f,base,352);write_floats(f,zero4,4);
    assert(fclose(f)==0);
}

int main(void) {
    char path[256];
#ifdef _WIN32
    snprintf(path,sizeof(path),"test_gguf_granite_%ld.gguf",(long)getpid());
#else
    snprintf(path,sizeof(path),"/tmp/test_gguf_granite_%ld.gguf",(long)getpid());
#endif
    write_fixture(path);
    GraniteModel m;GraniteScratch s={0};char err[512];
    ColiExec exec={COLI_BACKEND_CPU,0};
    assert(load_model(&m,path,2,0,exec,err,sizeof(err)));
    assert(scratch_alloc(&m,&s,err,sizeof(err)));
    assert(model_forward(&m,&s,1,0,err,sizeof(err)));
    assert(argmax(s.logits,m.vocab)==1); /* A remains the greedy tied-embedding token. */
    scratch_free(&s);model_free(&m);unlink(path);
    puts("test_gguf_granite: ok");return 0;
}
