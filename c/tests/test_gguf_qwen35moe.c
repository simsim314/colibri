#include "../gguf_qwen35moe.c"

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

static void put_u32(FILE*f,uint32_t v){unsigned char p[4];for(int i=0;i<4;i++)p[i]=(unsigned char)(v>>(8*i));assert(fwrite(p,1,4,f)==4);}
static void put_u64(FILE*f,uint64_t v){unsigned char p[8];for(int i=0;i<8;i++)p[i]=(unsigned char)(v>>(8*i));assert(fwrite(p,1,8,f)==8);}
static void put_f32(FILE*f,float v){uint32_t u;memcpy(&u,&v,4);put_u32(f,u);}
static void put_str(FILE*f,const char*s){put_u64(f,strlen(s));assert(fwrite(s,1,strlen(s),f)==strlen(s));}
static void kv_str(FILE*f,const char*k,const char*v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_STRING);put_str(f,v);}
static void kv_u32(FILE*f,const char*k,uint32_t v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_UINT32);put_u32(f,v);}
static void kv_f32(FILE*f,const char*k,float v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_FLOAT32);put_f32(f,v);}
static void kv_bool(FILE*f,const char*k,int v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_BOOL);fputc(v?1:0,f);}
static void kv_str_array(FILE*f,const char*k,const char **v,int n){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_ARRAY);put_u32(f,COLI_GGUF_TYPE_STRING);put_u64(f,n);for(int i=0;i<n;i++)put_str(f,v[i]);}
static void kv_u32_array(FILE*f,const char*k,const uint32_t*v,int n){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_ARRAY);put_u32(f,COLI_GGUF_TYPE_UINT32);put_u64(f,n);for(int i=0;i<n;i++)put_u32(f,v[i]);}
static uint64_t align32(uint64_t x){return(x+31u)&~31u;}

typedef struct{const char*name;int nd;uint64_t d[3];uint64_t off;float*data;size_t count;}TD;
static void desc(FILE*f,const TD*t){put_str(f,t->name);put_u32(f,t->nd);for(int i=0;i<t->nd;i++)put_u64(f,t->d[i]);put_u32(f,0);put_u64(f,t->off);}
static void pad32(FILE*f){while(ftell(f)%32)fputc(0,f);}
static void pad_to(FILE*f,long base,uint64_t off){while((uint64_t)(ftell(f)-base)<off)fputc(0,f);assert((uint64_t)(ftell(f)-base)==off);}

static void write_fixture(const char *path){
    float eye4[16]={0},ones4[4]={1,1,1,1},ones2[2]={1,1};
    for(int i=0;i<4;i++)eye4[i*4+i]=1.f;
    float z32[32]={0},z24[24]={0},z16[16]={0},z8[8]={0},z6[6]={0},z4[4]={0},z2[2]={0},z1[1]={0};
    float a1[1]={-1.f},eh[32]={0};for(int i=0;i<4;i++)eh[i*8+i]=1.f;
#define MOE0(prefix) \
      {prefix ".ffn_gate_inp.weight",2,{4,1},0,z4,4}, \
      {prefix ".ffn_gate_exps.weight",3,{4,2,1},0,z8,8}, \
      {prefix ".ffn_up_exps.weight",3,{4,2,1},0,z8,8}, \
      {prefix ".ffn_down_exps.weight",3,{2,4,1},0,z8,8}, \
      {prefix ".ffn_gate_inp_shexp.weight",1,{4},0,z4,4}, \
      {prefix ".ffn_gate_shexp.weight",2,{4,2},0,z8,8}, \
      {prefix ".ffn_up_shexp.weight",2,{4,2},0,z8,8}, \
      {prefix ".ffn_down_shexp.weight",2,{2,4},0,z8,8}
    TD t[]={
      {"token_embd.weight",2,{4,4},0,eye4,16},{"output_norm.weight",1,{4},0,ones4,4},{"output.weight",2,{4,4},0,eye4,16},
      {"blk.0.attn_norm.weight",1,{4},0,ones4,4},{"blk.0.post_attention_norm.weight",1,{4},0,ones4,4},
      {"blk.0.attn_qkv.weight",2,{4,6},0,z24,24},{"blk.0.attn_gate.weight",2,{4,2},0,z8,8},
      {"blk.0.ssm_beta.weight",2,{4,1},0,z4,4},{"blk.0.ssm_alpha.weight",2,{4,1},0,z4,4},
      {"blk.0.ssm_conv1d.weight",2,{2,6},0,z24,12},{"blk.0.ssm_dt.bias",1,{1},0,z1,1},{"blk.0.ssm_a",1,{1},0,a1,1},
      {"blk.0.ssm_norm.weight",1,{2},0,ones2,2},{"blk.0.ssm_out.weight",2,{2,4},0,z8,8},
      MOE0("blk.0"),
      {"blk.1.attn_norm.weight",1,{4},0,ones4,4},{"blk.1.post_attention_norm.weight",1,{4},0,ones4,4},
      {"blk.1.attn_q.weight",2,{4,8},0,z32,32},{"blk.1.attn_k.weight",2,{4,2},0,z8,8},{"blk.1.attn_v.weight",2,{4,2},0,z8,8},
      {"blk.1.attn_q_norm.weight",1,{2},0,ones2,2},{"blk.1.attn_k_norm.weight",1,{2},0,ones2,2},{"blk.1.attn_output.weight",2,{4,4},0,z16,16},
      MOE0("blk.1"),
      {"blk.1.nextn.eh_proj.weight",2,{8,4},0,eh,32},{"blk.1.nextn.enorm.weight",1,{4},0,ones4,4},
      {"blk.1.nextn.hnorm.weight",1,{4},0,ones4,4},{"blk.1.nextn.shared_head_norm.weight",1,{4},0,ones4,4},
    };
#undef MOE0
    (void)z6;(void)z2;
    const int nt=(int)(sizeof(t)/sizeof(t[0]));uint64_t off=0;
    for(int i=0;i<nt;i++){t[i].off=off;off=align32(off+t[i].count*4);}
    FILE*f=fopen(path,"wb");assert(f);fwrite("GGUF",1,4,f);put_u32(f,3);put_u64(f,nt);put_u64(f,30);
    kv_str(f,"general.architecture","qwen35moe");kv_u32(f,"qwen35moe.block_count",2);kv_u32(f,"qwen35moe.nextn_predict_layers",1);
    kv_u32(f,"qwen35moe.context_length",16);kv_u32(f,"qwen35moe.embedding_length",4);
    kv_u32(f,"qwen35moe.attention.head_count",2);kv_u32(f,"qwen35moe.attention.head_count_kv",1);
    kv_u32(f,"qwen35moe.attention.key_length",2);kv_u32(f,"qwen35moe.attention.value_length",2);
    kv_u32(f,"qwen35moe.expert_count",1);kv_u32(f,"qwen35moe.expert_used_count",1);
    kv_u32(f,"qwen35moe.expert_feed_forward_length",2);kv_u32(f,"qwen35moe.expert_shared_feed_forward_length",2);
    kv_u32(f,"qwen35moe.ssm.conv_kernel",2);kv_u32(f,"qwen35moe.ssm.state_size",2);kv_u32(f,"qwen35moe.ssm.group_count",1);
    kv_u32(f,"qwen35moe.ssm.time_step_rank",1);kv_u32(f,"qwen35moe.ssm.inner_size",2);kv_u32(f,"qwen35moe.full_attention_interval",4);
    kv_u32(f,"qwen35moe.rope.dimension_count",2);const uint32_t sec[4]={1,0,0,0};kv_u32_array(f,"qwen35moe.rope.dimension_sections",sec,4);
    kv_f32(f,"qwen35moe.rope.freq_base",10000.f);kv_f32(f,"qwen35moe.attention.layer_norm_rms_epsilon",1e-6f);
    kv_str(f,"tokenizer.ggml.pre","refact");const char*toks[]={"<eos>","A","B","C"};const uint32_t types[]={3,1,1,1};
    kv_str_array(f,"tokenizer.ggml.tokens",toks,4);kv_u32_array(f,"tokenizer.ggml.token_type",types,4);kv_str_array(f,"tokenizer.ggml.merges",NULL,0);
    kv_u32(f,"tokenizer.ggml.eos_token_id",0);kv_u32(f,"tokenizer.ggml.bos_token_id",0);kv_bool(f,"tokenizer.ggml.add_bos_token",0);
    for(int i=0;i<nt;i++)desc(f,&t[i]);pad32(f);long base=ftell(f);
    for(int i=0;i<nt;i++){pad_to(f,base,t[i].off);for(size_t j=0;j<t[i].count;j++)put_f32(f,t[i].data[j]);}
    assert(fclose(f)==0);
}

static void bind_vec(ColiTensor*t,float*p,int n){memset(t,0,sizeof(*t));t->dtype=COLI_DTYPE_F32;t->n_dims=1;t->dims[0]=n;t->row_count=1;t->row_bytes=n*4;t->storage_bytes=n*4;t->data=(const uint8_t*)p;}
static void test_separate_delta(void){
    Q35Model m={0};m.n_key_heads=1;m.n_value_heads=1;m.state_size=2;m.value_head_dim=2;m.key_dim=2;m.value_dim=2;m.eps=1e-6f;
    Q35Layer l={0};float dt[1]={0},a[1]={-1},nw[2]={1,1};bind_vec(&l.dt_bias,dt,1);bind_vec(&l.a,a,1);bind_vec(&l.ssm_norm,nw,2);
    float conv[6]={1,0,1,0,2,3},z[2]={0,0},beta[1]={0},alpha[1]={0},state[4]={0},out[2]={0};char err[128];
    assert(q35_delta_decode_cpu(&m,&l,conv,z,beta,alpha,state,out,err,sizeof(err)));
    assert(isfinite(out[0])&&isfinite(out[1]));
}

/* Current Qwen3.5/3.6 GGUF conversion tiles V heads across K heads.  With
 * nk=2,nv=4 the mapping must be 0,1,0,1, not the grouped 0,0,1,1. */
static void test_tiled_value_head_mapping(void){
    Q35Model m={0};m.n_key_heads=2;m.n_value_heads=4;m.state_size=2;m.value_head_dim=2;
    m.key_dim=4;m.value_dim=8;m.eps=1e-6f;
    Q35Layer l={0};
    float dt[4]={0,0,0,0},a[4]={-1,-1,-1,-1},nw[2]={1,1};
    bind_vec(&l.dt_bias,dt,4);bind_vec(&l.a,a,4);bind_vec(&l.ssm_norm,nw,2);
    /* q0=q1=[1,0], k0=[1,0], k1=[0,1].  Thus only value heads mapped
     * to k0 produce a nonzero result.  Tiled order means heads 0 and 2. */
    float conv[16]={
        1,0, 1,0,                 /* q0,q1 */
        1,0, 0,1,                 /* k0,k1 */
        1,1, 1,1, 1,1, 1,1       /* v0..v3 */
    };
    float z[8]={1,1,1,1,1,1,1,1};
    float beta[4]={20,20,20,20},alpha[4]={0,0,0,0};
    float state[16]={0},out[8]={0};char err[128];
    assert(q35_delta_decode_cpu(&m,&l,conv,z,beta,alpha,state,out,err,sizeof(err)));
    assert(fabsf(out[0])>0.5f&&fabsf(out[1])>0.5f);   /* vh0 -> kh0 */
    assert(fabsf(out[2])<1e-6f&&fabsf(out[3])<1e-6f);/* vh1 -> kh1 */
    assert(fabsf(out[4])>0.5f&&fabsf(out[5])>0.5f);   /* vh2 -> kh0 */
    assert(fabsf(out[6])<1e-6f&&fabsf(out[7])<1e-6f);/* vh3 -> kh1 */
}

int main(void){
    test_separate_delta();
    test_tiled_value_head_mapping();
    char path[256];
#ifdef _WIN32
    snprintf(path,sizeof(path),"test_qwen35moe_%ld.gguf",(long)getpid());
#else
    snprintf(path,sizeof(path),"/tmp/test_qwen35moe_%ld.gguf",(long)getpid());
#endif
    write_fixture(path);
    Q35Model m;Q35Scratch s={0};char err[512];ColiExec exec={COLI_BACKEND_CPU,0};
    assert(q35_load_model(&m,path,8,0,2,exec,err,sizeof(err)));
    assert(m.n_layers_all==2&&m.n_layers==1&&m.n_mtp_layers==1);
    assert(m.n_recurrent==1&&m.n_attention==0&&m.n_attention_all==1);
    assert(m.layers[1].attention_index==0&&m.layers[1].has_nextn_head_norm);
    assert(q35_scratch_alloc(&m,&s,err,sizeof(err)));
    /* Prompt synchronization needs only MTP K/V.  It must not execute the
     * decoder MLP/head or overwrite the draft logits buffer. */
    s.mtp_logits[0]=123.f;
    assert(q35_mtp_update_kv(&m,&s,1,0,err,sizeof(err))); /* token 0 + zero pending h */
    assert(m.mtp_steps==0&&m.mtp_kv_updates==1&&s.mtp_logits[0]==123.f);
    assert(q35_forward(&m,&s,1,0,err,sizeof(err)));
    assert(q35_argmax(s.logits,m.vocab)==1);
    assert(q35_mtp_update_kv(&m,&s,2,1,err,sizeof(err))); /* token 1 + target h[0] */
    assert(m.mtp_steps==0&&m.mtp_kv_updates==2);
    assert(q35_forward(&m,&s,2,1,err,sizeof(err)));
    int draft=-1;assert(q35_mtp_make_drafts(&m,&s,2,2,1,&draft,err,sizeof(err))==1);assert(draft==2);
    assert(m.mtp_steps==1&&m.mtp_kv_updates==2);

    /* A two-position target verification block must be numerically identical
     * to restoring the recurrent checkpoint and decoding the same inputs one
     * by one. */
    { int vt[2]={2,2}; float blogits[8], slogits0[4], slogits1[4];
      assert(s.verify_cap>=2);
      assert(q35_checkpoint_target_state(&m,&s,err,sizeof(err)));
      assert(q35_forward_batch(&m,&s,vt,2,2,err,sizeof(err)));
      memcpy(blogits,s.vlogits,sizeof(blogits));
      { uint64_t before=m.mtp_kv_updates;
        assert(q35_mtp_catchup_batch(&m,&s,vt,2,2,err,sizeof(err)));
        assert(m.mtp_kv_updates==before+1); /* row zero was already correct */
      }
      assert(q35_restore_target_state(&m,&s,err,sizeof(err)));
      assert(q35_forward(&m,&s,vt[0],2,err,sizeof(err)));memcpy(slogits0,s.logits,sizeof(slogits0));
      assert(q35_forward(&m,&s,vt[1],3,err,sizeof(err)));memcpy(slogits1,s.logits,sizeof(slogits1));
      for(int i=0;i<4;i++){assert(fabsf(blogits[i]-slogits0[i])<1e-5f);assert(fabsf(blogits[4+i]-slogits1[i])<1e-5f);}
    }
    q35_scratch_free(&s);q35_model_free(&m);

    /* The appended block is still bound for compatibility, but an ordinary
     * non-MTP run must not reserve its KV row, dense residency, or scheduler
     * layer. */
    memset(&s,0,sizeof(s));
    assert(q35_load_model(&m,path,8,0,0,exec,err,sizeof(err)));
    assert(!m.mtp_enabled&&m.n_exec_layers==1&&m.n_attention_exec==0);
    assert(q35_scratch_alloc(&m,&s,err,sizeof(err)));
    assert(!s.mtp_hidden&&!s.mtp_x&&!s.mtp_cat&&!s.mtp_logits);
#ifdef COLI_CUDA
    assert(!s.dmtp_hidden&&!s.dmtp_x&&!s.dmtp_cat&&!s.dmtp_logits);
#endif
    assert(q35_forward(&m,&s,1,0,err,sizeof(err)));
    q35_scratch_free(&s);q35_model_free(&m);unlink(path);
    puts("test_gguf_qwen35moe: ok");return 0;
}
