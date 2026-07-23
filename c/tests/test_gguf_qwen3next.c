#include "../gguf_qwen3next.c"

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

static void put_u32(FILE *f,uint32_t v){unsigned char p[4];for(int i=0;i<4;i++)p[i]=(unsigned char)(v>>(8*i));assert(fwrite(p,1,4,f)==4);}
static void put_u64(FILE *f,uint64_t v){unsigned char p[8];for(int i=0;i<8;i++)p[i]=(unsigned char)(v>>(8*i));assert(fwrite(p,1,8,f)==8);}
static void put_f32(FILE *f,float v){uint32_t u;memcpy(&u,&v,4);put_u32(f,u);}
static void put_str(FILE *f,const char*s){put_u64(f,strlen(s));assert(fwrite(s,1,strlen(s),f)==strlen(s));}
static void kv_str(FILE*f,const char*k,const char*v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_STRING);put_str(f,v);}
static void kv_u32(FILE*f,const char*k,uint32_t v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_UINT32);put_u32(f,v);}
static void kv_f32(FILE*f,const char*k,float v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_FLOAT32);put_f32(f,v);}
static void kv_bool(FILE*f,const char*k,int v){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_BOOL);fputc(v?1:0,f);}
static void kv_str_array(FILE*f,const char*k,const char **v,int n){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_ARRAY);put_u32(f,COLI_GGUF_TYPE_STRING);put_u64(f,n);for(int i=0;i<n;i++)put_str(f,v[i]);}
static void kv_u32_array(FILE*f,const char*k,const uint32_t*v,int n){put_str(f,k);put_u32(f,COLI_GGUF_TYPE_ARRAY);put_u32(f,COLI_GGUF_TYPE_UINT32);put_u64(f,n);for(int i=0;i<n;i++)put_u32(f,v[i]);}
static uint64_t align32(uint64_t x){return (x+31u)&~31u;}

typedef struct{const char*name;int nd;uint64_t d[3];uint64_t off;float *data;size_t count;} TD;
static void tensor_desc(FILE*f,const TD*t){put_str(f,t->name);put_u32(f,t->nd);for(int i=0;i<t->nd;i++)put_u64(f,t->d[i]);put_u32(f,0);put_u64(f,t->off);}
static void pad32(FILE*f){while(ftell(f)%32)fputc(0,f);}
static void pad_to(FILE*f,long base,uint64_t off){while((uint64_t)(ftell(f)-base)<off)fputc(0,f);assert((uint64_t)(ftell(f)-base)==off);}

static void write_fixture(const char *path){
    float emb[16]={0},out[16]={0},ones4[4]={1,1,1,1},ones2[2]={1,1};
    for(int i=0;i<4;i++){emb[i*4+i]=1.f;out[i*4+i]=1.f;}
    float z24[24]={0},z12[12]={0},z8[8]={0},z4[4]={0},z1[1]={0},a1[1]={-1.f};
    TD t[]={
      {"token_embd.weight",2,{4,4},0,emb,16},{"output_norm.weight",1,{4},0,ones4,4},{"output.weight",2,{4,4},0,out,16},
      {"blk.0.attn_norm.weight",1,{4},0,ones4,4},{"blk.0.post_attention_norm.weight",1,{4},0,ones4,4},
      {"blk.0.attn_qkv.weight",2,{4,6},0,z24,24},{"blk.0.attn_gate.weight",2,{4,2},0,z8,8},
      {"blk.0.ssm_ba.weight",2,{4,2},0,z8,8},{"blk.0.ssm_conv1d.weight",2,{2,6},0,z12,12},
      {"blk.0.ssm_dt.bias",1,{1},0,z1,1},{"blk.0.ssm_a",1,{1},0,a1,1},
      {"blk.0.ssm_norm.weight",1,{2},0,ones2,2},{"blk.0.ssm_out.weight",2,{2,4},0,z8,8},
      {"blk.0.ffn_gate_inp.weight",2,{4,1},0,z4,4},{"blk.0.ffn_gate_exps.weight",3,{4,2,1},0,z8,8},
      {"blk.0.ffn_up_exps.weight",3,{4,2,1},0,z8,8},{"blk.0.ffn_down_exps.weight",3,{2,4,1},0,z8,8},
      {"blk.0.ffn_gate_inp_shexp.weight",1,{4},0,z4,4},{"blk.0.ffn_gate_shexp.weight",2,{4,2},0,z8,8},
      {"blk.0.ffn_up_shexp.weight",2,{4,2},0,z8,8},{"blk.0.ffn_down_shexp.weight",2,{2,4},0,z8,8},
    };
    const int nt=(int)(sizeof(t)/sizeof(t[0]));uint64_t off=0;
    for(int i=0;i<nt;i++){t[i].off=off;off=align32(off+t[i].count*4);}
    FILE*f=fopen(path,"wb");assert(f);fwrite("GGUF",1,4,f);put_u32(f,3);put_u64(f,nt);put_u64(f,28);
    kv_str(f,"general.architecture","qwen3next");kv_u32(f,"qwen3next.block_count",1);kv_u32(f,"qwen3next.context_length",16);
    kv_u32(f,"qwen3next.embedding_length",4);kv_u32(f,"qwen3next.attention.head_count",2);kv_u32(f,"qwen3next.attention.head_count_kv",1);
    kv_u32(f,"qwen3next.attention.key_length",4);kv_u32(f,"qwen3next.attention.value_length",4);
    kv_u32(f,"qwen3next.expert_count",1);kv_u32(f,"qwen3next.expert_used_count",1);kv_u32(f,"qwen3next.expert_feed_forward_length",2);
    kv_u32(f,"qwen3next.expert_shared_feed_forward_length",2);kv_u32(f,"qwen3next.ssm.conv_kernel",2);kv_u32(f,"qwen3next.ssm.state_size",2);
    kv_u32(f,"qwen3next.ssm.group_count",1);kv_u32(f,"qwen3next.ssm.time_step_rank",1);kv_u32(f,"qwen3next.ssm.inner_size",2);
    kv_u32(f,"qwen3next.full_attention_interval",4);kv_u32(f,"qwen3next.rope.dimension_count",2);kv_f32(f,"qwen3next.rope.freq_base",10000.f);
    kv_f32(f,"qwen3next.attention.layer_norm_rms_epsilon",1e-6f);kv_str(f,"tokenizer.ggml.pre","refact");
    const char*toks[]={"<eos>","A","B","C"};const uint32_t types[]={3,1,1,1};
    kv_str_array(f,"tokenizer.ggml.tokens",toks,4);kv_u32_array(f,"tokenizer.ggml.token_type",types,4);kv_str_array(f,"tokenizer.ggml.merges",NULL,0);
    kv_u32(f,"tokenizer.ggml.eos_token_id",0);kv_u32(f,"tokenizer.ggml.bos_token_id",0);kv_bool(f,"tokenizer.ggml.add_bos_token",0);
    for(int i=0;i<nt;i++)tensor_desc(f,&t[i]);pad32(f);long base=ftell(f);
    for(int i=0;i<nt;i++){pad_to(f,base,t[i].off);for(size_t j=0;j<t[i].count;j++)put_f32(f,t[i].data[j]);}
    assert(fclose(f)==0);
}

static void write_attention_fixture(const char *path){
    float emb[4]={1,1,0,0},out[4]={1,0,0,1},ones2[2]={1,1};
    /* Per-head q_proj output rows are [Q0,Q1,G0,G1]. For input [1,1],
     * gates become [-20,+20], so only the second attention channel survives. */
    float qg[8]={1,0,0,1,-20,0,20,0},eye[4]={1,0,0,1};
    float z2[2]={0},z1[1]={0};
    TD t[]={
      {"token_embd.weight",2,{2,2},0,emb,4},{"output_norm.weight",1,{2},0,ones2,2},{"output.weight",2,{2,2},0,out,4},
      {"blk.0.attn_norm.weight",1,{2},0,ones2,2},{"blk.0.post_attention_norm.weight",1,{2},0,ones2,2},
      {"blk.0.attn_q.weight",2,{2,4},0,qg,8},{"blk.0.attn_k.weight",2,{2,2},0,eye,4},
      {"blk.0.attn_v.weight",2,{2,2},0,eye,4},{"blk.0.attn_q_norm.weight",1,{2},0,ones2,2},
      {"blk.0.attn_k_norm.weight",1,{2},0,ones2,2},{"blk.0.attn_output.weight",2,{2,2},0,eye,4},
      {"blk.0.ffn_gate_inp.weight",2,{2,1},0,z2,2},{"blk.0.ffn_gate_exps.weight",3,{2,1,1},0,z2,2},
      {"blk.0.ffn_up_exps.weight",3,{2,1,1},0,z2,2},{"blk.0.ffn_down_exps.weight",3,{1,2,1},0,z2,2},
      {"blk.0.ffn_gate_inp_shexp.weight",1,{2},0,z2,2},{"blk.0.ffn_gate_shexp.weight",2,{2,1},0,z2,2},
      {"blk.0.ffn_up_shexp.weight",2,{2,1},0,z2,2},{"blk.0.ffn_down_shexp.weight",2,{1,2},0,z2,2},
    };
    const int nt=(int)(sizeof(t)/sizeof(t[0]));uint64_t off=0;for(int i=0;i<nt;i++){t[i].off=off;off=align32(off+t[i].count*4);}
    FILE*f=fopen(path,"wb");assert(f);fwrite("GGUF",1,4,f);put_u32(f,3);put_u64(f,nt);put_u64(f,28);
    kv_str(f,"general.architecture","qwen3next");kv_u32(f,"qwen3next.block_count",1);kv_u32(f,"qwen3next.context_length",8);
    kv_u32(f,"qwen3next.embedding_length",2);kv_u32(f,"qwen3next.attention.head_count",1);kv_u32(f,"qwen3next.attention.head_count_kv",1);
    kv_u32(f,"qwen3next.attention.key_length",2);kv_u32(f,"qwen3next.attention.value_length",2);
    kv_u32(f,"qwen3next.expert_count",1);kv_u32(f,"qwen3next.expert_used_count",1);kv_u32(f,"qwen3next.expert_feed_forward_length",1);
    kv_u32(f,"qwen3next.expert_shared_feed_forward_length",1);kv_u32(f,"qwen3next.ssm.conv_kernel",2);kv_u32(f,"qwen3next.ssm.state_size",1);
    kv_u32(f,"qwen3next.ssm.group_count",1);kv_u32(f,"qwen3next.ssm.time_step_rank",1);kv_u32(f,"qwen3next.ssm.inner_size",1);
    kv_u32(f,"qwen3next.full_attention_interval",1);kv_u32(f,"qwen3next.rope.dimension_count",2);kv_f32(f,"qwen3next.rope.freq_base",10000.f);
    kv_f32(f,"qwen3next.attention.layer_norm_rms_epsilon",1e-6f);kv_str(f,"tokenizer.ggml.pre","refact");
    const char*toks[]={"<eos>","A"};const uint32_t types[]={3,1};kv_str_array(f,"tokenizer.ggml.tokens",toks,2);
    kv_u32_array(f,"tokenizer.ggml.token_type",types,2);kv_str_array(f,"tokenizer.ggml.merges",NULL,0);
    kv_u32(f,"tokenizer.ggml.eos_token_id",0);kv_u32(f,"tokenizer.ggml.bos_token_id",0);kv_bool(f,"tokenizer.ggml.add_bos_token",0);
    for(int i=0;i<nt;i++)tensor_desc(f,&t[i]);pad32(f);long base=ftell(f);
    for(int i=0;i<nt;i++){pad_to(f,base,t[i].off);for(size_t j=0;j<t[i].count;j++)put_f32(f,t[i].data[j]);}
    assert(fclose(f)==0);(void)z1;
}

static void test_neox(void){float x[4]={1,2,3,4};qwen_rope_neox(x,1,4,4,0,10000.f);assert(x[0]==1&&x[1]==2&&x[2]==3&&x[3]==4);
    float y[4]={1,2,3,4};qwen_rope_neox(y,1,4,2,1,10000.f);assert(y[2]==3&&y[3]==4);}

static void test_qg_layout(void){
    const float raw[12]={1,2,3,10,20,30,4,5,6,40,50,60};
    float q[6]={0},g[6]={0};
    const float qw[6]={1,2,3,4,5,6},gw[6]={10,20,30,40,50,60};
    qwen_split_q_gate(q,g,raw,2,3);
    for(int i=0;i<6;i++){assert(q[i]==qw[i]);assert(g[i]==gw[i]);}
}

static void test_expert_511_view(void){
    float data[2*3*512];for(int i=0;i<(int)(sizeof(data)/sizeof(data[0]));i++)data[i]=(float)i;
    ColiTensor parent={0},view={0};parent.dtype=COLI_DTYPE_F32;parent.n_dims=3;
    parent.dims[0]=2;parent.dims[1]=3;parent.dims[2]=512;parent.element_count=2*3*512;
    parent.row_count=3*512;parent.row_bytes=2*sizeof(float);parent.storage_bytes=sizeof(data);parent.data=(const uint8_t*)data;
    assert(coli_tensor_rows_view(&parent,511u*3u,3,&view));
    assert(view.data==(const uint8_t*)&data[511*3*2]);assert(view.row_count==3&&view.dims[0]==2);
    float row[2];assert(coli_tensor_read_row_f32(&view,2,row,2));
    assert(row[0]==data[(511*3+2)*2]&&row[1]==data[(511*3+2)*2+1]);
    coli_tensor_destroy(&view);
}

static void bind_vec(ColiTensor*t,float*p,int n){memset(t,0,sizeof(*t));t->dtype=COLI_DTYPE_F32;t->n_dims=1;t->dims[0]=n;t->row_count=1;t->row_bytes=n*4;t->storage_bytes=n*4;t->data=(const uint8_t*)p;}
static void test_delta(void){
    QwenModel m={0};m.n_key_heads=1;m.n_value_heads=1;m.state_size=2;m.value_head_dim=2;m.key_dim=2;m.value_dim=2;m.eps=1e-6f;
    QwenLayer l={0};float dt[1]={0},a[1]={-1},nw[2]={1,1};bind_vec(&l.dt_bias,dt,1);bind_vec(&l.a,a,1);bind_vec(&l.ssm_norm,nw,2);
    float conv[6]={1,0,1,0,2,3},z[2]={1,1},ba[2]={0,0},state[4]={0},out[2];char err[128];
    assert(qwen_delta_decode_cpu(&m,&l,conv,z,ba,state,out,err,sizeof(err)));
    assert(fabsf(state[0]-1.f)<1e-5f&&fabsf(state[2]-1.5f)<1e-5f);
    assert(fabsf(out[0]-0.5734892f)<1e-5f&&fabsf(out[1]-0.8602338f)<1e-5f);
}

int main(void){
    test_neox();test_qg_layout();test_expert_511_view();test_delta();
    char *qp=coli_gguf_tokenizer_format_qwen3next_prompt(NULL,"hello");
    assert(qp&&strcmp(qp,"<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n")==0);free(qp);
    char path[256];
#ifdef _WIN32
    snprintf(path,sizeof(path),"test_qwen3next_%ld.gguf",(long)getpid());
#else
    snprintf(path,sizeof(path),"/tmp/test_qwen3next_%ld.gguf",(long)getpid());
#endif
    write_fixture(path);QwenModel m;QwenScratch s={0};char err[512];ColiExec exec={COLI_BACKEND_CPU,0};
    assert(qwen_load_model(&m,path,2,0,exec,err,sizeof(err)));assert(m.head_dim==4);
    assert(qwen_scratch_alloc(&m,&s,err,sizeof(err)));
    assert(qwen_forward(&m,&s,1,0,err,sizeof(err)));assert(qwen_argmax(s.logits,m.vocab)==1);
    qwen_scratch_free(&s);qwen_model_free(&m);unlink(path);

#ifdef _WIN32
    snprintf(path,sizeof(path),"test_qwen3next_attn_%ld.gguf",(long)getpid());
#else
    snprintf(path,sizeof(path),"/tmp/test_qwen3next_attn_%ld.gguf",(long)getpid());
#endif
    write_attention_fixture(path);memset(&s,0,sizeof(s));
    assert(qwen_load_model(&m,path,2,0,exec,err,sizeof(err)));assert(m.n_attention==1&&m.head_dim==2);
    assert(qwen_scratch_alloc(&m,&s,err,sizeof(err)));assert(qwen_forward(&m,&s,0,0,err,sizeof(err)));
    /* Correct [Q,gate]-per-head split and post-attention sigmoid gate make
     * channel 1 larger. Treating q_proj as [all Q][all gate] fails here. */
    assert(qwen_argmax(s.logits,m.vocab)==1);
    qwen_scratch_free(&s);qwen_model_free(&m);unlink(path);
    puts("test_gguf_qwen3next: ok");return 0;
}
