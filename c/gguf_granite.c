#include "gguf_granite.h"
#include "gguf_f32_loader.h"
#include "gguf_tokenizer.h"
#include "f32_kernels.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif

typedef struct {
    ColiF32Tensor attn_norm, q, k, v, o;
    ColiF32Tensor ffn_norm, router, gate, up, down;
} GraniteLayer;

typedef struct {
    ColiGgufFile gguf;
    ColiGgufTokenizer *tokenizer;
    char *architecture;
    int n_layers, hidden, n_heads, n_kv_heads, head_dim, kv_dim;
    int n_experts, n_expert_used, expert_ff, vocab, rope_dims;
    float rope_base, eps, attention_scale, embedding_scale;
    float residual_scale, logit_scale, expert_weights_scale;
    int rope_enabled;
    ColiF32Tensor token_embd, output_norm, output;
    int tied_output;
    GraniteLayer *layers;
    int context_capacity;
    float *k_cache, *v_cache;
    int verbose;
} GraniteModel;

typedef struct {
    float *x, *norm, *q, *k, *v, *attn, *proj, *ffn_in;
    float *router, *gate, *up, *expert_out, *moe, *scores, *logits;
    int *top_idx;
    float *top_w;
} GraniteScratch;

static double now_sec(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int errf(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) { va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap); }
    return 0;
}

static int kv_u64(const ColiGgufFile *g, const char *key, uint64_t *out, int required,
                  char *err, size_t cap) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g, key);
    if (!kv) return required ? errf(err,cap,"missing metadata: %s",key) : 0;
    return coli_gguf_kv_read_u64(g,kv,out) ? 1 : errf(err,cap,"invalid integer metadata: %s",key);
}
static int kv_f32(const ColiGgufFile *g, const char *key, float *out, int required,
                  char *err, size_t cap) {
    const ColiGgufKV *kv = coli_gguf_find_kv(g,key); double v;
    if (!kv) return required ? errf(err,cap,"missing metadata: %s",key) : 0;
    if (!coli_gguf_kv_read_f64(g,kv,&v)) return errf(err,cap,"invalid float metadata: %s",key);
    *out=(float)v; return 1;
}
static int kv_bool_default(const ColiGgufFile *g, const char *key, int fallback) {
    const ColiGgufKV *kv=coli_gguf_find_kv(g,key); int v;
    return kv && coli_gguf_kv_read_bool(g,kv,&v) ? v : fallback;
}
static int to_int(uint64_t v, int *out, const char *name, char *err, size_t cap) {
    if (v==0 || v>INT_MAX) return errf(err,cap,"invalid %s: %llu",name,(unsigned long long)v);
    *out=(int)v; return 1;
}
static int read_count(const ColiGgufFile *g, const char *a, const char *b, int *out,
                      char *err, size_t cap) {
    uint64_t v;
    if (kv_u64(g,a,&v,0,err,cap) || (b && kv_u64(g,b,&v,0,err,cap))) return to_int(v,out,a,err,cap);
    return errf(err,cap,"missing metadata: %s",a);
}

static void tensor_free(ColiF32Tensor *t) { coli_f32_tensor_destroy(t); }
static void layer_free(GraniteLayer *l) {
    tensor_free(&l->attn_norm); tensor_free(&l->q); tensor_free(&l->k); tensor_free(&l->v); tensor_free(&l->o);
    tensor_free(&l->ffn_norm); tensor_free(&l->router); tensor_free(&l->gate); tensor_free(&l->up); tensor_free(&l->down);
}
static void model_free(GraniteModel *m) {
    if (!m) return;
    if (m->layers) for (int i=0;i<m->n_layers;i++) layer_free(&m->layers[i]);
    free(m->layers); tensor_free(&m->token_embd); tensor_free(&m->output_norm); tensor_free(&m->output);
    free(m->k_cache); free(m->v_cache); free(m->architecture);
    coli_gguf_tokenizer_destroy(m->tokenizer); coli_gguf_close(&m->gguf);
    memset(m,0,sizeof(*m));
}

static int tensor_dims(const ColiF32Tensor *t, int nd, const uint64_t *dims) {
    if (!t || (int)t->n_dims != nd) return 0;
    for (int i=0;i<nd;i++) if (t->dims[i] != dims[i]) return 0;
    return 1;
}
static int load_tensor(GraniteModel *m, const char *name, ColiF32Tensor *out,
                       int nd, const uint64_t *dims, char *err, size_t cap) {
    const ColiGgufTensorInfo *ti=coli_gguf_find_tensor(&m->gguf,name);
    if (!ti) return errf(err,cap,"missing tensor: %s",name);
    if (!coli_gguf_tensor_to_f32(&m->gguf,ti,out,err,(uint64_t)cap))
        return 0;
    if (!tensor_dims(out,nd,dims)) {
        tensor_free(out); return errf(err,cap,"wrong dimensions for tensor: %s",name);
    }
    return 1;
}
static int load_layer_tensor(GraniteModel *m, int layer, const char *suffix,
                             ColiF32Tensor *out, int nd, const uint64_t *dims,
                             char *err, size_t cap) {
    char name[128]; snprintf(name,sizeof(name),"blk.%d.%s",layer,suffix);
    return load_tensor(m,name,out,nd,dims,err,cap);
}

static int model_config(GraniteModel *m, char *err, size_t cap) {
    const ColiGgufKV *akv=coli_gguf_find_kv(&m->gguf,"general.architecture");
    if (!akv || !coli_gguf_kv_read_string(&m->gguf,akv,&m->architecture))
        return errf(err,cap,"general.architecture missing");
    if (strcmp(m->architecture,"granitemoe") != 0)
        return errf(err,cap,"unsupported GGUF architecture '%s' (expected granitemoe)",m->architecture);

    uint64_t u;
#define READI(key, field) do { if(!kv_u64(&m->gguf,key,&u,1,err,cap)||!to_int(u,&m->field,key,err,cap)) return 0; } while(0)
    READI("granitemoe.block_count",n_layers);
    READI("granitemoe.embedding_length",hidden);
    READI("granitemoe.attention.head_count",n_heads);
    READI("granitemoe.attention.head_count_kv",n_kv_heads);
    READI("granitemoe.expert_count",n_experts);
    READI("granitemoe.expert_used_count",n_expert_used);
    if (!read_count(&m->gguf,"granitemoe.expert_feed_forward_length","granitemoe.feed_forward_length",
                    &m->expert_ff,err,cap)) return 0;
    if (!kv_u64(&m->gguf,"granitemoe.rope.dimension_count",&u,0,err,cap)) u=(uint64_t)(m->hidden/m->n_heads);
    if (!to_int(u,&m->rope_dims,"rope dimensions",err,cap)) return 0;
#undef READI
    if (m->hidden % m->n_heads || m->n_heads % m->n_kv_heads)
        return errf(err,cap,"invalid GQA dimensions");
    m->head_dim=m->hidden/m->n_heads; m->kv_dim=m->n_kv_heads*m->head_dim;
    if (m->rope_dims>m->head_dim || (m->rope_dims&1)) return errf(err,cap,"invalid RoPE dimension count");
    m->vocab=0;
    const ColiGgufTensorInfo *emb=coli_gguf_find_tensor(&m->gguf,"token_embd.weight");
    if (!emb || emb->n_dims!=2 || emb->dims[0]!=(uint64_t)m->hidden || emb->dims[1]>INT_MAX)
        return errf(err,cap,"invalid token_embd.weight dimensions");
    m->vocab=(int)emb->dims[1];

    m->rope_base=10000.f; kv_f32(&m->gguf,"granitemoe.rope.freq_base",&m->rope_base,0,err,cap);
    m->eps=1e-6f; kv_f32(&m->gguf,"granitemoe.attention.layer_norm_rms_epsilon",&m->eps,0,err,cap);
    m->attention_scale=1.f/sqrtf((float)m->head_dim); kv_f32(&m->gguf,"granitemoe.attention.scale",&m->attention_scale,0,err,cap);
    m->embedding_scale=1.f; kv_f32(&m->gguf,"granitemoe.embedding_scale",&m->embedding_scale,0,err,cap);
    m->residual_scale=1.f; kv_f32(&m->gguf,"granitemoe.residual_scale",&m->residual_scale,0,err,cap);
    m->logit_scale=1.f; kv_f32(&m->gguf,"granitemoe.logit_scale",&m->logit_scale,0,err,cap);
    m->expert_weights_scale=1.f; kv_f32(&m->gguf,"granitemoe.expert_weights_scale",&m->expert_weights_scale,0,err,cap);
    m->rope_enabled=kv_bool_default(&m->gguf,"granitemoe.rope.scaling.finetuned",1);
    if (m->logit_scale==0.f) return errf(err,cap,"granitemoe.logit_scale cannot be zero");
    if (m->n_expert_used>m->n_experts) return errf(err,cap,"expert_used_count exceeds expert_count");
    return 1;
}

static int model_load_weights(GraniteModel *m, char *err, size_t cap) {
    uint64_t d1[3];
    d1[0]=m->hidden; d1[1]=m->vocab;
    if (!load_tensor(m,"token_embd.weight",&m->token_embd,2,d1,err,cap)) return 0;
    d1[0]=m->hidden;
    if (!load_tensor(m,"output_norm.weight",&m->output_norm,1,d1,err,cap)) return 0;
    const ColiGgufTensorInfo *out=coli_gguf_find_tensor(&m->gguf,"output.weight");
    if (out) {
        d1[0]=m->hidden; d1[1]=m->vocab;
        if (!load_tensor(m,"output.weight",&m->output,2,d1,err,cap)) return 0;
    } else m->tied_output=1;

    m->layers=(GraniteLayer*)calloc((size_t)m->n_layers,sizeof(*m->layers));
    if (!m->layers) return errf(err,cap,"out of memory allocating layers");
    for(int l=0;l<m->n_layers;l++){
        GraniteLayer *x=&m->layers[l];
        d1[0]=m->hidden;
        if(!load_layer_tensor(m,l,"attn_norm.weight",&x->attn_norm,1,d1,err,cap))return 0;
        d1[0]=m->hidden;d1[1]=m->hidden;
        if(!load_layer_tensor(m,l,"attn_q.weight",&x->q,2,d1,err,cap))return 0;
        d1[0]=m->hidden;d1[1]=m->kv_dim;
        if(!load_layer_tensor(m,l,"attn_k.weight",&x->k,2,d1,err,cap))return 0;
        if(!load_layer_tensor(m,l,"attn_v.weight",&x->v,2,d1,err,cap))return 0;
        d1[0]=m->hidden;d1[1]=m->hidden;
        if(!load_layer_tensor(m,l,"attn_output.weight",&x->o,2,d1,err,cap))return 0;
        d1[0]=m->hidden;
        if(!load_layer_tensor(m,l,"ffn_norm.weight",&x->ffn_norm,1,d1,err,cap))return 0;
        d1[0]=m->hidden;d1[1]=m->n_experts;
        if(!load_layer_tensor(m,l,"ffn_gate_inp.weight",&x->router,2,d1,err,cap))return 0;
        d1[0]=m->hidden;d1[1]=m->expert_ff;d1[2]=m->n_experts;
        if(!load_layer_tensor(m,l,"ffn_gate_exps.weight",&x->gate,3,d1,err,cap))return 0;
        if(!load_layer_tensor(m,l,"ffn_up_exps.weight",&x->up,3,d1,err,cap))return 0;
        d1[0]=m->expert_ff;d1[1]=m->hidden;d1[2]=m->n_experts;
        if(!load_layer_tensor(m,l,"ffn_down_exps.weight",&x->down,3,d1,err,cap))return 0;
        if(m->verbose)fprintf(stderr,"[GGUF] loaded layer %d/%d\r",l+1,m->n_layers);
    }
    if(m->verbose)fputc('\n',stderr);
    return 1;
}

static int alloc_cache(GraniteModel *m, int context, char *err, size_t cap) {
    if(context<=0)return errf(err,cap,"invalid context capacity");
    size_t elems=(size_t)m->n_layers;
    if(elems>SIZE_MAX/(size_t)context)return errf(err,cap,"KV cache overflow");
    elems*=context;
    if(elems>SIZE_MAX/(size_t)m->kv_dim)return errf(err,cap,"KV cache overflow");
    elems*=m->kv_dim;
    if(elems>SIZE_MAX/sizeof(float))return errf(err,cap,"KV cache overflow");
    m->k_cache=(float*)calloc(elems,sizeof(float));m->v_cache=(float*)calloc(elems,sizeof(float));
    if(!m->k_cache||!m->v_cache)return errf(err,cap,"out of memory allocating %.2f MiB KV cache",2.0*elems*sizeof(float)/(1024.0*1024.0));
    m->context_capacity=context;return 1;
}

static void scratch_free(GraniteScratch *s){
    free(s->x);free(s->norm);free(s->q);free(s->k);free(s->v);free(s->attn);free(s->proj);free(s->ffn_in);
    free(s->router);free(s->gate);free(s->up);free(s->expert_out);free(s->moe);free(s->scores);free(s->logits);free(s->top_idx);free(s->top_w);
    memset(s,0,sizeof(*s));
}
static int scratch_alloc(const GraniteModel*m,GraniteScratch*s,char*err,size_t cap){
#define ALLOC(field,n,type) do{s->field=(type*)calloc((size_t)(n),sizeof(type));if(!s->field){scratch_free(s);return errf(err,cap,"out of memory allocating inference scratch");}}while(0)
    ALLOC(x,m->hidden,float);ALLOC(norm,m->hidden,float);ALLOC(q,m->hidden,float);ALLOC(k,m->kv_dim,float);ALLOC(v,m->kv_dim,float);
    ALLOC(attn,m->hidden,float);ALLOC(proj,m->hidden,float);ALLOC(ffn_in,m->hidden,float);ALLOC(router,m->n_experts,float);
    ALLOC(gate,m->expert_ff,float);ALLOC(up,m->expert_ff,float);ALLOC(expert_out,m->hidden,float);ALLOC(moe,m->hidden,float);
    ALLOC(scores,m->context_capacity,float);ALLOC(logits,m->vocab,float);ALLOC(top_idx,m->n_expert_used,int);ALLOC(top_w,m->n_expert_used,float);
#undef ALLOC
    return 1;
}

static int model_forward(GraniteModel*m,GraniteScratch*s,int token,int pos,char*err,size_t cap){
    if(token<0||token>=m->vocab)return errf(err,cap,"token id %d outside vocabulary",token);
    if(pos<0||pos>=m->context_capacity)return errf(err,cap,"position %d outside context",pos);
    const float *emb=m->token_embd.data+(int64_t)token*m->hidden;
    for(int i=0;i<m->hidden;i++)s->x[i]=emb[i]*m->embedding_scale;
    for(int l=0;l<m->n_layers;l++){
        GraniteLayer*L=&m->layers[l];
        coli_f32_rmsnorm(s->norm,s->x,L->attn_norm.data,m->hidden,m->eps);
        coli_f32_matmul(s->q,s->norm,L->q.data,1,m->hidden,m->hidden);
        coli_f32_matmul(s->k,s->norm,L->k.data,1,m->hidden,m->kv_dim);
        coli_f32_matmul(s->v,s->norm,L->v.data,1,m->hidden,m->kv_dim);
        if(m->rope_enabled){
            coli_f32_rope(s->q,m->n_heads,m->head_dim,m->rope_dims,pos,m->rope_base);
            coli_f32_rope(s->k,m->n_kv_heads,m->head_dim,m->rope_dims,pos,m->rope_base);
        }
        const size_t layer_stride=(size_t)m->context_capacity*m->kv_dim;
        float*kc=m->k_cache+(size_t)l*layer_stride;float*vc=m->v_cache+(size_t)l*layer_stride;
        memcpy(kc+(size_t)pos*m->kv_dim,s->k,(size_t)m->kv_dim*sizeof(float));
        memcpy(vc+(size_t)pos*m->kv_dim,s->v,(size_t)m->kv_dim*sizeof(float));
        coli_f32_gqa_attention(s->attn,s->q,kc,vc,pos,m->n_heads,m->n_kv_heads,m->head_dim,m->attention_scale,s->scores);
        coli_f32_matmul(s->proj,s->attn,L->o.data,1,m->hidden,m->hidden);
        for(int i=0;i<m->hidden;i++)s->ffn_in[i]=s->x[i]+m->residual_scale*s->proj[i];
        coli_f32_rmsnorm(s->norm,s->ffn_in,L->ffn_norm.data,m->hidden,m->eps);
        coli_f32_matmul(s->router,s->norm,L->router.data,1,m->hidden,m->n_experts);
        const int nk=coli_f32_router_topk(s->router,m->n_experts,m->n_expert_used,s->top_idx,s->top_w);
        memset(s->moe,0,(size_t)m->hidden*sizeof(float));
        const int64_t gu_stride=(int64_t)m->hidden*m->expert_ff;
        const int64_t down_stride=(int64_t)m->expert_ff*m->hidden;
        for(int j=0;j<nk;j++){
            const int e=s->top_idx[j];const float rw=s->top_w[j]*m->expert_weights_scale;
            coli_f32_matmul(s->gate,s->norm,L->gate.data+(int64_t)e*gu_stride,1,m->hidden,m->expert_ff);
            coli_f32_matmul(s->up,s->norm,L->up.data+(int64_t)e*gu_stride,1,m->hidden,m->expert_ff);
            for(int i=0;i<m->expert_ff;i++)s->gate[i]=coli_f32_silu(s->gate[i])*s->up[i];
            coli_f32_matmul(s->expert_out,s->gate,L->down.data+(int64_t)e*down_stride,1,m->expert_ff,m->hidden);
            for(int i=0;i<m->hidden;i++)s->moe[i]+=rw*s->expert_out[i];
        }
        for(int i=0;i<m->hidden;i++)s->x[i]=s->ffn_in[i]+m->residual_scale*s->moe[i];
    }
    coli_f32_rmsnorm(s->norm,s->x,m->output_norm.data,m->hidden,m->eps);
    const float*outw=m->tied_output?m->token_embd.data:m->output.data;
    coli_f32_matmul(s->logits,s->norm,outw,1,m->hidden,m->vocab);
    const float inv=1.f/m->logit_scale;for(int i=0;i<m->vocab;i++)s->logits[i]*=inv;
    return 1;
}
static int argmax(const float*x,int n){int b=0;for(int i=1;i<n;i++)if(x[i]>x[b])b=i;return b;}

static int load_model(GraniteModel*m,const char*path,int context,int verbose,char*err,size_t cap){
    memset(m,0,sizeof(*m));m->gguf.fd=-1;m->verbose=verbose;
    if(!coli_gguf_open(&m->gguf,path))return errf(err,cap,"cannot open GGUF: %s",coli_gguf_error(&m->gguf));
    if(!model_config(m,err,cap))return 0;
    if(!coli_gguf_tokenizer_load(&m->tokenizer,&m->gguf,err,cap))return 0;
    if(coli_gguf_tokenizer_vocab_size(m->tokenizer)!=m->vocab)return errf(err,cap,"tokenizer/model vocabulary mismatch");
    const double t0=now_sec();if(!model_load_weights(m,err,cap))return 0;
    if(!alloc_cache(m,context,err,cap))return 0;
    if(verbose)fprintf(stderr,"[GGUF] granitemoe: layers=%d hidden=%d heads=%d/%d experts=%d top=%d vocab=%d\n[GGUF] F32 weights loaded in %.2fs; context=%d; threads=%d\n",
        m->n_layers,m->hidden,m->n_heads,m->n_kv_heads,m->n_experts,m->n_expert_used,m->vocab,now_sec()-t0,context,omp_get_max_threads());
    return 1;
}

static void usage(const char*prog){
    fprintf(stderr,"Usage: %s [--gguf] MODEL.gguf --prompt TEXT [--max-tokens N] [--raw-prompt] [--verbose]\n",prog);
}
int coli_gguf_cli_requested(int argc,char**argv){
    if(argc>1&&!strcmp(argv[1],"--gguf"))return 1;
    if(argc>1){size_t n=strlen(argv[1]);if(n>=5&&!strcmp(argv[1]+n-5,".gguf"))return 1;}
    return 0;
}
int coli_gguf_run_cli(int argc,char**argv){
    const char*model_path=NULL,*prompt=NULL;int max_tokens=24,raw=0,verbose=0;
    int i=1;if(i<argc&&!strcmp(argv[i],"--gguf"))i++;
    if(i<argc)model_path=argv[i++];
    for(;i<argc;i++){
        if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"--max-tokens")&&i+1<argc)max_tokens=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--raw-prompt"))raw=1;
        else if(!strcmp(argv[i],"--verbose")||!strcmp(argv[i],"-v"))verbose=1;
        else {fprintf(stderr,"unknown GGUF option: %s\n",argv[i]);usage(argv[0]);return 2;}
    }
    if(!model_path||!prompt||max_tokens<0){usage(argv[0]);return 2;}
    /* Open metadata/tokenizer first so prompt formatting follows the template embedded in this GGUF. */
    ColiGgufFile probe;probe.fd=-1;ColiGgufTokenizer*pt=NULL;char err[512];
    if(!coli_gguf_open(&probe,model_path)||!coli_gguf_tokenizer_load(&pt,&probe,err,sizeof(err))){
        fprintf(stderr,"%s\n",probe.fd>=0?err:coli_gguf_error(&probe));coli_gguf_close(&probe);return 1;}
    char*formatted=raw?strdup(prompt):coli_gguf_tokenizer_format_granite_prompt(pt,prompt);
    if(!formatted){fprintf(stderr,"cannot format prompt\n");coli_gguf_tokenizer_destroy(pt);coli_gguf_close(&probe);return 1;}
    /* Conservative capacity: byte-level BPE cannot emit more tokens than input bytes plus BOS. */
    size_t max_prompt=strlen(formatted)+2;
    if(max_prompt>INT_MAX){free(formatted);coli_gguf_tokenizer_destroy(pt);coli_gguf_close(&probe);fprintf(stderr,"prompt too long\n");return 1;}
    int*ids=(int*)malloc(max_prompt*sizeof(int));
    if(!ids){free(formatted);coli_gguf_tokenizer_destroy(pt);coli_gguf_close(&probe);fprintf(stderr,"out of memory\n");return 1;}
    int n_prompt=coli_gguf_tokenizer_encode(pt,formatted,ids,(int)max_prompt);
    coli_gguf_tokenizer_destroy(pt);coli_gguf_close(&probe);free(formatted);
    if(n_prompt<=0){fprintf(stderr,"prompt tokenization failed\n");free(ids);return 1;}
    if(max_tokens>INT_MAX-n_prompt){fprintf(stderr,"context overflow\n");free(ids);return 1;}

    GraniteModel m;GraniteScratch s={0};
    if(!load_model(&m,model_path,n_prompt+max_tokens,verbose,err,sizeof(err))){fprintf(stderr,"%s\n",err);model_free(&m);free(ids);return 1;}
    if(!scratch_alloc(&m,&s,err,sizeof(err))){fprintf(stderr,"%s\n",err);model_free(&m);free(ids);return 1;}
    const double t0=now_sec();
    for(int p=0;p<n_prompt;p++)if(!model_forward(&m,&s,ids[p],p,err,sizeof(err))){fprintf(stderr,"%s\n",err);goto fail;}
    int pos=n_prompt,generated=0;
    while(generated<max_tokens){
        const int next=argmax(s.logits,m.vocab);
        if(next==coli_gguf_tokenizer_eos(m.tokenizer))break;
        if(!coli_gguf_tokenizer_is_control(m.tokenizer,next)){
            char piece[4096];int n=coli_gguf_tokenizer_decode(m.tokenizer,&next,1,piece,sizeof(piece));
            if(n>0){fwrite(piece,1,(size_t)n,stdout);fflush(stdout);}
        }
        generated++;
        if(generated>=max_tokens)break;
        if(!model_forward(&m,&s,next,pos++,err,sizeof(err))){fprintf(stderr,"\n%s\n",err);goto fail;}
    }
    fputc('\n',stdout);
    if(verbose)fprintf(stderr,"[GGUF] prompt=%d generated=%d elapsed=%.2fs (%.3f tok/s decode+prefill)\n",n_prompt,generated,now_sec()-t0,
        (n_prompt+generated)/(now_sec()-t0));
    scratch_free(&s);model_free(&m);free(ids);return 0;
fail:
    scratch_free(&s);model_free(&m);free(ids);return 1;
}
