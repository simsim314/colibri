#include "gguf_granite.h"
#include "gguf_qwen3next.h"
#include "tensor.h"
#include "gguf_tokenizer.h"
#include "f32_kernels.h"
#include "expert_scheduler.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

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
    int eid;
    ColiTensor *gate, *up, *down;
    uint64_t used;
} GraniteExpertSlot;

static const ColiExpertSlotLayout g_granite_slot_layout = {
    sizeof(GraniteExpertSlot), offsetof(GraniteExpertSlot, eid), offsetof(GraniteExpertSlot, used)
};

typedef struct {
    ColiTensor attn_norm, q, k, v, o;
    ColiTensor ffn_norm, router, gate, up, down;
    ColiTensor *gate_expert, *up_expert, *down_expert;
    int expert_views;
    GraniteExpertSlot *pin, *cache;
    int npin, ncache, cache_cap;
    uint32_t *heat, *last, *usage;
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
    ColiTensor token_embd, output_norm, output;
    int tied_output;
    GraniteLayer *layers;
    int context_capacity;
    float *k_cache, *v_cache;
    float *k_cache_dev, *v_cache_dev;
    size_t cuda_weight_bytes, cuda_dense_bytes, cuda_expert_bytes;
    size_t expert_bytes;
    uint64_t expert_clock;
    uint32_t expert_access_clock;
    ColiExpertSchedulerStats scheduler_stats;
    int scheduler_evict_guard;
    int repin_interval, tokens_since_repin;
    int pilot, pilot_real, pilot_k;
    int couple, couple_k, couple_d;
    ColiExpertCoupling coupling;
    char usage_path[2048];
    int64_t usage_history;
    int verbose;
    ColiExec exec;
} GraniteModel;

typedef struct {
    float *x, *norm, *q, *k, *v, *attn, *proj, *ffn_in;
    float *router, *gate, *up, *expert_out, *moe, *scores, *logits, *weight;
    int *top_idx;
    float *top_w;
    float *dx, *dnorm, *dq, *dk, *dv, *dattn, *dproj, *dffn_in;
    float *drouter, *dgate, *dup, *dexpert_out, *dmoe, *dscores, *dlogits;
    int cuda_device, cuda_allocated;
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

static void tensor_free(ColiTensor *t) { coli_tensor_destroy(t); }
static void layer_free(GraniteLayer *l) {
    if (!l) return;
    for (int e=0;e<l->expert_views;e++) {
        tensor_free(&l->gate_expert[e]); tensor_free(&l->up_expert[e]); tensor_free(&l->down_expert[e]);
    }
    free(l->gate_expert); free(l->up_expert); free(l->down_expert);
    free(l->pin); free(l->cache); free(l->heat); free(l->last); free(l->usage);
    tensor_free(&l->attn_norm); tensor_free(&l->q); tensor_free(&l->k); tensor_free(&l->v); tensor_free(&l->o);
    tensor_free(&l->ffn_norm); tensor_free(&l->router); tensor_free(&l->gate); tensor_free(&l->up); tensor_free(&l->down);
}
static void model_free(GraniteModel *m) {
    if (!m) return;
    if (m->layers) for (int i=0;i<m->n_layers;i++) layer_free(&m->layers[i]);
    free(m->layers); tensor_free(&m->token_embd); tensor_free(&m->output_norm); tensor_free(&m->output);
#ifdef COLI_CUDA
    if (m->k_cache_dev) coli_cuda_pipe_free(m->exec.device,m->k_cache_dev);
    if (m->v_cache_dev) coli_cuda_pipe_free(m->exec.device,m->v_cache_dev);
#endif
    free(m->k_cache); free(m->v_cache); free(m->architecture);
    coli_expert_coupling_destroy(&m->coupling);
    coli_gguf_tokenizer_destroy(m->tokenizer); coli_gguf_close(&m->gguf);
    memset(m,0,sizeof(*m));
}

static int tensor_dims(const ColiTensor *t, int nd, const uint64_t *dims) {
    if (!t || (int)t->n_dims != nd) return 0;
    for (int i=0;i<nd;i++) if (t->dims[i] != dims[i]) return 0;
    return 1;
}
static int load_tensor(GraniteModel *m, const char *name, ColiTensor *out,
                       int nd, const uint64_t *dims, char *err, size_t cap) {
    const ColiGgufTensorInfo *ti=coli_gguf_find_tensor(&m->gguf,name);
    if (!ti) return errf(err,cap,"missing tensor: %s",name);
    if (!coli_tensor_bind_gguf(&m->gguf,ti,out,err,cap)) return 0;
    if (!tensor_dims(out,nd,dims)) {
        tensor_free(out); return errf(err,cap,"wrong dimensions for tensor: %s",name);
    }
    return 1;
}
static int load_layer_tensor(GraniteModel *m, int layer, const char *suffix,
                             ColiTensor *out, int nd, const uint64_t *dims,
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

static int model_reside_cuda(GraniteModel *m,char *err,size_t cap){
    if(m->exec.kind!=COLI_BACKEND_CUDA)return 1;
#ifdef COLI_CUDA
#define RESIDE_DENSE(t) do{if(!coli_tensor_reside(&m->exec,(t)))return errf(err,cap,"CUDA residency failed for %s",(t)->name?(t)->name:"<unnamed>");m->cuda_dense_bytes+=(t)->storage_bytes;}while(0)
    RESIDE_DENSE(&m->token_embd); RESIDE_DENSE(&m->output_norm); if(!m->tied_output)RESIDE_DENSE(&m->output);
    for(int l=0;l<m->n_layers;l++){
        GraniteLayer *x=&m->layers[l];
        RESIDE_DENSE(&x->attn_norm);RESIDE_DENSE(&x->q);RESIDE_DENSE(&x->k);RESIDE_DENSE(&x->v);RESIDE_DENSE(&x->o);
        RESIDE_DENSE(&x->ffn_norm);RESIDE_DENSE(&x->router);
        if(m->verbose)fprintf(stderr,"[CUDA] dense resident layer %d/%d\r",l+1,m->n_layers);
    }
    if(m->verbose)fputc('\n',stderr);
#undef RESIDE_DENSE
    m->cuda_weight_bytes=m->cuda_dense_bytes;
    return 1;
#else
    return errf(err,cap,"CUDA backend unavailable");
#endif
}

static int model_build_expert_views(GraniteModel *m,char *err,size_t cap){
    for(int l=0;l<m->n_layers;l++){
        GraniteLayer *x=&m->layers[l];
        x->gate_expert=(ColiTensor*)calloc((size_t)m->n_experts,sizeof(ColiTensor));
        x->up_expert=(ColiTensor*)calloc((size_t)m->n_experts,sizeof(ColiTensor));
        x->down_expert=(ColiTensor*)calloc((size_t)m->n_experts,sizeof(ColiTensor));
        x->heat=(uint32_t*)calloc((size_t)m->n_experts,sizeof(uint32_t));
        x->last=(uint32_t*)calloc((size_t)m->n_experts,sizeof(uint32_t));
        x->usage=(uint32_t*)calloc((size_t)m->n_experts,sizeof(uint32_t));
        if(!x->gate_expert||!x->up_expert||!x->down_expert||!x->heat||!x->last||!x->usage)
            return errf(err,cap,"out of memory allocating native expert views");
        for(int e=0;e<m->n_experts;e++){
            if(!coli_tensor_rows_view(&x->gate,(uint64_t)e*m->expert_ff,(uint64_t)m->expert_ff,&x->gate_expert[e])||
               !coli_tensor_rows_view(&x->up,(uint64_t)e*m->expert_ff,(uint64_t)m->expert_ff,&x->up_expert[e])||
               !coli_tensor_rows_view(&x->down,(uint64_t)e*m->hidden,(uint64_t)m->hidden,&x->down_expert[e])){
                tensor_free(&x->gate_expert[e]);tensor_free(&x->up_expert[e]);tensor_free(&x->down_expert[e]);
                return errf(err,cap,"invalid expert tensor view for layer %d expert %d",l,e);
            }
            x->expert_views=e+1;
        }
    }
    if(m->n_layers>0&&m->n_experts>0){GraniteLayer*x=&m->layers[0];
        m->expert_bytes=(size_t)x->gate_expert[0].storage_bytes+(size_t)x->up_expert[0].storage_bytes+(size_t)x->down_expert[0].storage_bytes;}
    return 1;
}

static ColiExpertLayerStore granite_store(GraniteModel*m,int layer){
    GraniteLayer*l=&m->layers[layer];ColiExpertLayerStore s;memset(&s,0,sizeof(s));
    s.pin=l->pin;s.npin=l->npin;s.cache=l->cache;s.ncache=&l->ncache;s.cache_cap=l->cache_cap;
    s.n_experts=m->n_experts;s.layout=g_granite_slot_layout;s.clock=&m->expert_clock;
    s.heat=l->heat;s.last=l->last;s.usage=l->usage;s.access_clock=&m->expert_access_clock;return s;
}
static void granite_slot_bind(GraniteModel*m,int layer,int eid,GraniteExpertSlot*s){
    GraniteLayer*l=&m->layers[layer];s->eid=eid;s->gate=&l->gate_expert[eid];s->up=&l->up_expert[eid];s->down=&l->down_expert[eid];
}
static int granite_storage_load(void*ctx,int layer,int eid,void*slot,int demand){
    GraniteModel*m=(GraniteModel*)ctx;GraniteExpertSlot*s=(GraniteExpertSlot*)slot;(void)demand;
    granite_slot_bind(m,layer,eid,s);coli_tensor_prefetch_host(s->gate);coli_tensor_prefetch_host(s->up);coli_tensor_prefetch_host(s->down);
    if(m->exec.kind==COLI_BACKEND_CUDA){
        if(!coli_tensor_reside(&m->exec,s->gate)||!coli_tensor_reside(&m->exec,s->up)||!coli_tensor_reside(&m->exec,s->down)){
            coli_tensor_release_backend(&m->exec,s->gate);coli_tensor_release_backend(&m->exec,s->up);coli_tensor_release_backend(&m->exec,s->down);
            s->eid=-1;s->gate=s->up=s->down=NULL;return 0;
        }
        m->cuda_expert_bytes+=m->expert_bytes;m->cuda_weight_bytes=m->cuda_dense_bytes+m->cuda_expert_bytes;
    }
    return 1;
}
static void granite_storage_evict(void*ctx,int layer,void*slot){
    GraniteModel*m=(GraniteModel*)ctx;GraniteExpertSlot*s=(GraniteExpertSlot*)slot;(void)layer;
    if(s->eid>=0&&m->exec.kind==COLI_BACKEND_CUDA){
        coli_tensor_release_backend(&m->exec,s->gate);coli_tensor_release_backend(&m->exec,s->up);coli_tensor_release_backend(&m->exec,s->down);
        if(m->cuda_expert_bytes>=m->expert_bytes)m->cuda_expert_bytes-=m->expert_bytes;
        m->cuda_weight_bytes=m->cuda_dense_bytes+m->cuda_expert_bytes;
    }
    s->eid=-1;s->gate=s->up=s->down=NULL;s->used=0;
}
static size_t granite_storage_bytes(void*ctx,int layer,const void*slot){(void)layer;(void)slot;return ((GraniteModel*)ctx)->expert_bytes;}
static const ColiExpertStorageOps g_granite_storage={granite_storage_load,granite_storage_evict,granite_storage_bytes};

static void granite_usage_path(GraniteModel *m, const char *model_path) {
    if (!m || !model_path) return;
    const size_t n = strlen(model_path);
    if (n + sizeof(".coli_usage") > sizeof(m->usage_path)) return;
    memcpy(m->usage_path, model_path, n);
    memcpy(m->usage_path + n, ".coli_usage", sizeof(".coli_usage"));
}
static void granite_usage_rows(GraniteModel *m, uint32_t **rows) {
    for (int l = 0; l < m->n_layers; ++l) rows[l] = m->layers[l].usage;
}
static int64_t granite_usage_load(GraniteModel *m, const char *model_path) {
    granite_usage_path(m, model_path);
    if (!m->usage_path[0]) return 0;
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return 0;
    granite_usage_rows(m, rows);
    const int64_t total = coli_expert_usage_load(m->usage_path, rows, m->n_layers, m->n_experts);
    free(rows);
    m->usage_history = total;
    return total;
}
static void granite_usage_save(GraniteModel *m) {
    if (!m || !m->usage_path[0] || !m->layers) return;
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return;
    granite_usage_rows(m, rows);
    if (!coli_expert_usage_save(m->usage_path, rows, m->n_layers, m->n_experts) && m->verbose)
        fprintf(stderr, "[USAGE] cannot save %s\n", m->usage_path);
    free(rows);
}
static int granite_usage_top(GraniteModel *m, int *ids, int cap) {
    uint32_t **rows = (uint32_t **)malloc((size_t)m->n_layers * sizeof(*rows));
    if (!rows) return 0;
    granite_usage_rows(m, rows);
    const int n = coli_expert_usage_top(rows, m->n_layers, m->n_experts, ids, cap);
    free(rows);
    return n;
}
static int granite_stats_fallback(GraniteModel *m, char *path, size_t cap) {
    if (!m->usage_path[0] || !path || cap == 0) return 0;
    const char *slash = strrchr(m->usage_path, '/');
    if (!slash) return snprintf(path, cap, "stats.txt") > 0;
    const size_t dir = (size_t)(slash - m->usage_path);
    if (dir + sizeof("/stats.txt") > cap) return 0;
    memcpy(path, m->usage_path, dir);
    memcpy(path + dir, "/stats.txt", sizeof("/stats.txt"));
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); const long size = ftell(f); fclose(f);
    return size > 0;
}
static int granite_scheduler_init(GraniteModel*m,char*err,size_t cap){
    m->scheduler_evict_guard=getenv("PILOT_EVICT_GUARD")?atoi(getenv("PILOT_EVICT_GUARD")):1;
    m->repin_interval=getenv("REPIN")?atoi(getenv("REPIN")):0;
    m->pilot=getenv("PILOT")?atoi(getenv("PILOT")):0;
    m->pilot_real=getenv("PILOT_REAL")?atoi(getenv("PILOT_REAL")):0;
    if(m->pilot_real)m->pilot=1;
    m->pilot_k=getenv("PILOT_K")?atoi(getenv("PILOT_K")):(m->pilot_real?6:8);
    if(m->pilot_k<1)m->pilot_k=1;
    if(m->pilot_k>m->n_experts)m->pilot_k=m->n_experts;
    m->couple_k=getenv("COUPLE_K")?atoi(getenv("COUPLE_K")):8;if(m->couple_k<1)m->couple_k=1;if(m->couple_k>32)m->couple_k=32;
    m->couple_d=getenv("COUPLE_D")?atoi(getenv("COUPLE_D")):1;if(m->couple_d<1)m->couple_d=1;if(m->couple_d>2)m->couple_d=2;
    if(getenv("COUPLE")&&*getenv("COUPLE")){long used=0;m->couple=coli_expert_coupling_load(&m->coupling,getenv("COUPLE"),m->n_layers,m->n_experts,&used);
        if(m->verbose)fprintf(stderr,"[COUPLE] GGUF %s: %ld conditioning entries, K=%d depth=%d\n",getenv("COUPLE"),used,m->couple_k,m->couple_d);}
    const int total=m->n_layers*m->n_experts;int slots=total;
#ifdef COLI_CUDA
    if(m->exec.kind==COLI_BACKEND_CUDA){
        size_t free_b=0,total_b=0;double reserve=getenv("CUDA_RESERVE_GB")?atof(getenv("CUDA_RESERVE_GB")):0.5;
        double budget=getenv("CUDA_EXPERT_GB")?atof(getenv("CUDA_EXPERT_GB"))*1e9:-1.0;
        if(budget<0&&coli_cuda_mem_info(m->exec.device,&free_b,&total_b))budget=(double)free_b-reserve*1e9;
        if(budget<0)budget=0;
        slots=m->expert_bytes?(int)(budget/(double)m->expert_bytes):0;
        if(slots>total)slots=total;
        if(slots<m->n_layers)return errf(err,cap,"CUDA expert budget fits %d slots; need at least %d (set CUDA_EXPERT_GB or reduce CUDA_RESERVE_GB)",slots,m->n_layers);
    }
#endif
    int pin_total=slots==total?total:0;const char*pinfile=getenv("PIN");int*pinids=NULL,npinids=0;
    if(slots<total){
        const int maxpins=slots>m->n_layers?slots-m->n_layers:0;
        int want=0;char auto_stats[2048];const char*source=pinfile;
        if(pinfile){
            const char*pgs=getenv("PIN_GB");
            if(pgs&&!strcmp(pgs,"all"))want=maxpins;
            else if(pgs&&atof(pgs)>0&&m->expert_bytes)want=(int)(atof(pgs)*1e9/m->expert_bytes);
            else want=slots/2;
            if(want>maxpins)want=maxpins;
            if(want<0)want=0;
            if(!strcmp(pinfile,"auto")){
                if(m->usage_history>0)source=m->usage_path;
                else if(granite_stats_fallback(m,auto_stats,sizeof(auto_stats)))source=auto_stats;
                else source=NULL;
            }
        }else{
            const int autopin=getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
            if(autopin&&m->usage_history>=5000){
                double confidence=(double)m->usage_history/200000.0;if(confidence>1.0)confidence=1.0;
                want=(int)(0.5*confidence*slots);if(want>maxpins)want=maxpins;
                source=m->usage_path;
            }
        }
        if(want>0&&source){
            pinids=(int*)malloc((size_t)want*sizeof(int));
            if(!pinids)return errf(err,cap,"scheduler pin ranking OOM");
            if(source==m->usage_path)npinids=granite_usage_top(m,pinids,want);
            else npinids=coli_expert_usage_top_file(source,m->n_layers,m->n_experts,pinids,want,NULL);
            pin_total=npinids;
            if(m->verbose)fprintf(stderr,"[PIN] GGUF: %d experts from %s\n",npinids,source);
        }else if(pinfile&&m->verbose)fprintf(stderr,"[PIN] GGUF: no usable history for %s\n",pinfile);
    }
    int*pc=(int*)calloc((size_t)m->n_layers,sizeof(int));if(!pc){free(pinids);return errf(err,cap,"scheduler OOM");}
    if(pin_total==total)for(int l=0;l<m->n_layers;l++)pc[l]=m->n_experts;else for(int i=0;i<npinids;i++)pc[pinids[i]/m->n_experts]++;
    int cache_slots=slots-pin_total,base=cache_slots/m->n_layers,extra=cache_slots%m->n_layers;
    for(int l=0;l<m->n_layers;l++){
        GraniteLayer*x=&m->layers[l];x->npin=pc[l];x->cache_cap=base+(l<extra);if(x->npin)x->pin=(GraniteExpertSlot*)calloc((size_t)x->npin,sizeof(*x->pin));if(x->cache_cap)x->cache=(GraniteExpertSlot*)calloc((size_t)x->cache_cap,sizeof(*x->cache));
        if((x->npin&&!x->pin)||(x->cache_cap&&!x->cache)){free(pc);free(pinids);return errf(err,cap,"scheduler slot OOM");}
        for(int z=0;z<x->npin;z++)x->pin[z].eid=-1;
        for(int z=0;z<x->cache_cap;z++)x->cache[z].eid=-1;
    }
    if(pin_total==total){for(int l=0;l<m->n_layers;l++)for(int e=0;e<m->n_experts;e++){GraniteExpertSlot*q=&m->layers[l].pin[e];if(!granite_storage_load(m,l,e,q,0)){free(pc);free(pinids);return errf(err,cap,"expert pin residency failed");}q->used=++m->expert_clock;}}
    else{int*next=(int*)calloc((size_t)m->n_layers,sizeof(int));for(int i=0;i<npinids;i++){int l=pinids[i]/m->n_experts,e=pinids[i]%m->n_experts;GraniteExpertSlot*q=&m->layers[l].pin[next[l]++];if(!granite_storage_load(m,l,e,q,0)){free(next);free(pc);free(pinids);return errf(err,cap,"expert pin residency failed");}q->used=++m->expert_clock;}free(next);}
    free(pc);free(pinids);
    if(m->verbose)fprintf(stderr,"[SCHED] one native scheduler: %d pin + %d LRU expert slots, %.2f MiB/slot%s; REPIN=%d PILOT=%s COUPLE=%s\n",
        pin_total,cache_slots,m->expert_bytes/(1024.0*1024.0),slots==total?", all experts resident":"",
        m->repin_interval,m->pilot?(m->pilot_real?"real":"hint"):"off",m->couple?"on":"off");
    return 1;
}
static GraniteExpertSlot*granite_expert_acquire(GraniteModel*m,int layer,int eid,int demand){
    ColiExpertLayerStore st=granite_store(m,layer);
    return (GraniteExpertSlot*)coli_expert_acquire(&st,layer,eid,demand,m->scheduler_evict_guard,&g_granite_storage,m,&m->scheduler_stats);
}
static void granite_prefetch_ids(GraniteModel*m,int layer,const int*ids,int n){
    if(layer<0||layer>=m->n_layers)return;
    GraniteLayer*l=&m->layers[layer];
    for(int i=0;i<n;i++){
        const int eid=ids[i];if(eid<0||eid>=m->n_experts)continue;
        if(m->pilot_real)(void)granite_expert_acquire(m,layer,eid,0);
        else{
            coli_tensor_prefetch_host(&l->gate_expert[eid]);
            coli_tensor_prefetch_host(&l->up_expert[eid]);
            coli_tensor_prefetch_host(&l->down_expert[eid]);
        }
    }
}
static void granite_couple_prefetch(GraniteModel*m,int layer,const int*routed,int nrouted){
    if(!m->couple)return;
    for(int d=1;d<=m->couple_d;d++){int target=layer+d;if(target>=m->n_layers)break;int pred[32];
        int n=coli_expert_coupling_predict(&m->coupling,layer,d,routed,nrouted,pred,m->couple_k);
        granite_prefetch_ids(m,target,pred,n);}
}
static void granite_repin(GraniteModel*m){
    if(m->repin_interval<=0||++m->tokens_since_repin<m->repin_interval)return;
    m->tokens_since_repin=0;
    for(int l=0;l<m->n_layers;l++){
        ColiExpertLayerStore st=granite_store(m,l);int pi,e;long gain;
        if(!coli_expert_repin_pick(&st,&pi,&e,&gain))continue;
        GraniteExpertSlot*q=&m->layers[l].pin[pi];const int old=q->eid;
        int moved=coli_expert_repin_promote_cached(&st,pi,e,l,&g_granite_storage,m);
        if(moved>0){
            if(m->verbose)fprintf(stderr,"[REPIN] GGUF layer %d: %d <- cached %d (gain %ld)\n",l,old,e,gain);
            coli_expert_decay_heat(&st);continue;
        }
        if(moved<0)continue;
        granite_storage_evict(m,l,q);
        if(granite_storage_load(m,l,e,q,0)){
            q->used=++m->expert_clock;
            if(m->verbose)fprintf(stderr,"[REPIN] GGUF layer %d: %d <- %d (gain %ld)\n",l,old,e,gain);
        }else if(old>=0){
            (void)granite_storage_load(m,l,old,q,0);
            q->used=++m->expert_clock;
            if(m->verbose)fprintf(stderr,"[REPIN] GGUF layer %d: upload of %d failed; restored %d\n",l,e,old);
        }
        coli_expert_decay_heat(&st);
    }
}

static int alloc_cache(GraniteModel *m, int context, char *err, size_t cap) {
    if(context<=0)return errf(err,cap,"invalid context capacity");
    size_t elems=(size_t)m->n_layers;
    if(elems>SIZE_MAX/(size_t)context)return errf(err,cap,"KV cache overflow");
    elems*=context;
    if(elems>SIZE_MAX/(size_t)m->kv_dim)return errf(err,cap,"KV cache overflow");
    elems*=m->kv_dim;
    if(elems>SIZE_MAX/sizeof(float))return errf(err,cap,"KV cache overflow");
    if(m->exec.kind==COLI_BACKEND_CUDA){
#ifdef COLI_CUDA
        size_t bytes=elems*sizeof(float);
        m->k_cache_dev=(float*)coli_cuda_pipe_alloc(m->exec.device,bytes);
        m->v_cache_dev=(float*)coli_cuda_pipe_alloc(m->exec.device,bytes);
        if(!m->k_cache_dev||!m->v_cache_dev)return errf(err,cap,"out of VRAM allocating %.2f MiB native KV cache",2.0*bytes/(1024.0*1024.0));
#else
        return errf(err,cap,"CUDA backend unavailable");
#endif
    }else{
        m->k_cache=(float*)calloc(elems,sizeof(float));m->v_cache=(float*)calloc(elems,sizeof(float));
        if(!m->k_cache||!m->v_cache)return errf(err,cap,"out of memory allocating %.2f MiB KV cache",2.0*elems*sizeof(float)/(1024.0*1024.0));
    }
    m->context_capacity=context;return 1;
}

static void scratch_free(GraniteScratch *s){
#ifdef COLI_CUDA
    if(s->cuda_allocated){
#define DFREE(x) do{if(s->x)coli_cuda_pipe_free(s->cuda_device,s->x);}while(0)
        DFREE(dx);DFREE(dnorm);DFREE(dq);DFREE(dk);DFREE(dv);DFREE(dattn);DFREE(dproj);DFREE(dffn_in);
        DFREE(drouter);DFREE(dgate);DFREE(dup);DFREE(dexpert_out);DFREE(dmoe);DFREE(dscores);DFREE(dlogits);
#undef DFREE
    }
#endif
    free(s->x);free(s->norm);free(s->q);free(s->k);free(s->v);free(s->attn);free(s->proj);free(s->ffn_in);
    free(s->router);free(s->gate);free(s->up);free(s->expert_out);free(s->moe);free(s->scores);free(s->logits);free(s->weight);free(s->top_idx);free(s->top_w);
    memset(s,0,sizeof(*s));
}
static int scratch_alloc(const GraniteModel*m,GraniteScratch*s,char*err,size_t cap){
#define ALLOC(field,n,type) do{s->field=(type*)calloc((size_t)(n),sizeof(type));if(!s->field){scratch_free(s);return errf(err,cap,"out of memory allocating inference scratch");}}while(0)
    ALLOC(x,m->hidden,float);ALLOC(norm,m->hidden,float);ALLOC(q,m->hidden,float);ALLOC(k,m->kv_dim,float);ALLOC(v,m->kv_dim,float);
    ALLOC(attn,m->hidden,float);ALLOC(proj,m->hidden,float);ALLOC(ffn_in,m->hidden,float);ALLOC(router,m->n_experts,float);
    ALLOC(gate,m->expert_ff,float);ALLOC(up,m->expert_ff,float);ALLOC(expert_out,m->hidden,float);ALLOC(moe,m->hidden,float);
    ALLOC(scores,m->context_capacity,float);ALLOC(logits,m->vocab,float);ALLOC(weight,m->hidden,float);ALLOC(top_idx,m->n_expert_used,int);ALLOC(top_w,m->n_expert_used,float);
#undef ALLOC
#ifdef COLI_CUDA
    if(m->exec.kind==COLI_BACKEND_CUDA){
        s->cuda_device=m->exec.device;s->cuda_allocated=1;
#define DALLOC(field,n) do{s->field=(float*)coli_cuda_pipe_alloc(s->cuda_device,(size_t)(n)*sizeof(float));if(!s->field){scratch_free(s);return errf(err,cap,"out of VRAM allocating Granite pipeline scratch");}}while(0)
        DALLOC(dx,m->hidden);DALLOC(dnorm,m->hidden);DALLOC(dq,m->hidden);DALLOC(dk,m->kv_dim);DALLOC(dv,m->kv_dim);
        DALLOC(dattn,m->hidden);DALLOC(dproj,m->hidden);DALLOC(dffn_in,m->hidden);DALLOC(drouter,m->n_experts);
        DALLOC(dgate,m->expert_ff);DALLOC(dup,m->expert_ff);DALLOC(dexpert_out,m->hidden);DALLOC(dmoe,m->hidden);
        DALLOC(dscores,(size_t)m->n_heads*m->context_capacity);DALLOC(dlogits,m->vocab);
#undef DALLOC
    }
#endif
    return 1;
}

static int tensor_vector(const ColiTensor *t,float *dst,int n,char *err,size_t cap){
    if(t->row_count!=1 || t->dims[0]!=(uint64_t)n || !coli_tensor_read_row_f32(t,0,dst,(uint64_t)n))
        return errf(err,cap,"failed reading tensor vector: %s",t->name?t->name:"<unnamed>");
    return 1;
}
static int tensor_mm(GraniteModel*m,float*y,const float*x,ColiTensor*t,int I,int O,
                     char*err,size_t cap){
    if(!coli_tensor_matmul(&m->exec,y,x,t,1,I,O))
        return errf(err,cap,"%s matmul failed for %s",
                    m->exec.kind==COLI_BACKEND_CUDA?"CUDA":"CPU",t->name?t->name:"<unnamed>");
    return 1;
}

static int model_forward_cpu(GraniteModel*m,GraniteScratch*s,int token,int pos,char*err,size_t cap){
    if(token<0||token>=m->vocab)return errf(err,cap,"token id %d outside vocabulary",token);
    if(pos<0||pos>=m->context_capacity)return errf(err,cap,"position %d outside context",pos);
    if(!coli_tensor_read_row_f32(&m->token_embd,(uint64_t)token,s->x,(uint64_t)m->hidden))
        return errf(err,cap,"failed reading token embedding row %d",token);
    for(int i=0;i<m->hidden;i++)s->x[i]*=m->embedding_scale;
    for(int l=0;l<m->n_layers;l++){
        GraniteLayer*L=&m->layers[l];
        if(!tensor_vector(&L->attn_norm,s->weight,m->hidden,err,cap))return 0;
        coli_f32_rmsnorm(s->norm,s->x,s->weight,m->hidden,m->eps);
        if(!tensor_mm(m,s->q,s->norm,&L->q,m->hidden,m->hidden,err,cap))return 0;
        if(!tensor_mm(m,s->k,s->norm,&L->k,m->hidden,m->kv_dim,err,cap))return 0;
        if(!tensor_mm(m,s->v,s->norm,&L->v,m->hidden,m->kv_dim,err,cap))return 0;
        if(m->rope_enabled){
            coli_f32_rope(s->q,m->n_heads,m->head_dim,m->rope_dims,pos,m->rope_base);
            coli_f32_rope(s->k,m->n_kv_heads,m->head_dim,m->rope_dims,pos,m->rope_base);
        }
        const size_t layer_stride=(size_t)m->context_capacity*m->kv_dim;
        float*kc=m->k_cache+(size_t)l*layer_stride;float*vc=m->v_cache+(size_t)l*layer_stride;
        memcpy(kc+(size_t)pos*m->kv_dim,s->k,(size_t)m->kv_dim*sizeof(float));
        memcpy(vc+(size_t)pos*m->kv_dim,s->v,(size_t)m->kv_dim*sizeof(float));
        coli_f32_gqa_attention(s->attn,s->q,kc,vc,pos,m->n_heads,m->n_kv_heads,m->head_dim,m->attention_scale,s->scores);
        if(!tensor_mm(m,s->proj,s->attn,&L->o,m->hidden,m->hidden,err,cap))return 0;
        for(int i=0;i<m->hidden;i++)s->ffn_in[i]=s->x[i]+m->residual_scale*s->proj[i];
        if(!tensor_vector(&L->ffn_norm,s->weight,m->hidden,err,cap))return 0;
        coli_f32_rmsnorm(s->norm,s->ffn_in,s->weight,m->hidden,m->eps);
        if(!tensor_mm(m,s->router,s->norm,&L->router,m->hidden,m->n_experts,err,cap))return 0;
        const int nk=coli_f32_router_topk(s->router,m->n_experts,m->n_expert_used,s->top_idx,s->top_w);
        granite_couple_prefetch(m,l,s->top_idx,nk);
        if(m->pilot&&l+1<m->n_layers){int pidx[64];float pw[64];int keep=m->pilot_k<64?m->pilot_k:64;
            if(tensor_mm(m,s->router,s->norm,&m->layers[l+1].router,m->hidden,m->n_experts,err,cap)){
                int pn=coli_f32_router_topk(s->router,m->n_experts,keep,pidx,pw);granite_prefetch_ids(m,l+1,pidx,pn);
            }else return 0;}
        memset(s->moe,0,(size_t)m->hidden*sizeof(float));
        for(int j=0;j<nk;j++){
            const int e=s->top_idx[j];const float rw=s->top_w[j]*m->expert_weights_scale;
            GraniteExpertSlot *slot=granite_expert_acquire(m,l,e,1);
            if(!slot)return errf(err,cap,"expert scheduler admission failed at layer %d expert %d",l,e);
            if(!tensor_mm(m,s->gate,s->norm,slot->gate,m->hidden,m->expert_ff,err,cap))return 0;
            if(!tensor_mm(m,s->up,s->norm,slot->up,m->hidden,m->expert_ff,err,cap))return 0;
            for(int i=0;i<m->expert_ff;i++)s->gate[i]=coli_f32_silu(s->gate[i])*s->up[i];
            if(!tensor_mm(m,s->expert_out,s->gate,slot->down,m->expert_ff,m->hidden,err,cap))return 0;
            for(int i=0;i<m->hidden;i++)s->moe[i]+=rw*s->expert_out[i];
        }
        for(int i=0;i<m->hidden;i++)s->x[i]=s->ffn_in[i]+m->residual_scale*s->moe[i];
    }
    if(!tensor_vector(&m->output_norm,s->weight,m->hidden,err,cap))return 0;
    coli_f32_rmsnorm(s->norm,s->x,s->weight,m->hidden,m->eps);
    ColiTensor*outw=m->tied_output?&m->token_embd:&m->output;
    if(!tensor_mm(m,s->logits,s->norm,outw,m->hidden,m->vocab,err,cap))return 0;
    const float inv=1.f/m->logit_scale;for(int i=0;i<m->vocab;i++)s->logits[i]*=inv;
    return 1;
}


#ifdef COLI_CUDA
static const float *tensor_f32_device(GraniteModel*m,ColiTensor*t,int n,char*err,size_t cap){
    if(t->dtype!=COLI_DTYPE_F32||t->row_count!=1||t->dims[0]!=(uint64_t)n||
       !coli_tensor_reside(&m->exec,t)){
        errf(err,cap,"invalid resident F32 vector: %s",t->name?t->name:"<unnamed>");return NULL;
    }
    return (const float*)coli_tensor_device_data(t);
}
static int tensor_mm_device(GraniteModel*m,float*y,const float*x,ColiTensor*t,char*err,size_t cap){
    if(!coli_tensor_matmul_device(&m->exec,y,x,t,1))
        return errf(err,cap,"native CUDA matmul failed for %s",t->name?t->name:"<unnamed>");
    return 1;
}
static int model_forward_cuda(GraniteModel*m,GraniteScratch*s,int token,int pos,char*err,size_t cap){
    const int dev=m->exec.device,H=m->hidden;
    if(token<0||token>=m->vocab)return errf(err,cap,"token id %d outside vocabulary",token);
    if(pos<0||pos>=m->context_capacity)return errf(err,cap,"position %d outside context",pos);
    if(!coli_tensor_read_row_device(&m->exec,&m->token_embd,(uint64_t)token,s->dx,m->embedding_scale))
        return errf(err,cap,"CUDA embedding row decode failed");
    const size_t layer_stride=(size_t)m->context_capacity*m->kv_dim;
    for(int l=0;l<m->n_layers;l++){
        GraniteLayer*L=&m->layers[l];
        const float *w=tensor_f32_device(m,&L->attn_norm,H,err,cap);if(!w)return 0;
        if(!coli_cuda_pipe_rmsnorm(dev,s->dnorm,s->dx,w,1,H,m->eps))return errf(err,cap,"CUDA attention RMSNorm failed");
        if(!tensor_mm_device(m,s->dq,s->dnorm,&L->q,err,cap)||
           !tensor_mm_device(m,s->dk,s->dnorm,&L->k,err,cap)||
           !tensor_mm_device(m,s->dv,s->dnorm,&L->v,err,cap))return 0;
        if(m->rope_enabled&&
           (!coli_cuda_pipe_rope_interleaved(dev,s->dq,pos,m->n_heads,m->head_dim,m->rope_dims,m->rope_base)||
            !coli_cuda_pipe_rope_interleaved(dev,s->dk,pos,m->n_kv_heads,m->head_dim,m->rope_dims,m->rope_base)))
            return errf(err,cap,"CUDA RoPE failed");
        float *kc=m->k_cache_dev+(size_t)l*layer_stride,*vc=m->v_cache_dev+(size_t)l*layer_stride;
        if(!coli_cuda_pipe_gqa_decode(dev,s->dattn,s->dq,s->dk,s->dv,kc,vc,s->dscores,
                pos,m->context_capacity,m->n_heads,m->n_kv_heads,m->head_dim,m->attention_scale))
            return errf(err,cap,"CUDA GQA failed at layer %d",l);
        if(!tensor_mm_device(m,s->dproj,s->dattn,&L->o,err,cap))return 0;
        if(!coli_cuda_pipe_residual(dev,s->dffn_in,s->dx,s->dproj,m->residual_scale,(size_t)H))
            return errf(err,cap,"CUDA attention residual failed");
        w=tensor_f32_device(m,&L->ffn_norm,H,err,cap);if(!w)return 0;
        if(!coli_cuda_pipe_rmsnorm(dev,s->dnorm,s->dffn_in,w,1,H,m->eps))
            return errf(err,cap,"CUDA FFN RMSNorm failed");
        if(!tensor_mm_device(m,s->drouter,s->dnorm,&L->router,err,cap)||
           !coli_cuda_pipe_download(dev,s->drouter,s->router,(size_t)m->n_experts*sizeof(float)))
            return errf(err,cap,"CUDA router failed");
        const int nk=coli_f32_router_topk(s->router,m->n_experts,m->n_expert_used,s->top_idx,s->top_w);
        granite_couple_prefetch(m,l,s->top_idx,nk);
        if(m->pilot&&l+1<m->n_layers){int pidx[64];float pw[64];int keep=m->pilot_k<64?m->pilot_k:64;
            if(!tensor_mm_device(m,s->drouter,s->dnorm,&m->layers[l+1].router,err,cap)||
               !coli_cuda_pipe_download(dev,s->drouter,s->router,(size_t)m->n_experts*sizeof(float)))
                return errf(err,cap,"CUDA PILOT router failed at layer %d",l);
            int pn=coli_f32_router_topk(s->router,m->n_experts,keep,pidx,pw);granite_prefetch_ids(m,l+1,pidx,pn);}
        if(!coli_cuda_pipe_zero(dev,s->dmoe,(size_t)H))return errf(err,cap,"CUDA MoE zero failed");
        for(int j=0;j<nk;j++){
            const int e=s->top_idx[j];const float rw=s->top_w[j]*m->expert_weights_scale;
            GraniteExpertSlot *slot=granite_expert_acquire(m,l,e,1);
            if(!slot)return errf(err,cap,"expert scheduler admission failed at layer %d expert %d",l,e);
            if(!tensor_mm_device(m,s->dgate,s->dnorm,slot->gate,err,cap)||
               !tensor_mm_device(m,s->dup,s->dnorm,slot->up,err,cap)||
               !coli_cuda_pipe_silu_mul(dev,s->dgate,s->dup,(size_t)m->expert_ff)||
               !tensor_mm_device(m,s->dexpert_out,s->dgate,slot->down,err,cap)||
               !coli_cuda_pipe_axpy(dev,s->dmoe,s->dexpert_out,rw,(size_t)H))
                return errf(err,cap,"CUDA expert %d failed at layer %d",e,l);
        }
        if(!coli_cuda_pipe_residual(dev,s->dx,s->dffn_in,s->dmoe,m->residual_scale,(size_t)H))
            return errf(err,cap,"CUDA MoE residual failed");
    }
    const float *w=tensor_f32_device(m,&m->output_norm,H,err,cap);if(!w)return 0;
    if(!coli_cuda_pipe_rmsnorm(dev,s->dnorm,s->dx,w,1,H,m->eps))return errf(err,cap,"CUDA output RMSNorm failed");
    ColiTensor*outw=m->tied_output?&m->token_embd:&m->output;
    if(!tensor_mm_device(m,s->dlogits,s->dnorm,outw,err,cap)||
       !coli_cuda_pipe_download(dev,s->dlogits,s->logits,(size_t)m->vocab*sizeof(float)))
        return errf(err,cap,"CUDA output projection failed");
    const float inv=1.f/m->logit_scale;for(int i=0;i<m->vocab;i++)s->logits[i]*=inv;
    return 1;
}
#endif

static int model_forward(GraniteModel*m,GraniteScratch*s,int token,int pos,char*err,size_t cap){
    int ok;
#ifdef COLI_CUDA
    if(m->exec.kind==COLI_BACKEND_CUDA)ok=model_forward_cuda(m,s,token,pos,err,cap);
    else
#endif
    ok=model_forward_cpu(m,s,token,pos,err,cap);
    if(ok)granite_repin(m);
    return ok;
}

static int argmax(const float*x,int n){int b=0;for(int i=1;i<n;i++)if(x[i]>x[b])b=i;return b;}

static int load_model(GraniteModel*m,const char*path,int context,int verbose,ColiExec exec,char*err,size_t cap){
    memset(m,0,sizeof(*m));m->gguf.fd=-1;m->verbose=verbose;m->exec=exec;
    if(!coli_gguf_open(&m->gguf,path))return errf(err,cap,"cannot open GGUF: %s",coli_gguf_error(&m->gguf));
    if(!model_config(m,err,cap))return 0;
    if(!coli_gguf_tokenizer_load(&m->tokenizer,&m->gguf,err,cap))return 0;
    if(coli_gguf_tokenizer_vocab_size(m->tokenizer)!=m->vocab)return errf(err,cap,"tokenizer/model vocabulary mismatch");
    const double t0=now_sec();if(!model_load_weights(m,err,cap))return 0;
    const double mapped_s=now_sec()-t0,t1=now_sec();
    if(!model_build_expert_views(m,err,cap)||!model_reside_cuda(m,err,cap))return 0;
    const int64_t history=granite_usage_load(m,path);
    if(history>0&&verbose)fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)history,m->usage_path);
    if(!granite_scheduler_init(m,err,cap)||!alloc_cache(m,context,err,cap))return 0;
    if(verbose){
        fprintf(stderr,"[GGUF] granitemoe: layers=%d hidden=%d heads=%d/%d experts=%d top=%d vocab=%d\n",
            m->n_layers,m->hidden,m->n_heads,m->n_kv_heads,m->n_experts,m->n_expert_used,m->vocab);
        if(exec.kind==COLI_BACKEND_CUDA)fprintf(stderr,"[GGUF] mapped in %.2fs; scheduled native residency %.2fs, %.2f MiB; backend=cuda (Colibri scheduler); context=%d\n",
            mapped_s,now_sec()-t1,m->cuda_weight_bytes/(1024.0*1024.0),context);
        else fprintf(stderr,"[GGUF] mapped quantized weights in %.2fs; backend=cpu (direct mmap); context=%d; threads=%d\n",
            mapped_s,context,omp_get_max_threads());
    }
    return 1;
}

static void usage(const char*prog){
    fprintf(stderr,"Usage: %s [--gguf] MODEL.gguf --prompt TEXT [--max-tokens N] [--device cpu|cuda[:N]] [--raw-prompt] [--verbose]\n",prog);
}
int coli_gguf_cli_requested(int argc,char**argv){
    if(argc>1&&!strcmp(argv[1],"--gguf"))return 1;
    if(argc>1){size_t n=strlen(argv[1]);if(n>=5&&!strcmp(argv[1]+n-5,".gguf"))return 1;}
    return 0;
}
int coli_gguf_run_cli(int argc,char**argv){
    const char*model_path=NULL,*prompt=NULL,*device_arg="cpu";int max_tokens=24,raw=0,verbose=0;
    int i=1;if(i<argc&&!strcmp(argv[i],"--gguf"))i++;
    if(i<argc)model_path=argv[i++];
    for(;i<argc;i++){
        if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"--max-tokens")&&i+1<argc)max_tokens=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--device")&&i+1<argc)device_arg=argv[++i];
        else if(!strcmp(argv[i],"--raw-prompt"))raw=1;
        else if(!strcmp(argv[i],"--verbose")||!strcmp(argv[i],"-v"))verbose=1;
        else {fprintf(stderr,"unknown GGUF option: %s\n",argv[i]);usage(argv[0]);return 2;}
    }
    if(!model_path||!prompt||max_tokens<0){usage(argv[0]);return 2;}
    /* Architecture dispatch stays in the single GGUF CLI entry point. */
    {
        ColiGgufFile arch_probe; arch_probe.fd=-1;
        if(coli_gguf_open(&arch_probe,model_path)){
            const ColiGgufKV *akv=coli_gguf_find_kv(&arch_probe,"general.architecture");
            char *arch=NULL;
            if(akv&&coli_gguf_kv_read_string(&arch_probe,akv,&arch)&&!strcmp(arch,"qwen3next")){
                free(arch);coli_gguf_close(&arch_probe);return coli_qwen3next_run_cli(argc,argv);
            }
            free(arch);coli_gguf_close(&arch_probe);
        }
    }
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

    ColiExec exec={COLI_BACKEND_CPU,0}; int cuda_started=0; (void)cuda_started;
    if(!strcmp(device_arg,"cuda")||!strncmp(device_arg,"cuda:",5)){
        exec.kind=COLI_BACKEND_CUDA; if(device_arg[4]==':')exec.device=atoi(device_arg+5);
#ifdef COLI_CUDA
        if(!coli_cuda_init(&exec.device,1)){fprintf(stderr,"cannot initialize CUDA device %d\n",exec.device);free(ids);return 1;}
        cuda_started=1;
#else
        fprintf(stderr,"this binary was built without CUDA support\n");free(ids);return 1;
#endif
    } else if(strcmp(device_arg,"cpu")){fprintf(stderr,"invalid --device: %s\n",device_arg);free(ids);return 2;}

    GraniteModel m;GraniteScratch s={0};
    if(!load_model(&m,model_path,n_prompt+max_tokens,verbose,exec,err,sizeof(err))){fprintf(stderr,"%s\n",err);model_free(&m);free(ids);
#ifdef COLI_CUDA
        if(cuda_started)coli_cuda_shutdown();
#endif
        return 1;}
    if(!scratch_alloc(&m,&s,err,sizeof(err))){fprintf(stderr,"%s\n",err);model_free(&m);free(ids);
#ifdef COLI_CUDA
        if(cuda_started)coli_cuda_shutdown();
#endif
        return 1;}
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
    if(verbose){fprintf(stderr,"[GGUF] prompt=%d generated=%d elapsed=%.2fs (%.3f tok/s decode+prefill)\n",n_prompt,generated,now_sec()-t0,
        (n_prompt+generated)/(now_sec()-t0));
        fprintf(stderr,"[SCHED] hits=%llu (pin=%llu LRU=%llu) misses=%llu admissions=%llu evictions=%llu speculative=%llu/%llu\n",
            (unsigned long long)m.scheduler_stats.hits,(unsigned long long)m.scheduler_stats.pin_hits,
            (unsigned long long)m.scheduler_stats.cache_hits,(unsigned long long)m.scheduler_stats.misses,
            (unsigned long long)m.scheduler_stats.admissions,(unsigned long long)m.scheduler_stats.evictions,
            (unsigned long long)m.scheduler_stats.speculative_loads,(unsigned long long)m.scheduler_stats.speculative_drops);}
    granite_usage_save(&m);scratch_free(&s);model_free(&m);free(ids);
#ifdef COLI_CUDA
    if(cuda_started)coli_cuda_shutdown();
#endif
    return 0;
fail:
    granite_usage_save(&m);scratch_free(&s);model_free(&m);free(ids);
#ifdef COLI_CUDA
    if(cuda_started)coli_cuda_shutdown();
#endif
    return 1;
}
