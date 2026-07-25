/* backend_loader.c — Windows runtime loader for coli_cuda.dll.
 *
 * Why this exists: the engine is built with MinGW-w64 (gcc), but CUDA kernels
 * must be compiled with MSVC + nvcc. We cannot link a CUDA .o into a gcc binary
 * reliably across the MSVC/MinGW ABI, and nvcc requires cl.exe as its host
 * compiler. The clean cross-toolchain split is: build the CUDA backend into a
 * standalone coli_cuda.dll with nvcc+MSVC, then load it here at runtime via
 * LoadLibrary/GetProcAddress. The host (glm.exe) never links cudart directly.
 *
 * On Linux this file is not compiled (the Makefile links backend_cuda.o
 * directly). On Windows, when COLI_CUDA is defined, glm.c calls the
 * coli_cuda_* wrappers below, which forward through function pointers resolved
 * from the DLL at first use. If the DLL is absent, every call safely returns
 * the "not initialized" sentinel (0 / no-op) and the engine falls back to CPU.
 *
 * ABI note: ColiCudaTensor* is opaque to the host (it stores the pointer,
 * never dereferences it), so the MSVC-allocated struct is safe to pass across
 * the boundary as an opaque handle. All scalar types (int, size_t, pointers)
 * agree between MSVC and MinGW-w64 on x86-64.
 */
#ifdef _WIN32

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <windows.h>

#include "backend_cuda.h"

/* Function-pointer typedefs matching each exported symbol. */
typedef int            (*fn_init)(const int *devices, int count);
typedef void           (*fn_shutdown)(void);
typedef int            (*fn_device_count)(void);
typedef int            (*fn_device_at)(int index);
typedef int            (*fn_mem_info)(int device, size_t *free_bytes, size_t *total_bytes);
typedef void           (*fn_stats)(int device, size_t *tensor_count, size_t *tensor_bytes);
typedef void           (*fn_group_stats)(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                         double *h2d_ms, double *kernel_ms, double *d2h_ms);
typedef int            (*fn_expert_mlp)(ColiCudaTensor *gate, ColiCudaTensor *up,
                                        ColiCudaTensor *down, float *y, const float *x, int S);
typedef int            (*fn_expert_group)(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                                          ColiCudaTensor *const *downs, const int *rows, int count,
                                          float *y, const float *x);
typedef int            (*fn_expert_group_issue)(ColiCudaTensor *const *gates,
                                                ColiCudaTensor *const *ups,
                                                ColiCudaTensor *const *downs,
                                                const int *rows, int count, const float *x);
typedef const float *  (*fn_expert_group_take)(int device);
typedef int            (*fn_attention_absorb)(ColiCudaTensor *kv_b, float *ctx, const float *q,
                                              const float *latent, const float *rope, int H, int Q,
                                              int R, int V, int K, int T, float attention_scale);
typedef int            (*fn_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                           const float *scales, int fmt, int I, int O, int device);
typedef int            (*fn_tensor_upload_g)(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device, int gs);
typedef int            (*fn_matmul)(ColiCudaTensor **tensor, float *y, const float *x,
                                    const void *weights, const float *scales,
                                    int fmt, int S, int I, int O, int device, int gs);
typedef int            (*fn_ggml_matmul)(float *y, const float *x, const void *weights,
                                         uint32_t dtype, uint64_t weight_bytes,
                                         int S, int I, int O, int device);
typedef int            (*fn_tensor_upload_ggml)(ColiCudaTensor **tensor,const void *weights,
                                         uint32_t dtype,uint64_t weight_bytes,int I,int O,int device);
typedef int            (*fn_tensor_upload_sgguf)(ColiCudaTensor **tensor,
                                         const uint8_t *block_offsets_le,const uint8_t *blocks,
                                         uint64_t first_block,uint64_t block_count,uint32_t codec_id,
                                         uint32_t sparse_layout,uint32_t offset_width,
                                         uint16_t auxiliary_bytes_per_block,uint16_t retained_value_bits,
                                         int I,int O,int device);
typedef int            (*fn_tensor_view_rows)(ColiCudaTensor *base,uint64_t first_row,
                                         uint64_t row_count,ColiCudaTensor **view);
typedef const void *   (*fn_tensor_data)(const ColiCudaTensor *tensor);
typedef int            (*fn_tensor_matmul_host)(ColiCudaTensor *tensor,float *y,const float *x,int S);
typedef void           (*fn_tensor_free)(ColiCudaTensor *tensor);
typedef size_t         (*fn_tensor_bytes)(const ColiCudaTensor *tensor);
typedef int            (*fn_tensor_device)(const ColiCudaTensor *tensor);

/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */


/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */
typedef int (*fn_attention_absorb_batch)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale);
typedef int (*fn_attention_absorb_batch_dev)(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_absorb_kvdev)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale);
typedef int (*fn_attention_project_batch)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale);
typedef int (*fn_attention_project_ragged)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);
typedef int (*fn_attention_project_batch_dev)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_project_batch_dev_out)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_pipe_add)(int device,float *x_dev,const float *t_dev,size_t n);
typedef int (*fn_pipe_axpy)(int device,float *y_dev,const float *x_dev,float alpha,size_t n);
typedef int (*fn_pipe_residual)(int device,float *y_dev,const float *base_dev,const float *delta_dev,float alpha,size_t n);
typedef int (*fn_pipe_zero)(int device,float *x_dev,size_t n);
typedef int (*fn_pipe_decode_row)(ColiCudaTensor *tensor,uint64_t row,float *out_dev,float scale);
typedef int (*fn_pipe_gqa_decode)(int device,float *out_dev,const float *q_dev,const float *k_dev,
        const float *v_dev,float *k_cache_dev,float *v_cache_dev,float *scores_dev,int pos,
        int context_capacity,int n_heads,int n_kv_heads,int head_dim,float attention_scale);
typedef void * (*fn_pipe_alloc)(int device,size_t bytes);
typedef int (*fn_pipe_copy2d)(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height);
typedef int (*fn_pipe_download)(int device,const void *src,void *dst,size_t bytes);
typedef int (*fn_pipe_copy)(int device,void *dst,const void *src,size_t bytes);
typedef void (*fn_pipe_free)(int device,void *p);
typedef int (*fn_pipe_gemm)(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
typedef int (*fn_pipe_peer_copy)(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes);
typedef int (*fn_pipe_rmsnorm)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps);
typedef int (*fn_pipe_rmsnorm_s)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride);
typedef int (*fn_group_resident_issue)(ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,ColiCudaTensor *const *downs,const float *weights,int count,int home_device,const float *x_src_dev,float *partial_slot_dev);
typedef int (*fn_group_resident_take)(int home_device,const int *devices,int n_issued,float *slots_dev,float *acc_dev,int D);
typedef int (*fn_pipe_router)(int device,const float *x_dev,const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,float topp,int norm_topk,float routed_scale,int *idx_host,float *w_host,int *keff_host);
typedef int (*fn_pipe_rope)(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rope_base)(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rope_interleaved)(int device,float *v_dev,int position,int n_heads,int head_dim,int rope_dims,float theta);
typedef int (*fn_pipe_qg_split)(int device,float *q_dev,float *gate_dev,const float *qg_dev,int n_heads,int head_dim);
typedef int (*fn_pipe_rope_neox)(int device,float *v_dev,int position,int n_heads,int head_dim,int rope_dims,float theta);
typedef int (*fn_pipe_sigmoid_mul)(int device,float *x_dev,const float *gate_dev,size_t n);
typedef int (*fn_pipe_sigmoid_scale)(int device,float *x_dev,const float *logit_dev,size_t n);
typedef int (*fn_pipe_gated_delta_decode)(int device,float *out_dev,const float *qkv_dev,const float *z_dev,const float *ba_dev,const float *conv_weight_dev,const float *dt_bias_dev,const float *a_dev,const float *norm_weight_dev,float *conv_state_dev,float *recurrent_state_dev,int n_key_heads,int n_value_heads,int head_dim,int conv_kernel,float eps);
typedef int (*fn_pipe_gated_delta_decode_separate)(int device,float *out_dev,const float *qkv_dev,const float *z_dev,const float *beta_dev,const float *alpha_dev,const float *conv_weight_dev,const float *dt_bias_dev,const float *a_dev,const float *norm_weight_dev,float *conv_state_dev,float *recurrent_state_dev,int n_key_heads,int n_value_heads,int head_dim,int conv_kernel,float eps);
typedef int (*fn_pipe_qwen_topk)(int device,const float *logits_dev,int E,int Ksel,int *idx_host,float *w_host);
typedef int (*fn_pipe_rows_add)(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D);
typedef float * (*fn_pipe_scratch)(int device,int slot,size_t bytes);
typedef int (*fn_pipe_silu_mul)(int device,float *gate_dev,const float *up_dev,size_t n);
typedef int (*fn_pipe_sync)(int device);
typedef int (*fn_pipe_upload)(int device,void *dst,const void *src,size_t bytes);
typedef int (*fn_shared_mlp_w4a16)(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S);
typedef int (*fn_tensor_update)(ColiCudaTensor *tensor, const void *weights, const float *scales);

/* Resolved pointers, plus a flag so we attempt the load at most once. */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_expert_group_issue expert_group_issue;
    fn_expert_group_take expert_group_take;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_tensor_upload_g tensor_upload_g;
    fn_matmul          matmul;
    fn_ggml_matmul     ggml_matmul;
    fn_tensor_upload_ggml tensor_upload_ggml;
    fn_tensor_upload_sgguf tensor_upload_sgguf;
    fn_tensor_view_rows tensor_view_rows;
    fn_tensor_data tensor_data;
    fn_tensor_matmul_host tensor_matmul_host;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;

    fn_attention_absorb_batch attention_absorb_batch;
    fn_attention_absorb_batch_dev attention_absorb_batch_dev;
    fn_attention_absorb_kvdev attention_absorb_kvdev;
    fn_attention_project_batch attention_project_batch;
    fn_attention_project_ragged attention_project_ragged;
    fn_attention_project_batch_dev attention_project_batch_dev;
    fn_attention_project_batch_dev_out attention_project_batch_dev_out;
    fn_pipe_add pipe_add;
    fn_pipe_axpy pipe_axpy;
    fn_pipe_residual pipe_residual;
    fn_pipe_zero pipe_zero;
    fn_pipe_decode_row pipe_decode_row;
    fn_pipe_gqa_decode pipe_gqa_decode;
    fn_pipe_alloc pipe_alloc;
    fn_pipe_copy2d pipe_copy2d;
    fn_pipe_download pipe_download;
    fn_pipe_copy pipe_copy;
    fn_pipe_free pipe_free;
    fn_pipe_gemm pipe_gemm;
    fn_pipe_peer_copy pipe_peer_copy;
    fn_pipe_rmsnorm pipe_rmsnorm;
    fn_pipe_rmsnorm_s pipe_rmsnorm_s;
    fn_group_resident_issue expert_group_resident_issue;
    fn_group_resident_take expert_group_resident_take;
    fn_pipe_router pipe_router;
    fn_pipe_rope pipe_rope;
    fn_pipe_rope_base pipe_rope_base;
    fn_pipe_rope_interleaved pipe_rope_interleaved;
    fn_pipe_qg_split pipe_qg_split;
    fn_pipe_rope_neox pipe_rope_neox;
    fn_pipe_sigmoid_mul pipe_sigmoid_mul;
    fn_pipe_sigmoid_scale pipe_sigmoid_scale;
    fn_pipe_gated_delta_decode pipe_gated_delta_decode;
    fn_pipe_gated_delta_decode_separate pipe_gated_delta_decode_separate;
    fn_pipe_qwen_topk pipe_qwen_topk;
    fn_pipe_rows_add pipe_rows_add;
    fn_pipe_scratch pipe_scratch;
    fn_pipe_silu_mul pipe_silu_mul;
    fn_pipe_sync pipe_sync;
    fn_pipe_upload pipe_upload;
    fn_shared_mlp_w4a16 shared_mlp_w4a16;
    fn_tensor_update tensor_update;
} g_cuda;

/* Resolve the DLL and all 11 symbols. Returns 1 on success, 0 otherwise.
 * Idempotent: the first call (success or fail) sticks; later calls are no-ops
 * that return the cached result. The engine treats a 0 return as "CUDA
 * unavailable" and falls back to the CPU path without aborting. */
static int coli_cuda_load(void){
    if(g_cuda.loaded) return g_cuda.available;
    g_cuda.loaded = 1;

    /* Load coli_cuda.dll from the engine's OWN directory, by absolute path —
     * never a bare name. LoadLibraryA("coli_cuda.dll") searches the current
     * working directory (and, without SafeDllSearchMode, other writable dirs):
     * an attacker who drops a coli_cuda.dll where the user launches glm.exe (or
     * inside a downloaded model directory the user cd's into) would get their
     * DllMain run at load — classic DLL hijacking -> arbitrary code execution.
     * Resolving the path next to glm.exe and loading THAT specific file with
     * LOAD_WITH_ALTERED_SEARCH_PATH anchors both the DLL and its dependency
     * search to the trusted install directory instead of the CWD. */
    char dllpath[MAX_PATH];
    DWORD mn = GetModuleFileNameA(NULL, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + sizeof("coli_cuda.dll") <= sizeof(dllpath)){
            strcpy(slash + 1, "coli_cuda.dll");
            g_cuda.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    if(!g_cuda.dll){
        /* fallback (GetModuleFileNameA praticamente non fallisce): cerca solo
         * nella dir dell'applicazione e in System32, MAI la CWD. */
        g_cuda.dll = LoadLibraryExA("coli_cuda.dll", NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if(!g_cuda.dll){
        fprintf(stderr, "[CUDA] coli_cuda.dll not found; GPU tier disabled "
                        "(CPU path remains active).\n");
        return 0;
    }

    #define RESOLVE(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_cuda.name){ \
            fprintf(stderr, "[CUDA] coli_cuda.dll missing symbol coli_cuda_" #name "\n"); \
            FreeLibrary(g_cuda.dll); g_cuda.dll=NULL; return 0; }

    RESOLVE(init,           fn_init)
    RESOLVE(shutdown,       fn_shutdown)
    RESOLVE(device_count,   fn_device_count)
    RESOLVE(device_at,      fn_device_at)
    RESOLVE(mem_info,       fn_mem_info)
    RESOLVE(stats,          fn_stats)
    RESOLVE(group_stats,    fn_group_stats)
    RESOLVE(expert_mlp,     fn_expert_mlp)
    RESOLVE(expert_group,   fn_expert_group)
    RESOLVE(expert_group_issue, fn_expert_group_issue)
    RESOLVE(expert_group_take, fn_expert_group_take)
    RESOLVE(attention_absorb, fn_attention_absorb)
    RESOLVE(tensor_upload,  fn_tensor_upload)
    RESOLVE(tensor_upload_g, fn_tensor_upload_g)
    RESOLVE(matmul,         fn_matmul)
    RESOLVE(ggml_matmul,    fn_ggml_matmul)
    RESOLVE(tensor_upload_ggml, fn_tensor_upload_ggml)
    RESOLVE(tensor_upload_sgguf, fn_tensor_upload_sgguf)
    RESOLVE(tensor_view_rows, fn_tensor_view_rows)
    RESOLVE(tensor_data, fn_tensor_data)
    RESOLVE(tensor_matmul_host, fn_tensor_matmul_host)
    RESOLVE(tensor_free,    fn_tensor_free)
    RESOLVE(tensor_bytes,   fn_tensor_bytes)
    RESOLVE(tensor_device,  fn_tensor_device)

    RESOLVE(attention_absorb_batch, fn_attention_absorb_batch)
    RESOLVE(attention_absorb_batch_dev, fn_attention_absorb_batch_dev)
    RESOLVE(attention_absorb_kvdev, fn_attention_absorb_kvdev)
    RESOLVE(attention_project_batch, fn_attention_project_batch)
    RESOLVE(attention_project_ragged, fn_attention_project_ragged)
    RESOLVE(attention_project_batch_dev, fn_attention_project_batch_dev)
    RESOLVE(attention_project_batch_dev_out, fn_attention_project_batch_dev_out)
    RESOLVE(pipe_add, fn_pipe_add)
    RESOLVE(pipe_axpy, fn_pipe_axpy)
    RESOLVE(pipe_residual, fn_pipe_residual)
    RESOLVE(pipe_zero, fn_pipe_zero)
    RESOLVE(pipe_decode_row, fn_pipe_decode_row)
    RESOLVE(pipe_gqa_decode, fn_pipe_gqa_decode)
    RESOLVE(pipe_alloc, fn_pipe_alloc)
    RESOLVE(pipe_copy2d, fn_pipe_copy2d)
    RESOLVE(pipe_download, fn_pipe_download)
    RESOLVE(pipe_copy, fn_pipe_copy)
    RESOLVE(pipe_free, fn_pipe_free)
    RESOLVE(pipe_gemm, fn_pipe_gemm)
    RESOLVE(pipe_peer_copy, fn_pipe_peer_copy)
    RESOLVE(pipe_rmsnorm, fn_pipe_rmsnorm)
    RESOLVE(pipe_rmsnorm_s, fn_pipe_rmsnorm_s)
    RESOLVE(expert_group_resident_issue, fn_group_resident_issue)
    RESOLVE(expert_group_resident_take, fn_group_resident_take)
    RESOLVE(pipe_router, fn_pipe_router)
    RESOLVE(pipe_rope, fn_pipe_rope)
    RESOLVE(pipe_rope_base, fn_pipe_rope_base)
    RESOLVE(pipe_rope_interleaved, fn_pipe_rope_interleaved)
    RESOLVE(pipe_qg_split, fn_pipe_qg_split)
    RESOLVE(pipe_rope_neox, fn_pipe_rope_neox)
    RESOLVE(pipe_sigmoid_mul, fn_pipe_sigmoid_mul)
    RESOLVE(pipe_sigmoid_scale, fn_pipe_sigmoid_scale)
    RESOLVE(pipe_gated_delta_decode, fn_pipe_gated_delta_decode)
    RESOLVE(pipe_gated_delta_decode_separate, fn_pipe_gated_delta_decode_separate)
    RESOLVE(pipe_qwen_topk, fn_pipe_qwen_topk)
    RESOLVE(pipe_rows_add, fn_pipe_rows_add)
    RESOLVE(pipe_scratch, fn_pipe_scratch)
    RESOLVE(pipe_silu_mul, fn_pipe_silu_mul)
    RESOLVE(pipe_sync, fn_pipe_sync)
    RESOLVE(pipe_upload, fn_pipe_upload)
    RESOLVE(shared_mlp_w4a16, fn_shared_mlp_w4a16)
    RESOLVE(tensor_update, fn_tensor_update)
    #undef RESOLVE

    g_cuda.available = 1;
    return 1;
}

/* ---- Public wrappers: match backend_cuda.h signatures exactly.
 * Each forwards to the resolved pointer; if the DLL never loaded, return the
 * "not initialized" result the engine already handles (init returns 0, matmul
 * returns 0 so the caller marks the tensor cuda_failed and uses CPU). ---- */

int coli_cuda_init(const int *devices, int count){
    if(!coli_cuda_load()) return 0;
    return g_cuda.init(devices, count);
}

void coli_cuda_shutdown(void){
    if(g_cuda.available && g_cuda.shutdown) g_cuda.shutdown();
}

int coli_cuda_device_count(void){
    if(!g_cuda.available) return 0;
    return g_cuda.device_count();
}

int coli_cuda_device_at(int index){
    if(!g_cuda.available) return -1;
    return g_cuda.device_at(index);
}

int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_cuda.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_cuda.mem_info(device, free_bytes, total_bytes);
}

void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_cuda.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_cuda.stats(device, tensor_count, tensor_bytes);
}

void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_cuda.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_cuda.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_mlp(gate, up, down, y, x, S);
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs, const int *rows, int count,
                           float *y, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group(gates, ups, downs, rows, count, y, x);
}

int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                 ColiCudaTensor *const *ups,
                                 ColiCudaTensor *const *downs,
                                 const int *rows, int count, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group_issue(gates, ups, downs, rows, count, x);
}

const float *coli_cuda_expert_group_take(int device){
    if(!g_cuda.available) return NULL;
    return g_cuda.expert_group_take(device);
}

int coli_cuda_attention_absorb(ColiCudaTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_cuda.available) return 0;
    return g_cuda.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device, int gs){
    if(!g_cuda.available || !g_cuda.tensor_upload_g){ return 0; }
    return g_cuda.tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs);
}

int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs){
    if(!g_cuda.available) return 0;
    return g_cuda.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device, gs);
}

int coli_cuda_ggml_matmul(float *y,const float *x,const void *weights,
                          uint32_t dtype,uint64_t weight_bytes,
                          int S,int I,int O,int device){
    if(!g_cuda.available || !g_cuda.ggml_matmul) return 0;
    return g_cuda.ggml_matmul(y,x,weights,dtype,weight_bytes,S,I,O,device);
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(g_cuda.available && g_cuda.tensor_free) g_cuda.tensor_free(tensor);
}

size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_bytes(tensor);
}

int coli_cuda_tensor_device(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return -1;
    return g_cuda.tensor_device(tensor);
}

/* ---- #111 pipeline wrappers ---- */


/* ---- #111 pipeline wrappers (see header for semantics) ---- */

int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch(kv_b, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch_dev(kv_b_shard, ctx_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_tensor_upload_ggml(ColiCudaTensor **tensor,const void *weights,uint32_t dtype,
        uint64_t weight_bytes,int I,int O,int device){
    if(!g_cuda.available)return 0;return g_cuda.tensor_upload_ggml(tensor,weights,dtype,weight_bytes,I,O,device);
}
int coli_cuda_tensor_upload_sgguf(ColiCudaTensor **tensor,const uint8_t *block_offsets_le,
        const uint8_t *blocks,uint64_t first_block,uint64_t block_count,uint32_t codec_id,
        uint32_t sparse_layout,uint32_t offset_width,
        uint16_t auxiliary_bytes_per_block,uint16_t retained_value_bits,
        int I,int O,int device){
    if(!g_cuda.available)return 0;
    return g_cuda.tensor_upload_sgguf(tensor,block_offsets_le,blocks,first_block,block_count,
                                      codec_id,sparse_layout,offset_width,
                                      auxiliary_bytes_per_block,retained_value_bits,I,O,device);
}
int coli_cuda_tensor_view_rows(ColiCudaTensor *base,uint64_t first_row,uint64_t row_count,ColiCudaTensor **view){
    if(!g_cuda.available)return 0;return g_cuda.tensor_view_rows(base,first_row,row_count,view);
}
const void *coli_cuda_tensor_data(const ColiCudaTensor *tensor){
    if(!g_cuda.available)return NULL;return g_cuda.tensor_data(tensor);
}
int coli_cuda_tensor_matmul_host(ColiCudaTensor *tensor,float *y,const float *x,int S){
    if(!g_cuda.available)return 0;return g_cuda.tensor_matmul_host(tensor,y,x,S);
}

int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale){
    if(!coli_cuda_load()) return 0;
    return g_cuda.attention_project_ragged(kv_b,o_proj,out,q,keys,latent,rope,lengths,
        S,H,Q,R,V,K,max_t,attention_scale);
}

int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev(kv_b, o_proj, out, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev_out(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_add(device, x_dev, t_dev, n);
}
int coli_cuda_pipe_axpy(int device,float *y_dev,const float *x_dev,float alpha,size_t n){
    if(!g_cuda.available)return 0;return g_cuda.pipe_axpy(device,y_dev,x_dev,alpha,n);
}
int coli_cuda_pipe_residual(int device,float *y_dev,const float *base_dev,const float *delta_dev,float alpha,size_t n){
    if(!g_cuda.available)return 0;return g_cuda.pipe_residual(device,y_dev,base_dev,delta_dev,alpha,n);
}
int coli_cuda_pipe_zero(int device,float *x_dev,size_t n){
    if(!g_cuda.available)return 0;return g_cuda.pipe_zero(device,x_dev,n);
}
int coli_cuda_pipe_decode_row(ColiCudaTensor *tensor,uint64_t row,float *out_dev,float scale){
    if(!g_cuda.available)return 0;return g_cuda.pipe_decode_row(tensor,row,out_dev,scale);
}
int coli_cuda_pipe_gqa_decode(int device,float *out_dev,const float *q_dev,const float *k_dev,
        const float *v_dev,float *kc,float *vc,float *scores,int pos,int context_capacity,
        int H,int HK,int D,float scale){
    if(!g_cuda.available)return 0;return g_cuda.pipe_gqa_decode(device,out_dev,q_dev,k_dev,v_dev,kc,vc,scores,pos,context_capacity,H,HK,D,scale);
}


void * coli_cuda_pipe_alloc(int device,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_alloc(device, bytes);
}

int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy2d(device, dst, dpitch, src, spitch, width, height);
}

int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_download(device, src, dst, bytes);
}
int coli_cuda_pipe_copy(int device,void *dst,const void *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy(device, dst, src, bytes);
}

void coli_cuda_pipe_free(int device,void *p){
    if(!g_cuda.available){ return; }
    g_cuda.pipe_free(device, p);
}

int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_gemm(t, y_dev, x_dev, S);
}

int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_peer_copy(dst_dev, dst, src_dev, src, bytes);
}

int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm(device, y_dev, x_dev, w_dev, S, D, eps);
}

int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,ColiCudaTensor *const *downs,const float *weights,int count,int home_device,const float *x_src_dev,float *partial_slot_dev){
    if(!g_cuda.available || !g_cuda.expert_group_resident_issue){ return 0; }
    return g_cuda.expert_group_resident_issue(gates, ups, downs, weights, count, home_device, x_src_dev, partial_slot_dev);
}

int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,float *slots_dev,float *acc_dev,int D){
    if(!g_cuda.available || !g_cuda.expert_group_resident_take){ return 0; }
    return g_cuda.expert_group_resident_take(home_device, devices, n_issued, slots_dev, acc_dev, D);
}

int coli_cuda_pipe_router(int device,const float *x_dev,const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,float topp,int norm_topk,float routed_scale,int *idx_host,float *w_host,int *keff_host){
    if(!g_cuda.available || !g_cuda.pipe_router){ return 0; }
    return g_cuda.pipe_router(device, x_dev, rw_dev, rb_dev, D, E, Ksel, topp, norm_topk, routed_scale, idx_host, w_host, keff_host);
}

int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm_s(device, y_dev, x_dev, w_dev, S, D, eps, xstride, ystride);
}

int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope(device, v_dev, pos_dev, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope_base(device, v_dev, pos_base, rows, stride, offset, R, heads, theta);
}
int coli_cuda_pipe_rope_interleaved(int device,float *v_dev,int position,int n_heads,int head_dim,int rope_dims,float theta){
    if(!g_cuda.available)return 0;return g_cuda.pipe_rope_interleaved(device,v_dev,position,n_heads,head_dim,rope_dims,theta);
}
int coli_cuda_pipe_qg_split(int device,float *q_dev,float *gate_dev,const float *qg_dev,int n_heads,int head_dim){
    if(!g_cuda.available)return 0;return g_cuda.pipe_qg_split(device,q_dev,gate_dev,qg_dev,n_heads,head_dim);
}
int coli_cuda_pipe_rope_neox(int device,float *v_dev,int position,int n_heads,int head_dim,int rope_dims,float theta){
    if(!g_cuda.available)return 0;return g_cuda.pipe_rope_neox(device,v_dev,position,n_heads,head_dim,rope_dims,theta);
}
int coli_cuda_pipe_sigmoid_mul(int device,float *x_dev,const float *gate_dev,size_t n){
    if(!g_cuda.available)return 0;return g_cuda.pipe_sigmoid_mul(device,x_dev,gate_dev,n);
}
int coli_cuda_pipe_sigmoid_scale(int device,float *x_dev,const float *logit_dev,size_t n){
    if(!g_cuda.available)return 0;return g_cuda.pipe_sigmoid_scale(device,x_dev,logit_dev,n);
}
int coli_cuda_pipe_gated_delta_decode(int device,float *out_dev,const float *qkv_dev,const float *z_dev,const float *ba_dev,const float *conv_weight_dev,const float *dt_bias_dev,const float *a_dev,const float *norm_weight_dev,float *conv_state_dev,float *recurrent_state_dev,int n_key_heads,int n_value_heads,int head_dim,int conv_kernel,float eps){
    if(!g_cuda.available)return 0;return g_cuda.pipe_gated_delta_decode(device,out_dev,qkv_dev,z_dev,ba_dev,conv_weight_dev,dt_bias_dev,a_dev,norm_weight_dev,conv_state_dev,recurrent_state_dev,n_key_heads,n_value_heads,head_dim,conv_kernel,eps);
}
int coli_cuda_pipe_gated_delta_decode_separate(int device,float *out_dev,const float *qkv_dev,const float *z_dev,const float *beta_dev,const float *alpha_dev,const float *conv_weight_dev,const float *dt_bias_dev,const float *a_dev,const float *norm_weight_dev,float *conv_state_dev,float *recurrent_state_dev,int n_key_heads,int n_value_heads,int head_dim,int conv_kernel,float eps){
    if(!g_cuda.available)return 0;return g_cuda.pipe_gated_delta_decode_separate(device,out_dev,qkv_dev,z_dev,beta_dev,alpha_dev,conv_weight_dev,dt_bias_dev,a_dev,norm_weight_dev,conv_state_dev,recurrent_state_dev,n_key_heads,n_value_heads,head_dim,conv_kernel,eps);
}
int coli_cuda_pipe_qwen_topk(int device,const float *logits_dev,int E,int Ksel,int *idx_host,float *w_host){
    if(!g_cuda.available)return 0;return g_cuda.pipe_qwen_topk(device,logits_dev,E,Ksel,idx_host,w_host);
}


int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rows_add(device, x_dev, partial_dev, rows_dev, nrows, D);
}

float * coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_scratch(device, slot, bytes);
}

int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_silu_mul(device, gate_dev, up_dev, n);
}

int coli_cuda_pipe_sync(int device){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_sync(device);
}

int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_upload(device, dst, src, bytes);
}

int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.shared_mlp_w4a16(gate, up, down, y, x, S);
}

int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales){
    if(!g_cuda.available){ return 0; }
    return g_cuda.tensor_update(tensor, weights, scales);
}

#endif /* _WIN32 */
