#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
        const void *weights, const float *scales,
        int fmt, int I, int O, int device, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2, 4=grouped int4.
 * gs is the group size for fmt=4 (0 for all other formats).
 * The first successful call uploads W and its scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs);

/* Native Colibri/GGUF dtype matmul. `weights` points to encoded GGML rows
 * (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K). The encoded
 * bytes are staged to reusable VRAM scratch and decoded inside the kernel; no
 * persistent F32 weight copy is created. */
COLI_CUDA_DLLEXPORT int coli_cuda_ggml_matmul(float *y, const float *x,
                           const void *weights, uint32_t dtype,
                           uint64_t weight_bytes, int S, int I, int O, int device);

/* Native encoded GGML/GGUF tensor residency. The uploaded representation is
 * byte-identical to the mmap-backed source; no F32 weight copy is created. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_ggml(ColiCudaTensor **tensor,
                           const void *weights, uint32_t dtype,
                           uint64_t weight_bytes, int I, int O, int device);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_view_rows(ColiCudaTensor *base,
                           uint64_t first_row, uint64_t row_count,
                           ColiCudaTensor **view);
COLI_CUDA_DLLEXPORT const void *coli_cuda_tensor_data(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_matmul_host(ColiCudaTensor *tensor,
                           float *y, const float *x, int S);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Prefill-oriented shared expert path.  INT4 weights stay packed in global
 * memory, activations are converted to FP16 per tile, and Tensor Cores
 * accumulate into FP32.  Unlike COLI_CUDA_TC_INT4 this does not quantize the
 * activation to INT4. */
COLI_CUDA_DLLEXPORT int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up,
                               ColiCudaTensor *down, float *y,
                               const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
/* Async issue/take split of the group call below (Inc.4): issue launches on the
 * device stream and returns; take syncs and returns the pinned result rows (valid
 * until the next issue on that device). Small totals only (<=8 rows); one
 * outstanding issue per device. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                               ColiCudaTensor *const *ups,
                               ColiCudaTensor *const *downs,
                               const int *rows, int count, const float *x);
COLI_CUDA_DLLEXPORT const float *coli_cuda_expert_group_take(int device);

COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

/* Causal MLA absorption for S contiguous rows from one sequence.  The KV
 * arrays contain T rows ending at the final query; query s attends T-S+s+1
 * rows.  One transfer and one launch replace S host round-trips. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q,
                                     const float *latent,const float *rope,int S,
                                     int H,int Q,int R,int V,int K,int T,
                                     float attention_scale);

/* Same attention batch followed immediately by resident o_proj on the same
 * device.  Only the final [S,D] tensor crosses back to the host. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
                                      float *out,const float *q,const float *latent,
                                      const float *rope,int S,int H,int Q,int R,
                                      int V,int K,int T,float attention_scale);

COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* Replace a resident tensor's contents without reallocating its device slot. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                            const void *weights, const float *scales);

/* ---- resident-pipeline primitives (Inc.0): device-pointer entry points ---- */
COLI_CUDA_DLLEXPORT float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes);
COLI_CUDA_DLLEXPORT void *coli_cuda_pipe_alloc(int device,size_t bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_pipe_free(int device,void *p);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy(int device,void *dst,const void *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                           const float *w_dev,int S,int D,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows,
                        int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_axpy(int device,float *y_dev,const float *x_dev,float alpha,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_residual(int device,float *y_dev,const float *base_dev,
                           const float *delta_dev,float alpha,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_zero(int device,float *x_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_decode_row(ColiCudaTensor *tensor,uint64_t row,
                           float *out_dev,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gqa_decode(int device,float *out_dev,
                           const float *q_dev,const float *k_dev,const float *v_dev,
                           float *k_cache_dev,float *v_cache_dev,float *scores_dev,
                           int pos,int context_capacity,int n_heads,int n_kv_heads,
                           int head_dim,float attention_scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                            const int *rows_dev,int nrows,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                             const float *w_dev,int S,int D,float eps,
                             int xstride,int ystride);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                             int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope_interleaved(int device,float *v_dev,int position,
                             int n_heads,int head_dim,int rope_dims,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_qg_split(int device,float *q_dev,float *gate_dev,
        const float *qg_dev,int n_heads,int head_dim);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope_neox(int device,float *v_dev,int position,
                             int n_heads,int head_dim,int rope_dims,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_sigmoid_mul(int device,float *x_dev,const float *gate_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_sigmoid_scale(int device,float *x_dev,const float *logit_dev,size_t n);
/* One-token Gated DeltaNet update. qkv is family-major [all Q, all K, all V];
 * ba is grouped [beta..., alpha...] inside each key-head group. GGUF `a` is
 * already -exp(A_log). State is kept transposed per head: [column][key_dim]. */
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gated_delta_decode(int device,float *out_dev,
        const float *qkv_dev,const float *z_dev,const float *ba_dev,
        const float *conv_weight_dev,const float *dt_bias_dev,const float *a_dev,
        const float *norm_weight_dev,float *conv_state_dev,float *recurrent_state_dev,
        int n_key_heads,int n_value_heads,int head_dim,int conv_kernel,float eps);
/* Qwen3.5-MoE variant: beta and alpha are independent family-major
 * projections [n_value_heads], rather than Qwen3-Next's grouped BA tensor. */
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gated_delta_decode_separate(int device,float *out_dev,
        const float *qkv_dev,const float *z_dev,const float *beta_dev,const float *alpha_dev,
        const float *conv_weight_dev,const float *dt_bias_dev,const float *a_dev,
        const float *norm_weight_dev,float *conv_state_dev,float *recurrent_state_dev,
        int n_key_heads,int n_value_heads,int head_dim,int conv_kernel,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_take(int home_device,const int *devices,
        int n_issued,float *slots_dev,float *acc_dev,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host);
/* Qwen router selection from logits that are already resident on-device.
 * Qwen uses global softmax followed by top-k and selected-mass
 * renormalization; the global denominator cancels, so this kernel selects by
 * raw logit and normalizes exp(logit) over the selected experts. */
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_qwen_topk(int device,
        const float *logits_dev,int E,int Ksel,
        int *idx_host,float *w_host);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                          int spitch,int width,int height);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                             const float *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_sync(int device);

#ifdef __cplusplus
}
#endif

#endif
