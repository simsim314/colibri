#ifndef COLIBRI_GGUF_GPTOSS_H
#define COLIBRI_GGUF_GPTOSS_H

#ifdef __cplusplus
extern "C" {
#endif

int coli_gptoss_run_cli(int argc, char **argv);
float coli_gptoss_oai_swiglu(float gate, float linear);
void coli_gptoss_yarn_rotate(float *vector, int heads, int head_dim, int position,
                             const float *inv_freq, float concentration);
void coli_gptoss_attention_sink_reference(float *out, const float *q,
                                           const float *k_cache, const float *v_cache,
                                           const float *sinks, float *scores,
                                           int pos, int window, int n_heads,
                                           int n_kv_heads, int head_dim, float scale);
int coli_gptoss_topk_softmax(const float *logits, int n, int k,
                             int *indices, float *weights);

#ifdef __cplusplus
}
#endif

#endif
