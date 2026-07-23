#ifndef COLIBRI_GGUF_TOKENIZER_H
#define COLIBRI_GGUF_TOKENIZER_H

#include "gguf_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiGgufTokenizer ColiGgufTokenizer;

int coli_gguf_tokenizer_load(ColiGgufTokenizer **out, const ColiGgufFile *gguf,
                             char *error, size_t error_size);
void coli_gguf_tokenizer_destroy(ColiGgufTokenizer *tok);
int coli_gguf_tokenizer_encode(ColiGgufTokenizer *tok, const char *text,
                               int *ids, int capacity);
int coli_gguf_tokenizer_decode(ColiGgufTokenizer *tok, const int *ids, int count,
                               char *out, int capacity);
int coli_gguf_tokenizer_id(const ColiGgufTokenizer *tok, const char *text);
int coli_gguf_tokenizer_eos(const ColiGgufTokenizer *tok);
int coli_gguf_tokenizer_bos(const ColiGgufTokenizer *tok);
int coli_gguf_tokenizer_is_control(const ColiGgufTokenizer *tok, int id);
int coli_gguf_tokenizer_vocab_size(const ColiGgufTokenizer *tok);

/* Concrete Granite 3.x chat formatting; no generic Jinja interpreter.
 * The tokenizer-aware form detects the known dated/non-dated Granite template. */
char *coli_gguf_tokenizer_format_granite_prompt(const ColiGgufTokenizer *tok,
                                                 const char *user_prompt);
char *coli_granite_format_prompt(const char *user_prompt);

#ifdef __cplusplus
}
#endif
#endif
