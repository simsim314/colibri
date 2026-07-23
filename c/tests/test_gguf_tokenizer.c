#include "../gguf_tokenizer.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void u32(FILE *f,uint32_t v){ for(int i=0;i<4;i++) fputc((v>>(8*i))&255,f); }
static void u64(FILE *f,uint64_t v){ for(int i=0;i<8;i++) fputc((v>>(8*i))&255,f); }
static void str(FILE *f,const char*s){ u64(f,strlen(s)); fwrite(s,1,strlen(s),f); }
static void kv_str(FILE*f,const char*k,const char*v){ str(f,k);u32(f,COLI_GGUF_TYPE_STRING);str(f,v); }
static void kv_u32(FILE*f,const char*k,uint32_t v){ str(f,k);u32(f,COLI_GGUF_TYPE_UINT32);u32(f,v); }
static void kv_bool(FILE*f,const char*k,int v){ str(f,k);u32(f,COLI_GGUF_TYPE_BOOL);fputc(v,f); }

static void fixture(const char *path){
    const char *t[]={"<|end_of_text|>","<|start_of_role|>","<|end_of_role|>","x","y","xy","1","2"};
    uint32_t ty[]={3,3,3,1,1,1,1,1};
    FILE*f=fopen(path,"wb"); assert(f);
    fwrite("GGUF",1,4,f);u32(f,3);u64(f,0);u64(f,8);
    kv_str(f,"general.architecture","granitemoe");
    kv_str(f,"tokenizer.ggml.pre","refact");
    kv_str(f,"tokenizer.chat_template","Knowledge Cutoff Date: April 2024. You are Granite.");
    str(f,"tokenizer.ggml.tokens");u32(f,COLI_GGUF_TYPE_ARRAY);u32(f,COLI_GGUF_TYPE_STRING);u64(f,8);
    for(int i=0;i<8;i++)str(f,t[i]);
    str(f,"tokenizer.ggml.token_type");u32(f,COLI_GGUF_TYPE_ARRAY);u32(f,COLI_GGUF_TYPE_UINT32);u64(f,8);
    for(int i=0;i<8;i++)u32(f,ty[i]);
    str(f,"tokenizer.ggml.merges");u32(f,COLI_GGUF_TYPE_ARRAY);u32(f,COLI_GGUF_TYPE_STRING);u64(f,0);
    kv_u32(f,"tokenizer.ggml.bos_token_id",0);
    kv_u32(f,"tokenizer.ggml.eos_token_id",0);
    kv_bool(f,"tokenizer.ggml.add_bos_token",0);
    while(ftell(f)%32)fputc(0,f);
    fclose(f);
}
int main(void){
    char path[128];snprintf(path,sizeof(path),"/tmp/coli_tok_%d.gguf",getpid());fixture(path);
    ColiGgufFile g;assert(coli_gguf_open(&g,path));
    ColiGgufTokenizer*t=NULL;char err[256];assert(coli_gguf_tokenizer_load(&t,&g,err,sizeof(err)));
    assert(coli_gguf_tokenizer_vocab_size(t)==8);assert(coli_gguf_tokenizer_eos(t)==0);
    int ids[8];int n=coli_gguf_tokenizer_encode(t,"xy12",ids,8);
    /* GGUF GPT-2 BPE does not use the tokenizer.json ignore_merges shortcut. */
    assert(n==4&&ids[0]==3&&ids[1]==4&&ids[2]==6&&ids[3]==7);
    n=coli_gguf_tokenizer_encode(t,"<|start_of_role|>xy",ids,8);
    assert(n==3&&ids[0]==1&&ids[1]==3&&ids[2]==4);
    char out[64];assert(coli_gguf_tokenizer_decode(t,ids,n,out,sizeof(out))>0);
    assert(strcmp(out,"<|start_of_role|>xy")==0);
    char*p=coli_gguf_tokenizer_format_granite_prompt(t,"Hi");assert(p);
    assert(strstr(p,"<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date"));
    assert(!strstr(p,"Today's Date: "));
    assert(strstr(p,"<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n"));
    assert(strcmp(p+strlen(p)-strlen("<|start_of_role|>assistant<|end_of_role|>"),
                  "<|start_of_role|>assistant<|end_of_role|>")==0);
    free(p);
    coli_gguf_tokenizer_destroy(t);coli_gguf_close(&g);unlink(path);
    puts("test_gguf_tokenizer: ok");return 0;
}
