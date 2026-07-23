#define _GNU_SOURCE
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../gguf_f32_loader.h"

static void put_f32(uint8_t *p, float f) {
    uint32_t u; memcpy(&u,&f,4);
    p[0]=(uint8_t)u; p[1]=(uint8_t)(u>>8); p[2]=(uint8_t)(u>>16); p[3]=(uint8_t)(u>>24);
}

int main(void) {
    char path[] = "/tmp/coli-loader-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    uint8_t raw[48];
    for (int i=0;i<12;i++) put_f32(raw+4*i,(float)(i+1));
    assert(write(fd,raw,sizeof(raw)) == (ssize_t)sizeof(raw));

    ColiGgufFile file; memset(&file,0,sizeof(file));
    file.fd=fd; file.file_size=sizeof(raw);
    ColiGgufTensorInfo ti; memset(&ti,0,sizeof(ti));
    ti.name="three_d"; ti.n_dims=3; ti.dims[0]=3; ti.dims[1]=2; ti.dims[2]=2;
    ti.type=0; ti.absolute_offset=0;
    ColiF32Tensor out; char error[256];
    assert(coli_gguf_tensor_to_f32(&file,&ti,&out,error,sizeof(error)));
    assert(out.element_count==12 && out.dims[0]==3 && out.dims[2]==2);
    for(int i=0;i<12;i++) assert(fabsf(out.data[i]-(i+1))<1e-6f);
    coli_f32_tensor_destroy(&out);

    ti.dims[0]=4; ti.dims[1]=4; ti.dims[2]=1;
    assert(!coli_gguf_tensor_to_f32(&file,&ti,&out,error,sizeof(error)));
    assert(strstr(error,"exceeds"));

    close(fd); unlink(path);
    puts("test_gguf_f32_loader: ok");
    return 0;
}
