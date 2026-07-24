#include "../backend_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
/* MSVC has no POSIX setenv/unsetenv */
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
static int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif

static int close_enough(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - want[i]) > 1e-4f) {
            std::fprintf(stderr, "mismatch %d: got %.6f want %.6f\n", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

static int relative_rms(const float *got,const float *want,int n,float limit){
    double err=0,ref=0; for(int i=0;i<n;i++){double d=got[i]-want[i];err+=d*d;ref+=(double)want[i]*want[i];}
    float r=(float)std::sqrt(err/(ref+1e-20));
    if(r>limit){std::fprintf(stderr,"relative RMS %.5f exceeds %.5f\n",r,limit);return 0;} return 1;
}

static float host_silu(float x){return x/(1.f+std::exp(-x));}
static float host_softplus(float x){if(x>20.f)return x;if(x<-20.f)return std::exp(x);return std::log1p(std::exp(x));}
static void qwen_gdn_oracle(float *out,const float *qkv,const float *z,const float *ba,
        const float *cw,const float *dt,const float *a,const float *nw,float *cs,float *rs,
        int nk,int nv,int D,int K,float eps){
    const int cd=(2*nk+nv)*D;float conv[128]={0};
    if(cd>(int)(sizeof(conv)/sizeof(conv[0])))std::abort();
    for(int c=0;c<cd;c++){float *st=cs+c*K;for(int j=0;j+1<K;j++)st[j]=st[j+1];st[K-1]=qkv[c];
        float v=0;for(int j=0;j<K;j++)v+=st[j]*cw[c*K+j];conv[c]=host_silu(v);}
    const int ratio=nv/nk;
    for(int vh=0;vh<nv;vh++){int kh=vh/ratio,sub=vh%ratio,base=kh*2*ratio;
        float q[16],k[16];double q2=0,k2=0;for(int i=0;i<D;i++){q[i]=conv[kh*D+i];k[i]=conv[nk*D+kh*D+i];q2+=(double)q[i]*q[i];k2+=(double)k[i]*k[i];}
        float qi=1.f/std::sqrt((float)q2+1e-6f),ki=1.f/std::sqrt((float)k2+1e-6f);
        for(int i=0;i<D;i++){q[i]*=qi;k[i]*=ki;}
        float b=1.f/(1.f+std::exp(-ba[base+sub]));float decay=std::exp(a[vh]*host_softplus(ba[base+ratio+sub]+dt[vh]));
        float core[16];for(int col=0;col<D;col++){float *Sc=rs+(vh*D+col)*D;float kv=0;for(int i=0;i<D;i++)kv+=Sc[i]*k[i];
            float delta=(conv[2*nk*D+vh*D+col]-decay*kv)*b,y=0;
            for(int i=0;i<D;i++){Sc[i]=decay*Sc[i]+k[i]*delta;y+=Sc[i]*q[i];}core[col]=y/std::sqrt((float)D);}
        double ss=0;for(int i=0;i<D;i++)ss+=(double)core[i]*core[i];float inv=1.f/std::sqrt((float)(ss/D)+eps);
        for(int i=0;i<D;i++)out[vh*D+i]=core[i]*inv*nw[i]*host_silu(z[vh*D+i]);
    }
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    if (coli_cuda_device_count() != ndev) return 1;
    int d0 = devices[0], d1 = devices[ndev > 1 ? 1 : 0];
    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    const float x[8] = {1, -2, 3, -4, 2, 1, -1, 0.5f};
    float got[4];

    /* Native GGUF dtype parity: encoded K-quant blocks are uploaded as-is and
     * dequantized inside the CUDA kernel. Each fixture decodes to 256 ones. */
    {
        float qx[256],qy[1]; for(int i=0;i<256;i++) qx[i]=1.f;
        uint8_t q8[34]={0},q4k[144]={0},q5k[176]={0},q6k[210]={0};
        auto put16=[](uint8_t*p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);};
        auto scales=[](uint8_t*p){for(int i=0;i<4;i++)p[i]=1;for(int i=8;i<12;i++)p[i]=1;};
        put16(q8,0x3c00);std::memset(q8+2,1,32);
        put16(q4k,0x3c00);scales(q4k+4);std::memset(q4k+16,0x11,128);
        put16(q5k,0x3c00);scales(q5k+4);std::memset(q5k+48,0x11,128);
        std::memset(q6k,0x11,128);std::memset(q6k+128,0xaa,64);
        std::memset(q6k+192,1,16);put16(q6k+208,0x3c00);
        float qx8[32];for(int i=0;i<32;i++)qx8[i]=1.f;
        if(!coli_cuda_ggml_matmul(qy,qx8,q8,8,sizeof(q8),1,32,1,d0)||
           std::fabs(qy[0]-32.f)>1e-4f)return 1;
        if(!coli_cuda_ggml_matmul(qy,qx,q4k,12,sizeof(q4k),1,256,1,d0)||
           std::fabs(qy[0]-256.f)>1e-3f)return 1;
        if(!coli_cuda_ggml_matmul(qy,qx,q5k,13,sizeof(q5k),1,256,1,d0)||
           std::fabs(qy[0]-256.f)>1e-3f)return 1;
        if(!coli_cuda_ggml_matmul(qy,qx,q6k,14,sizeof(q6k),1,256,1,d0)||
           std::fabs(qy[0]-256.f)>1e-3f)return 1;
        if(coli_cuda_ggml_matmul(qy,qx,q4k,12,sizeof(q4k)-1,1,256,1,d0))return 1;

        uint8_t bf16[16];
        for(int i=0;i<8;i++)put16(bf16+2*i,0x3f80);
        float bx[8]={1,1,1,1,1,1,1,1},by[1]={0};
        if(!coli_cuda_ggml_matmul(by,bx,bf16,30,sizeof(bf16),1,8,1,d0)||
           std::fabs(by[0]-8.f)>1e-4f)return 1;

        ColiCudaTensor *rq8=nullptr;
        float *dx8=(float*)coli_cuda_pipe_alloc(d0,sizeof(qx8));
        float *dy8=(float*)coli_cuda_pipe_alloc(d0,sizeof(float));
        float ry8=0.f;
        if(!dx8||!dy8||!coli_cuda_tensor_upload_ggml(&rq8,q8,8,sizeof(q8),32,1,d0)||
           !coli_cuda_pipe_upload(d0,dx8,qx8,sizeof(qx8))||
           !coli_cuda_pipe_gemm(rq8,dy8,dx8,1)||
           !coli_cuda_pipe_download(d0,dy8,&ry8,sizeof(ry8))||
           std::fabs(ry8-32.f)>1e-4f)return 1;
        coli_cuda_pipe_free(d0,dx8);coli_cuda_pipe_free(d0,dy8);
        coli_cuda_tensor_free(rq8);
    }

    /* Native resident tensor + zero-copy row view + device-resident pipeline. */
    {
        float qx[256];for(int i=0;i<256;i++)qx[i]=1.f;
        uint8_t w[288]={0};
        auto put16=[](uint8_t*p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);};
        auto one=[](uint8_t*p){for(int i=0;i<4;i++)p[4+i]=1;for(int i=8;i<12;i++)p[4+i]=1;std::memset(p+16,0x11,128);};
        put16(w,0x3c00);one(w);put16(w+144,0x3c00);one(w+144);
        ColiCudaTensor *base=nullptr,*view=nullptr;
        if(!coli_cuda_tensor_upload_ggml(&base,w,12,sizeof(w),256,2,d0)||
           !coli_cuda_tensor_view_rows(base,1,1,&view))return 1;
        float *dx=(float*)coli_cuda_pipe_alloc(d0,sizeof(qx));
        float *dy=(float*)coli_cuda_pipe_alloc(d0,2*sizeof(float));
        float *dr=(float*)coli_cuda_pipe_alloc(d0,256*sizeof(float));
        if(!dx||!dy||!dr||!coli_cuda_pipe_upload(d0,dx,qx,sizeof(qx))||
           !coli_cuda_pipe_gemm(base,dy,dx,1))return 1;
        float out[2];if(!coli_cuda_pipe_download(d0,dy,out,sizeof(out))||
           std::fabs(out[0]-256.f)>1e-3f||std::fabs(out[1]-256.f)>1e-3f)return 1;
        if(!coli_cuda_pipe_gemm(view,dy,dx,1)||!coli_cuda_pipe_download(d0,dy,out,sizeof(float))||
           std::fabs(out[0]-256.f)>1e-3f)return 1;
        if(!coli_cuda_pipe_decode_row(base,1,dr,2.f))return 1;
        float decoded[256];if(!coli_cuda_pipe_download(d0,dr,decoded,sizeof(decoded)))return 1;
        for(int i=0;i<256;i++)if(std::fabs(decoded[i]-2.f)>1e-5f)return 1;
        if(!coli_cuda_pipe_zero(d0,dr,256)||!coli_cuda_pipe_axpy(d0,dr,dx,3.f,256)||
           !coli_cuda_pipe_download(d0,dr,decoded,sizeof(decoded)))return 1;
        for(int i=0;i<256;i++)if(std::fabs(decoded[i]-3.f)>1e-5f)return 1;
        coli_cuda_pipe_free(d0,dx);coli_cuda_pipe_free(d0,dy);coli_cuda_pipe_free(d0,dr);
        coli_cuda_tensor_free(view);coli_cuda_tensor_free(base);
    }

    /* Qwen3-Next resident primitives: partial NeoX RoPE, sigmoid gates,
     * causal-conv state, and one-token Gated DeltaNet recurrence. */
    {
        {
            const int H=2,D=2;float raw[8]={1,2,10,20,3,4,30,40},q[4],g[4];
            float *dr=(float*)coli_cuda_pipe_alloc(d0,sizeof(raw));
            float *dq=(float*)coli_cuda_pipe_alloc(d0,sizeof(q));
            float *dg0=(float*)coli_cuda_pipe_alloc(d0,sizeof(g));
            if(!dr||!dq||!dg0||!coli_cuda_pipe_upload(d0,dr,raw,sizeof(raw))||
               !coli_cuda_pipe_qg_split(d0,dq,dg0,dr,H,D)||
               !coli_cuda_pipe_download(d0,dq,q,sizeof(q))||!coli_cuda_pipe_download(d0,dg0,g,sizeof(g)))return 1;
            float qw[4]={1,2,3,4},gw[4]={10,20,30,40};
            if(!close_enough(q,qw,4)||!close_enough(g,gw,4))return 1;
            coli_cuda_pipe_free(d0,dr);coli_cuda_pipe_free(d0,dq);coli_cuda_pipe_free(d0,dg0);
        }
        float hx[4]={1,2,3,4},gate[4]={0,20,-20,0},logit[1]={0};
        float *dx=(float*)coli_cuda_pipe_alloc(d0,sizeof(hx));
        float *dg=(float*)coli_cuda_pipe_alloc(d0,sizeof(gate));
        float *dl=(float*)coli_cuda_pipe_alloc(d0,sizeof(logit));
        if(!dx||!dg||!dl||!coli_cuda_pipe_upload(d0,dx,hx,sizeof(hx))||
           !coli_cuda_pipe_upload(d0,dg,gate,sizeof(gate))||
           !coli_cuda_pipe_rope_neox(d0,dx,0,1,4,4,10000.f)||
           !coli_cuda_pipe_sigmoid_mul(d0,dx,dg,4)||
           !coli_cuda_pipe_download(d0,dx,hx,sizeof(hx)))return 1;
        float swant[4]={.5f,2.f,0.f,2.f};if(!close_enough(hx,swant,4))return 1;
        float ones[4]={1,1,1,1};if(!coli_cuda_pipe_upload(d0,dx,ones,sizeof(ones))||
           !coli_cuda_pipe_upload(d0,dl,logit,sizeof(logit))||
           !coli_cuda_pipe_sigmoid_scale(d0,dx,dl,4)||
           !coli_cuda_pipe_download(d0,dx,hx,sizeof(hx)))return 1;
        float half[4]={.5f,.5f,.5f,.5f};if(!close_enough(hx,half,4))return 1;
        coli_cuda_pipe_free(d0,dx);coli_cuda_pipe_free(d0,dg);coli_cuda_pipe_free(d0,dl);

        const int D=2,NK=1,NV=1,K=2,CD=(2*NK+NV)*D;
        float qkv[CD]={1,0,1,0,2,3},z[2]={1,1},ba[2]={0,0};
        float cw[CD*K];for(int c=0;c<CD;c++){cw[c*K]=0.f;cw[c*K+1]=1.f;}
        float dt[1]={0},a[1]={-1},nw[2]={1,1};
        float *dqkv=(float*)coli_cuda_pipe_alloc(d0,sizeof(qkv));
        float *dz=(float*)coli_cuda_pipe_alloc(d0,sizeof(z));
        float *dba=(float*)coli_cuda_pipe_alloc(d0,sizeof(ba));
        float *dcw=(float*)coli_cuda_pipe_alloc(d0,sizeof(cw));
        float *ddt=(float*)coli_cuda_pipe_alloc(d0,sizeof(dt));
        float *da=(float*)coli_cuda_pipe_alloc(d0,sizeof(a));
        float *dnw=(float*)coli_cuda_pipe_alloc(d0,sizeof(nw));
        float *dcs=(float*)coli_cuda_pipe_alloc(d0,CD*K*sizeof(float));
        float *drs=(float*)coli_cuda_pipe_alloc(d0,NV*D*D*sizeof(float));
        float *doo=(float*)coli_cuda_pipe_alloc(d0,NV*D*sizeof(float));
        float hcs[CD*K]={0},hrs[NV*D*D]={0},want[2],gout[2],gstate[NV*D*D],gcstate[CD*K];
        qwen_gdn_oracle(want,qkv,z,ba,cw,dt,a,nw,hcs,hrs,NK,NV,D,K,1e-6f);
        if(!dqkv||!dz||!dba||!dcw||!ddt||!da||!dnw||!dcs||!drs||!doo||
           !coli_cuda_pipe_upload(d0,dqkv,qkv,sizeof(qkv))||!coli_cuda_pipe_upload(d0,dz,z,sizeof(z))||
           !coli_cuda_pipe_upload(d0,dba,ba,sizeof(ba))||!coli_cuda_pipe_upload(d0,dcw,cw,sizeof(cw))||
           !coli_cuda_pipe_upload(d0,ddt,dt,sizeof(dt))||!coli_cuda_pipe_upload(d0,da,a,sizeof(a))||
           !coli_cuda_pipe_upload(d0,dnw,nw,sizeof(nw))||!coli_cuda_pipe_zero(d0,dcs,CD*K)||
           !coli_cuda_pipe_zero(d0,drs,NV*D*D)||
           !coli_cuda_pipe_gated_delta_decode(d0,doo,dqkv,dz,dba,dcw,ddt,da,dnw,dcs,drs,NK,NV,D,K,1e-6f)||
           !coli_cuda_pipe_download(d0,doo,gout,sizeof(gout))||
           !coli_cuda_pipe_download(d0,drs,gstate,sizeof(gstate))||
           !coli_cuda_pipe_download(d0,dcs,gcstate,sizeof(gcstate)))return 1;
        if(!close_enough(gout,want,2)||!close_enough(gstate,hrs,NV*D*D)||!close_enough(gcstate,hcs,CD*K))return 1;
        float qkv2[CD]={0,1,0,1,1,-1};qwen_gdn_oracle(want,qkv2,z,ba,cw,dt,a,nw,hcs,hrs,NK,NV,D,K,1e-6f);
        if(!coli_cuda_pipe_upload(d0,dqkv,qkv2,sizeof(qkv2))||
           !coli_cuda_pipe_gated_delta_decode(d0,doo,dqkv,dz,dba,dcw,ddt,da,dnw,dcs,drs,NK,NV,D,K,1e-6f)||
           !coli_cuda_pipe_download(d0,doo,gout,sizeof(gout))||
           !coli_cuda_pipe_download(d0,drs,gstate,sizeof(gstate))||
           !coli_cuda_pipe_download(d0,dcs,gcstate,sizeof(gcstate)))return 1;
        if(!close_enough(gout,want,2)||!close_enough(gstate,hrs,NV*D*D)||!close_enough(gcstate,hcs,CD*K))return 1;

        /* Qwen3.5 uses independent beta and alpha projections instead of the
         * Qwen3-Next grouped BA vector. Verify identical math when the values
         * are arranged equivalently. */
        float beta[1]={0},alpha[1]={0};
        float *dbeta=(float*)coli_cuda_pipe_alloc(d0,sizeof(beta));
        float *dalpha=(float*)coli_cuda_pipe_alloc(d0,sizeof(alpha));
        std::memset(hcs,0,sizeof(hcs));std::memset(hrs,0,sizeof(hrs));
        qwen_gdn_oracle(want,qkv,z,ba,cw,dt,a,nw,hcs,hrs,NK,NV,D,K,1e-6f);
        if(!dbeta||!dalpha||!coli_cuda_pipe_upload(d0,dqkv,qkv,sizeof(qkv))||
           !coli_cuda_pipe_upload(d0,dbeta,beta,sizeof(beta))||
           !coli_cuda_pipe_upload(d0,dalpha,alpha,sizeof(alpha))||
           !coli_cuda_pipe_zero(d0,dcs,CD*K)||!coli_cuda_pipe_zero(d0,drs,NV*D*D)||
           !coli_cuda_pipe_gated_delta_decode_separate(d0,doo,dqkv,dz,dbeta,dalpha,dcw,ddt,da,dnw,dcs,drs,NK,NV,D,K,1e-6f)||
           !coli_cuda_pipe_download(d0,doo,gout,sizeof(gout))||
           !coli_cuda_pipe_download(d0,drs,gstate,sizeof(gstate))||
           !coli_cuda_pipe_download(d0,dcs,gcstate,sizeof(gcstate)))return 1;
        if(!close_enough(gout,want,2)||!close_enough(gstate,hrs,NV*D*D)||!close_enough(gcstate,hcs,CD*K))return 1;
        coli_cuda_pipe_free(d0,dbeta);coli_cuda_pipe_free(d0,dalpha);
        coli_cuda_pipe_free(d0,dqkv);coli_cuda_pipe_free(d0,dz);coli_cuda_pipe_free(d0,dba);
        coli_cuda_pipe_free(d0,dcw);coli_cuda_pipe_free(d0,ddt);coli_cuda_pipe_free(d0,da);
        coli_cuda_pipe_free(d0,dnw);coli_cuda_pipe_free(d0,dcs);coli_cuda_pipe_free(d0,drs);coli_cuda_pipe_free(d0,doo);
    }

    /* Standard GQA decode keeps K/V and attention entirely on the device. */
    {
        const int H=2,HK=1,D=2,C=3;float q[4]={1,0,0,1},k0[2]={1,0},v0[2]={2,3};
        float *dq=(float*)coli_cuda_pipe_alloc(d0,sizeof(q));
        float *dk=(float*)coli_cuda_pipe_alloc(d0,sizeof(k0));
        float *dv=(float*)coli_cuda_pipe_alloc(d0,sizeof(v0));
        float *doo=(float*)coli_cuda_pipe_alloc(d0,4*sizeof(float));
        float *kc=(float*)coli_cuda_pipe_alloc(d0,C*HK*D*sizeof(float));
        float *vc=(float*)coli_cuda_pipe_alloc(d0,C*HK*D*sizeof(float));
        float *sc=(float*)coli_cuda_pipe_alloc(d0,H*C*sizeof(float));
        if(!dq||!dk||!dv||!doo||!kc||!vc||!sc||
           !coli_cuda_pipe_upload(d0,dq,q,sizeof(q))||!coli_cuda_pipe_upload(d0,dk,k0,sizeof(k0))||
           !coli_cuda_pipe_upload(d0,dv,v0,sizeof(v0))||
           !coli_cuda_pipe_gqa_decode(d0,doo,dq,dk,dv,kc,vc,sc,0,C,H,HK,D,1.f))return 1;
        float out[4];if(!coli_cuda_pipe_download(d0,doo,out,sizeof(out)))return 1;
        const float want0[4]={2,3,2,3};if(!close_enough(out,want0,4))return 1;
        float k1[2]={0,1},v1[2]={4,5};
        if(!coli_cuda_pipe_upload(d0,dk,k1,sizeof(k1))||!coli_cuda_pipe_upload(d0,dv,v1,sizeof(v1))||
           !coli_cuda_pipe_gqa_decode(d0,doo,dq,dk,dv,kc,vc,sc,1,C,H,HK,D,1.f)||
           !coli_cuda_pipe_download(d0,doo,out,sizeof(out)))return 1;
        float a=std::exp(1.f)/(std::exp(1.f)+1.f),b=1.f-a;
        float want1[4]={a*2+b*4,a*3+b*5,b*2+a*4,b*3+a*5};
        if(!close_enough(out,want1,4))return 1;
        coli_cuda_pipe_free(d0,dq);coli_cuda_pipe_free(d0,dk);coli_cuda_pipe_free(d0,dv);
        coli_cuda_pipe_free(d0,doo);coli_cuda_pipe_free(d0,kc);coli_cuda_pipe_free(d0,vc);coli_cuda_pipe_free(d0,sc);
    }

    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float s8[2] = {0.5f, 2.0f};
    const float want8[4] = {-5.0f, -60.0f, 1.5f, 10.0f};
    ColiCudaTensor *t8 = nullptr;
    if (!coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d0)) return 1;
    if (coli_cuda_tensor_upload(&t8, q8, s8, 1, 5, 2, d0)) return 1;
    if (ndev > 1 && coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d1)) return 1;
    if (!coli_cuda_matmul(&t8, got, x, q8, s8, 1, 2, 4, 2, d0, 0) || !close_enough(got, want8, 4)) return 1;
    /* Cached tensor must stay callable without live host pointers
     * (CUDA_RELEASE_HOST slots null theirs after upload) — including
     * SUSTAINED reuse, not just the first call. */
    for (int rep = 0; rep < 64; rep++)
        if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
    /* A tensor uploaded from a TEMPORARY host buffer must survive the buffer
     * being scribbled and freed (the release-host lifecycle). */
    {
        int8_t *tmpw = static_cast<int8_t *>(std::malloc(8));
        float  *tmps = static_cast<float *>(std::malloc(2 * sizeof(float)));
        if (!tmpw || !tmps) return 2;
        for (int i = 0; i < 8; i++) tmpw[i] = q8[i];
        tmps[0] = s8[0]; tmps[1] = s8[1];
        ColiCudaTensor *tt = nullptr;
        if (!coli_cuda_tensor_upload(&tt, tmpw, tmps, 1, 4, 2, d0)) return 1;
        for (int i = 0; i < 8; i++) tmpw[i] = 99;
        std::free(tmpw); std::free(tmps);
        if (!coli_cuda_matmul(&tt, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
        coli_cuda_tensor_free(tt);
    }
    /* Upload failures must be graceful and must not corrupt accounting —
     * and must not poison LATER healthy launches (sticky-error regression). */
    {
        size_t c0 = 0, b0 = 0, c1 = 0, b1 = 0;
        coli_cuda_stats(-1, &c0, &b0);
        ColiCudaTensor *bad = nullptr;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 4, 2, 9999)) return 1;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 7, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, q8, nullptr, 1, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, nullptr, s8, 1, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 1 << 20, 1 << 24, d0)) return 1; /* ~16 TB */
        if (bad) return 1;
        coli_cuda_stats(-1, &c1, &b1);
        if (c0 != c1 || b0 != b1) return 1;
        /* healthy launch immediately after the failed allocation */
        if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
    }
    /* Fault injection hook: on/off, restores cleanly. */
    if (setenv("COLI_GPU_FAIL_AFTER", "0", 1)) return 2;
    if (coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0)) return 1;
    if (unsetenv("COLI_GPU_FAIL_AFTER")) return 2;
    if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
        !close_enough(got, want8, 4)) return 1;
    const int8_t q8b[8]={-1,-2,-3,-4, 1,-2,3,-4};
    const float s8b[2]={1.f,.5f},want8b[4]={10.f,15.f,-3.f,-2.5f};
    if(!coli_cuda_tensor_update(t8,q8b,s8b)||
       !coli_cuda_matmul(&t8,got,x,q8b,s8b,1,2,4,2,d0,0)||
       !close_enough(got,want8b,4))return 1;

    /* Rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first. */
    const uint8_t q4[4] = {0x70, 0xf8, 0xa9, 0xcb};
    const float s4[2] = {1.0f, 0.25f};
    const float want4[2] = {-34.0f, -2.5f};
    ColiCudaTensor *t4 = nullptr;
    if (!coli_cuda_matmul(&t4, got, x, q4, s4, 2, 1, 4, 2, d1, 0) || !close_enough(got, want4, 2)) return 1;

    const uint8_t q2[2] = {0xe4, 0x1b};
    const float s2[2] = {0.5f, 2.0f};
    const float want2[2] = {-2.0f, 12.0f};
    ColiCudaTensor *t2 = nullptr;
    if (!coli_cuda_matmul(&t2, got, x, q2, s2, 3, 1, 4, 2, d1, 0) || !close_enough(got, want2, 2)) return 1;

    const float wf[8] = {1, 0, -1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wantf[2] = {-10.0f, -1.0f};
    ColiCudaTensor *tf = nullptr;
    if (!coli_cuda_matmul(&tf, got, x, wf, nullptr, 0, 1, 4, 2, d0, 0) || !close_enough(got, wantf, 2)) return 1;

    const float eg[8] = {1,0,0,0, 0,1,0,0};
    const float eu[8] = {1,0,0,0, 0,1,0,0};
    const float ed[8] = {1,0, 0,1, 1,1, 1,-1};
    ColiCudaTensor *tg=nullptr,*tu=nullptr,*td=nullptr;
    if (!coli_cuda_tensor_upload_g(&tg,eg,nullptr,0,4,2,d0,0) ||
        !coli_cuda_tensor_upload_g(&tu,eu,nullptr,0,4,2,d0,0) ||
        !coli_cuda_tensor_upload_g(&td,ed,nullptr,0,2,4,d0,0)) return 1;
    float expert[8], want_expert[8];
    for(int s=0;s<2;s++){
        float a=x[s*4], b=x[s*4+1];
        a=(a/(1.0f+std::exp(-a)))*a; b=(b/(1.0f+std::exp(-b)))*b;
        want_expert[s*4]=a; want_expert[s*4+1]=b;
        want_expert[s*4+2]=a+b; want_expert[s*4+3]=a-b;
    }
    if (!coli_cuda_expert_mlp(tg,tu,td,expert,x,2) ||
        !close_enough(expert,want_expert,8)) return 1;
    ColiCudaTensor *gates[2]={tg,tg},*ups[2]={tu,tu},*downs[2]={td,td};
    int group_rows[2]={1,1}; float grouped[8];
    if (!coli_cuda_expert_group(gates,ups,downs,group_rows,2,grouped,x) ||
        !close_enough(grouped,want_expert,8)) return 1;

    const float aw[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    const float aq[4]={1,2,.5f,-.5f},al[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float ar[6]={1,0, 0,1, 1,1};float actx[2],aref[2];
    ColiCudaTensor *at=nullptr;if(!coli_cuda_tensor_upload_g(&at,aw,nullptr,0,4,4,d0,0))return 1;
    float score[3];for(int t=0;t<3;t++)score[t]=aq[0]*al[t*4]+aq[1]*al[t*4+1]+aq[2]*ar[t*2]+aq[3]*ar[t*2+1];
    float mx=score[0],z=0;for(int t=1;t<3;t++)mx=score[t]>mx?score[t]:mx;
    for(int t=0;t<3;t++){score[t]=std::exp(score[t]-mx);z+=score[t];}for(int t=0;t<3;t++)score[t]/=z;
    for(int v=0;v<2;v++){aref[v]=0;for(int t=0;t<3;t++)aref[v]+=score[t]*al[t*4+2+v];}
    if(!coli_cuda_attention_absorb(at,actx,aq,al,ar,1,2,2,2,4,3,1.f)||
       !close_enough(actx,aref,2))return 1;
    coli_cuda_tensor_free(at);

    /* Native s4 WMMA path: compare the quantized-activation result against the
       existing FP32-activation/s4-weight grouped implementation. */
    uint8_t w4[32*32/2]; float ws4[32], gx4[64], scalar4[64], tensor4[64];
    for(int i=0;i<(int)sizeof(w4);i++){
        int lo=((i%15)-7)&15,hi=(((i*3)%15)-7)&15;
        w4[i]=(uint8_t)(lo|(hi<<4));
    }
    for(int i=0;i<32;i++)ws4[i]=0.01f+(i%5)*0.002f;
    for(int i=0;i<64;i++)gx4[i]=std::sin((float)(i+1)*0.17f)*2.f;
    ColiCudaTensor *g4=nullptr,*u4=nullptr,*d4=nullptr;
    if(!coli_cuda_tensor_upload_g(&g4,w4,ws4,2,32,32,d0,0)||
       !coli_cuda_tensor_upload_g(&u4,w4,ws4,2,32,32,d0,0)||
       !coli_cuda_tensor_upload_g(&d4,w4,ws4,2,32,32,d0,0))return 1;
    ColiCudaTensor *gg4[2]={g4,g4},*ug4[2]={u4,u4},*dg4[2]={d4,d4};
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,scalar4,gx4))return 1;
    setenv("COLI_CUDA_TC_INT4","1",1);
    setenv("COLI_CUDA_TC_MIN_ROWS","1",1);
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,tensor4,gx4)||
       !relative_rms(tensor4,scalar4,64,0.30f))return 1;
    unsetenv("COLI_CUDA_TC_INT4");
    unsetenv("COLI_CUDA_TC_MIN_ROWS");
    coli_cuda_tensor_free(g4);coli_cuda_tensor_free(u4);coli_cuda_tensor_free(d4);

    /* Resident grouped execution for native GGUF Q4_K experts. This is the
       path used by focused Qwen layers; no weight conversion is permitted. */
    {
        const int D=256,I=256,C=2; const size_t rb=144,wb=rb*I;
        uint8_t *enc=(uint8_t*)std::calloc(1,wb); if(!enc)return 1;
        auto put16=[](uint8_t*p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);};
        for(int r=0;r<I;r++){
            uint8_t *row=enc+(size_t)r*rb;put16(row,0x3c00);
            for(int i=0;i<4;i++)row[4+i]=1;
            for(int i=8;i<12;i++)row[4+i]=1;
            std::memset(row+16,0x11,128);
        }
        ColiCudaTensor *ng[C]={},*nu[C]={},*nd[C]={};
        for(int c=0;c<C;c++)if(!coli_cuda_tensor_upload_ggml(&ng[c],enc,12,wb,D,I,d0)||
            !coli_cuda_tensor_upload_ggml(&nu[c],enc,12,wb,D,I,d0)||
            !coli_cuda_tensor_upload_ggml(&nd[c],enc,12,wb,I,D,d0))return 1;
        float hx[D];for(int i=0;i<D;i++)hx[i]=1.f;
        float *dx=(float*)coli_cuda_pipe_alloc(d0,sizeof(hx));
        float *slots=(float*)coli_cuda_pipe_alloc(d0,D*sizeof(float));
        float *acc=(float*)coli_cuda_pipe_alloc(d0,D*sizeof(float));
        float ww[C]={.25f,.75f};int devs[1]={d0};
        if(!dx||!slots||!acc||!coli_cuda_pipe_upload(d0,dx,hx,sizeof(hx))||
           !coli_cuda_expert_group_resident_issue(ng,nu,nd,ww,C,d0,dx,slots)||
           !coli_cuda_expert_group_resident_take(d0,devs,1,slots,acc,D))return 1;
        float out[D];if(!coli_cuda_pipe_download(d0,acc,out,sizeof(out)))return 1;
        float h=256.f/(1.f+std::exp(-256.f))*256.f,want=256.f*h;
        for(int i=0;i<D;i++)if(std::fabs(out[i]-want)>1e-4f*(std::fabs(want)+1.f))return 1;
        coli_cuda_pipe_free(d0,dx);coli_cuda_pipe_free(d0,slots);coli_cuda_pipe_free(d0,acc);
        for(int c=0;c<C;c++){coli_cuda_tensor_free(ng[c]);coli_cuda_tensor_free(nu[c]);coli_cuda_tensor_free(nd[c]);}
        std::free(enc);
    }

    uint64_t group_calls=0,group_experts=0,group_total_rows=0;
    coli_cuda_group_stats(&group_calls,&group_experts,&group_total_rows,nullptr,nullptr,nullptr);
    if(group_calls!=3||group_experts!=6||group_total_rows!=6) return 1;

    coli_cuda_stats(-1, &count, &bytes);
    if (count != 7 || bytes != 166) {
        std::fprintf(stderr, "unexpected CUDA stats: %zu tensors, %zu bytes\n", count, bytes);
        return 1;
    }
    if (coli_cuda_tensor_device(t8) != d0 || coli_cuda_tensor_device(tf) != d0 ||
        coli_cuda_tensor_device(t4) != d1 || coli_cuda_tensor_device(t2) != d1) return 1;
    coli_cuda_stats(d0, &count, &bytes);
    if (ndev > 1) {
        if (count != 5 || bytes != 144) return 1;
        coli_cuda_stats(d1, &count, &bytes);
        if (count != 2 || bytes != 22) return 1;
    } else if (count != 7 || bytes != 166) return 1;

    coli_cuda_tensor_free(t8);
    coli_cuda_tensor_free(t4);
    coli_cuda_tensor_free(t2);
    coli_cuda_tensor_free(tf);
    coli_cuda_tensor_free(tg);
    coli_cuda_tensor_free(tu);
    coli_cuda_tensor_free(td);
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    coli_cuda_shutdown();
    std::printf("cuda backend: native GGUF K-quants + q8/q4/q2/f32 correctness ok on %d device(s)\n", ndev);
    return 0;
}
