#include "../expert_scheduler.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { int eid; int payload; uint64_t used; } Slot;
typedef struct { int loads, evicts; } Ctx;

static int load_cb(void *v,int layer,int eid,void *slot,int demand){
    Ctx*c=(Ctx*)v;Slot*s=(Slot*)slot;(void)layer;(void)demand;c->loads++;s->payload=eid*10;return 1;
}
static void evict_cb(void*v,int layer,void*slot){Ctx*c=(Ctx*)v;(void)layer;c->evicts++;((Slot*)slot)->payload=0;}
static size_t bytes_cb(void*v,int layer,const void*slot){(void)v;(void)layer;(void)slot;return sizeof(Slot);}

int main(void){
    Slot pin[1]={{1,10,1}}, cache[2]={{-1,0,0},{-1,0,0}};
    int nc=0;uint64_t clock=1;uint32_t heat[8]={0},last[8]={0},ac=0;
    uint32_t usage[8]={0};
    ColiExpertLayerStore st={pin,1,cache,&nc,2,8,{sizeof(Slot),offsetof(Slot,eid),offsetof(Slot,used)},&clock,heat,last,usage,&ac};
    ColiExpertStorageOps ops={load_cb,evict_cb,bytes_cb};Ctx ctx={0};ColiExpertSchedulerStats stats={0};
    Slot*s=(Slot*)coli_expert_acquire(&st,0,1,1,1,&ops,&ctx,&stats);assert(s==&pin[0]);assert(stats.pin_hits==1);
    s=(Slot*)coli_expert_acquire(&st,0,2,1,1,&ops,&ctx,&stats);assert(s&&s->eid==2&&s->payload==20&&nc==1);
    s=(Slot*)coli_expert_acquire(&st,0,3,1,1,&ops,&ctx,&stats);assert(s&&s->eid==3&&nc==2);
    /* A backend circuit breaker must preserve hits and demand accounting but
     * must not evict/load on a miss while admissions are disabled. */
    int loads_before=ctx.loads,evicts_before=ctx.evicts;
    s=(Slot*)coli_expert_acquire_controlled(&st,0,3,1,0,1,&ops,&ctx,&stats);
    assert(s&&s->eid==3&&ctx.loads==loads_before&&ctx.evicts==evicts_before);
    s=(Slot*)coli_expert_acquire_controlled(&st,0,7,1,0,1,&ops,&ctx,&stats);
    assert(!s&&ctx.loads==loads_before&&ctx.evicts==evicts_before);
    assert(heat[7]==1&&usage[7]==1);
    /* Make expert 2 the LRU, then admit 4 and evict exactly one slot. */
    cache[0].used=1;cache[1].used=20;
    s=(Slot*)coli_expert_acquire(&st,0,4,1,1,&ops,&ctx,&stats);assert(s==&cache[0]&&s->eid==4);assert(ctx.evicts==1);
    assert(heat[1]==1&&heat[2]==1&&heat[3]==2&&heat[4]==1);
    assert(usage[1]==1&&usage[2]==1&&usage[3]==2&&usage[4]==1);
    /* The shared PILOT guard protects a warm LRU victim from a cold speculation. */
    cache[0].used=1;cache[1].used=20;heat[4]=20;last[4]=ac;heat[5]=0;last[5]=0;
    ColiExpertAdmission a;assert(!coli_expert_begin_admission(&st,5,1,1,&a));
    /* Guard-off preserves the old plain-LRU policy, and reservations prevent duplicates. */
    assert(coli_expert_begin_admission(&st,5,1,0,&a));assert(coli_expert_resident(&st,5,1));
    coli_expert_finish_admission(&st,&a,5,0);assert(!coli_expert_resident(&st,5,1));
    /* REPIN selection is the same LFRU choice for every storage format. */
    heat[1]=1;last[1]=1;heat[6]=100;last[6]=++ac;
    int pin_index=-1,candidate=-1;long gain=0;
    assert(coli_expert_repin_pick(&st,&pin_index,&candidate,&gain));
    assert(pin_index==0&&candidate==6&&gain>0);
    /* A cached REPIN candidate must move, not be duplicated across pin/LRU. */
    pin[0]=(Slot){1,10,1};cache[0]=(Slot){6,60,30};cache[1]=(Slot){3,30,20};nc=2;
    int ev0=ctx.evicts;
    assert(coli_expert_repin_promote_cached(&st,0,6,0,&ops,&ctx)==1);
    assert(pin[0].eid==6&&pin[0].payload==60);
    assert(cache[0].eid==-1&&cache[0].payload==0&&ctx.evicts==ev0+1);
    assert(coli_expert_lookup(&st,6,0).from_pin==1);

    {
        const char *path="tmp_expert_usage.txt";
        uint32_t row0[8]={0},row1[8]={0};uint32_t*rows[2]={row0,row1};
        row0[3]=7;row1[2]=11;row1[5]=4;
        assert(coli_expert_usage_save(path,rows,2,8));
        memset(row0,0,sizeof(row0));memset(row1,0,sizeof(row1));
        assert(coli_expert_usage_load(path,rows,2,8)==22);
        int ids[3]={-1,-1,-1};assert(coli_expert_usage_top(rows,2,8,ids,3)==3);
        assert(ids[0]==10&&ids[1]==3&&ids[2]==13);
        int64_t total=0;memset(ids,-1,sizeof(ids));
        assert(coli_expert_usage_top_file(path,2,8,ids,2,&total)==2&&total==22);
        assert(ids[0]==10&&ids[1]==3);remove(path);
    }
    {
        uint32_t a[4]={100,90,10,1},b[4]={80,70,2,1};
        uint32_t c[4]={60,50,40,1},d[4]={30,20,10,1};
        uint32_t *rows[4]={a,b,c,d};
        int ids[6]={-1,-1,-1,-1,-1,-1};
        int selected[2]={-1,-1},counts[4]={0};
        int n=coli_expert_usage_focus(rows,4,4,2,ids,4,selected,2,counts);
        assert(n==4);
        assert(selected[0]==0&&selected[1]==1);
        assert(counts[0]==2&&counts[1]==2&&counts[2]==0&&counts[3]==0);
        assert(ids[0]==0&&ids[1]==4&&ids[2]==1&&ids[3]==5);
        memset(ids,-1,sizeof(ids));memset(counts,0,sizeof(counts));selected[0]=selected[1]=-1;
        n=coli_expert_usage_focus(rows,4,4,1,ids,3,selected,2,counts);
        assert(n==3&&selected[0]==0&&counts[0]==3);
        assert(ids[0]==0&&ids[1]==1&&ids[2]==2);
        memset(ids,-1,sizeof(ids));memset(counts,0,sizeof(counts));
        int selected4[4]={-1,-1,-1,-1};
        n=coli_expert_usage_focus(rows,4,4,4,ids,4,selected4,4,counts);
        assert(n==4);
        for(int l=0;l<4;l++)assert(counts[l]==1);
    }
    {
        const char *path="tmp_expert_pairs.txt";FILE*f=fopen(path,"w");assert(f);
        fputs("COLIPAIRS 1 2\n0 1 2 5:3.0 6:1.0\n0 1 3 6:4.0 5:1.0\n",f);fclose(f);
        ColiExpertCoupling cp={0};long used=0;assert(coli_expert_coupling_load(&cp,path,2,8,&used));assert(used==2);
        int routed[2]={2,3},pred[2]={-1,-1};int pn=coli_expert_coupling_predict(&cp,0,1,routed,2,pred,2);
        assert(pn==2&&pred[0]==6&&pred[1]==5);coli_expert_coupling_destroy(&cp);remove(path);
    }
    printf("test_expert_scheduler: one policy for pin/LRU/admission/eviction/REPIN/coupling/usage ok\n");
    return 0;
}
