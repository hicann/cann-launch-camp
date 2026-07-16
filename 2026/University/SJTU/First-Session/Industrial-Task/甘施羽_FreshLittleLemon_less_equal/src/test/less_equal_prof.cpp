// msprof 采集用：单形状多次调用，采集 device 侧 kernel 时间。
// 用法: le_prof <case>  其中 case ∈ {t1,t2,t3,t4,t5}
#include <acl/acl.h>
#include <aclnn_less_equal.h>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>

enum DType { DT_F16, DT_F32, DT_I32, DT_I8 };
static aclDataType AclType(DType t){switch(t){case DT_F16:return ACL_FLOAT16;case DT_F32:return ACL_FLOAT;case DT_I32:return ACL_INT32;case DT_I8:return ACL_INT8;}return ACL_FLOAT;}
static size_t TS(DType t){switch(t){case DT_F16:return 2;case DT_F32:return 4;case DT_I32:return 4;case DT_I8:return 1;}return 4;}
static std::vector<int64_t> Contig(const std::vector<int64_t>&s){std::vector<int64_t> st(s.size(),1);for(int i=(int)s.size()-2;i>=0;i--)st[i]=st[i+1]*s[i+1];return st;}
static int64_t Numel(const std::vector<int64_t>&s){int64_t n=1;for(auto d:s)n*=d;return n;}

static aclrtStream g_stream=nullptr;
static void Run(std::vector<int64_t> s1,std::vector<int64_t> s2,std::vector<int64_t> so,DType t,int iters){
    int64_t n1=Numel(s1),n2=Numel(s2),no=Numel(so);
    void *d1,*d2,*dy;
    aclrtMalloc(&d1,n1*TS(t),ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d2,n2*TS(t),ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dy,no,ACL_MEM_MALLOC_HUGE_FIRST);
    auto st1=Contig(s1),st2=Contig(s2),sto=Contig(so);
    aclTensor*x1=aclCreateTensor(s1.data(),s1.size(),AclType(t),st1.data(),0,ACL_FORMAT_ND,s1.data(),s1.size(),d1);
    aclTensor*x2=aclCreateTensor(s2.data(),s2.size(),AclType(t),st2.data(),0,ACL_FORMAT_ND,s2.data(),s2.size(),d2);
    aclTensor*y=aclCreateTensor(so.data(),so.size(),ACL_BOOL,sto.data(),0,ACL_FORMAT_ND,so.data(),so.size(),dy);
    uint64_t ws=0;aclOpExecutor*e=nullptr;
    aclnnLessEqualGetWorkspaceSize(x1,x2,y,&ws,&e);
    void*wsp=nullptr;if(ws>0)aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnLessEqual(wsp,ws,e,g_stream);aclrtSynchronizeStream(g_stream);
    for(int i=0;i<iters;i++){uint64_t w2=0;aclOpExecutor*e2=nullptr;aclnnLessEqualGetWorkspaceSize(x1,x2,y,&w2,&e2);aclnnLessEqual(wsp,w2,e2,g_stream);}
    aclrtSynchronizeStream(g_stream);
    if(wsp)aclrtFree(wsp);aclDestroyTensor(x1);aclDestroyTensor(x2);aclDestroyTensor(y);aclrtFree(d1);aclrtFree(d2);aclrtFree(dy);
}

int main(int argc,char**argv){
    std::string c = argc>1?argv[1]:"t1";
    aclInit(nullptr);aclrtSetDevice(0);aclrtCreateStream(&g_stream);
    int N=50;
    // 猜测的测试点代表形状（覆盖 dtype/维度/广播）
    if(c=="t1") Run({2048},{2048},{2048},DT_F32,N);              // 小 fp32 一维
    else if(c=="t2") Run({256,512},{256,512},{256,512},DT_F16,N); // 中 fp16 二维
    else if(c=="t3") Run({1024,1024},{1024,1024},{1024,1024},DT_F32,N); // 大 fp32
    else if(c=="t4") Run({4096},{4096},{4096},DT_I32,N);          // 小 int32
    else if(c=="t5") Run({2048,2048},{2048,2048},{2048,2048},DT_F16,N); // 大 fp16
    else if(c=="bc") Run({1024,1024},{1024},{1024,1024},DT_F32,N);      // 广播
    aclrtSynchronizeStream(g_stream);
    aclrtDestroyStream(g_stream);aclrtResetDevice(0);aclFinalize();
    return 0;
}
