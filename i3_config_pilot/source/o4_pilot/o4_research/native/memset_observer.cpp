// Standalone CUPTI callback observer. Never load alongside Nsight/CUPTI tools.
#include <cuda_runtime_api.h>
#include <cupti.h>
// cupti.h already includes generated_cuda_runtime_api_meta.h (no include guard).
#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <sys/syscall.h>
struct Record { unsigned corr,site; unsigned long long count,ns; int value,rc,tid; char scope[160],name[80]; };
static Record records[32768];static size_t used=0;static bool overflow=false,active=false;static std::mutex lock;static CUpti_SubscriberHandle subscriber;
static thread_local char scope[160]="outside";
extern "C" void observer_scope(const char* s){std::snprintf(scope,sizeof(scope),"%s",s);}
static void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void* data){
 if(!active||domain!=CUPTI_CB_DOMAIN_RUNTIME_API)return;
 if(id!=CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020&&id!=CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000)return;
 auto cb=static_cast<const CUpti_CallbackData*>(data);Record row{};row.corr=cb->correlationId;row.site=cb->callbackSite;row.tid=syscall(SYS_gettid);row.rc=-1;
 timespec ts;clock_gettime(CLOCK_MONOTONIC_RAW,&ts);row.ns=1000000000ULL*ts.tv_sec+ts.tv_nsec;
 // Copy arguments NOW: functionParams must never be dereferenced after callback.
 if(id==CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020){auto p=static_cast<const cudaMemsetAsync_v3020_params*>(cb->functionParams);row.count=p->count;row.value=p->value;}
 else {auto p=static_cast<const cudaMemsetAsync_ptsz_v7000_params*>(cb->functionParams);row.count=p->count;row.value=p->value;}
 if(cb->callbackSite==CUPTI_API_EXIT&&cb->functionReturnValue)row.rc=*static_cast<const cudaError_t*>(cb->functionReturnValue);
 std::snprintf(row.scope,sizeof(row.scope),"%s",scope);std::snprintf(row.name,sizeof(row.name),"%s",cb->functionName);
 std::lock_guard<std::mutex> guard(lock);if(used<32768)records[used++]=row;else overflow=true;
}
extern "C" int observer_version(){uint32_t v=0;auto r=cuptiGetVersion(&v);return r==CUPTI_SUCCESS?int(v):-int(r);}
extern "C" int observer_start(){
 if(active)return -1;used=0;overflow=false;auto r=cuptiSubscribe(&subscriber,callback,nullptr);if(r!=CUPTI_SUCCESS)return int(r);
 active=true;r=cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020);if(r==CUPTI_SUCCESS)r=cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000);
 if(r!=CUPTI_SUCCESS){active=false;cuptiUnsubscribe(subscriber);}return int(r);
}
extern "C" int observer_stop(const char* path){
 if(!active)return -2;auto r=cuptiUnsubscribe(subscriber);active=false;if(r!=CUPTI_SUCCESS)return int(r);
 FILE* f=std::fopen(path,"wx");if(!f)return -3;
 std::fprintf(f,"correlation_id,site,count_bytes,value,return_code,tid,monotonic_raw_ns,scope,function\n");
 for(size_t i=0;i<used;i++){auto &x=records[i];std::fprintf(f,"%u,%u,%llu,%d,%d,%d,%llu,%s,%s\n",x.corr,x.site,x.count,x.value,x.rc,x.tid,x.ns,x.scope,x.name);}
 std::fclose(f);return overflow?-4:0;
}
