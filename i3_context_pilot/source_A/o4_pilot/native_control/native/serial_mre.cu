// Proposed diagnostic only. Compiled on GB10; no GPU execution in this analysis.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA failure %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 2;} } while(0)
__global__ void first(const float* a,float* b,unsigned long long cycles){
 unsigned long long begin=clock64();while(clock64()-begin<cycles){};int i=blockIdx.x*blockDim.x+threadIdx.x;b[i]=a[i]*2.f;
}
__global__ void second(const float* b,float* c,unsigned long long cycles){
 unsigned long long begin=clock64();while(clock64()-begin<cycles){};int i=blockIdx.x*blockDim.x+threadIdx.x;c[i]=b[i]+1.f;
}
int main(int argc,char**argv){
 bool ref=false;int count=8,warmup=8;unsigned long long cycles=20000;
 for(int i=1;i<argc;i++){
  if(!std::strcmp(argv[i],"--reference"))ref=true;
  else if(!std::strcmp(argv[i],"--measure")&&i+1<argc)count=std::atoi(argv[++i]);
  else if(!std::strcmp(argv[i],"--warmup")&&i+1<argc)warmup=std::atoi(argv[++i]);
  else {std::fprintf(stderr,"Usage: %s [--reference] [--measure 8] [--warmup 8]\n",argv[0]);return 2;}
 }
 if(count<1||count>16||warmup<1||warmup>16)return 2;
 float *a=nullptr,*b=nullptr,*c=nullptr;unsigned char*scratch=nullptr;std::vector<float>host(256,1.f);
 CHECK(cudaSetDevice(0));CHECK(cudaMalloc(&a,1024));CHECK(cudaMalloc(&b,1024));CHECK(cudaMalloc(&c,1024));CHECK(cudaMalloc(&scratch,64));CHECK(cudaMemcpy(a,host.data(),1024,cudaMemcpyHostToDevice));
 cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));CHECK(cudaEventRecord(start));CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));
 nvtxRangePushA("diag.setup");nvtxRangePop();
 for(int i=0;i<warmup;i++){first<<<4,64>>>(a,b,cycles);second<<<4,64>>>(b,c,cycles);CHECK(cudaMemsetAsync(scratch,0,64));CHECK(cudaMemsetAsync(scratch,0,0));CHECK(cudaDeviceSynchronize());}
 std::vector<float>times;CHECK(cudaProfilerStart());
 for(int i=0;i<count;i++){
  char name[96];std::snprintf(name,sizeof(name),"diag.iter/%06d",i);nvtxRangePushA(name);CHECK(cudaEventRecord(start));
  if(ref){std::snprintf(name,sizeof(name),"diag.op/%06d/first",i);nvtxRangePushA(name);}first<<<4,64>>>(a,b,cycles);CHECK(cudaPeekAtLastError());if(ref)nvtxRangePop();
  if(ref){std::snprintf(name,sizeof(name),"diag.op/%06d/second",i);nvtxRangePushA(name);}second<<<4,64>>>(b,c,cycles);CHECK(cudaPeekAtLastError());if(ref)nvtxRangePop();
  if(ref){std::snprintf(name,sizeof(name),"diag.op/%06d/memset64",i);nvtxRangePushA(name);}CHECK(cudaMemsetAsync(scratch,0,64));if(ref)nvtxRangePop();
  if(ref){std::snprintf(name,sizeof(name),"diag.op/%06d/memset0",i);nvtxRangePushA(name);}CHECK(cudaMemsetAsync(scratch,0,0));if(ref)nvtxRangePop();
  CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));times.push_back(ms*1000.f);nvtxRangePop();
 }
 CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));CHECK(cudaProfilerStop());
 // Correctness and host printing are outside captured scope.
 CHECK(cudaMemcpy(host.data(),c,1024,cudaMemcpyDeviceToHost));for(float x:host)if(x!=3.f){std::fprintf(stderr,"wrong output\n");return 3;}
 std::puts("iteration,event_elapsed_us,kernel_calls,positive_memset_bytes,zero_memset_bytes,reference");
 for(int i=0;i<count;i++)std::printf("%d,%.6f,2,64,0,%d\n",i,times[i],ref?1:0);
 CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));CHECK(cudaFree(scratch));CHECK(cudaDeviceReset());return 0;
}
