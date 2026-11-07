#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <thread>
#include <chrono>
int main(){
 nvtxInitialize(nullptr);
 for(int i=0;i<8;i++){char name[64];std::snprintf(name,sizeof(name),"audit.cpu/%06d",i);nvtxRangePushA(name);std::this_thread::sleep_for(std::chrono::milliseconds(1));nvtxRangePop();}
 std::puts("known_NVTX_push=8,known_NVTX_pop=8,no_CUDA_calls=1");return 0;
}
