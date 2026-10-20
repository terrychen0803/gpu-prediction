#include <cuda_runtime.h>
#include <stdint.h>

__global__ void e2_empty_kernel() { asm volatile("" ::: "memory"); }
__global__ void e2_short_kernel(unsigned int *counter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(counter, 1u);
}
__global__ void e2_copy_kernel(const uint32_t *src, uint32_t *dst, size_t words) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}
__global__ void e2_read_kernel(const uint32_t *src, unsigned int *sum, size_t words) {
    unsigned int local = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
        local += src[i];
    atomicAdd(sum, local);
}
__global__ void e2_write_kernel(uint32_t *dst, size_t words) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
        dst[i] = 0xA5A5A5A5u;
}

extern "C" int e2_launch_empty(cudaStream_t stream) {
    e2_empty_kernel<<<1, 1, 0, stream>>>();
    return static_cast<int>(cudaGetLastError());
}
extern "C" int e2_launch_short(void *counter, cudaStream_t stream) {
    e2_short_kernel<<<1, 1, 0, stream>>>(static_cast<unsigned int *>(counter));
    return static_cast<int>(cudaGetLastError());
}
extern "C" int e2_launch_memory(int kind, void *src, void *dst, void *counter,
                                  size_t bytes, cudaStream_t stream) {
    if (bytes % 4) return static_cast<int>(cudaErrorInvalidValue);
    size_t words = bytes / 4;
    if (kind == 0) e2_copy_kernel<<<256, 256, 0, stream>>>(static_cast<const uint32_t *>(src), static_cast<uint32_t *>(dst), words);
    else if (kind == 1) e2_read_kernel<<<256, 256, 0, stream>>>(static_cast<const uint32_t *>(src), static_cast<unsigned int *>(counter), words);
    else if (kind == 2) e2_write_kernel<<<256, 256, 0, stream>>>(static_cast<uint32_t *>(dst), words);
    else return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaGetLastError());
}
