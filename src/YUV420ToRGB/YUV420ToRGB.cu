#include "YUV420ToRGB.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"

__global__ void yuv420toRGBKernel(uint8_t* yuv420, int width, int height, uint8_t* rgb) {
    // sperate Y, U, and V planes
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (i >= pixels) return;
    uint8_t* y = yuv420 + i;
    uint8_t* u = yuv420 + width * height + (i / width / 2) * (width / 2)  + i % width / 2;
    uint8_t* v = yuv420 + width * height + (width * height / 4) + (i / width / 2) * (width / 2) + i % width / 2;
    int r = *y + 1.402 * (*v - 128);
    int g = *y - 0.344136 * (*u - 128) - 0.714136 * (*v - 128);
    int b = *y + 1.772 * (*u - 128);
    rgb[i * 3 + 0] = min(max(r, 0), 255);
    rgb[i * 3 + 1] = min(max(g, 0), 255);
    rgb[i * 3 + 2] = min(max(b, 0), 255);
}

void yuv420toRGBInPlace(uint8_t* yuv420, int width, int height, uint8_t* rgb, cudaStream_t stream) {
    // Launch kernel
    int pixels = width * height;
    int numThreads = 256;
    int numBlocks = ceil(pixels / (float)numThreads);
    yuv420toRGBKernel<<<numBlocks, numThreads, 0 , stream>>>(yuv420, width, height, rgb);
    // 等待 GPU 完成處理
    CUDA_CHECK(cudaStreamSynchronize(stream));
}