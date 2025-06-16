#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void yuv420toRGBInPlace(uint8_t* yuv420, int width, int height, uint8_t* rgb_buffer = nullptr, cudaStream_t stream = 0);