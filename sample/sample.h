#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

struct Detection {
    float conf;
    int classId; // renamed from class_id
    cv::Rect bbox; // ensure cv::Rect namespace
};

class YOLOv11 {
public:
    YOLOv11(const std::string &modelPath,
            float confThreshold,
            nvinfer1::ILogger &logger);
    ~YOLOv11();

    // gpuRgbBuffer already contains properly formatted / normalized data
    // im0W / im0H: original image size
    void preprocess(uint8_t* gpuRgbBuffer, int im0W, int im0H, bool block = true);

    void inference();

    void postprocess(std::vector<Detection>& output);

    float getConfThreshold() const { return confThreshold; }
    cudaStream_t getStream() const { return stream; }

    int inputW{0}; // renamed from input_w
    int inputH{0}; // renamed from input_h

private:
    void loadEngine(const std::string &modelPath, nvinfer1::ILogger &logger);
    void allocateBuffers();

private:
    float confThreshold{0.f};

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream{nullptr};

    std::vector<void*> gpuBuffers;
    std::vector<size_t> bufferBytes;
    int inputBinding{-1};
    std::vector<int> outputBindings;

    bool ownsStream{false};
};
