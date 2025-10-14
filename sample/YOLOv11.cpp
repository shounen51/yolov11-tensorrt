#include "YOLOv11.h"
#include "preprocess.h"
#include <fstream>
#include <stdexcept>

namespace {
inline size_t typeSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
#if NV_TENSORRT_MAJOR >= 8
        case nvinfer1::DataType::kBOOL:  return 1;
#endif
        default: return 0;
    }
}
inline size_t volume(const nvinfer1::Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}
}

YOLOv11::YOLOv11(const std::string &modelPath,
                 float confThreshold_,
                 nvinfer1::ILogger &logger) : confThreshold(confThreshold_) {
    loadEngine(modelPath, logger);
    allocateBuffers();
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
    ownsStream = true;
}

YOLOv11::~YOLOv11() {
    for (void* p : gpuBuffers) {
        if (p) cudaFree(p);
    }
    if (ownsStream && stream) cudaStreamDestroy(stream);
}

void YOLOv11::loadEngine(const std::string &modelPath, nvinfer1::ILogger &logger) {
    std::ifstream f(modelPath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open engine file: " + modelPath);
    }
    f.seekg(0, f.end);
    size_t fsize = static_cast<size_t>(f.tellg());
    f.seekg(0, f.beg);
    std::vector<char> engineData(fsize);
    f.read(engineData.data(), fsize);
    f.close();

    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
    context.reset(engine->createExecutionContext());
    if (!context) throw std::runtime_error("createExecutionContext failed");
}

void YOLOv11::allocateBuffers() {
    int nb = engine->getNbBindings();
    gpuBuffers.resize(nb, nullptr);
    bufferBytes.resize(nb, 0);
    for (int i = 0; i < nb; ++i) {
        auto dims = engine->getBindingDimensions(i);
        auto dt = engine->getBindingDataType(i);
        size_t bytes = volume(dims) * typeSize(dt);
        bufferBytes[i] = bytes;
        if (cudaMalloc(&gpuBuffers[i], bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for binding " + std::to_string(i));
        }
        if (engine->bindingIsInput(i)) {
            inputBinding = i;
            // Expect dims like [N,C,H,W]
            if (dims.nbDims == 4) {
                inputH = dims.d[2];
                inputW = dims.d[3];
            }
        } else {
            outputBindings.push_back(i);
        }
    }
    if (inputBinding < 0 || outputBindings.empty()) {
        throw std::runtime_error("Failed to identify input/output bindings");
    }
}

void YOLOv11::preprocess(uint8_t* gpuRgbBuffer, int im0W, int im0H, bool block) {
    // Preprocessing data on gpu
    cuda_preprocess(gpuRgbBuffer, im0W, im0H, gpuBuffers[inputBinding], inputW, inputH, stream);
    if (block) cudaStreamSynchronize(stream);
}

void YOLOv11::inference() {
#if NV_TENSORRT_MAJOR < 10
    if (!context->enqueueV2(gpuBuffers.data(), stream, nullptr)) {
        throw std::runtime_error("enqueueV2 failed");
    }
#else
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }
#endif
}

void YOLOv11::postprocess(std::vector<Detection>& output) {
    output.clear(); // placeholder for detection decoding in future

    // Host-side storage of each float output
    std::vector<std::vector<float>> hostOutputs; // owns memory
    std::vector<float*> outputPtrs;              // raw pointers to hostOutputs[i].data()
    std::vector<size_t> outputElemCounts;        // element counts per output (float elements)

    hostOutputs.clear();
    outputPtrs.clear();
    outputElemCounts.clear();
    hostOutputs.reserve(outputBindings.size());
    outputPtrs.reserve(outputBindings.size());
    outputElemCounts.reserve(outputBindings.size());

    for (int idx : outputBindings) {
        nvinfer1::Dims d = engine->getBindingDimensions(idx);
        nvinfer1::DataType dt = engine->getBindingDataType(idx);
        if (dt != nvinfer1::DataType::kFLOAT) {
            continue; // skip non-float outputs
        }
        size_t elemCount = volume(d);
        size_t bytes = elemCount * sizeof(float);
        hostOutputs.emplace_back();
        hostOutputs.back().resize(elemCount);
        float* dstPtr = hostOutputs.back().data();
        if (cudaMemcpyAsync(dstPtr, gpuBuffers[idx], bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpyAsync failed for binding " + std::to_string(idx));
        }
        // sync to ensure data ready before exposing pointer
        cudaStreamSynchronize(stream);
        outputPtrs.push_back(dstPtr);
        outputElemCounts.push_back(elemCount);
    }
}
