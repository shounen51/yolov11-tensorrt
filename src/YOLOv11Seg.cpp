#include "YOLOv11Seg.h"
#include "preprocess.h"
#include "Logger.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
// Long utility function implementations ----------------------------------
void utils::letterBox(const cv::Mat &image,
                      cv::Mat &outImage,
                      const cv::Size &newShape,
                      const cv::Scalar &color,
                      bool auto_,
                      bool scaleFill,
                      bool scaleUp,
                      int stride) {
    float r = std::min((float)newShape.height / (float)image.rows,
                       (float)newShape.width  / (float)image.cols);
    if (!scaleUp) r = std::min(r, 1.0f);

    int newW = static_cast<int>(std::round(image.cols * r));
    int newH = static_cast<int>(std::round(image.rows * r));

    int dw = newShape.width  - newW;
    int dh = newShape.height - newH;

    if (auto_) {
        dw = dw % stride;
        dh = dh % stride;
    } else if (scaleFill) {
        newW = newShape.width;
        newH = newShape.height;
        dw = 0;
        dh = 0;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    cv::copyMakeBorder(resized, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

BoundingBox utils::scaleCoords(const cv::Size &letterboxShape,
                               const BoundingBox &coords,
                               const cv::Size &originalShape,
                               bool p_Clip) {
    float gain = std::min((float)letterboxShape.height / (float)originalShape.height,
                          (float)letterboxShape.width  / (float)originalShape.width);

    int padW = static_cast<int>(std::round(((float)letterboxShape.width  - (float)originalShape.width  * gain) / 2.f));
    int padH = static_cast<int>(std::round(((float)letterboxShape.height - (float)originalShape.height * gain) / 2.f));

    BoundingBox ret;
    ret.x      = static_cast<int>(std::round(((float)coords.x      - (float)padW) / gain));
    ret.y      = static_cast<int>(std::round(((float)coords.y      - (float)padH) / gain));
    ret.width  = static_cast<int>(std::round((float)coords.width   / gain));
    ret.height = static_cast<int>(std::round((float)coords.height  / gain));

    if (p_Clip) {
        ret.x = utils::clamp(ret.x, 0, originalShape.width);
        ret.y = utils::clamp(ret.y, 0, originalShape.height);
        ret.width  = utils::clamp(ret.width,  0, originalShape.width  - ret.x);
        ret.height = utils::clamp(ret.height, 0, originalShape.height - ret.y);
    }
    return ret;
}

std::vector<cv::Scalar> utils::generateColors(const std::vector<std::string> &classNames, int seed) {
    static std::unordered_map<size_t, std::vector<cv::Scalar>> cache;
    size_t key = 0;
    for (const auto &name : classNames) {
        size_t h = std::hash<std::string>{}(name);
        key ^= (h + 0x9e3779b9 + (key << 6) + (key >> 2));
    }
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<cv::Scalar> colors;
    colors.reserve(classNames.size());
    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.emplace_back(cv::Scalar(dist(rng), dist(rng), dist(rng)));
    }
    cache[key] = colors;
    return colors;
}

void utils::NMSBoxes(const std::vector<BoundingBox> &boxes,
                     const std::vector<float> &scores,
                     float scoreThreshold,
                     float nmsThreshold,
                     std::vector<int> &indices) {
    indices.clear();
    if (boxes.empty()) return;

    std::vector<int> order;
    order.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (scores[i] >= scoreThreshold) order.push_back((int)i);
    }
    if (order.empty()) return;

    std::sort(order.begin(), order.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<float> areas(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        areas[i] = (float)(boxes[i].width * boxes[i].height);
    }

    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;

        indices.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j) {
            int idx2 = order[j];
            if (suppressed[idx2]) continue;

            const BoundingBox &a = boxes[idx];
            const BoundingBox &b = boxes[idx2];
            int interX1 = std::max(a.x, b.x);
            int interY1 = std::max(a.y, b.y);
            int interX2 = std::min(a.x + a.width,  b.x + b.width);
            int interY2 = std::min(a.y + a.height, b.y + b.height);
            int w = interX2 - interX1;
            int h = interY2 - interY1;
            if (w > 0 && h > 0) {
                float interArea = (float)(w * h);
                float unionArea = areas[idx] + areas[idx2] - interArea;
                float iou = (unionArea > 0.f) ? (interArea / unionArea) : 0.f;
                if (iou > nmsThreshold) suppressed[idx2] = true;
            }
        }
    }
}

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

ClothesSegDetector::ClothesSegDetector(const std::string &modelPath,
                 float confThreshold_,
                 nvinfer1::ILogger &logger) : confThreshold(confThreshold_) {
    loadEngine(modelPath, logger);
    allocateBuffers();
    classNames  = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames);
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
    ownsStream = true;
    // Default class groups
    upperClothClasses = {
        static_cast<int>(ClothesClass::UpperClothes),
        static_cast<int>(ClothesClass::Coat),
        static_cast<int>(ClothesClass::Dress)
    };
    lowerClothClasses = {
        static_cast<int>(ClothesClass::Pants),
        static_cast<int>(ClothesClass::Skirt)
    };
}

ClothesSegDetector::~ClothesSegDetector() {
    for (void* p : gpuBuffers) {
        if (p) cudaFree(p);
    }
    if (ownsStream && stream) cudaStreamDestroy(stream);
}

void ClothesSegDetector::loadEngine(const std::string &modelPath, nvinfer1::ILogger &logger) {
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

void ClothesSegDetector::allocateBuffers() {
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
                letterboxSize = cv::Size(inputW, inputH);
            }
        } else {
            outputBindings.push_back(i);
        }
    }
    if (inputBinding < 0 || outputBindings.empty()) {
        throw std::runtime_error("Failed to identify input/output bindings");
    }
}

void ClothesSegDetector::preprocess(uint8_t* gpuRgbBuffer, int im0W, int im0H, bool block) {
    // Ensure input binding is FP32 as expected by cuda_preprocess
    if (engine->getBindingDataType(inputBinding) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("cuda_preprocess expects float input buffer");
    }
    float* dst = static_cast<float*>(gpuBuffers[inputBinding]);
    this->im0W = im0W;
    this->im0H = im0H;
    cuda_preprocess(gpuRgbBuffer, im0W, im0H, dst, inputW, inputH, stream);
    if (block) cudaStreamSynchronize(stream);
}

void ClothesSegDetector::inference() {
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

void ClothesSegDetector::postprocess(std::vector<Segmentation>& output) {
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
    // Ensure we have at least two output tensors
    if (hostOutputs.size() < 2) {
        throw std::runtime_error("Expected at least 2 output tensors for segmentation model");
    }
    auto shape0 = engine->getBindingDimensions(outputBindings[0]); // [1, 32, maskH, maskW]
    auto shape1 = engine->getBindingDimensions(outputBindings[1]); // [1, 116, num_detections]
    const int numDetections = shape1.d[2]; // e.g. 8400

    const int maskH = shape0.d[2]; // 160
    const int maskW = shape0.d[3]; // 160

    const float* output0_ptr = outputPtrs[0];
    const float* output1_ptr = outputPtrs[1];

    const size_t num_features = shape1.d[1]; // e.g 80 class + 4 bbox parms + 32 seg masks = 116
    const int numClasses = static_cast<int>(num_features - 4 - 32); // Corrected number of classes
    // Constants
    constexpr int BOX_OFFSET = 0;          // xc,yc,w,h stored sequentially
    constexpr int CLASS_CONF_OFFSET = 4;   // class scores start
    const int MASK_COEFF_OFFSET = CLASS_CONF_OFFSET + numClasses; // mask coeffs start

    // Prepare prototype masks (32, maskH, maskW)
    std::vector<cv::Mat> prototypeMasks;
    prototypeMasks.reserve(32);
    for (int m = 0; m < 32; ++m) {
        const float* p = output0_ptr + m * maskH * maskW;
        cv::Mat proto(maskH, maskW, CV_32F, const_cast<float*>(p));
        prototypeMasks.emplace_back(proto.clone());
    }

    // Collect candidate boxes
    std::vector<BoundingBox> boxes;
    boxes.reserve(numDetections);
    std::vector<float> confidences;
    confidences.reserve(numDetections);
    std::vector<int> classIds;
    classIds.reserve(numDetections);
    std::vector<std::vector<float>> maskCoefficientsList;
    maskCoefficientsList.reserve(numDetections);

    for (int i = 0; i < numDetections; ++i) {
        float xc = output1_ptr[BOX_OFFSET * numDetections + i];
        float yc = output1_ptr[(BOX_OFFSET + 1) * numDetections + i];
        float w  = output1_ptr[(BOX_OFFSET + 2) * numDetections + i];
        float h  = output1_ptr[(BOX_OFFSET + 3) * numDetections + i];

        BoundingBox box{
            static_cast<int>(std::round(xc - w / 2.0f)),
            static_cast<int>(std::round(yc - h / 2.0f)),
            static_cast<int>(std::round(w)),
            static_cast<int>(std::round(h))
        };

        float maxConf = 0.f;
        int bestCls = -1;
        for (int c = 0; c < numClasses; ++c) {
            float conf = output1_ptr[(CLASS_CONF_OFFSET + c) * numDetections + i];
            if (conf > maxConf) {
                maxConf = conf;
                bestCls = c;
            }
        }
        if (maxConf < confThreshold) continue;

        boxes.push_back(box);
        confidences.push_back(maxConf);
        classIds.push_back(bestCls);

        std::vector<float> maskCoeffs(32);
        for (int m = 0; m < 32; ++m) {
            maskCoeffs[m] = output1_ptr[(MASK_COEFF_OFFSET + m) * numDetections + i];
        }
        maskCoefficientsList.emplace_back(std::move(maskCoeffs));
    }

    if (boxes.empty()) return;

    std::vector<int> nmsIndices;
    utils::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, nmsIndices);
    if (nmsIndices.empty()) return;

    output.reserve(nmsIndices.size());

    // Precompute scaling from letterbox to original
    cv::Size origSize = cv::Size(im0W, im0H);
    const float gain = std::min(static_cast<float>(letterboxSize.height) / origSize.height,
                                static_cast<float>(letterboxSize.width)  / origSize.width);
    const int scaledW = static_cast<int>(origSize.width * gain);
    const int scaledH = static_cast<int>(origSize.height * gain);
    const float padW = (letterboxSize.width  - scaledW) / 2.f;
    const float padH = (letterboxSize.height - scaledH) / 2.f;
    const float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
    const float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

    for (int idx : nmsIndices) {
        Segmentation seg;
        seg.box = boxes[idx];
        seg.conf = confidences[idx];
        seg.classId = classIds[idx];
        seg.box = utils::scaleCoords(letterboxSize, seg.box, origSize, true);

        const auto &maskCoeffs = maskCoefficientsList[idx];
        cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
        for (int m = 0; m < 32; ++m) finalMask += maskCoeffs[m] * prototypeMasks[m];
        finalMask = utils::sigmoid(finalMask);

        int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
        int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
        int x2 = static_cast<int>(std::round((letterboxSize.width - padW + 0.1f) * maskScaleX));
        int y2 = static_cast<int>(std::round((letterboxSize.height - padH + 0.1f) * maskScaleY));
        x1 = std::max(0, std::min(x1, maskW - 1));
        y1 = std::max(0, std::min(y1, maskH - 1));
        x2 = std::max(x1, std::min(x2, maskW));
        y2 = std::max(y1, std::min(y2, maskH));
        if (x2 <= x1 || y2 <= y1) continue;

        cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat croppedMask = finalMask(cropRect).clone();
        cv::Mat resizedMask;
        cv::resize(croppedMask, resizedMask, origSize, 0, 0, cv::INTER_LINEAR);
        cv::Mat binaryMask;
        cv::threshold(resizedMask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U);

        cv::Mat finalBinaryMask = cv::Mat::zeros(origSize, CV_8U);
        cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
        roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows);
        if (roi.area() > 0) binaryMask(roi).copyTo(finalBinaryMask(roi));
        seg.mask = finalBinaryMask;
        output.push_back(seg);
    }
}
void ClothesSegDetector::drawSegmentations(cv::Mat &image,
                                           const std::vector<Segmentation> &results,
                                           float maskAlpha) const {
    for (const auto &seg : results) {
        cv::Scalar color = classColors[seg.classId % classColors.size()];
        if (!seg.mask.empty()) {
            cv::Mat mask_gray;
            if (seg.mask.channels() == 3) cv::cvtColor(seg.mask, mask_gray, cv::COLOR_BGR2GRAY);
            else mask_gray = seg.mask.clone();

            cv::Mat mask_binary;
            cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

            cv::Mat colored_mask;
            cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
            colored_mask.setTo(color, mask_binary);
            cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
        }
    }
}
bool ClothesSegDetector::isUpperCloth(int classId) const {
    return upperClothClasses.find(classId) != upperClothClasses.end();
}

bool ClothesSegDetector::isLowerCloth(int classId) const {
    return lowerClothClasses.find(classId) != lowerClothClasses.end();
}

void ClothesSegDetector::setUpperClothClasses(const std::vector<int>& classIds) {
    upperClothClasses.clear();
    upperClothClasses.insert(classIds.begin(), classIds.end());
}

void ClothesSegDetector::setLowerClothClasses(const std::vector<int>& classIds) {
    lowerClothClasses.clear();
    lowerClothClasses.insert(classIds.begin(), classIds.end());
}