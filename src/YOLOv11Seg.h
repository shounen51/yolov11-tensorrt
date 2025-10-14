#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

enum class ClothesClass : int {
    Hat = 0,
    Hair = 1,
    Glove = 2,
    Sunglasses = 3,
    UpperClothes = 4,
    Dress = 5,
    Coat = 6,
    Socks = 7,
    Pants = 8,
    Jumpsuits = 9,
    Scarf = 10,
    Skirt = 11,
    Face = 12,
    LeftArm = 13,
    RightArm = 14,
    LeftLeg = 15,
    RightLeg = 16,
    LeftShoe = 17,
    RightShoe = 18
};

// Bounding box structure
struct BoundingBox {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    BoundingBox() = default;
    BoundingBox(int _x, int _y, int w, int h) : x(_x), y(_y), width(w), height(h) {}

    float area() const { return static_cast<float>(width * height); }
    BoundingBox intersect(const BoundingBox &other) const;
};

// Segmentation result
struct Segmentation {
    BoundingBox box;       // Bounding box
    float       conf{0.f}; // Confidence
    int         classId{0};
    cv::Mat     mask;      // Single-channel 8UC1 mask in full resolution
};

// Utility namespace
namespace utils {
    template <typename T>
    T clamp(const T &val, const T &low, const T &high) {
        return std::max(low, std::min(val, high));
    }

    void letterBox(const cv::Mat &image,
                   cv::Mat &outImage,
                   const cv::Size &newShape,
                   const cv::Scalar &color = cv::Scalar(114,114,114),
                   bool auto_      = true,
                   bool scaleFill  = false,
                   bool scaleUp    = true,
                   int stride      = 32);
    std::vector<std::string> getClassNames(const std::string &path);
    BoundingBox scaleCoords(const cv::Size &letterboxShape,
                            const BoundingBox &coords,
                            const cv::Size &originalShape,
                            bool p_Clip = true);

    std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42);
    cv::Mat sigmoid(const cv::Mat &src);

    void NMSBoxes(const std::vector<BoundingBox> &boxes,
                  const std::vector<float> &scores,
                  float scoreThreshold,
                  float nmsThreshold,
                  std::vector<int> &indices);
}

class ClothesSegDetector {
public:
    ClothesSegDetector(const std::string &modelPath,
            float confThreshold,
            nvinfer1::ILogger &logger);
    ~ClothesSegDetector();

    // gpuRgbBuffer already contains properly formatted / normalized data
    // im0W / im0H: original image size
    void preprocess(uint8_t* gpuRgbBuffer, int im0W, int im0H, bool block = true);

    void inference();

    void postprocess(std::vector<Segmentation>& output);

    void drawSegmentations(cv::Mat &image,
                           const std::vector<Segmentation> &results,
                           float maskAlpha = 0.5f) const;

    float getConfThreshold() const { return confThreshold; }
    cudaStream_t getStream() const { return stream; }

    // Clothing category helpers
    bool isUpperCloth(int classId) const;
    bool isLowerCloth(int classId) const;
    void setUpperClothClasses(const std::vector<int>& classIds);
    void setLowerClothClasses(const std::vector<int>& classIds);

    int im0W{0};
    int im0H{0};
    int inputW{0};
    int inputH{0};
    cv::Size letterboxSize{}; // Width / Height

private:
    void loadEngine(const std::string &modelPath, nvinfer1::ILogger &logger);
    void allocateBuffers();
    inline cv::Mat sigmoid(const cv::Mat &src) {
        cv::Mat dst;
        cv::exp(-src, dst);
        dst = 1.0 / (1.0 + dst);
        return dst;
    }
private:
    float confThreshold{0.f};
    float iouThreshold{0.5f};
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream{nullptr};

    std::vector<void*> gpuBuffers;
    std::vector<size_t> bufferBytes;
    int inputBinding{-1};
    std::vector<int> outputBindings;
    bool ownsStream{false};
    std::string labelsPath{"classes.txt"};
    std::vector<std::string> classNames;
    std::vector<cv::Scalar>  classColors;

    // Configurable clothing class sets
    std::unordered_set<int> upperClothClasses;
    std::unordered_set<int> lowerClothClasses;
};

inline BoundingBox BoundingBox::intersect(const BoundingBox &other) const {
    int xStart = std::max(x, other.x);
    int yStart = std::max(y, other.y);
    int xEnd   = std::min(x + width,  other.x + other.width);
    int yEnd   = std::min(y + height, other.y + other.height);
    int iw     = std::max(0, xEnd - xStart);
    int ih     = std::max(0, yEnd - yStart);
    return BoundingBox(xStart, yStart, iw, ih);
}

// Utility inline definitions (short) ----------------------------------
namespace utils {
inline std::vector<std::string> getClassNames(const std::string &path) {
    std::vector<std::string> classNames;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[ERROR] Could not open class names file: " << path << std::endl;
        return classNames;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        classNames.push_back(line);
    }
    return classNames;
}

inline cv::Mat sigmoid(const cv::Mat &src) {
    cv::Mat dst;
    cv::exp(-src, dst);
    dst = 1.0 / (1.0 + dst);
    return dst;
}
} // namespace utils

// Remaining utility functions (long) implemented in YOLO11Seg.cpp
// - letterBox
// - scaleCoords
// - generateColors
// - NMSBoxes