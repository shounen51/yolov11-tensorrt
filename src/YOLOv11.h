#pragma once

#include "cuda_utils.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class YOLOv11
{
public:

    YOLOv11(string model_path, float conf_threshold, nvinfer1::ILogger& logger);
    ~YOLOv11();

    uint8_t* getGpuRgbBuffer(int, int);
    int input_w;
    int input_h;
    int num_classes = 11;
    int person_on_wheelchair_class_id = 10; // pw special
    int person_class_id = 0;

    void preprocess(uint8_t* gpu_rgb_buffer, int im0_w, int im0_h, bool block=true);
    void infer();
    void postprocess(vector<Detection>& output);
    void draw(Mat& image, const vector<Detection>& output);
    void joinGPUStream() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
private:
    void init(std::string engine_path, float conf_threshold, nvinfer1::ILogger& logger);

    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;
    cudaStream_t stream;
    uint8_t* gpu_rgb_buffer = nullptr;         //!< The device buffer for RGB input
    int gpu_rgb_buffer_size = 1920*1920*3*sizeof(uint8_t); //!< The size of the device buffer for RGB input

    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int num_detections;
    int detection_attribute_size;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.7f;

    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};