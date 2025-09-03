#pragma once

#include "cuda_utils.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <unordered_map>

using namespace nvinfer1;
using namespace std;
using namespace cv;

// Custom model classes (18 classes)
enum class CustomClass : int {
    PERSON = 0,
    BICYCLE = 1,
    CAR = 2,
    MOTORCYCLE = 3,
    BUS = 4,
    TRAIN = 5,
    BACKPACK = 6,
    HANDBAG = 7,
    SUITCASE = 8,
    WHEELCHAIR = 9,
    PERSON_ON_WHEELCHAIR = 10,
    BENCH = 11,
    UMBRELLA = 12,
    BOTTLE = 13,
    WINE_GLASS = 14,
    CHAIR = 15,
    COUCH = 16,
    VASE = 17
};

// COCO standard classes (82 classes)
enum class CocoClass : int {
    PERSON = 0,
    BICYCLE = 1,
    CAR = 2,
    MOTORCYCLE = 3,
    AIRPLANE = 4,
    BUS = 5,
    TRAIN = 6,
    TRUCK = 7,
    BOAT = 8,
    TRAFFIC_LIGHT = 9,
    FIRE_HYDRANT = 10,
    STOP_SIGN = 11,
    PARKING_METER = 12,
    BENCH = 13,
    BIRD = 14,
    CAT = 15,
    DOG = 16,
    HORSE = 17,
    SHEEP = 18,
    COW = 19,
    ELEPHANT = 20,
    BEAR = 21,
    ZEBRA = 22,
    GIRAFFE = 23,
    BACKPACK = 24,
    UMBRELLA = 25,
    HANDBAG = 26,
    TIE = 27,
    SUITCASE = 28,
    FRISBEE = 29,
    SKIS = 30,
    SNOWBOARD = 31,
    SPORTS_BALL = 32,
    KITE = 33,
    BASEBALL_BAT = 34,
    BASEBALL_GLOVE = 35,
    SKATEBOARD = 36,
    SURFBOARD = 37,
    TENNIS_RACKET = 38,
    BOTTLE = 39,
    WINE_GLASS = 40,
    CUP = 41,
    FORK = 42,
    KNIFE = 43,
    SPOON = 44,
    BOWL = 45,
    BANANA = 46,
    APPLE = 47,
    SANDWICH = 48,
    ORANGE = 49,
    BROCCOLI = 50,
    CARROT = 51,
    HOT_DOG = 52,
    PIZZA = 53,
    DONUT = 54,
    CAKE = 55,
    CHAIR = 56,
    COUCH = 57,
    POTTED_PLANT = 58,
    BED = 59,
    DINING_TABLE = 60,
    TOILET = 61,
    TV = 62,
    LAPTOP = 63,
    MOUSE = 64,
    REMOTE = 65,
    KEYBOARD = 66,
    CELL_PHONE = 67,
    MICROWAVE = 68,
    OVEN = 69,
    TOASTER = 70,
    SINK = 71,
    REFRIGERATOR = 72,
    BOOK = 73,
    CLOCK = 74,
    VASE = 75,
    SCISSORS = 76,
    TEDDY_BEAR = 77,
    HAIR_DRIER = 78,
    TOOTHBRUSH = 79,
    WHEELCHAIR = 80,
    PERSON_ON_WHEELCHAIR = 81
};

// Custom model class names mapping
static const unordered_map<int, int> CUSTOM_to_COCO = {
    {static_cast<int>(CustomClass::PERSON), static_cast<int>(CocoClass::PERSON)},
    {static_cast<int>(CustomClass::BICYCLE), static_cast<int>(CocoClass::BICYCLE)},
    {static_cast<int>(CustomClass::CAR), static_cast<int>(CocoClass::CAR)},
    {static_cast<int>(CustomClass::MOTORCYCLE), static_cast<int>(CocoClass::MOTORCYCLE)},
    {static_cast<int>(CustomClass::BUS), static_cast<int>(CocoClass::BUS)},
    {static_cast<int>(CustomClass::TRAIN), static_cast<int>(CocoClass::TRAIN)},
    {static_cast<int>(CustomClass::BACKPACK), static_cast<int>(CocoClass::BACKPACK)},
    {static_cast<int>(CustomClass::HANDBAG), static_cast<int>(CocoClass::HANDBAG)},
    {static_cast<int>(CustomClass::SUITCASE), static_cast<int>(CocoClass::SUITCASE)},
    {static_cast<int>(CustomClass::WHEELCHAIR), static_cast<int>(CocoClass::WHEELCHAIR)},
    {static_cast<int>(CustomClass::PERSON_ON_WHEELCHAIR), static_cast<int>(CocoClass::PERSON_ON_WHEELCHAIR)},
    {static_cast<int>(CustomClass::BENCH), static_cast<int>(CocoClass::BENCH)},
    {static_cast<int>(CustomClass::UMBRELLA), static_cast<int>(CocoClass::UMBRELLA)},
    {static_cast<int>(CustomClass::BOTTLE), static_cast<int>(CocoClass::BOTTLE)},
    {static_cast<int>(CustomClass::WINE_GLASS), static_cast<int>(CocoClass::WINE_GLASS)},
    {static_cast<int>(CustomClass::CHAIR), static_cast<int>(CocoClass::CHAIR)},
    {static_cast<int>(CustomClass::COUCH), static_cast<int>(CocoClass::COUCH)},
    {static_cast<int>(CustomClass::VASE), static_cast<int>(CocoClass::VASE)}
};

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
    int num_classes = 18;

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
    float nms_threshold = 0.5f;

    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};