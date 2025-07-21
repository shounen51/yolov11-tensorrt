#include "yolov11.h"
#include "yolov11_dll.h"
#include "logging.h"
#include "ColorClassifier.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include <cuda_utils.h>
#include <opencv2/opencv.hpp>

using namespace std;
namespace YoloWithColor {
    extern unique_ptr<YOLOv11> model;
    extern thread inferenceThread; // 用於執行 inference_thread 的執行緒
    extern bool stopThread;     // 用於控制執行緒的停止

    extern queue<InputData> inputQueue;
    extern mutex inputQueueMutex;
    extern condition_variable inputQueueCondition;
    extern vector<queue<OutputData>> outputQueues;
    extern vector<unique_ptr<mutex>> outputQueueMutexes;
    extern vector<unique_ptr<condition_variable>> outputQueueConditions;
    void createModelAndStartThread(const char* engine_path, int camera_amount, float conf_threshold, const char* logFilePath);
    void inference_thread();

    extern Logger logger;
    extern HSVColorClassifier colorClassifier;
}