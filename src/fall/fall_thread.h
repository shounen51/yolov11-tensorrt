#include "yolov11.h"
#include "yolov11_dll.h"
#include "logging.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "unordered_map"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fall {
    extern unique_ptr<YOLOv11> model;
    extern ICudaEngine* engine;
    extern IExecutionContext* context;
    extern unordered_map<int, string> fall_classname;

    extern thread inferenceThread; // 用於執行 inference_thread 的執行緒
    extern bool stopThread;     // 用於控制執行緒的停止

    extern queue<InputData> inputQueue;
    extern mutex inputQueueMutex;
    extern condition_variable inputQueueCondition;
    extern vector<queue<OutputData>> outputQueues;
    extern vector<unique_ptr<mutex>> outputQueueMutexes;
    extern vector<unique_ptr<condition_variable>> outputQueueConditions;
    void createModelAndStartThread(const char* det_engine_path, const char* cls_engine_path, int camera_amount, float conf_threshold, const char* logFilePath);
    void inference_thread();

    extern Logger logger;

    // 常數定義：小於等於 fall_index 的類別被視為跌倒類別
    constexpr int fall_index = 1;
}