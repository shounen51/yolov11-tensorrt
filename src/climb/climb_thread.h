#include "yolov11.h"
#include "yolov11_dll.h"
#include "logging.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include <fstream>
#include <iostream>
#include <cuda_utils.h>
#include <opencv2/opencv.hpp>

using namespace std;
namespace climb {
    extern ICudaEngine* engine;
    extern IExecutionContext* context;
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
    struct skeleton_points
    {
        cv::Point LeftShoulder;
        cv::Point RightShoulder;
        cv::Point LeftHip;
        cv::Point RightHip;
        // cv::Point LeftWrist;
        // cv::Point RightWrist;
        // cv::Point LeftKnee;
        // cv::Point RightKnee;
        // cv::Point LeftAnkle;
        // cv::Point RightAnkle;
    };

    enum kps_index {
        Nose = 8400 * 5,
        LeftEye = 8400 * 5 + 8400 * 1 * 3,
        RightEye = 8400 * 5 + 8400 * 2 * 3,
        LeftEar = 8400 * 5 + 8400 * 3 * 3,
        RightEar = 8400 * 5 + 8400 * 4 * 3,
        LeftShoulder = 8400 * 5 + 8400 * 5 * 3,
        RightShoulder = 8400 * 5 + 8400 * 6 * 3,
        LeftElbow = 8400 * 5 + 8400 * 7 * 3,
        RightElbow = 8400 * 5 + 8400 * 8 * 3,
        LeftWrist = 8400 * 5 + 8400 * 9 * 3,
        RightWrist = 8400 * 5 + 8400 * 10 * 3,
        LeftHip = 8400 * 5 + 8400 * 11 * 3,
        RightHip = 8400 * 5 + 8400 * 12 * 3,
        LeftKnee = 8400 * 5 + 8400 * 13 * 3,
        RightKnee = 8400 * 5 + 8400 * 14 * 3,
        LeftAnkle = 8400 * 5 + 8400 * 15 * 3,
        RightAnkle = 8400 * 5 + 8400 * 16 * 3
    };
}