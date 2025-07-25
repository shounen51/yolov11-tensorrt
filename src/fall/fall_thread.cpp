#include "detection_color_thread.h"
#include "fall_thread.h"
#include "yolov11.h"
#include "logging.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "yolov11_dll.h"
#include "unordered_map"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

static int default_yuv_size = 1920*1920*3/2 * sizeof(uint8_t); // Example size for 1920x1920 YUV420 image
static uint8_t* yuv_buffer_device = nullptr;

// Helper function to read engine file
std::vector<char> readEngineFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file");
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

namespace fall {
    ICudaEngine* engine;
    IExecutionContext* context;
    unordered_map<int, string> fall_classname = {
        {0, "fall"},
        {1, "sitonground"},
        {2, "stand"},
        {3, "other"}
    };
    std::unique_ptr<YOLOv11> model;
    queue<InputData> inputQueue;
    std::mutex inputQueueMutex;
    condition_variable inputQueueCondition;
    vector<queue<OutputData>> outputQueues;
    vector<std::unique_ptr<std::mutex>> outputQueueMutexes;
    vector<std::unique_ptr<std::condition_variable>> outputQueueConditions;
    Logger logger;
    bool stopThread = false;
    std::thread inferenceThread;

    float fallInput[1 * 3 * 224 * 224];
    float fallOutput[1 * 4] = {0};
    void* fallBuffers[2];

    void inference_thread();

    void createModelAndStartThread(const char* det_engine_path, const char* cls_engine_path, int camera_amount, float conf_threshold, const char* logFilePath) {
        AILogger::init(std::string(logFilePath));
        if (std::string(logFilePath) == "") {
            AILOG_INFO("No log file path specified, using default console logging.");
            AILogger::setConsoleOnly(true); // 如果沒有指定 logFilePath，則只輸出到 console
        } else {
            AILOG_INFO("Logging to file: " + std::string(logFilePath));
        }
        // 初始化 cls model
        auto runtime = createInferRuntime(logger);
        auto engineData = readEngineFile(cls_engine_path);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        context = engine->createExecutionContext();
        cudaMalloc(&fallBuffers[0], sizeof(fallInput));
        cudaMalloc(&fallBuffers[1], sizeof(fallOutput));

        model = std::make_unique<YOLOv11>(det_engine_path, conf_threshold, logger);

        // 初始化 outputQueues - 使用智能指針避免複製問題
        outputQueues.reserve(camera_amount);
        outputQueueMutexes.reserve(camera_amount);
        outputQueueConditions.reserve(camera_amount);

        for (int i = 0; i < camera_amount; ++i) {
            outputQueues.emplace_back();
            outputQueueMutexes.emplace_back(std::make_unique<std::mutex>());
            outputQueueConditions.emplace_back(std::make_unique<std::condition_variable>());
        }

        // 啟動執行緒執行 inference_thread
        inferenceThread = std::thread([&]() {
            inference_thread();
        });
        AILOG_INFO("Done initializing fall detection model and started inference thread.");
    }

    uint8_t* GetYuvGpuBuffer(uint8_t* yuv, int width, int height) {
        int img_size = width * height * 3 / 2 * sizeof(uint8_t);
        if (img_size > default_yuv_size) {
            AILOG_INFO("YUV buffer size changed from " + std::to_string(default_yuv_size) + " to " + std::to_string(img_size));

            if (yuv_buffer_device) CUDA_CHECK(cudaFree(yuv_buffer_device));
            CUDA_CHECK(cudaMalloc(&yuv_buffer_device, img_size));
            default_yuv_size = img_size;
        }
        if (yuv_buffer_device == nullptr) {
            AILOG_INFO("Allocating GPU YUV buffer of size: " + std::to_string(default_yuv_size));
            CUDA_CHECK(cudaMalloc(&yuv_buffer_device, default_yuv_size));
        }
        // Copy YUV data to device
        CUDA_CHECK(cudaMemcpy(yuv_buffer_device, yuv, img_size, cudaMemcpyHostToDevice));
        return yuv_buffer_device;
    }

    void inference_thread() {
        AILOG_INFO("Inference thread started.");
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        while (!stopThread) {
            std::unique_lock<std::mutex> lock(inputQueueMutex);

            // 如果 inputQueue 為空，則等待通知
            inputQueueCondition.wait(lock, [] { return !inputQueue.empty() || stopThread; });

            // 再次檢查條件，避免虛假喚醒
            if (stopThread && inputQueue.empty()) break;

            // 從 inputQueue 中取出資料
            InputData input = inputQueue.front();
            inputQueue.pop();
            lock.unlock();
            // 取得 GPU buffers
            uint8_t* gpu_yuv_buffer = GetYuvGpuBuffer(input.image_data, input.width, input.height);
            uint8_t* gpu_rgb_buffer = model->getGpuRgbBuffer(input.width, input.height);
            // 將 yuv 轉換成 rgb
            yuv420toRGBInPlace(gpu_yuv_buffer, input.width, input.height, gpu_rgb_buffer, stream);

            std::vector<Detection> detections;

            model->preprocess(gpu_rgb_buffer, input.width, input.height, false);
            // 利用 CPU 空檔時間將 GPU buffer 從 device 轉成 Mat
            std::vector<uint8_t> rgb_host(input.width * input.height * 3);
            CUDA_CHECK(cudaMemcpy(rgb_host.data(), gpu_rgb_buffer, input.width * input.height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            cv::Mat rgb_image(input.height, input.width, CV_8UC3, rgb_host.data()); // for fall detection crop person image
            // 等待 preprocess 完成
            model->joinGPUStream();
            model->infer();
            model->postprocess(detections);

            int count = std::min(static_cast<int>(detections.size()), input.max_output);
            AILOG_INFO("Inference camera " + std::to_string(input.camera_id) + ", detected " + std::to_string(count) + " objects.");

            // YOLO input size（根據模型固定）
            const int INPUT_W = model->input_w;
            const int INPUT_H = model->input_h;

            // 計算 resize 比例與 padding
            float r = std::min(1.0f * INPUT_W / input.width, 1.0f * INPUT_H / input.height);
            int unpad_w = static_cast<int>(r * input.width);
            int unpad_h = static_cast<int>(r * input.height);
            int pad_x = (INPUT_W - unpad_w) / 2;
            int pad_y = (INPUT_H - unpad_h) / 2;
            svObjData_t* output = new svObjData_t[count];

            // 初始化所有元素
            for (int i = 0; i < count; ++i) {
                svObjData_init(&output[i]);
            }

            // 取得 roi 引用
            ROI* roi_ptr = nullptr;
            if (input.roi_id != -1 && roi_map.find(input.roi_id) != roi_map.end()) {
                roi_ptr = &roi_map[input.roi_id];  // 取得指針，指向原始數據
                roi_ptr->alarm <<= 1; // 左移一位，丟棄最舊的警報
            }

            for (int i = 0; i < count; ++i) {
                const auto& det = detections[i];

                // 模型輸出的 bbox 相對於 640x640，要先扣 padding 再除以 r
                float x1 = (det.bbox.x - pad_x) / r;
                float y1 = (det.bbox.y - pad_y) / r;
                float x2 = (det.bbox.x + det.bbox.width - pad_x) / r;
                float y2 = (det.bbox.y + det.bbox.height - pad_y) / r;

                // 正規化成 [0~1]
                float norm_x1 = std::clamp(x1 / input.width, 0.0f, 1.0f);
                float norm_y1 = std::clamp(y1 / input.height, 0.0f, 1.0f);
                float norm_x2 = std::clamp(x2 / input.width, 0.0f, 1.0f);
                float norm_y2 = std::clamp(y2 / input.height, 0.0f, 1.0f);

                // 將結果放入 output
                output[i].bbox_xmin = norm_x1;
                output[i].bbox_ymin = norm_y1;
                output[i].bbox_xmax = norm_x2;
                output[i].bbox_ymax = norm_y2;
                output[i].class_id = det.class_id;
                output[i].confidence = det.conf;
                // 檢查是否在 ROI 內
                if (input.roi_id != -1 && det.class_id == model->person_class_id && roi_ptr != nullptr) {
                    cv::Point bottom_middle = cv::Point(int((x1+x2)/2), int(y2));
                    output[i].in_roi_id = (roi_ptr->mask.at<uchar>(bottom_middle) != 0) ? input.roi_id : -1; // 設定 ROI ID
                    if (output[i].in_roi_id != -1) {
                        // 如果在 ROI 內，做跌倒辨識
                        cv::Mat personCrop = rgb_image(Rect(Point(x1, y1),
                                                            Point(x2, y2)));
                        cv::resize(personCrop, personCrop, cv::Size(224, 224));
                        personCrop.convertTo(personCrop, CV_32FC3, 1.0 / 255); // 歸一化到0~1
                        std::vector<cv::Mat> channels(3);
                        cv::split(personCrop, channels);
                        for (int c = 0; c < 3; ++c) {
                            memcpy(fallInput + c * 224 * 224, channels[c].data, 224 * 224 * sizeof(float));
                        }
                        cudaMemcpy(fallBuffers[0], fallInput, sizeof(fallInput), cudaMemcpyHostToDevice);
                        context->executeV2(fallBuffers);
                        cudaMemcpy(fallOutput, fallBuffers[1], sizeof(fallOutput), cudaMemcpyDeviceToHost);
                        // 取 fallOutput 最大值作為跌倒判斷
                        float max_conf = fallOutput[0];
                        int max_index = 0;
                        for (int j = 1; j < 4; ++j) {
                            if (fallOutput[j] > max_conf) {
                                max_conf = fallOutput[j];
                                max_index = j;
                            }
                        }
                        std::string fall_label = fall_classname[max_index];
                        if (max_index <= fall_index){
                            roi_ptr->alarm[0] = 1; // 設定此 frame 有人跌倒
                            if (roi_ptr->alarm.count() > int(roi_ptr->alarm.size()/2)) {
                                // 如果連續三個 frame 都有跌倒，則觸發警報
                                AILOG_WARN("Fall detected in ROI " + std::to_string(input.roi_id) + " for camera " + std::to_string(input.camera_id));
                                fall_label = "fall"; // 強制標記為跌倒
                            }else {
                                fall_label = "falling"; // 跌倒投票中
                            }
                        }
                        // 將跌倒類別寫入 output
                        strncpy(output[i].pose, fall_label.c_str(), sizeof(output[i].pose) - 1);
                        output[i].pose[sizeof(output[i].pose) - 1] = '\0'; // 確保字串結尾
                    }
                }
            }

            // 將結果放入 outputQueue
            int camera_id = input.camera_id;
            {
                std::lock_guard<std::mutex> lock(*outputQueueMutexes[camera_id]);
                outputQueues[camera_id].push({output, count});
            }
            outputQueueConditions[camera_id]->notify_one(); // 通知等待的執行緒有新結果可用
        }
    }
}