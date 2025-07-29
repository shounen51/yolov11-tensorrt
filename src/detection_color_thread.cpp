#include "detection_color_thread.h"
#include "yolov11.h"
#include "logging.h"
#include "ColorClassifier.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "yolov11_dll.h"
#include <cuda_utils.h>
#include <opencv2/opencv.hpp>

static int default_yuv_size = 1920*1920*3/2 * sizeof(uint8_t); // Example size for 1920x1920 YUV420 image
static uint8_t* yuv_buffer_device = nullptr;

namespace YoloWithColor {
    std::unique_ptr<YOLOv11> model;
    queue<InputData> inputQueue;
    std::mutex inputQueueMutex;
    condition_variable inputQueueCondition;
    vector<queue<OutputData>> outputQueues;
    vector<std::unique_ptr<std::mutex>> outputQueueMutexes;
    vector<std::unique_ptr<std::condition_variable>> outputQueueConditions;
    HSVColorClassifier colorClassifier;
    Logger logger;
    bool stopThread = false;
    std::thread inferenceThread;

    void createModelAndStartThread(const char* engine_path, int camera_amount, float conf_threshold, const char* logFilePath) {
        AILogger::init(std::string(logFilePath));
        if (std::string(logFilePath) == "") {
            AILOG_INFO("No log file path specified, using default console logging.");
            AILogger::setConsoleOnly(true); // 如果沒有指定 logFilePath，則只輸出到 console
        } else {
            AILOG_INFO("Logging to file: " + std::string(logFilePath));
        }
        AILOG_INFO("Initializing YOLOv11 model with engine: " + std::string(engine_path));
        model = std::make_unique<YOLOv11>(engine_path, conf_threshold, logger);

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
        colorClassifier.setDefaultColorRange();
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
        int frame_counter = 0;
        while (!stopThread) {
            std::unique_lock<std::mutex> lock(inputQueueMutex);

            // 如果 inputQueue 為空，則等待通知
            inputQueueCondition.wait(lock, [] { return !inputQueue.empty() || stopThread; });

            // 再次檢查條件，避免虛假喚醒
            if (stopThread && inputQueue.empty()) {
                AILOG_INFO("Stop signal received, exiting inference thread.");
                break;
            }

            // 從 inputQueue 中取出資料
            InputData input = inputQueue.front();
            inputQueue.pop();
            frame_counter++;
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
            cv::Mat rgb_image(input.height, input.width, CV_8UC3, rgb_host.data());
            // 等待 preprocess 完成
            model->joinGPUStream();
            model->infer();
            model->postprocess(detections);

            int count = std::min(static_cast<int>(detections.size()), input.max_output);

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

            // 取得 roi
            std::unordered_map<int, ROI> roi_map;
            if (camera_function_roi_map.find(input.camera_id) != camera_function_roi_map.end()) {
                if (camera_function_roi_map[input.camera_id].find(functions::YOLO_COLOR) != camera_function_roi_map[input.camera_id].end())
                    roi_map = camera_function_roi_map[input.camera_id][functions::YOLO_COLOR];
            }

            for (int i = 0; i < count; ++i) {
                const auto& det = detections[i];

                // 模型輸出的 bbox 相對於 640x640，要先扣 padding 再除以 r
                float x1 = (det.bbox.x - pad_x) / r;
                float y1 = (det.bbox.y - pad_y) / r;
                float x2 = (det.bbox.x + det.bbox.width - pad_x) / r;
                float y2 = (det.bbox.y + det.bbox.height - pad_y) / r;

                // 計算顏色
                std::vector<std::vector<float>> box_points = {{0, 0, 1, 1}}; // 預設為整個 bbox
                if (det.class_id == model->person_class_id)
                    box_points = {{0.14, 0.2, 0.78, 0.45}, {0.2, 0.5, 0.87, 0.8}}; // 如果類別是人則分上半身和下半身
                std::vector<std::string> color_labels;
                for (auto& point : box_points) {
                    cv::Mat personCrop = rgb_image(Rect(Point(x1 + (det.bbox.width*point[0])/r, y1 + (det.bbox.height*point[1])/r),
                                                        Point(x1 + (det.bbox.width*point[2])/r, y1 + (det.bbox.height*point[3])/r)));

                    vector<unsigned char> color = colorClassifier.classifyStatistics(personCrop, 500, cv::COLOR_BGR2HSV);
                    int maxIndex = 0;
                    int maxCount = 0;
                    for (int j = 0; j < color.size(); j++) {
                        if (color[j] > maxCount && ColorLabelsString[j] != "unknow") {
                            maxCount = color[j];
                            maxIndex = j;
                        }
                    }
                    color_labels.push_back(ColorLabelsString[maxIndex]);
                }

                // 正規化成 [0~1]
                x1 = std::clamp(x1 / input.width, 0.0f, 1.0f);
                y1 = std::clamp(y1 / input.height, 0.0f, 1.0f);
                x2 = std::clamp(x2 / input.width, 0.0f, 1.0f);
                y2 = std::clamp(y2 / input.height, 0.0f, 1.0f);

                // 將結果放入 output
                output[i].bbox_xmin = x1;
                output[i].bbox_ymin = y1;
                output[i].bbox_xmax = x2;
                output[i].bbox_ymax = y2;
                output[i].class_id = det.class_id;
                output[i].confidence = det.conf;
                strncpy(output[i].color_label_first, color_labels[0].c_str(), sizeof(output[i].color_label_first) - 1);
                if (det.class_id == model->person_class_id)
                    strncpy(output[i].color_label_second, color_labels[1].c_str(), sizeof(output[i].color_label_second) - 1);
                output[i].color_label_first[sizeof(output[i].color_label_first) - 1] = '\0'; // 確保以空字元結尾
                output[i].color_label_second[sizeof(output[i].color_label_second) - 1] = '\0'; // 確保以空字元結尾
                // 檢查是否在 ROI 內
                output[i].in_roi_id = -1; // 預設不在 ROI 內
                for (auto& roi_pair : roi_map) {
                    cv::Point bottom_middle = (cv::Point(x1 * input.width, y2 * input.height) + cv::Point(x2 * input.width, y2 * input.height)) / 2;
                    if (roi_pair.second.mask.at<uchar>(bottom_middle) != 0) {
                        output[i].in_roi_id = roi_pair.first;
                        if (det.class_id == model->person_on_wheelchair_class_id) {
                            AILOG_INFO("Object " + std::to_string(i) + " (class: person on wheelchair" +
                                    ") is in ROI " + std::to_string(roi_pair.first));
                        }
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