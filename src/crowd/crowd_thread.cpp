#include "crowd_thread.h"
#include "yolov11.h"
#include "logging.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "yolov11_dll.h"
#include <cuda_utils.h>
#include <opencv2/opencv.hpp>

static int default_yuv_size = 1920*1920*3/2*sizeof(uint8_t); // Example size for 1920x1920 YUV420 image
static uint8_t* yuv_buffer_device = nullptr;

namespace Crowd {
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

    void createModelAndStartThread(const char* engine_path, int camera_amount, float conf_threshold, const char* logFilePath) {
        AILogger::init(std::string(logFilePath));
        if (std::string(logFilePath) == "") {
            AILOG_INFO("No log file path specified, using default console logging.");
            AILogger::setConsoleOnly(true);
        } else {
            AILOG_INFO("Logging to file: " + std::string(logFilePath));
        }
        AILOG_INFO("Initializing YOLOv11 model with engine: " + std::string(engine_path));
        model = std::make_unique<YOLOv11>(engine_path, conf_threshold, logger);

        outputQueues.reserve(camera_amount);
        outputQueueMutexes.reserve(camera_amount);
        outputQueueConditions.reserve(camera_amount);
        for (int i = 0; i < camera_amount; ++i) {
            outputQueues.emplace_back();
            outputQueueMutexes.emplace_back(std::make_unique<std::mutex>());
            outputQueueConditions.emplace_back(std::make_unique<std::condition_variable>());
        }

        inferenceThread = std::thread([&]() { inference_thread(); });
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
        CUDA_CHECK(cudaMemcpy(yuv_buffer_device, yuv, img_size, cudaMemcpyHostToDevice));
        return yuv_buffer_device;
    }

    // Helper: compute bounding box (normalized) from ROI points (assumed normalized)
    inline void roiBoundingBox(const std::vector<cv::Point2f>& pts, float& x1, float& y1, float& x2, float& y2) {
        x1 = 1.f; y1 = 1.f; x2 = 0.f; y2 = 0.f;
        for (const auto& p : pts) {
            x1 = std::min(x1, p.x); y1 = std::min(y1, p.y);
            x2 = std::max(x2, p.x); y2 = std::max(y2, p.y);
        }
        x1 = std::clamp(x1, 0.f, 1.f); y1 = std::clamp(y1, 0.f, 1.f);
        x2 = std::clamp(x2, 0.f, 1.f); y2 = std::clamp(y2, 0.f, 1.f);
    }

    void inference_thread() {
        AILOG_INFO("Inference thread started.");
        cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
        int frame_counter = 0;
        while (!stopThread) {
            std::unique_lock<std::mutex> lock(inputQueueMutex);
            inputQueueCondition.wait(lock, [] { return !inputQueue.empty() || stopThread; });
            if (stopThread && inputQueue.empty()) {
                AILOG_INFO("frame:" + std::to_string(frame_counter) + " Stop signal received, exiting inference thread.");
                break;
            }
            InputData input = inputQueue.front();
            inputQueue.pop();
            frame_counter++;
            lock.unlock();

            uint8_t* gpu_yuv_buffer = GetYuvGpuBuffer(input.image_data, input.width, input.height);
            uint8_t* gpu_rgb_buffer = model->getGpuRgbBuffer(input.width, input.height);
            yuv420toRGBInPlace(gpu_yuv_buffer, input.width, input.height, gpu_rgb_buffer, stream);

            std::vector<Detection> detections;
            model->preprocess(gpu_rgb_buffer, input.width, input.height, false);
            model->joinGPUStream();
            model->infer();
            model->postprocess(detections);

            // 取得 ROI 映射 (若無 ROI 則直接輸出空結果)
            std::unordered_map<int, ROI> roi_map;
            if (camera_function_roi_map.find(input.camera_id) != camera_function_roi_map.end()) {
                if (camera_function_roi_map[input.camera_id].find(functions::CROWD) != camera_function_roi_map[input.camera_id].end())
                    roi_map = camera_function_roi_map[input.camera_id][functions::CROWD];
            }

            if (roi_map.empty()) {
                // 無 ROI -> 不分析，回傳空輸出
                svObjData_t* output = new svObjData_t[0];
                {
                    std::lock_guard<std::mutex> lk(*outputQueueMutexes[input.camera_id]);
                    outputQueues[input.camera_id].push({output, 0});
                }
                outputQueueConditions[input.camera_id]->notify_one();
                AILOG_DEBUG("frame:" + std::to_string(frame_counter) + " No ROI for CROWD, skip analysis");
                continue;
            }

            // 收集所有 person 類別的 normalized bbox (已經映射為標準 COCO id, person=0)
            struct PersonBox { float x1,y1,x2,y2; }; // normalized
            std::vector<PersonBox> person_boxes;
            person_boxes.reserve(detections.size());

            // YOLO 模型輸出 bbox 與 640x640 尺寸相關，需要逆向還原 (與其他線程一致)
            const int INPUT_W = model->input_w;
            const int INPUT_H = model->input_h;
            // 重新做一次與其他模組一致的 padding / scale 邏輯 (複製自其他線程的計算)
            float r = std::min(1.0f * INPUT_W / input.width, 1.0f * INPUT_H / input.height);
            int unpad_w = static_cast<int>(r * input.width);
            int unpad_h = static_cast<int>(r * input.height);
            int pad_x = (INPUT_W - unpad_w) / 2;
            int pad_y = (INPUT_H - unpad_h) / 2;

            for (const auto& det : detections) {
                if (det.class_id != static_cast<int>(CustomClass::PERSON)) continue;
                float x1 = (det.bbox.x - pad_x) / r;
                float y1 = (det.bbox.y - pad_y) / r;
                float x2 = (det.bbox.x + det.bbox.width - pad_x) / r;
                float y2 = (det.bbox.y + det.bbox.height - pad_y) / r;
                // Clamp to image size
                x1 = std::clamp(x1, 0.0f, (float)input.width - 1);
                y1 = std::clamp(y1, 0.0f, (float)input.height - 1);
                x2 = std::clamp(x2, 0.0f, (float)input.width - 1);
                y2 = std::clamp(y2, 0.0f, (float)input.height - 1);
                if (x1 >= x2 || y1 >= y2) continue;
                // Normalize
                person_boxes.push_back({x1 / input.width, y1 / input.height, x2 / input.width, y2 / input.height});
            }

            // 若沒有任何 person，也直接返回空結果
            if (person_boxes.empty()) {
                svObjData_t* output = new svObjData_t[0];
                {
                    std::lock_guard<std::mutex> lk(*outputQueueMutexes[input.camera_id]);
                    outputQueues[input.camera_id].push({output, 0});
                }
                outputQueueConditions[input.camera_id]->notify_one();
                AILOG_DEBUG("frame:" + std::to_string(frame_counter) + " No person detections, skip CROWD output");
                continue;
            }

            // 為每個 ROI 計算覆蓋率 (person 矩形的 union 與 ROI mask 的交集 / ROI 面積)
            std::vector<std::pair<int,float>> roi_coverages; // roi_id, coverage
            roi_coverages.reserve(roi_map.size());

            for (auto& roi_pair : roi_map) {
                int roi_id = roi_pair.first;
                ROI& roi = roi_pair.second;
                if (roi.mask.empty()) continue; // 沒有 mask 無法計算

                // 建立一個臨時遮罩，標記所有 person box 的 union
                cv::Mat union_mask = cv::Mat::zeros(roi.mask.size(), CV_8U);
                for (const auto& pb : person_boxes) {
                    int rx1 = std::clamp(int(pb.x1 * roi.mask.cols), 0, roi.mask.cols - 1);
                    int ry1 = std::clamp(int(pb.y1 * roi.mask.rows), 0, roi.mask.rows - 1);
                    int rx2 = std::clamp(int(pb.x2 * roi.mask.cols), 0, roi.mask.cols - 1);
                    int ry2 = std::clamp(int(pb.y2 * roi.mask.rows), 0, roi.mask.rows - 1);
                    if (rx2 <= rx1 || ry2 <= ry1) continue;
                    cv::rectangle(union_mask, cv::Rect(cv::Point(rx1, ry1), cv::Point(rx2, ry2)), cv::Scalar(255), cv::FILLED);
                }
                cv::Mat intersection_mask;
                cv::bitwise_and(union_mask, roi.mask, intersection_mask);
                int covered = cv::countNonZero(intersection_mask);
                int total = cv::countNonZero(roi.mask);
                float coverage = (total > 0) ? (float)covered / (float)total : 0.f;
                roi_coverages.emplace_back(roi_id, coverage);
            }

            // 建立輸出：每個覆蓋率 > 0.3 的 ROI 產生一個預設物件
            std::vector<svObjData_t> temp_outputs; temp_outputs.reserve(roi_coverages.size());
            for (auto& rc : roi_coverages) {
                if (rc.second > 0.3f) {
                    auto& roi = roi_map[rc.first];
                    float bx1, by1, bx2, by2; roiBoundingBox(roi.points, bx1, by1, bx2, by2);
                    svObjData_t obj; svObjData_init(&obj);
                    obj.bbox_xmin = bx1; obj.bbox_ymin = by1; obj.bbox_xmax = bx2; obj.bbox_ymax = by2;
                    obj.class_id = static_cast<int>(CustomClass::PERSON); // 使用 person 類別代表人群
                    obj.confidence = rc.second; // 覆蓋率作為 confidence
                    obj.in_roi_id = rc.first;
                    strncpy(obj.color_label_first, "", sizeof(obj.color_label_first)-1);
                    strncpy(obj.color_label_second, "", sizeof(obj.color_label_second)-1);
                    strncpy(obj.pose, "none", sizeof(obj.pose)-1);
                    strncpy(obj.climb, "none", sizeof(obj.climb)-1);
                    temp_outputs.push_back(obj);
                }
            }

            int out_count = std::min((int)temp_outputs.size(), input.max_output);
            svObjData_t* output = new svObjData_t[out_count];
            for (int i = 0; i < out_count; ++i) output[i] = temp_outputs[i];

            {
                std::lock_guard<std::mutex> lk(*outputQueueMutexes[input.camera_id]);
                outputQueues[input.camera_id].push({output, out_count});
            }
            outputQueueConditions[input.camera_id]->notify_one();
            AILOG_DEBUG("frame:" + std::to_string(frame_counter) + " CROWD outputs: " + std::to_string(out_count));
        }
    }
}