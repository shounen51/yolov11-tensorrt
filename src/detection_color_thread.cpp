#include "detection_color_thread.h"
#include "yolov11.h"
#include "logging.h"
#include "ColorClassifier.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "yolov11_dll.h"
#include "tracker.h"
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

    // 為每個攝像頭創建獨立的追蹤器
    std::vector<std::unique_ptr<ObjectTracker>> trackers;

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

        // 初始化 outputQueues 和 trackers - 使用智能指針避免複製問題
        outputQueues.reserve(camera_amount);
        outputQueueMutexes.reserve(camera_amount);
        outputQueueConditions.reserve(camera_amount);
        trackers.reserve(camera_amount);

        for (int i = 0; i < camera_amount; ++i) {
            outputQueues.emplace_back();
            outputQueueMutexes.emplace_back(std::make_unique<std::mutex>());
            outputQueueConditions.emplace_back(std::make_unique<std::condition_variable>());
            trackers.emplace_back(std::make_unique<ObjectTracker>(5, 0.1f)); // max_skip_frames=5, max_distance_threshold=0.3
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
            std::unordered_map<int, int> roi_people_count;
            if (camera_function_roi_map.find(input.camera_id) != camera_function_roi_map.end()) {
                if (camera_function_roi_map[input.camera_id].find(functions::YOLO_COLOR) != camera_function_roi_map[input.camera_id].end())
                    roi_map = camera_function_roi_map[input.camera_id][functions::YOLO_COLOR];
            }
            // 初始化 roi_people_count
            for (const auto& roi_pair : roi_map) {
                roi_people_count[roi_pair.first] = 0; // 初始化每個 ROI 的人數計數為 0
            }

            // 取得 crossing line
            std::unordered_map<int, CrossingLineROI> crossing_line_map;
            if (CrossingLineROI_map.find(input.camera_id) != CrossingLineROI_map.end()) {
                crossing_line_map = CrossingLineROI_map[input.camera_id][functions::YOLO_COLOR];
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
                    // 計算裁剪區域的座標，並確保在圖像邊界內
                    int crop_x1 = std::max(0, static_cast<int>(x1 + (det.bbox.width*point[0])/r));
                    int crop_y1 = std::max(0, static_cast<int>(y1 + (det.bbox.height*point[1])/r));
                    int crop_x2 = std::min(rgb_image.cols, static_cast<int>(x1 + (det.bbox.width*point[2])/r));
                    int crop_y2 = std::min(rgb_image.rows, static_cast<int>(y1 + (det.bbox.height*point[3])/r));

                    // 確保裁剪區域有效（寬度和高度都大於0）
                    if (crop_x2 <= crop_x1 || crop_y2 <= crop_y1) {
                        color_labels.push_back("unknown");
                        continue;
                    }
                    cv::Mat personCrop = rgb_image(Rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1));

                    vector<unsigned char> color = colorClassifier.classifyStatistics(personCrop, 500, cv::COLOR_BGR2HSV);
                    int maxIndex = 0;
                    int maxCount = 0;
                    for (int j = 0; j < color.size(); j++) {
                        if (color[j] > maxCount && ColorLabelsString[j] != "unknown") {
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
                    // 確保 bottom_middle 在 mask 邊界內
                    if (bottom_middle.x >= 0 && bottom_middle.x < roi_pair.second.mask.cols &&
                        bottom_middle.y >= 0 && bottom_middle.y < roi_pair.second.mask.rows &&
                        roi_pair.second.mask.at<uchar>(bottom_middle) != 0) {
                        output[i].in_roi_id = roi_pair.first;
                        if (det.class_id == model->person_class_id) roi_people_count[roi_pair.first]++; // 增加該 ROI 的人數計數
                        else if (det.class_id == model->person_on_wheelchair_class_id) {
                            AILOG_INFO("Object " + std::to_string(i) + " (class: person on wheelchair" +
                                    ") is in ROI " + std::to_string(roi_pair.first) + " for camera " + std::to_string(input.camera_id));
                        }
                    }
                }
            }
            std::string log_message = "Inference for camera " + std::to_string(input.camera_id) + " completed. Detections: " + std::to_string(count);
            for (const auto& roi_pair : roi_people_count) {
                log_message += ", ROI " + std::to_string(roi_pair.first) + ": " + std::to_string(roi_pair.second) + " people";
            }
            AILOG_DEBUG(log_message);

            // 應用目標追蹤
            if (input.camera_id < trackers.size() && crossing_line_map.size() > 0) {
                AILOG_DEBUG("Starting tracking for camera " + std::to_string(input.camera_id) + " with " + std::to_string(count) + " detections");
                try {
                    std::vector<TrackerDetection> tracker_detections;
                    int person_count = 0;
                    for (int i = 0; i < count; ++i) {
                        // 只對人類進行追蹤
                        if (output[i].class_id == model->person_class_id) {
                            float center_x = (output[i].bbox_xmin + output[i].bbox_xmax) / 2.0f;
                            // 將參考點改為上半部框的中心 (y座標向上移動1/8)
                            float bbox_height = output[i].bbox_ymax - output[i].bbox_ymin;
                            float center_y = output[i].bbox_ymin + bbox_height * 0.125f;
                            float width = output[i].bbox_xmax - output[i].bbox_xmin;
                            float height = output[i].bbox_ymax - output[i].bbox_ymin;

                            tracker_detections.emplace_back(center_x, center_y, width, height,
                                                           output[i].class_id, output[i].confidence);
                            person_count++;
                        }
                    }
                    auto tracking_result = trackers[input.camera_id]->update(tracker_detections);

                    // 使用新的映射結果直接分配track_id
                    // tracking_result.detection_to_track_id[i] 直接對應 tracker_detections[i]

                    for (int i = 0; i < count; ++i) {
                        output[i].track_id = -1; // 預設無追蹤ID
                    }

                    // 按順序分配追蹤ID給人類檢測
                    size_t tracker_detection_index = 0;
                    for (int i = 0; i < count; ++i) {
                        if (output[i].class_id == model->person_class_id) {
                            // 檢查是否還有可用的tracker_detection映射
                            if (tracker_detection_index < tracking_result.detection_to_track_id.size()) {
                                int track_id = tracking_result.detection_to_track_id[tracker_detection_index];
                                output[i].track_id = track_id;

                                if (track_id != -1) {
                                    // 處理crossing line檢測
                                    auto prev_detection = trackers[input.camera_id]->getPreviousDetection(track_id);
                                    if (prev_detection.valid) {
                                        // 計算前一幀的底邊中心 (p1)
                                        float prev_bottom_y = prev_detection.y + prev_detection.height * 0.875f;
                                        cv::Point2f p1(prev_detection.x, prev_bottom_y);

                                        // 計算當前幀的底邊中心 (p2)
                                        float curr_center_x = (output[i].bbox_xmin + output[i].bbox_xmax) / 2.0f;
                                        float curr_bottom_y = output[i].bbox_ymax;
                                        cv::Point2f p2(curr_center_x, curr_bottom_y);

                                        for (const auto& [roi_id, roi] : crossing_line_map) {
                                            // 遍歷所有線段 (point_count - 1 個線段)
                                            for (int segment_idx = 0; segment_idx < roi.points.size() - 1; ++segment_idx) {
                                                cv::Point2f q1 = roi.points[segment_idx];
                                                cv::Point2f q2 = roi.points[segment_idx + 1];
                                                int dir = GeometryUtils::doIntersect(q1, q2, p1, p2);
                                                if (dir != 0) {
                                                    output[i].crossing_line_id = roi_id;
                                                    output[i].crossing_line_direction = (dir == roi.in_area_direction[segment_idx]) ? 1 : -1;
                                                    AILOG_INFO("Track ID " + std::to_string(track_id) +
                                                              " crossed line segment " + std::to_string(segment_idx) +
                                                              " in ROI " + std::to_string(roi_id) +
                                                              ", direction: " + std::to_string(output[i].crossing_line_direction));
                                                    break; // 找到相交線段後跳出內層循環
                                                }
                                            }
                                        }
                                    }
                                }
                                tracker_detection_index++; // 移動到下一個tracker_detection
                            }
                        }
                    }
                    AILOG_DEBUG("Camera " + std::to_string(input.camera_id) + " tracking: " +
                              std::to_string(tracking_result.total_active_tracks) + " objects tracked");
                }
                catch (const std::exception& e) {
                    AILOG_ERROR("Exception in tracking for camera " + std::to_string(input.camera_id) + ": " + std::string(e.what()));
                    // 如果追蹤失敗，給所有檢測設置無效的追蹤ID
                    for (int i = 0; i < count; ++i) {
                        output[i].track_id = -1;
                    }
                }
            } else {
                for (int i = 0; i < count; ++i) {
                    output[i].track_id = -1;
                }
            }

            // 將結果放入 outputQueue
            {
                std::lock_guard<std::mutex> lock(*outputQueueMutexes[input.camera_id]);
                outputQueues[input.camera_id].push({output, count});
            }
            outputQueueConditions[input.camera_id]->notify_one(); // 通知等待的執行緒有新結果可用
        }
    }
}