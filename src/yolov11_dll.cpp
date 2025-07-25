#include "yolov11_dll.h"
#include "yolov11.h"
#include "logging.h"
#include "ColorClassifier.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "detection_color_thread.h"
#include "fall_thread.h"
#include <cuda_utils.h>
#include <memory>
#include <opencv2/opencv.hpp>

std::unordered_map<int, ROI> roi_map;

cv::Mat createROI(int width, int height, std::vector<cv::Point2f>& points) {
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    if (points.size() < 3) {
        AILOG_ERROR("ROI points must be at least 3 to form a polygon.");
        return mask;
    }
    std::vector<cv::Point> scaled_points;
    for (const auto& pt : points) {
        scaled_points.emplace_back(pt.x * width, pt.y * height);
    }
    cv::fillConvexPoly(mask, scaled_points, cv::Scalar(255));
    // show mask for debugging
    // cv::imshow("ROI Mask", mask);
    // cv::waitKey(1); // 確保 mask 能夠顯示出來
    return mask;
}

extern "C" {
    YOLOV11_API void svCreate_ObjectModules(int function, int camera_amount, const char* engine_path1, const char* engine_path2, float conf_threshold, const char* logFilePath) {
        AILogger::init(std::string(logFilePath));
        if (std::string(logFilePath) == "") {
            AILOG_INFO("No log file path specified, using default console logging.");
            AILogger::setConsoleOnly(true); // 如果沒有指定 logFilePath，則只輸出到 console
        }
        else {
            AILOG_INFO("Logging to file: " + std::string(logFilePath));
        }
        if (function == functions::YOLO_COLOR) {
            AILOG_INFO("Initializing YOLOv11 model with engine: " + std::string(engine_path1));
            YoloWithColor::createModelAndStartThread(engine_path1, camera_amount, conf_threshold, logFilePath);
        }
        else if (function == functions::FALL) {
            AILOG_INFO("Initializing Fall Detection model with engine: " + std::string(engine_path1) + " and " + std::string(engine_path2));
            fall::createModelAndStartThread(engine_path1, engine_path2, camera_amount, conf_threshold, logFilePath);
        }
    }

    YOLOV11_API int svObjectModules_inputImageYUV(int function, int camera_id, int roi_id,
        unsigned char* image_data, int width, int height, int channels, int max_output) {

        if (roi_map.find(roi_id) != roi_map.end()) {
            ROI roi = roi_map[roi_id];
            if (roi.width != width || roi.height != height) {
                AILOG_WARN("ROI size does not match input image size, creating new ROI.");
                std::vector<cv::Point2f> points;
                for (int i = 0; i < roi.points.size(); ++i) {
                    points.emplace_back(roi.points[i].x, roi.points[i].y);
                }
                cv::Mat mask = createROI(width, height, points);
                roi_map[roi_id] = {mask, points, width, height};
            }
        }else {
            AILOG_WARN("ROI with id " + std::to_string(roi_id) + " does not exist, ignoring.");
        }
        if (function == functions::YOLO_COLOR) {
            if (YoloWithColor::stopThread || camera_id >= YoloWithColor::outputQueues.size()) return -1;
            std::lock_guard<std::mutex> lock(YoloWithColor::inputQueueMutex);
            YoloWithColor::inputQueue.push({camera_id, roi_id, image_data, width, height, channels, max_output});
            YoloWithColor::inputQueueCondition.notify_one();
            if (YoloWithColor::inputQueue.size() > 100)
                AILOG_WARN("Input queue size exceeds 100, input too fast.");
        }
        else if (function == functions::FALL) {
            if (fall::stopThread || camera_id >= fall::outputQueues.size()) return -1;
            std::lock_guard<std::mutex> lock(fall::inputQueueMutex);
            fall::inputQueue.push({camera_id, roi_id, image_data, width, height, channels, max_output});
            fall::inputQueueCondition.notify_one();
            if (fall::inputQueue.size() > 100)
                AILOG_WARN("Input queue size exceeds 100, input too fast.");
        }
        else {
            AILOG_ERROR("Unknown function type: " + std::to_string(function));
            return -1;
        }
        return 1;
    }

    YOLOV11_API int svObjectModules_getResult(int function, int camera_id, svObjData_t* output, int max_output, bool wait) {
        int count = 0;
        if (function == functions::YOLO_COLOR) {
            if (YoloWithColor::stopThread || camera_id > YoloWithColor::outputQueues.size()) return -1;
            if (YoloWithColor::outputQueues[camera_id].empty()) {
                if (wait) { // 等待直到有結果可用
                    std::unique_lock<std::mutex> lock(*YoloWithColor::outputQueueMutexes[camera_id]);
                    YoloWithColor::outputQueueConditions[camera_id]->wait(lock, [&] { return !YoloWithColor::outputQueues[camera_id].empty() || YoloWithColor::stopThread; });
                }
                else {
                    return -1; // 如果不等待且沒有結果，則返回 -1
                }
            }
            else {
                std::unique_lock<std::mutex> lock(*YoloWithColor::outputQueueMutexes[camera_id]);
            }
            OutputData result = YoloWithColor::outputQueues[camera_id].front();
            YoloWithColor::outputQueues[camera_id].pop();
            // 複製結果到輸出
            count = std::min(result.count, max_output);
            memcpy(output, result.output, count * sizeof(svObjData_t));
        }
        else if (function == functions::FALL) {
            if (fall::stopThread || camera_id > fall::outputQueues.size()) return -1;
            if (fall::outputQueues[camera_id].empty()) {
                if (wait) { // 等待直到有結果可用
                    std::unique_lock<std::mutex> lock(*fall::outputQueueMutexes[camera_id]);
                    fall::outputQueueConditions[camera_id]->wait(lock, [&] { return !fall::outputQueues[camera_id].empty() || fall::stopThread; });
                }
                else {
                    return -1; // 如果不等待且沒有結果，則返回 -1
                }
            }
            else {
                std::unique_lock<std::mutex> lock(*fall::outputQueueMutexes[camera_id]);
            }
            OutputData result = fall::outputQueues[camera_id].front();
            fall::outputQueues[camera_id].pop();
            // 複製結果到輸出
            count = std::min(result.count, max_output);
            memcpy(output, result.output, count * sizeof(svObjData_t));
        } else {
            AILOG_ERROR("Unknown function type: " + std::to_string(function));
            return -1;
        }
        return count; // 返回檢測到的物件數量
    }

    YOLOV11_API void svCreate_ROI(int roi_id, int width, int height, float* points_x, float* points_y, int point_count) {
        if (roi_id == -1) {
            AILOG_ERROR("ROI ID cannot be -1, please provide a valid ROI ID.");
            return;
        }
        if (roi_map.find(roi_id) != roi_map.end()) {
            AILOG_WARN("ROI with id " + std::to_string(roi_id) + " already exists, updating.");
        }
        // 轉換 C 數組為 C++ vector
        std::vector<cv::Point2f> points;
        for (int i = 0; i < point_count; ++i) {
            points.emplace_back(points_x[i], points_y[i]);
        }
        cv::Mat mask = createROI(width, height, points);
        roi_map[roi_id] = {mask, points, width, height};
    }

    YOLOV11_API void svRemove_ROI(int roi_id) {
        auto it = roi_map.find(roi_id);
        if (it != roi_map.end()) {
            roi_map.erase(it);
            AILOG_INFO("Removed ROI with id " + std::to_string(roi_id));
        } else {
            AILOG_WARN("ROI with id " + std::to_string(roi_id) + " does not exist.");
        }
    }

    YOLOV11_API void release() {
        YoloWithColor::stopThread = true;
        YoloWithColor::inputQueueCondition.notify_all(); // 通知執行緒停止等待

        // 通知所有output condition variables
        for (auto& cv : YoloWithColor::outputQueueConditions) {
            cv->notify_all();
        }

        if (YoloWithColor::inferenceThread.joinable()) {
            YoloWithColor::inferenceThread.join();
        }
        AILOG_INFO("Inference thread stopped.");
        YoloWithColor::model.reset();
    }
}
