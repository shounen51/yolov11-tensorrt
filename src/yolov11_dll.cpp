#include "yolov11_dll.h"
#include "yolov11.h"
#include "logging.h"
#include "ColorClassifier.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "detection_color_thread.h"
#include "fall_thread.h"
#include "climb_thread.h"
#include <cuda_utils.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <filesystem>

std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, ROI>>> camera_function_roi_map;
std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, MRTRedlightROI>>> MRTRedlightROI_map;
std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, CrossingLineROI>>> CrossingLineROI_map;

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
    // cv::imshow("ROI Mask Full", mask);
    // cv::waitKey(1); // 確保 mask 能夠顯示出來
    return mask;
}

// 線段相交判斷函數實現
namespace GeometryUtils {
    // 計算向量叉積 (p1-p0) × (p2-p0)
    YOLOV11_API float crossProduct(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2) {
        return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
    }

    // 檢查點是否在線段上（假設點已經共線）
    YOLOV11_API bool onSegment(cv::Point2f p, cv::Point2f q, cv::Point2f r) {
        return (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
                q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y));
    }

    // 判斷兩條線段是否相交，並返回p1相對於q1q2線段的位置關係
    // 線段1: p1-p2, 線段2: q1-q2
    // 返回值：0=不相交, -1=p1在q1q2的A側, 1=p1在q1q2的B側
    YOLOV11_API int doIntersect(cv::Point2f q1, cv::Point2f q2, cv::Point2f p1, cv::Point2f p2) {
        // 計算p1和p2相對於q1q2線段的位置
        float d1 = crossProduct(q1, q2, p1);  // p1相對於q1q2的叉積
        float d2 = crossProduct(q1, q2, p2);  // p2相對於q1q2的叉積
        float d3 = crossProduct(p1, p2, q1);  // q1相對於p1p2的叉積
        float d4 = crossProduct(p1, p2, q2);  // q2相對於p1p2的叉積

        // 判斷線段是否相交
        bool intersect = false;
        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
            ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            intersect = true;
        }

        // 特殊情況：點共線且重疊
        if (d1 == 0 && onSegment(q1, p1, q2)) intersect = true;
        if (d2 == 0 && onSegment(q1, p2, q2)) intersect = true;
        if (d3 == 0 && onSegment(p1, q1, p2)) intersect = true;
        if (d4 == 0 && onSegment(p1, q2, p2)) intersect = true;

        // 如果不相交，根據p1相對於q1q2的位置返回值
        if (!intersect) {
            return 0;  // 不相交
        }

        // 如果相交，判斷p1的位置關係
        if (d1 == 0) {  // p1在q1q2線段上
            if (d2 > 0) {
                return 0;  // p1在線段上且p2在B側，返回0
            } else {
                return 0;   // p1在線段上且p2在A側，返回0
            }
        } else if (d1 > 0) {
            return 1;   // p1在q1q2線段的B側
        } else {
            return -1;  // p1在q1q2線段的A側
        }
    }
}

// File existence check function (internal use)
bool checkFileExists(const std::string& filepath) {
    try {
        return std::filesystem::exists(filepath);
    } catch (const std::filesystem::filesystem_error& ex) {
        AILOG_ERROR("Filesystem error checking file: " + std::string(ex.what()));
        return false;
    } catch (const std::exception& ex) {
        AILOG_ERROR("Error occurred while checking file existence: " + std::string(ex.what()));
        return false;
    }
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
            if (!checkFileExists(std::string(engine_path1))) {
                AILOG_ERROR("Engine file does not exist: " + std::string(engine_path1));
                return;
            }
            AILOG_INFO("Initializing YOLOv11 model with engine: " + std::string(engine_path1));
            YoloWithColor::createModelAndStartThread(engine_path1, camera_amount, conf_threshold, logFilePath);
        }
        else if (function == functions::FALL) {
            if (!checkFileExists(std::string(engine_path1))) {
                AILOG_ERROR("Engine file 1 does not exist: " + std::string(engine_path1));
                return;
            }
            if (!checkFileExists(std::string(engine_path2))) {
                AILOG_ERROR("Engine file 2 does not exist: " + std::string(engine_path2));
                return;
            }
            AILOG_INFO("Initializing Fall Detection model with engine: " + std::string(engine_path1) + " and " + std::string(engine_path2));
            fall::createModelAndStartThread(engine_path1, engine_path2, camera_amount, conf_threshold, logFilePath);
        }
        else if (function == functions::CLIMB) {
            if (!checkFileExists(std::string(engine_path1))) {
                AILOG_ERROR("Engine file does not exist: " + std::string(engine_path1));
                return;
            }
            AILOG_INFO("Initializing Climb Detection model with engine: " + std::string(engine_path1));
            climb::createModelAndStartThread(engine_path1, camera_amount, conf_threshold, logFilePath);
        }
    }

    YOLOV11_API int svObjectModules_inputImageYUV(int function, int camera_id,
        unsigned char* image_data, int width, int height, int channels, int max_output) {

        // if (camera_function_roi_map.find(roi_id) != camera_function_roi_map.end()) {
        //     ROI roi = camera_function_roi_map[roi_id];
        //     if (roi.width != width || roi.height != height) {
        //         AILOG_WARN("ROI size does not match input image size, creating new ROI.");
        //         std::vector<cv::Point2f> points;
        //         for (int i = 0; i < roi.points.size(); ++i) {
        //             points.emplace_back(roi.points[i].x, roi.points[i].y);
        //         }
        //         cv::Mat mask = createROI(width, height, points);
        //         camera_function_roi_map[roi_id] = {mask, points, width, height};
        //     }
        // }else {
        //     AILOG_WARN("ROI with id " + std::to_string(roi_id) + " does not exist, ignoring.");
        // }
        if (function == functions::YOLO_COLOR) {
            if (YoloWithColor::stopThread || camera_id >= YoloWithColor::outputQueues.size()) return -1;
            std::lock_guard<std::mutex> lock(YoloWithColor::inputQueueMutex);
            YoloWithColor::inputQueue.push({camera_id, image_data, width, height, channels, max_output});
            YoloWithColor::inputQueueCondition.notify_one();
            if (YoloWithColor::inputQueue.size() > 100)
                AILOG_WARN("Input queue size exceeds 100, input too fast.");
            return YoloWithColor::inputQueue.size();
        }
        else if (function == functions::FALL) {
            if (fall::stopThread || camera_id >= fall::outputQueues.size()) return -1;
            std::lock_guard<std::mutex> lock(fall::inputQueueMutex);
            fall::inputQueue.push({camera_id, image_data, width, height, channels, max_output});
            fall::inputQueueCondition.notify_one();
            if (fall::inputQueue.size() > 100)
                AILOG_WARN("Input queue size exceeds 100, input too fast.");
            return fall::inputQueue.size();
        }
        else if (function == functions::CLIMB) {
            if (climb::stopThread || camera_id >= climb::outputQueues.size()) return -1;
            std::lock_guard<std::mutex> lock(climb::inputQueueMutex);
            climb::inputQueue.push({camera_id, image_data, width, height, channels, max_output});
            climb::inputQueueCondition.notify_one();
            if (climb::inputQueue.size() > 100)
                AILOG_WARN("Input queue size exceeds 100, input too fast.");
            return climb::inputQueue.size();
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
            delete[] result.output; // 釋放記憶體
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
            delete[] result.output;
        }
        else if (function == functions::CLIMB) {
            if (climb::stopThread || camera_id > climb::outputQueues.size()) return -1;
            if (climb::outputQueues[camera_id].empty()) {
                if (wait) { // 等待直到有結果可用
                    std::unique_lock<std::mutex> lock(*climb::outputQueueMutexes[camera_id]);
                    climb::outputQueueConditions[camera_id]->wait(lock, [&] { return !climb::outputQueues[camera_id].empty() || climb::stopThread; });
                }
                else {
                    return -1; // 如果不等待且沒有結果，則返回 -1
                }
            }
            else {
                std::unique_lock<std::mutex> lock(*climb::outputQueueMutexes[camera_id]);
            }
            OutputData result = climb::outputQueues[camera_id].front();
            climb::outputQueues[camera_id].pop();
            // 複製結果到輸出
            count = std::min(result.count, max_output);
            memcpy(output, result.output, count * sizeof(svObjData_t));
            delete[] result.output;
        } else {
            AILOG_ERROR("Unknown function type: " + std::to_string(function));
            return -1;
        }
        return count; // 返回檢測到的物件數量
    }

    YOLOV11_API void svCreate_ROI(int camera_id, int function_id, int roi_id, int width, int height, float* points_x, float* points_y, int point_count) {
        if (roi_id == -1) {
            AILOG_ERROR("ROI ID cannot be -1, please provide a valid ROI ID.");
            return;
        }
        if (camera_function_roi_map.find(camera_id) != camera_function_roi_map.end()) {
            if (camera_function_roi_map[camera_id].find(function_id) != camera_function_roi_map[camera_id].end()) {
                if (camera_function_roi_map[camera_id][function_id].find(roi_id) != camera_function_roi_map[camera_id][function_id].end()) {
                    AILOG_WARN("ROI with id " + std::to_string(roi_id) + " already exists, updating.");
                }
            }
        }
        // 轉換 C 數組為 C++ vector
        std::vector<cv::Point2f> points;
        for (int i = 0; i < point_count; ++i) {
            points.emplace_back(points_x[i],points_y[i]);
        }
        cv::Mat mask;
        if (function_id == functions::CLIMB) {
            mask = cv::Mat::zeros(height, width, CV_8UC1);
        } else {
            mask = createROI(width, height, points);
        }
        camera_function_roi_map[camera_id][function_id][roi_id] = {mask, points, width, height};
    }

    YOLOV11_API void svRemove_ROIandWall(int camera_id, int function_id, int roi_id) {
        auto it = camera_function_roi_map.find(camera_id);
        if (it != camera_function_roi_map.end()) {
            auto it2 = it->second.find(function_id);
            if (it2 != it->second.end()) {
                auto it3 = it2->second.find(roi_id);
                if (it3 != it2->second.end()) {
                    it2->second.erase(it3);
                    AILOG_INFO("Removed ROI with id " + std::to_string(roi_id));
                } else {
                    AILOG_WARN("ROI with id " + std::to_string(roi_id) + " does not exist.");
                }
            }
        }
    }

    YOLOV11_API void svCreate_MRTRedlightROI(int camera_id, int function_id, int roi_id, int width, int height, float* points_x, float* points_y, int point_count) {
        if (roi_id == -1) {
            AILOG_ERROR("ROI ID cannot be -1, please provide a valid ROI ID.");
            return;
        }
        if (MRTRedlightROI_map.find(camera_id) != MRTRedlightROI_map.end()) {
            if (MRTRedlightROI_map[camera_id].find(function_id) != MRTRedlightROI_map[camera_id].end()) {
                if (MRTRedlightROI_map[camera_id][function_id].find(roi_id) != MRTRedlightROI_map[camera_id][function_id].end()) {
                    AILOG_WARN("ROI with id " + std::to_string(roi_id) + " already exists, updating.");
                }
            }
        }
        // 轉換 C 數組為 C++ vector
        std::vector<cv::Point2f> points;
        for (int i = 0; i < point_count; ++i) {
            points.emplace_back(points_x[i], points_y[i]);
        }
        cv::Mat mask = createROI(width, height, points);

        // Find the minimum and maximum coordinates from all points
        float min_x = points[0].x, max_x = points[0].x;
        float min_y = points[0].y, max_y = points[0].y;

        for (const auto& point : points) {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
        }

        cv::Point left_top = cv::Point(static_cast<int>(min_x * width), static_cast<int>(min_y * height));
        cv::Point right_bottom = cv::Point(static_cast<int>(max_x * width), static_cast<int>(max_y * height));
        MRTRedlightROI_map[camera_id][function_id][roi_id] = {mask, points, left_top, right_bottom, width, height, std::bitset<3>()};
        AILOG_INFO("Created MRTRedlight ROI with id " + std::to_string(roi_id) +
                 ", left_top: (" + std::to_string(left_top.x) + "," + std::to_string(left_top.y) + "), " +
                 "right_bottom: (" + std::to_string(right_bottom.x) + "," + std::to_string(right_bottom.y) + ")");
    }

    YOLOV11_API void svRemove_MRTRedlightROI(int camera_id, int function_id, int roi_id) {
        auto it = MRTRedlightROI_map.find(camera_id);
        if (it != MRTRedlightROI_map.end()) {
            auto it2 = it->second.find(function_id);
            if (it2 != it->second.end()) {
                auto it3 = it2->second.find(roi_id);
                if (it3 != it2->second.end()) {
                    it2->second.erase(it3);
                    AILOG_INFO("Removed MRTRedlight ROI with id " + std::to_string(roi_id));
                } else {
                    AILOG_WARN("MRTRedlight ROI with id " + std::to_string(roi_id) + " does not exist.");
                }
            }
        }
    }
    YOLOV11_API void svCreate_CrossingLine(int camera_id, int function_id, int roi_id,
            int width, int height, float* points_x, float* points_y, int point_count){
        if (roi_id == -1) {
            AILOG_ERROR("ROI ID cannot be -1, please provide a valid ROI ID.");
            return;
        }
        if (point_count < 2){
            AILOG_ERROR("Crossing Line ROI must have at least 2 points, received " + std::to_string(point_count) + " points.");
            return;
        }
        if (CrossingLineROI_map.find(camera_id) != CrossingLineROI_map.end()) {
            if (CrossingLineROI_map[camera_id].find(function_id) != CrossingLineROI_map[camera_id].end()) {
                if (CrossingLineROI_map[camera_id][function_id].find(roi_id) != CrossingLineROI_map[camera_id][function_id].end()) {
                    AILOG_WARN("Crossing Line ROI with id " + std::to_string(roi_id) + " already exists, updating.");
                }
            }
        }
        // 轉換 C 數組為 C++ vector
        std::vector<cv::Point2f> points;
        for (int i = 0; i < point_count; ++i) {
            points.emplace_back(points_x[i], points_y[i]);
        }

        // 計算每個線段的方向
        std::vector<int> in_area_direction;
        for (int i = 0; i < point_count - 1; ++i) {
            cv::Point2f q1 = points[i];
            cv::Point2f q2 = points[i + 1];

            // 計算q1和q2的中心點
            cv::Point2f center = cv::Point2f((q1.x + q2.x) / 2.0f, (q1.y + q2.y) / 2.0f);

            // 計算線段q1q2的方向向量
            cv::Point2f direction = cv::Point2f(q2.x - q1.x, q2.y - q1.y);

            // 將方向向量逆時針旋轉90度：(x, y) -> (-y, x)
            cv::Point2f perpendicular = cv::Point2f(-direction.y, direction.x);

            // 創建測試線段p1p2，從中心點延伸
            cv::Point2f p1 = cv::Point2f(center.x - perpendicular.x, center.y - perpendicular.y);
            cv::Point2f p2 = cv::Point2f(center.x + perpendicular.x, center.y + perpendicular.y);

            // 呼叫doIntersect得到方向
            int direction_result = GeometryUtils::doIntersect(q1, q2, p1, p2);
            in_area_direction.push_back(direction_result);
        }

        CrossingLineROI_map[camera_id][function_id][roi_id] = {points, width, height, in_area_direction};

        AILOG_INFO("Created Crossing Line ROI with id " + std::to_string(roi_id));
    }

    YOLOV11_API void svRemove_CrossingLine(int camera_id, int function_id, int roi_id){
        auto it = CrossingLineROI_map.find(camera_id);
        if (it != CrossingLineROI_map.end()) {
            auto it2 = it->second.find(function_id);
            if (it2 != it->second.end()) {
                auto it3 = it2->second.find(roi_id);
                if (it3 != it2->second.end()) {
                    it2->second.erase(it3);
                    AILOG_INFO("Removed Crossing Line ROI with id " + std::to_string(roi_id));
                } else {
                    AILOG_WARN("Crossing Line ROI with id " + std::to_string(roi_id) + " does not exist.");
                }
            }
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
