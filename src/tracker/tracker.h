#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <memory>

struct TrackerDetection {
    float x, y;          // 中心位置 (0~1)
    float width, height; // 框的寬高 (0~1)
    int class_id;
    float confidence;

    TrackerDetection() : x(0), y(0), width(0), height(0), class_id(-1), confidence(0) {}
    TrackerDetection(float cx, float cy, float w, float h, int cls, float conf)
        : x(cx), y(cy), width(w), height(h), class_id(cls), confidence(conf) {}
};

// 追蹤結果結構：包含檢測索引到追蹤ID的映射
struct TrackingResult {
    std::vector<int> detection_to_track_id;  // detection_to_track_id[i] = 檢測i對應的track_id，-1表示無匹配
    std::vector<int> new_track_ids;          // 新創建的track_id列表
    int total_active_tracks;                 // 當前活躍軌跡總數
};struct TrackedObject {
    int id;
    cv::KalmanFilter kalman;
    cv::Mat state;           // [x, y, vx, vy]
    cv::Mat measurement;     // [x, y]
    int frames_since_update; // 連續未配對的frame數
    bool is_confirmed;       // 是否已確認的軌跡
    float last_confidence;
    int class_id;

    // 儲存前一幀的邊界框資訊
    float prev_x, prev_y, prev_width, prev_height;        // 前一幀的實際輸入座標
    float prev_input_x, prev_input_y;                     // 前一幀的原始輸入座標（用於穿越線檢測）
    float current_width, current_height;                  // 當前幀的寬高
    bool has_previous_detection;                          // 是否有前一幀的檢測資訊

    TrackedObject(int obj_id, float x, float y, int cls_id, float conf);
    void predict();
    void update(float x, float y, float conf);
    void update(float x, float y, float width, float height, float conf); // 新增帶xywh的update函數
    cv::Point2f getPredictedPosition() const;
    bool isOutOfBounds() const;

    // 獲取前一幀的xywh
    struct PreviousDetection {
        float x, y, width, height;
        bool valid;
    };
    PreviousDetection getPreviousDetection() const;
};

class ObjectTracker {
private:
    int next_id;
    int max_frames_to_skip;
    float max_distance_threshold;
    std::unordered_map<int, std::unique_ptr<TrackedObject>> tracked_objects;

    // 匈牙利演算法相關
    std::vector<std::vector<float>> computeCostMatrix(
        const std::vector<TrackerDetection>& detections,
        const std::vector<int>& track_ids);

    std::vector<std::pair<int, int>> hungarianAssignment(
        const std::vector<std::vector<float>>& cost_matrix);    float euclideanDistance(float x1, float y1, float x2, float y2) const;

    // 匈牙利演算法實現
    bool findAugmentingPath(int u, std::vector<std::vector<float>>& cost,
                           std::vector<int>& match_x, std::vector<int>& match_y,
                           std::vector<bool>& used_y, std::vector<float>& slack_y,
                           std::vector<int>& slack_x, std::vector<float>& lx,
                           std::vector<float>& ly);

public:
    ObjectTracker(int max_skip_frames = 5, float max_dist_threshold = 0.3f);
    ~ObjectTracker() = default;

    // 主要接口：返回檢測索引到track_id的映射
    TrackingResult update(const std::vector<TrackerDetection>& detections);

    // 工具函數
    void reset();
    int getTrackedObjectCount() const;
    std::vector<int> getActiveTrackIds() const;

    // 獲取特定ID的前一幀檢測資訊
    TrackedObject::PreviousDetection getPreviousDetection(int track_id) const;

    // 設置參數
    void setMaxSkipFrames(int frames) { max_frames_to_skip = frames; }
    void setMaxDistanceThreshold(float threshold) { max_distance_threshold = threshold; }
};
