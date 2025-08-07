#include "tracker.h"
#include <algorithm>
#include <limits>
#include <cmath>

// TrackedObject 實現
TrackedObject::TrackedObject(int obj_id, float x, float y, int cls_id, float conf)
    : id(obj_id), frames_since_update(0), is_confirmed(false),
      last_confidence(conf), class_id(cls_id),
      prev_x(0), prev_y(0), prev_width(0), prev_height(0),
      current_width(0), current_height(0), has_previous_detection(false) {

    // 初始化卡爾曼濾波器 (4狀態變量: x, y, vx, vy; 2測量變量: x, y)
    kalman.init(4, 2, 0, CV_32F);

    // 狀態轉移矩陣 A (等速運動模型)
    kalman.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,   // x = x + vx
        0, 1, 0, 1,   // y = y + vy
        0, 0, 1, 0,   // vx = vx
        0, 0, 0, 1);  // vy = vy

    // 測量矩陣 H
    kalman.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,   // 測量 x
        0, 1, 0, 0);  // 測量 y

    // 過程噪聲協方差矩陣 Q
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-4));

    // 測量噪聲協方差矩陣 R
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-1));

    // 後驗錯誤估計協方差矩陣 P
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(0.1));

    // 初始化狀態 [x, y, 0, 0]
    kalman.statePre = (cv::Mat_<float>(4, 1) << x, y, 0, 0);
    kalman.statePost = (cv::Mat_<float>(4, 1) << x, y, 0, 0);

    // 初始化 state 變數
    state = kalman.statePost.clone();

    measurement = cv::Mat::zeros(2, 1, CV_32F);
}

void TrackedObject::predict() {
    state = kalman.predict();
    frames_since_update++;
}

void TrackedObject::update(float x, float y, float conf) {
    measurement.at<float>(0) = x;
    measurement.at<float>(1) = y;

    state = kalman.correct(measurement);
    frames_since_update = 0;
    last_confidence = conf;
    is_confirmed = true;
}

void TrackedObject::update(float x, float y, float width, float height, float conf) {
    // 儲存當前狀態作為前一幀資訊（在更新之前）
    if (is_confirmed) {
        prev_x = state.at<float>(0);
        prev_y = state.at<float>(1);
        prev_width = current_width;
        prev_height = current_height;
        has_previous_detection = true;
    }

    // 更新當前的xywh
    current_width = width;
    current_height = height;

    measurement.at<float>(0) = x;
    measurement.at<float>(1) = y;

    state = kalman.correct(measurement);
    frames_since_update = 0;
    last_confidence = conf;
    is_confirmed = true;
}

cv::Point2f TrackedObject::getPredictedPosition() const {
    return cv::Point2f(state.at<float>(0), state.at<float>(1));
}

TrackedObject::PreviousDetection TrackedObject::getPreviousDetection() const {
    PreviousDetection prev;
    prev.valid = has_previous_detection;
    if (has_previous_detection) {
        prev.x = prev_x;
        prev.y = prev_y;
        prev.width = prev_width;
        prev.height = prev_height;
    } else {
        prev.x = prev.y = prev.width = prev.height = 0;
    }
    return prev;
}

bool TrackedObject::isOutOfBounds() const {
    float x = state.at<float>(0);
    float y = state.at<float>(1);

    // 考慮一定的邊界容差
    const float margin = 0.1f;
    return (x < -margin || x > 1.0f + margin ||
            y < -margin || y > 1.0f + margin);
}

// ObjectTracker 實現
ObjectTracker::ObjectTracker(int max_skip_frames, float max_dist_threshold)
    : next_id(1), max_frames_to_skip(max_skip_frames),
      max_distance_threshold(max_dist_threshold) {
}

std::vector<std::pair<int, TrackerDetection>> ObjectTracker::update(const std::vector<TrackerDetection>& detections) {
    // 1. 對所有現有軌跡進行預測
    for (auto& pair : tracked_objects) {
        pair.second->predict();
    }

    // 簡化版本：使用最近鄰配對，不使用匈牙利演算法
    std::vector<bool> detection_matched(detections.size(), false);
    std::vector<int> track_ids;
    for (const auto& pair : tracked_objects) {
        track_ids.push_back(pair.first);
    }
    std::vector<bool> track_matched(track_ids.size(), false);

    // 2. 為每個檢測找到最近的軌跡
    for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
        if (detection_matched[det_idx]) continue;

        const TrackerDetection& det = detections[det_idx];
        float min_distance = std::numeric_limits<float>::max();
        int best_track_idx = -1;

        for (size_t track_idx = 0; track_idx < track_ids.size(); ++track_idx) {
            if (track_matched[track_idx]) continue;

            int track_id = track_ids[track_idx];
            auto predicted_pos = tracked_objects[track_id]->getPredictedPosition();
            float distance = euclideanDistance(predicted_pos.x, predicted_pos.y, det.x, det.y);

            if (distance < min_distance && distance <= max_distance_threshold) {
                min_distance = distance;
                best_track_idx = track_idx;
            }
        }

        // 如果找到了匹配的軌跡
        if (best_track_idx != -1) {
            int track_id = track_ids[best_track_idx];
            tracked_objects[track_id]->update(det.x, det.y, det.width, det.height, det.confidence);
            detection_matched[det_idx] = true;
            track_matched[best_track_idx] = true;
        }
    }

    // 5. 處理未配對的軌跡
    std::vector<int> tracks_to_remove;
    for (size_t i = 0; i < track_ids.size(); i++) {
        if (!track_matched[i]) {
            int track_id = track_ids[i];
            auto& track = tracked_objects[track_id];

            // 檢查是否應該刪除軌跡
            if (track->frames_since_update >= max_frames_to_skip || track->isOutOfBounds()) {
                tracks_to_remove.push_back(track_id);
            }
        }
    }

    // 刪除過期的軌跡
    for (int track_id : tracks_to_remove) {
        tracked_objects.erase(track_id);
    }

    // 6. 為未配對的檢測創建新軌跡
    for (size_t i = 0; i < detections.size(); i++) {
        if (!detection_matched[i]) {
            const TrackerDetection& det = detections[i];
            tracked_objects[next_id] = std::make_unique<TrackedObject>(
                next_id, det.x, det.y, det.class_id, det.confidence);
            next_id++;
        }
    }

    // 7. 返回當前所有活躍軌跡的結果
    std::vector<std::pair<int, TrackerDetection>> results;
    for (const auto& pair : tracked_objects) {
        const auto& track = pair.second;
        auto pos = track->getPredictedPosition();

        TrackerDetection det;
        det.x = pos.x;
        det.y = pos.y;
        det.class_id = track->class_id;
        det.confidence = track->last_confidence;

        results.emplace_back(track->id, det);
    }    return results;
}

std::vector<std::vector<float>> ObjectTracker::computeCostMatrix(
    const std::vector<TrackerDetection>& detections,
    const std::vector<int>& track_ids) {

    std::vector<std::vector<float>> cost_matrix(track_ids.size(),
                                               std::vector<float>(detections.size()));

    for (size_t i = 0; i < track_ids.size(); i++) {
        int track_id = track_ids[i];
        auto predicted_pos = tracked_objects[track_id]->getPredictedPosition();

        for (size_t j = 0; j < detections.size(); j++) {
            const TrackerDetection& det = detections[j];
            float distance = euclideanDistance(predicted_pos.x, predicted_pos.y, det.x, det.y);

            // 如果距離超過閾值，設置為很大的成本
            if (distance > max_distance_threshold) {
                cost_matrix[i][j] = std::numeric_limits<float>::max();
            } else {
                cost_matrix[i][j] = distance;
            }
        }
    }

    return cost_matrix;
}

float ObjectTracker::euclideanDistance(float x1, float y1, float x2, float y2) const {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<std::pair<int, int>> ObjectTracker::hungarianAssignment(
    const std::vector<std::vector<float>>& cost_matrix) {

    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return {};
    }

    int n = cost_matrix.size();
    int m = cost_matrix[0].size();

    // 如果軌跡數量大於檢測數量，需要擴展矩陣
    int size = std::max(n, m);
    std::vector<std::vector<float>> matrix(size, std::vector<float>(size, std::numeric_limits<float>::max()));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i][j] = cost_matrix[i][j];
        }
    }

    // 匈牙利演算法變量
    std::vector<float> lx(size, 0), ly(size, 0);
    std::vector<int> match_x(size, -1), match_y(size, -1);

    // 初始化標籤
    for (int i = 0; i < size; i++) {
        lx[i] = *std::max_element(matrix[i].begin(), matrix[i].end());
        if (lx[i] == std::numeric_limits<float>::max()) {
            lx[i] = 0;
        }
    }

    // 主循環
    for (int i = 0; i < size; i++) {
        std::vector<float> slack_y(size, std::numeric_limits<float>::max());
        std::vector<int> slack_x(size, -1);

        while (true) {
            std::vector<bool> used_y(size, false);
            if (findAugmentingPath(i, matrix, match_x, match_y, used_y, slack_y, slack_x, lx, ly)) {
                break;
            }

            // 更新標籤
            float delta = std::numeric_limits<float>::max();
            for (int j = 0; j < size; j++) {
                if (!used_y[j]) {
                    delta = std::min(delta, slack_y[j]);
                }
            }

            if (delta == std::numeric_limits<float>::max()) {
                break;
            }

            for (int j = 0; j < size; j++) {
                if (used_y[j]) {
                    ly[j] += delta;
                }
            }
            lx[i] -= delta;

            for (int j = 0; j < size; j++) {
                if (!used_y[j]) {
                    slack_y[j] -= delta;
                }
            }
        }
    }

    // 構建結果
    std::vector<std::pair<int, int>> assignments;
    for (int i = 0; i < n; i++) {
        if (match_x[i] != -1 && match_x[i] < m &&
            cost_matrix[i][match_x[i]] != std::numeric_limits<float>::max()) {
            assignments.emplace_back(i, match_x[i]);
        }
    }

    return assignments;
}

bool ObjectTracker::findAugmentingPath(int u, std::vector<std::vector<float>>& cost,
                                      std::vector<int>& match_x, std::vector<int>& match_y,
                                      std::vector<bool>& used_y, std::vector<float>& slack_y,
                                      std::vector<int>& slack_x, std::vector<float>& lx,
                                      std::vector<float>& ly) {

    int size = cost.size();
    for (int v = 0; v < size; v++) {
        if (used_y[v]) continue;

        float delta = lx[u] + ly[v] - cost[u][v];
        if (delta < 1e-6) {  // 相等邊
            used_y[v] = true;
            if (match_y[v] == -1 || findAugmentingPath(match_y[v], cost, match_x, match_y,
                                                      used_y, slack_y, slack_x, lx, ly)) {
                match_x[u] = v;
                match_y[v] = u;
                return true;
            }
        } else if (slack_y[v] > delta) {
            slack_y[v] = delta;
            slack_x[v] = u;
        }
    }
    return false;
}

void ObjectTracker::reset() {
    tracked_objects.clear();
    next_id = 1;
}

int ObjectTracker::getTrackedObjectCount() const {
    return tracked_objects.size();
}

std::vector<int> ObjectTracker::getActiveTrackIds() const {
    std::vector<int> ids;
    for (const auto& pair : tracked_objects) {
        ids.push_back(pair.first);
    }
    return ids;
}

TrackedObject::PreviousDetection ObjectTracker::getPreviousDetection(int track_id) const {
    auto it = tracked_objects.find(track_id);
    if (it != tracked_objects.end()) {
        return it->second->getPreviousDetection();
    }

    // 如果找不到該ID，返回無效的檢測
    TrackedObject::PreviousDetection invalid;
    invalid.valid = false;
    invalid.x = invalid.y = invalid.width = invalid.height = 0;
    return invalid;
}
