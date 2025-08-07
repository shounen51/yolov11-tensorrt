#include "tracker.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <iomanip>

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
    std::cout << "\n=== TRACKER UPDATE START ===" << std::endl;
    std::cout << "Current tracked objects: " << tracked_objects.size() << std::endl;
    std::cout << "New detections: " << detections.size() << std::endl;

    // 打印當前所有軌跡狀態
    for (const auto& pair : tracked_objects) {
        auto pos = pair.second->getPredictedPosition();
        std::cout << "Track ID " << pair.first
                  << " at (" << std::fixed << std::setprecision(3) << pos.x << ", " << pos.y << ")"
                  << " frames_since_update: " << pair.second->frames_since_update << std::endl;
    }

    // 打印檢測位置
    for (size_t i = 0; i < detections.size(); i++) {
        std::cout << "Detection " << i
                  << " at (" << std::fixed << std::setprecision(3) << detections[i].x << ", " << detections[i].y << ")" << std::endl;
    }

    // 1. 對所有現有軌跡進行預測
    for (auto& pair : tracked_objects) {
        pair.second->predict();
    }

    // 如果沒有現有軌跡，為所有檢測創建新軌跡
    if (tracked_objects.empty()) {
        std::cout << "No existing tracks, creating new tracks for all detections" << std::endl;
        for (const auto& det : detections) {
            tracked_objects[next_id] = std::make_unique<TrackedObject>(
                next_id, det.x, det.y, det.class_id, det.confidence);
            std::cout << "Created new track ID " << next_id
                      << " at (" << std::fixed << std::setprecision(3) << det.x << ", " << det.y << ")" << std::endl;
            next_id++;
        }
    }
    // 如果有軌跡但沒有檢測，處理軌跡過期
    else if (detections.empty()) {
        std::cout << "No detections, checking for expired tracks" << std::endl;
        std::vector<int> tracks_to_remove;
        for (const auto& pair : tracked_objects) {
            auto& track = pair.second;
            if (track->frames_since_update >= max_frames_to_skip || track->isOutOfBounds()) {
                tracks_to_remove.push_back(pair.first);
                std::cout << "Track ID " << pair.first << " will be removed (frames_since_update: "
                          << track->frames_since_update << ", out_of_bounds: " << track->isOutOfBounds() << ")" << std::endl;
            }
        }
        for (int track_id : tracks_to_remove) {
            tracked_objects.erase(track_id);
            std::cout << "Removed track ID " << track_id << std::endl;
        }
    }
    // 使用匈牙利演算法進行配對
    else {
        std::cout << "Using Hungarian algorithm for assignment" << std::endl;
        std::vector<int> track_ids;
        for (const auto& pair : tracked_objects) {
            track_ids.push_back(pair.first);
        }

        // 計算成本矩陣
        auto cost_matrix = computeCostMatrix(detections, track_ids);

        // 打印成本矩陣
        std::cout << "Cost Matrix:" << std::endl;
        for (size_t i = 0; i < cost_matrix.size(); i++) {
            std::cout << "Track " << track_ids[i] << ": ";
            for (size_t j = 0; j < cost_matrix[i].size(); j++) {
                if (cost_matrix[i][j] == std::numeric_limits<float>::max()) {
                    std::cout << "INF ";
                } else {
                    std::cout << std::fixed << std::setprecision(3) << cost_matrix[i][j] << " ";
                }
            }
            std::cout << std::endl;
        }

        // 使用匈牙利演算法進行最優配對
        auto assignments = hungarianAssignment(cost_matrix);

        std::cout << "Hungarian assignments (" << assignments.size() << " pairs):" << std::endl;
        for (const auto& assignment : assignments) {
            std::cout << "Track index " << assignment.first << " (ID " << track_ids[assignment.first]
                      << ") -> Detection " << assignment.second << std::endl;
        }

        // 記錄哪些檢測和軌跡已被匹配
        std::vector<bool> detection_matched(detections.size(), false);
        std::vector<bool> track_matched(track_ids.size(), false);

        // 處理匹配結果
        for (const auto& assignment : assignments) {
            int track_idx = assignment.first;
            int det_idx = assignment.second;

            if (track_idx < track_ids.size() && det_idx < detections.size()) {
                int track_id = track_ids[track_idx];
                const TrackerDetection& det = detections[det_idx];

                // 只有在距離閾值內才進行更新，避免錯誤匹配
                auto predicted_pos = tracked_objects[track_id]->getPredictedPosition();
                float distance = euclideanDistance(predicted_pos.x, predicted_pos.y, det.x, det.y);

                std::cout << "Checking assignment: Track ID " << track_id
                          << " predicted at (" << std::fixed << std::setprecision(3) << predicted_pos.x << ", " << predicted_pos.y << ")"
                          << " vs Detection at (" << det.x << ", " << det.y << ")"
                          << " distance: " << distance << " threshold: " << max_distance_threshold << std::endl;

                if (distance <= max_distance_threshold) {
                    tracked_objects[track_id]->update(det.x, det.y, det.width, det.height, det.confidence);
                    detection_matched[det_idx] = true;
                    track_matched[track_idx] = true;
                    std::cout << "✓ Track ID " << track_id << " updated with detection " << det_idx << std::endl;
                } else {
                    std::cout << "✗ Track ID " << track_id << " rejected (distance too large)" << std::endl;
                }
            }
        }

        // 處理未配對的軌跡
        std::vector<int> tracks_to_remove;
        for (size_t i = 0; i < track_ids.size(); i++) {
            if (!track_matched[i]) {
                int track_id = track_ids[i];
                auto& track = tracked_objects[track_id];

                std::cout << "Unmatched track ID " << track_id
                          << " frames_since_update: " << track->frames_since_update
                          << " out_of_bounds: " << track->isOutOfBounds() << std::endl;

                // 檢查是否應該刪除軌跡
                if (track->frames_since_update >= max_frames_to_skip || track->isOutOfBounds()) {
                    tracks_to_remove.push_back(track_id);
                    std::cout << "Track ID " << track_id << " will be removed" << std::endl;
                }
            }
        }

        // 刪除過期的軌跡
        for (int track_id : tracks_to_remove) {
            tracked_objects.erase(track_id);
            std::cout << "Removed track ID " << track_id << std::endl;
        }

        // 為未配對的檢測創建新軌跡
        for (size_t i = 0; i < detections.size(); i++) {
            if (!detection_matched[i]) {
                const TrackerDetection& det = detections[i];
                tracked_objects[next_id] = std::make_unique<TrackedObject>(
                    next_id, det.x, det.y, det.class_id, det.confidence);
                std::cout << "Created new track ID " << next_id
                          << " for unmatched detection " << i
                          << " at (" << std::fixed << std::setprecision(3) << det.x << ", " << det.y << ")" << std::endl;
                next_id++;
            }
        }
    }

    // 返回當前所有活躍軌跡的結果
    std::vector<std::pair<int, TrackerDetection>> results;
    for (const auto& pair : tracked_objects) {
        const auto& track = pair.second;
        auto pos = track->getPredictedPosition();

        TrackerDetection det;
        det.x = pos.x;
        det.y = pos.y;
        det.width = track->current_width;
        det.height = track->current_height;
        det.class_id = track->class_id;
        det.confidence = track->last_confidence;

        results.emplace_back(track->id, det);
    }

    std::cout << "Final active tracks: " << results.size() << std::endl;
    for (const auto& result : results) {
        std::cout << "Output Track ID " << result.first
                  << " at (" << std::fixed << std::setprecision(3) << result.second.x << ", " << result.second.y << ")" << std::endl;
    }
    std::cout << "=== TRACKER UPDATE END ===\n" << std::endl;

    return results;
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
        std::cout << "Empty cost matrix, returning empty assignments" << std::endl;
        return {};
    }

    int n = cost_matrix.size();    // 軌跡數量
    int m = cost_matrix[0].size(); // 檢測數量

    std::cout << "Hungarian algorithm: " << n << " tracks, " << m << " detections" << std::endl;

    // 確保是方形矩陣，匈牙利演算法需要方形矩陣
    int size = std::max(n, m);
    std::vector<std::vector<float>> matrix(size, std::vector<float>(size, 1000.0f));

    // 複製原始成本矩陣到方形矩陣
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (cost_matrix[i][j] == std::numeric_limits<float>::max()) {
                matrix[i][j] = 1000.0f; // 使用大數值代替無限大
            } else {
                matrix[i][j] = cost_matrix[i][j];
            }
        }
    }

    std::cout << "Converted matrix:" << std::endl;
    for (int i = 0; i < std::min(n, 5); i++) {
        for (int j = 0; j < std::min(m, 5); j++) {
            std::cout << std::fixed << std::setprecision(3) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // 匈牙利演算法變量
    std::vector<float> lx(size, 0), ly(size, 0);
    std::vector<int> match_x(size, -1), match_y(size, -1);

    // 步驟1：對於最小成本問題，初始化行標籤為0，我們通過減法來處理
    // 實際上我們需要將問題轉換為最大權重匹配問題
    float max_cost = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (matrix[i][j] < 999.0f) { // 忽略無效的大數值
                max_cost = std::max(max_cost, matrix[i][j]);
            }
        }
    }
    max_cost += 1.0f; // 增加一點邊距

    // 將最小成本問題轉換為最大權重問題：新權重 = max_cost - 原成本
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (matrix[i][j] < 999.0f) {
                matrix[i][j] = max_cost - matrix[i][j];
            } else {
                matrix[i][j] = 0; // 無效配對的權重為0
            }
        }
    }

    // 初始化行標籤為行最大值
    for (int i = 0; i < size; i++) {
        lx[i] = *std::max_element(matrix[i].begin(), matrix[i].end());
    }
    // 列標籤初始化為0

    std::cout << "Max cost: " << max_cost << std::endl;
    std::cout << "Converted to max weight matrix (first 3x3):" << std::endl;
    for (int i = 0; i < std::min(size, 3); i++) {
        for (int j = 0; j < std::min(size, 3); j++) {
            std::cout << std::fixed << std::setprecision(3) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Initial labels - lx: ";
    for (int i = 0; i < std::min(size, 5); i++) {
        std::cout << std::fixed << std::setprecision(3) << lx[i] << " ";
    }
    std::cout << std::endl;

    // 主循環：為每個軌跡找匹配
    for (int i = 0; i < size; i++) {
        std::vector<float> slack_y(size, std::numeric_limits<float>::max());
        std::vector<int> slack_x(size, -1);

        while (true) {
            std::vector<bool> used_y(size, false);
            if (findAugmentingPath(i, matrix, match_x, match_y, used_y, slack_y, slack_x, lx, ly)) {
                std::cout << "Found augmenting path for track " << i << std::endl;
                break;
            }

            // 找到最小的 delta 來更新標籤
            float delta = std::numeric_limits<float>::max();
            for (int j = 0; j < size; j++) {
                if (!used_y[j] && slack_y[j] < delta) {
                    delta = slack_y[j];
                }
            }

            std::cout << "Delta for track " << i << ": " << std::fixed << std::setprecision(6) << delta << std::endl;

            // 如果找不到有效的 delta，跳出
            if (delta == std::numeric_limits<float>::max() || delta <= 1e-9) {
                std::cout << "No valid delta found, breaking" << std::endl;
                break;
            }

            // 更新標籤
            lx[i] -= delta;
            for (int j = 0; j < size; j++) {
                if (used_y[j]) {
                    ly[j] += delta;
                } else {
                    slack_y[j] -= delta;
                }
            }
        }
    }

    // 構建結果，只返回有效的配對
    std::vector<std::pair<int, int>> assignments;
    std::cout << "Hungarian result matches:" << std::endl;
    for (int i = 0; i < n; i++) {
        if (match_x[i] != -1 && match_x[i] < m) {
            // 檢查原始成本矩陣中的值是否有效
            if (cost_matrix[i][match_x[i]] != std::numeric_limits<float>::max()) {
                assignments.emplace_back(i, match_x[i]);
                std::cout << "  Track index " << i << " -> Detection " << match_x[i]
                          << " (cost: " << std::fixed << std::setprecision(3) << cost_matrix[i][match_x[i]] << ")" << std::endl;
            } else {
                std::cout << "  Track index " << i << " -> Detection " << match_x[i]
                          << " (REJECTED: infinite cost)" << std::endl;
            }
        } else {
            std::cout << "  Track index " << i << " -> NO MATCH" << std::endl;
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

        // 對於最大權重問題：delta = lx[u] + ly[v] - weight[u][v]
        float delta = lx[u] + ly[v] - cost[u][v];

        // 如果這是一個"緊邊"（reduced cost為0）
        if (std::abs(delta) < 1e-6) {
            used_y[v] = true;
            // 如果這個檢測還沒有匹配，或者可以為它當前的匹配找到新的路徑
            if (match_y[v] == -1 || findAugmentingPath(match_y[v], cost, match_x, match_y,
                                                      used_y, slack_y, slack_x, lx, ly)) {
                match_x[u] = v;
                match_y[v] = u;
                return true;
            }
        } else if (delta > 0 && delta < slack_y[v]) {
            // 只有當delta為正時才更新slack（對於最大權重問題）
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
