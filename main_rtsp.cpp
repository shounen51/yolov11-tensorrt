#include "yolov11_dll.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

// 模式枚舉
enum DrawMode {
    NORMAL_ROI_MODE = 0,
    REDLIGHT_ROI_MODE = 1,
    CROSSINGLINE_ROI_MODE = 2
};

// Global variables for mouse callback
vector<Point2f> clicked_points;  // Store all clicked p oints
bool point_clicked = false;
bool close_polygon = false;     // Flag to close the polygon
int frame_width_global = 0;
int frame_height_global = 0;
functions current_function = functions::YOLO_COLOR;  // Store current function
int current_camera_id = 0;  // Store current camera ID

// Mode switching variables
DrawMode current_draw_mode = NORMAL_ROI_MODE;
vector<Point2f> redlight_clicked_points;  // Store redlight ROI points
bool redlight_close_polygon = false;
int next_redlight_roi_id = 0;  // For assigning redlight ROI IDs

// Store created redlight ROIs for display
vector<vector<Point2f>> created_redlight_rois;  // Store all created redlight ROI polygons

// Crossing line variables
vector<Point2f> crossingline_clicked_points;  // Store crossing line points (max 4)
bool crossingline_created = false;
vector<Point2f> created_crossing_line;  // Store the created crossing line points (4 points)

// Video orientation correction variables
bool video_needs_rotation = false;
int rotation_type = ROTATE_180;  // Default to 180 degree rotation for upside-down videos

// Global variables for cumulative red box counting
int total_red_box_count = 0;  // Cumulative count of red boxes

// Function to auto-correct frame orientation
Mat correctFrameOrientation(const Mat& input_frame, bool check_orientation = false, int user_rotation = 0) {
    Mat corrected_frame;

    if (check_orientation) {
        // Set rotation based on user input
        if (user_rotation == 90) {
            video_needs_rotation = true;
            rotation_type = ROTATE_90_CLOCKWISE;
            cout << "[INFO] User specified 90-degree rotation (clockwise)" << endl;
        } else if (user_rotation == 180) {
            video_needs_rotation = true;
            rotation_type = ROTATE_180;
            cout << "[INFO] User specified 180-degree rotation (upside-down correction)" << endl;
        } else if (user_rotation == 270) {
            video_needs_rotation = true;
            rotation_type = ROTATE_90_COUNTERCLOCKWISE;
            cout << "[INFO] User specified 270-degree rotation (counterclockwise)" << endl;
        } else {
            video_needs_rotation = false;
            cout << "[INFO] No rotation applied (user specified 0 or default)" << endl;
        }

        if (video_needs_rotation) {
            rotate(input_frame, corrected_frame, rotation_type);
            cout << "[INFO] Frame rotated: " << input_frame.cols << "x" << input_frame.rows
                 << " -> " << corrected_frame.cols << "x" << corrected_frame.rows << endl;
        } else {
            corrected_frame = input_frame.clone();
        }
    } else {
        // Subsequent frames - apply the determined rotation
        if (video_needs_rotation) {
            rotate(input_frame, corrected_frame, rotation_type);
        } else {
            corrected_frame = input_frame.clone();
        }
    }

    return corrected_frame;
}

// Mouse callback function
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point2f new_point;
        new_point.x = (float)x / frame_width_global;  // Normalize to 0-1
        new_point.y = (float)y / frame_height_global; // Normalize to 0-1

        if (current_draw_mode == NORMAL_ROI_MODE) {
            clicked_points.push_back(new_point);
            close_polygon = false;  // Reset close flag when adding new point
            cout << "[MOUSE] Normal ROI - Added point " << clicked_points.size()
                 << " at pixel(" << x << ", " << y << ") = normalized("
                 << new_point.x << ", " << new_point.y << ")" << endl;
        } else if (current_draw_mode == REDLIGHT_ROI_MODE) {
            redlight_clicked_points.push_back(new_point);
            redlight_close_polygon = false;  // Reset close flag when adding new point
            cout << "[MOUSE] Redlight ROI - Added point " << redlight_clicked_points.size()
                 << " at pixel(" << x << ", " << y << ") = normalized("
                 << new_point.x << ", " << new_point.y << ")" << endl;
        } else if (current_draw_mode == CROSSINGLINE_ROI_MODE) {
            if (crossingline_clicked_points.size() < 4) {  // Maximum 4 points for crossing line
                crossingline_clicked_points.push_back(new_point);
                cout << "[MOUSE] Crossing Line - Added point " << crossingline_clicked_points.size()
                     << " at pixel(" << x << ", " << y << ") = normalized("
                     << new_point.x << ", " << new_point.y << ")" << endl;
            } else {
                cout << "[MOUSE] Crossing Line - Maximum 4 points allowed" << endl;
            }
        }

        point_clicked = true;
    }
    else if (event == EVENT_RBUTTONDOWN) {
        if (current_draw_mode == NORMAL_ROI_MODE) {
            if (clicked_points.size() >= 3) {  // Need at least 3 points to create ROI
                close_polygon = true;

                // Remove existing ROI
                svRemove_ROIandWall(current_camera_id, current_function, 0);
                cout << "[MOUSE] Normal ROI - Removed existing ROI" << endl;

                // Create new ROI using clicked points
                vector<float> points_x, points_y;
                for (const auto& pt : clicked_points) {
                    points_x.push_back(pt.x);  // Already normalized 0-1
                    points_y.push_back(pt.y);  // Already normalized 0-1
                }

                // Create new ROI using clicked points (same for all functions)
                svCreate_ROI(current_camera_id, current_function, 0, frame_width_global, frame_height_global,
                            points_x.data(), points_y.data(), static_cast<int>(clicked_points.size()));

                if (current_function == functions::CLIMB) {
                    cout << "[MOUSE] Normal ROI - Created new WALL (non-closed) for CLIMB with " << clicked_points.size() << " points" << endl;
                } else {
                    cout << "[MOUSE] Normal ROI - Created new ROI with " << clicked_points.size() << " points" << endl;
                }
            } else {
                cout << "[MOUSE] Normal ROI - Need at least 3 points to create ROI (current: "
                     << clicked_points.size() << ")" << endl;
            }
        } else if (current_draw_mode == REDLIGHT_ROI_MODE) {
            if (redlight_clicked_points.size() >= 3) {  // Need at least 3 points to create ROI
                redlight_close_polygon = true;

                // Create new MRT Redlight ROI using clicked points
                vector<float> points_x, points_y;
                for (const auto& pt : redlight_clicked_points) {
                    points_x.push_back(pt.x);  // Already normalized 0-1
                    points_y.push_back(pt.y);  // Already normalized 0-1
                }

                // Create new MRT Redlight ROI
                svCreate_MRTRedlightROI(current_camera_id, current_function, next_redlight_roi_id,
                                       frame_width_global, frame_height_global,
                                       points_x.data(), points_y.data(), static_cast<int>(redlight_clicked_points.size()));

                cout << "[MOUSE] Redlight ROI - Created new MRT Redlight ROI " << next_redlight_roi_id
                     << " with " << redlight_clicked_points.size() << " points" << endl;

                // Save the created redlight ROI for display
                created_redlight_rois.push_back(redlight_clicked_points);

                next_redlight_roi_id++;

                // Clear points after creating ROI (ready for next ROI)
                redlight_clicked_points.clear();
                redlight_close_polygon = false;
            } else {
                cout << "[MOUSE] Redlight ROI - Need at least 3 points to create ROI (current: "
                     << redlight_clicked_points.size() << ")" << endl;
            }
        } else if (current_draw_mode == CROSSINGLINE_ROI_MODE) {
            if (crossingline_clicked_points.size() == 4) {  // Need exactly 4 points for crossing line
                // Remove existing crossing line
                svRemove_CrossingLine(current_camera_id, current_function, 0);
                cout << "[MOUSE] Crossing Line - Removed existing crossing line" << endl;

                // Create new crossing line using clicked points
                vector<float> points_x, points_y;
                for (const auto& pt : crossingline_clicked_points) {
                    points_x.push_back(pt.x);  // Already normalized 0-1
                    points_y.push_back(pt.y);  // Already normalized 0-1
                }

                // Create new crossing line
                svCreate_CrossingLine(current_camera_id, current_function, 0,
                                    frame_width_global, frame_height_global,
                                    points_x.data(), points_y.data(), 4);

                cout << "[MOUSE] Crossing Line - Created new crossing line with 4 points" << endl;

                // Save the created crossing line for display
                created_crossing_line = crossingline_clicked_points;
                crossingline_created = true;

                // Clear points after creating crossing line
                crossingline_clicked_points.clear();
            } else {
                cout << "[MOUSE] Crossing Line - Need exactly 4 points to create crossing line (current: "
                     << crossingline_clicked_points.size() << ")" << endl;
            }
        }
    }
    else if (event == EVENT_MBUTTONDOWN) {
        // Middle click to clear all points and remove ROI/Wall
        if (current_draw_mode == NORMAL_ROI_MODE) {
            clicked_points.clear();
            close_polygon = false;
            // Remove existing ROI/Wall without creating new one
            svRemove_ROIandWall(current_camera_id, current_function, 0);
            cout << "[MOUSE] Normal ROI - Cleared all points and removed ROI/Wall" << endl;
        } else if (current_draw_mode == REDLIGHT_ROI_MODE) {
            redlight_clicked_points.clear();
            redlight_close_polygon = false;
            created_redlight_rois.clear();  // Clear all created redlight ROIs from display
            // Remove all MRT Redlight ROIs
            for (int i = 0; i < next_redlight_roi_id; i++) {
                svRemove_MRTRedlightROI(current_camera_id, current_function, i);
            }
            next_redlight_roi_id = 0;
            cout << "[MOUSE] Redlight ROI - Cleared all points and removed all MRT Redlight ROIs" << endl;
        } else if (current_draw_mode == CROSSINGLINE_ROI_MODE) {
            crossingline_clicked_points.clear();
            created_crossing_line.clear();
            crossingline_created = false;
            // Remove crossing line
            svRemove_CrossingLine(current_camera_id, current_function, 0);
            cout << "[MOUSE] Crossing Line - Cleared all points and removed crossing line" << endl;
        }

        point_clicked = false;
    }
}// Helper function to parse function from string
functions parseFunction(const string& func_str) {
    if (func_str == "YOLO_COLOR" || func_str == "yolo") return functions::YOLO_COLOR;
    if (func_str == "FALL" || func_str == "fall") return functions::FALL;
    if (func_str == "CLIMB" || func_str == "climb") return functions::CLIMB;
    return functions::YOLO_COLOR; // default
}

// Helper function to get function name as string
const char* getFunctionName(functions func) {
    switch (func) {
        case functions::YOLO_COLOR: return "YOLO_COLOR";
        case functions::FALL: return "FALL";
        case functions::CLIMB: return "CLIMB";
        default: return "UNKNOWN";
    }
}

void printUsage(const char* program_name) {
    cout << "\n=== RTSP Detection System ===" << endl;
    cout << "用法: " << program_name << " <function> <input_source> <log_file> [target_fps] [rotation]" << endl;
    cout << "\n参数:" << endl;
    cout << "  function      检测功能 (yolo|fall|climb)" << endl;
    cout << "  input_source  RTSP流地址或视频文件路径" << endl;
    cout << "  log_file      日志文件路径" << endl;
    cout << "  target_fps    可选：目标FPS (跳帧处理，如5表示每秒处理5帧)" << endl;
    cout << "  rotation      可选：视频旋转角度 (0|90|180|270)" << endl;
    cout << "\n示例:" << endl;
    cout << "  " << program_name << " yolo rtsp://admin:admin123@192.168.1.100:554/stream1 log/log.log" << endl;
    cout << "  " << program_name << " fall rtsp://192.168.1.100:8554/stream log/fall.log" << endl;
    cout << "  " << program_name << " climb video.mp4 log/climb.log 10" << endl;
    cout << "  " << program_name << " yolo test.avi log/yolo.log 5 180" << endl;
    cout << "\n交互控制:" << endl;
    cout << "  鼠标左键    点击添加点到多边形/线段" << endl;
    cout << "  鼠标右键    创建ROI区域/穿越线段" << endl;
    cout << "  鼠标中键    清除所有点并重置" << endl;
    cout << "  Tab键       切换ROI模式(正常/红灯/穿越线)" << endl;
    cout << "  ESC键       退出程序" << endl;
    cout << "\n穿越线模式:" << endl;
    cout << "  - 需要4个点：前2个点构成第一条线段，后2个点构成第二条线段" << endl;
    cout << "  - 两条线段用不同颜色显示（青色和黄色）" << endl;
    cout << "  - 用于检测物体穿越特定线段的行为" << endl;
}

void drawDetectionResults(Mat& frame, svObjData_t* results, int num_objects, functions function_type) {
    int frame_width = frame.cols;
    int frame_height = frame.rows;

    // Draw mode information at the top
    string mode_text;
    Scalar mode_color;
    if (current_draw_mode == NORMAL_ROI_MODE) {
        mode_text = "Mode: Normal ROI (Tab to switch)";
        mode_color = Scalar(0, 255, 0);  // Green
    } else if (current_draw_mode == REDLIGHT_ROI_MODE) {
        mode_text = "Mode: Redlight ROI (Tab to switch)";
        mode_color = Scalar(0, 0, 255);  // Red
    } else {
        mode_text = "Mode: Crossing Line ROI (Tab to switch)";
        mode_color = Scalar(255, 0, 255);  // Magenta
    }
    putText(frame, mode_text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2);

    // Always draw normal ROI points if they exist
    if (!clicked_points.empty()) {
        // Convert normalized coordinates to pixel coordinates
        vector<Point> pixel_points;
        for (const auto& pt : clicked_points) {
            int x = static_cast<int>(pt.x * frame_width);
            int y = static_cast<int>(pt.y * frame_height);
            pixel_points.push_back(Point(x, y));
        }

        // Draw points as small circles
        for (size_t i = 0; i < pixel_points.size(); i++) {
            // Use different colors based on ROI state
            Scalar point_color = close_polygon ? Scalar(255, 188, 0) : Scalar(0, 225, 225);
            circle(frame, pixel_points[i], 5, point_color, -1);

            // Draw point number
            string point_text = to_string(i + 1);
            putText(frame, point_text, Point(pixel_points[i].x + 10, pixel_points[i].y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1);
        }

        // Draw lines between adjacent points
        for (size_t i = 0; i < pixel_points.size() - 1; i++) {
            Scalar line_color = close_polygon ? Scalar(255, 188, 0) : Scalar(0, 225, 225);
            line(frame, pixel_points[i], pixel_points[i + 1], line_color, 2);
        }

        // Draw closing line if polygon is closed (but not for CLIMB function)
        if (close_polygon && pixel_points.size() >= 3 && current_function != functions::CLIMB) {
            line(frame, pixel_points.back(), pixel_points.front(), Scalar(255, 188, 0), 2);
        }
    }

    // Always draw all created redlight ROIs
    for (size_t roi_idx = 0; roi_idx < created_redlight_rois.size(); roi_idx++) {
        const auto& roi_points = created_redlight_rois[roi_idx];

        // Convert normalized coordinates to pixel coordinates
        vector<Point> pixel_points;
        for (const auto& pt : roi_points) {
            int x = static_cast<int>(pt.x * frame_width);
            int y = static_cast<int>(pt.y * frame_height);
            pixel_points.push_back(Point(x, y));
        }

        // Draw points as small circles (red color for redlight ROI)
        for (size_t i = 0; i < pixel_points.size(); i++) {
            circle(frame, pixel_points[i], 4, Scalar(0, 0, 255), -1);  // Smaller red circles

            // Draw ROI ID and point number
            string point_text = "R" + to_string(roi_idx) + "." + to_string(i + 1);
            putText(frame, point_text, Point(pixel_points[i].x + 8, pixel_points[i].y - 8),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);
        }

        // Draw lines between adjacent points
        for (size_t i = 0; i < pixel_points.size() - 1; i++) {
            line(frame, pixel_points[i], pixel_points[i + 1], Scalar(0, 0, 255), 2);
        }

        // Draw closing line
        if (pixel_points.size() >= 3) {
            line(frame, pixel_points.back(), pixel_points.front(), Scalar(0, 0, 255), 2);
        }
    }

    // Draw current redlight ROI points being created (only in redlight mode)
    if (current_draw_mode == REDLIGHT_ROI_MODE && !redlight_clicked_points.empty()) {
        // Convert normalized coordinates to pixel coordinates
        vector<Point> pixel_points;
        for (const auto& pt : redlight_clicked_points) {
            int x = static_cast<int>(pt.x * frame_width);
            int y = static_cast<int>(pt.y * frame_height);
            pixel_points.push_back(Point(x, y));
        }

        // Draw points as small circles (bright red for current editing)
        for (size_t i = 0; i < pixel_points.size(); i++) {
            Scalar point_color = Scalar(0, 100, 255);  // Bright red
            circle(frame, pixel_points[i], 5, point_color, -1);

            // Draw point number
            string point_text = to_string(i + 1);
            putText(frame, point_text, Point(pixel_points[i].x + 10, pixel_points[i].y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1);
        }

        // Draw lines between adjacent points
        for (size_t i = 0; i < pixel_points.size() - 1; i++) {
            line(frame, pixel_points[i], pixel_points[i + 1], Scalar(0, 100, 255), 2);
        }

        // Draw closing line if polygon is closed
        if (redlight_close_polygon && pixel_points.size() >= 3) {
            line(frame, pixel_points.back(), pixel_points.front(), Scalar(0, 100, 255), 2);
        }
    }

    // Draw created crossing line (always show if exists)
    if (crossingline_created && created_crossing_line.size() == 4) {
        // Convert normalized coordinates to pixel coordinates
        vector<Point> pixel_points;
        for (const auto& pt : created_crossing_line) {
            int x = static_cast<int>(pt.x * frame_width);
            int y = static_cast<int>(pt.y * frame_height);
            pixel_points.push_back(Point(x, y));
        }

        // Draw first line segment (points 0-1) in cyan
        line(frame, pixel_points[0], pixel_points[1], Scalar(255, 255, 0), 3);
        // Draw second line segment (points 2-3) in yellow
        line(frame, pixel_points[2], pixel_points[3], Scalar(0, 255, 255), 3);

        // Draw points as small circles
        for (size_t i = 0; i < pixel_points.size(); i++) {
            Scalar point_color = (i < 2) ? Scalar(255, 255, 0) : Scalar(0, 255, 255);  // Cyan for first line, Yellow for second
            circle(frame, pixel_points[i], 4, point_color, -1);

            // Draw point number
            string point_text = "C" + to_string(i + 1);
            putText(frame, point_text, Point(pixel_points[i].x + 8, pixel_points[i].y - 8),
                    FONT_HERSHEY_SIMPLEX, 0.4, point_color, 1);
        }
    }

    // Draw current crossing line points being created (only in crossing line mode)
    if (current_draw_mode == CROSSINGLINE_ROI_MODE && !crossingline_clicked_points.empty()) {
        // Convert normalized coordinates to pixel coordinates
        vector<Point> pixel_points;
        for (const auto& pt : crossingline_clicked_points) {
            int x = static_cast<int>(pt.x * frame_width);
            int y = static_cast<int>(pt.y * frame_height);
            pixel_points.push_back(Point(x, y));
        }

        // Draw points as circles
        for (size_t i = 0; i < pixel_points.size(); i++) {
            Scalar point_color = (i < 2) ? Scalar(255, 255, 0) : Scalar(0, 255, 255);  // Cyan for first line, Yellow for second
            circle(frame, pixel_points[i], 5, point_color, -1);

            // Draw point number
            string point_text = to_string(i + 1);
            putText(frame, point_text, Point(pixel_points[i].x + 10, pixel_points[i].y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1);
        }

        // Draw line segments as they are being created
        if (pixel_points.size() >= 2) {
            // Draw first line segment (points 0-1)
            line(frame, pixel_points[0], pixel_points[1], Scalar(255, 255, 0), 3);
        }
        if (pixel_points.size() >= 4) {
            // Draw second line segment (points 2-3)
            line(frame, pixel_points[2], pixel_points[3], Scalar(0, 255, 255), 3);
        }
    }

    // Display instructions based on current mode
    if (current_draw_mode == NORMAL_ROI_MODE) {
        string instruction = "Normal ROI - Points: " + to_string(clicked_points.size()) +
                           " | L-Click: Add | R-Click: Create ROI | M-Click: Reset";
        putText(frame, instruction, Point(10, frame_height - 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    } else if (current_draw_mode == REDLIGHT_ROI_MODE) {
        string instruction = "Redlight ROI - Points: " + to_string(redlight_clicked_points.size()) +
                           " | Created: " + to_string(created_redlight_rois.size()) +
                           " | L-Click: Add | R-Click: Create MRT ROI | M-Click: Clear All";
        putText(frame, instruction, Point(10, frame_height - 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 100, 255), 1);
    } else {
        string instruction = "Crossing Line - Points: " + to_string(crossingline_clicked_points.size()) + "/4" +
                           " | Created: " + (crossingline_created ? "Yes" : "No") +
                           " | L-Click: Add (max 4) | R-Click: Create Line | M-Click: Clear";
        putText(frame, instruction, Point(10, frame_height - 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);
    }

    for (int i = 0; i < num_objects; i++) {
        const auto& obj = results[i];

        // 将归一化坐标 (0~1) 转换为像素坐标
        int x1 = static_cast<int>(obj.bbox_xmin * frame_width);
        int y1 = static_cast<int>(obj.bbox_ymin * frame_height);
        int x2 = static_cast<int>(obj.bbox_xmax * frame_width);
        int y2 = static_cast<int>(obj.bbox_ymax * frame_height);

        // 确保坐标在图像范围内
        x1 = max(0, min(x1, frame_width - 1));
        y1 = max(0, min(y1, frame_height - 1));
        x2 = max(0, min(x2, frame_width - 1));
        y2 = max(0, min(y2, frame_height - 1));

        // 根据不同功能设置不同颜色
        string label;
        Scalar color = Scalar(0, 255, 0); // 绿色
        Scalar text_color = Scalar(255, 255, 255); // 白色文字

        switch (function_type) {
            case functions::YOLO_COLOR:
                label = "object_" + to_string(obj.class_id) + "_in roi: " + to_string(obj.in_roi_id);
                if (obj.in_roi_id != -1) {
                    color = Scalar(0, 0, 225); // 红色
                }
                break;
            case functions::FALL:
                label = string(obj.pose) + "_in roi: " + to_string(obj.in_roi_id);
                if (string(obj.pose) == "falling" && obj.in_roi_id != -1){
                    color = Scalar(0, 225, 225); // 黄色
                }else if (string(obj.pose) == "fall" && obj.in_roi_id != -1) {
                    color = Scalar(0, 0, 225); // 红色
                }
                break;
            case functions::CLIMB:
                label = "object_" + to_string(obj.class_id);
                if (string(obj.climb) == "climbing" && obj.in_roi_id != -1){
                    color = Scalar(0, 225, 225); // 黄色
                    label = "climbing_" + to_string(obj.class_id);
                }else if (string(obj.climb) == "climb" && obj.in_roi_id != -1) {
                    color = Scalar(0, 0, 225); // 红色
                    label = "climb_" + to_string(obj.class_id);
                }
                break;
        }

        // 画边界框
        rectangle(frame, Point(x1, y1), Point(x2, y2), color, 2);

        // 添加追蹤ID到標籤
        if (obj.track_id != -1) {
            label = "ID:" + to_string(obj.track_id) + " " + label;
        }

        // 画标签背景
        int baseline = 0;
        Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        rectangle(frame, Point(x1, y1 - text_size.height - 5),
                 Point(x1 + text_size.width, y1), color, -1);

        // 画标签文字
        putText(frame, label, Point(x1, y1 - 5), FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);

        // 在目標中心畫一個追蹤點 - 使用上半部框的中心作為參考點
        if (obj.track_id != -1) {
            int center_x = (x1 + x2) / 2;
            // 將參考點改為上半部框的中心 (y座標向下移動1/8)
            int bbox_height = y2 - y1;
            int center_y = y1 + bbox_height / 8;
            circle(frame, Point(center_x, center_y), 3, Scalar(255, 255, 0), -1);
        }
    }
}int main(int argc, char* argv[]) {
    // Check minimum arguments
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    // Parse required arguments
    string function_str = argv[1];
    string input_source = argv[2];
    string log_file_path = argv[3];

    // Parse optional target FPS argument
    double target_fps = 0;  // Default: no frame skipping
    if (argc >= 5) {
        target_fps = stod(argv[4]);
    }

    // Parse optional rotation argument
    int user_rotation = 0;  // Default: no rotation
    if (argc >= 6) {
        user_rotation = stoi(argv[5]);
    }    // Parse function type
    functions selected_function = parseFunction(function_str);

    // Determine if input is RTSP stream or video file
    bool is_rtsp = (input_source.find("rtsp://") == 0);
    bool is_video_file = !is_rtsp;

    // Set default engine paths based on function
    string engine_path1, engine_path2;
    switch (selected_function) {
        case functions::YOLO_COLOR:
            engine_path1 = "wheelchair_m_1.3.0.engine";
            engine_path2 = "wheelchair_m_1.3.0.engine";
            break;
        case functions::FALL:
            engine_path1 = "wheelchair_m_1.3.0.engine";
            engine_path2 = "yolo-fall4-cls_1.3.engine";
            break;
        case functions::CLIMB:
            engine_path1 = "yolo11x-pose.engine";
            engine_path2 = "yolo11x-pose.engine";
            break;
    }

    // Validate input source
    if (is_rtsp && input_source.find("rtsp://") != 0) {
        cerr << "[ERROR] Invalid RTSP URL format. Must start with 'rtsp://', got: " << input_source << endl;
        return 1;
    }

    cout << "=== Detection System ===" << endl;
    cout << "Function: " << getFunctionName(selected_function) << endl;
    cout << "Engine 1: " << engine_path1 << endl;
    cout << "Engine 2: " << engine_path2 << endl;
    cout << "Input Source: " << input_source << endl;
    cout << "Source Type: " << (is_rtsp ? "RTSP Stream" : "Video File") << endl;
    cout << "Log File: " << log_file_path << endl;
    if (is_video_file && target_fps > 0) {
        cout << "Target FPS: " << target_fps << " (frame skipping enabled)" << endl;
    }
    if (is_video_file && user_rotation != 0) {
        cout << "Rotation: " << user_rotation << " degrees" << endl;
    }
    cout << "按ESC键退出..." << endl;

    // Open input source (RTSP stream or video file)
    VideoCapture cap(input_source);
    if (!cap.isOpened()) {
        if (is_rtsp) {
            cerr << "[ERROR] Failed to open RTSP stream: " << input_source << endl;
        } else {
            cerr << "[ERROR] Failed to open video file: " << input_source << endl;
        }
        return 1;
    }

    // Set capture properties for better performance (mainly for RTSP)
    if (is_rtsp) {
        cap.set(CAP_PROP_BUFFERSIZE, 1);  // Minimal buffer size for RTSP
    }

    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    if (is_video_file) {
        cout << "[INFO] Video FPS: " << fps << ", Total frames: " << total_frames << endl;
    }

    // Get first frame to determine dimensions
    Mat frame_bgr;
    if (!cap.read(frame_bgr) || frame_bgr.empty()) {
        if (is_rtsp) {
            cerr << "[ERROR] Failed to read first frame from RTSP stream" << endl;
        } else {
            cerr << "[ERROR] Failed to read first frame from video file" << endl;
        }
        return 1;
    }

    // Auto-correct frame orientation for video files
    if (is_video_file) {
        frame_bgr = correctFrameOrientation(frame_bgr, true, user_rotation);  // Check orientation on first frame
    }

    int width = frame_bgr.cols;
    int height = frame_bgr.rows;
    cout << "[INFO] Frame size: " << width << "x" << height << endl;

    // Initialize detection model
    const char* log_file = log_file_path.c_str();
    const int camera_id = 0;
    const int MAX_OBJECTS = 100;

    // Set global frame dimensions for mouse callback
    frame_width_global = width;
    frame_height_global = height;
    current_function = selected_function;
    current_camera_id = camera_id;

    cout << "[INFO] Initializing " << getFunctionName(selected_function) << " model..." << endl;
    svCreate_ObjectModules(selected_function, 1, engine_path1.c_str(), engine_path2.c_str(), 0.3f, log_file);

    cout << "[INFO] Starting detection and display..." << endl;

    // Create window
    string window_name = "Detection - " + string(getFunctionName(selected_function)) +
                        (is_rtsp ? " (RTSP)" : " (Video)");
    namedWindow(window_name, WINDOW_AUTOSIZE);

    // Set mouse callback
    setMouseCallback(window_name, onMouse, nullptr);

    Mat frame_yuv;
    svObjData_t results[MAX_OBJECTS];
    int frame_count = 0;
    int processed_frames = 0;

    // Frame skipping variables
    int frame_skip_interval = 0;
    if (is_video_file && target_fps > 0 && fps > 0) {
        frame_skip_interval = max(1, (int)(fps / target_fps));
        cout << "[INFO] Frame skipping enabled: processing every " << frame_skip_interval << " frames" << endl;
    }

    while (true) {
        // Read frame from input source
        if (!cap.read(frame_bgr) || frame_bgr.empty()) {
            if (is_rtsp) {
                cerr << "[WARNING] Failed to read frame, attempting to reconnect..." << endl;
                cap.release();
                cap.open(input_source);
                if (!cap.isOpened()) {
                    cerr << "[ERROR] Failed to reconnect to RTSP stream" << endl;
                    break;
                }
                cap.set(CAP_PROP_BUFFERSIZE, 1);
                continue;
            } else {
                // Video file finished
                cout << "[INFO] Video playback completed" << endl;
                break;
            }
        }

        frame_count++;

        // Apply frame skipping for video files if target FPS is set
        if (is_video_file && frame_skip_interval > 1) {
            if ((frame_count - 1) % frame_skip_interval != 0) {
                // Skip this frame, continue to next iteration
                continue;
            }
        }

        processed_frames++;

        // Auto-correct frame orientation for video files
        if (is_video_file) {
            frame_bgr = correctFrameOrientation(frame_bgr, false, user_rotation);  // Apply determined rotation
        }

        // Convert BGR to YUV420
        cvtColor(frame_bgr, frame_yuv, COLOR_BGR2YUV_I420);
        uint8_t* yuv_data = frame_yuv.data;

        auto start_time = chrono::high_resolution_clock::now();

        // Process YUV image
        int q_size = svObjectModules_inputImageYUV(selected_function, camera_id, yuv_data, width, height, 3, MAX_OBJECTS);
        if (q_size < 1) {
            cerr << "[WARNING] Failed to process image" << endl;
            continue;
        }

        // Get detection results
        int num_objects = svObjectModules_getResult(selected_function, camera_id, results, MAX_OBJECTS, true);
        if (num_objects == -1) {
            cerr << "[INFO] Detection stopped" << endl;
            break;
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto inference_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

        // Draw detection results on frame
        drawDetectionResults(frame_bgr, results, num_objects, selected_function);

        // Count red bounding boxes (objects in ROI)
        int red_box_count = 0;
        for (int i = 0; i < num_objects; i++) {
            const auto& obj = results[i];

            // Count objects that trigger red boxes (in ROI)
            switch (selected_function) {
                case functions::YOLO_COLOR:
                    if (obj.crossing_line_id != -1) {
                        red_box_count += obj.crossing_line_direction;
                    }
                    break;
                case functions::FALL:
                    if ((string(obj.pose) == "falling" || string(obj.pose) == "fall") && obj.in_roi_id != -1) {
                        red_box_count++;
                    }
                    break;
                case functions::CLIMB:
                    if ((string(obj.climb) == "climbing" || string(obj.climb) == "climb") && obj.in_roi_id != -1) {
                        red_box_count++;
                    }
                    break;
            }
        }

        // Add to cumulative count
        total_red_box_count += red_box_count;

        // Display cumulative red box count in red text
        if (total_red_box_count > 0) {
            string cumulative_text = "Total Alert Count: " + to_string(total_red_box_count);
            putText(frame_bgr, cumulative_text, Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }

        // Display current frame red box count if any
        if (red_box_count > 0) {
            string current_text = "Current: " + to_string(red_box_count);
            putText(frame_bgr, current_text, Point(10, 135), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        }

        // Add frame counter and progress for video files
        if (is_video_file && total_frames > 0) {
            string progress_text = "Frame: " + to_string(frame_count) + "/" + to_string(total_frames) +
                                 " (" + to_string(int((double)frame_count / total_frames * 100)) + "%)";
            putText(frame_bgr, progress_text, Point(10, 170), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        } else {
            putText(frame_bgr, "Frame: " + to_string(frame_count), Point(10, 170), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        }        // Display frame
        imshow(window_name, frame_bgr);

        // For video files, add delay to maintain target frame rate (subtract processing time)
        int wait_time = 1;
        if (is_video_file && target_fps > 0) {
            int target_frame_time = int(1000.0 / target_fps);  // Target time per frame in ms
            wait_time = max(1, target_frame_time - static_cast<int>(inference_time));
        } else if (is_video_file && fps > 0) {
            int original_frame_time = int(1000.0 / fps);  // Original time per frame in ms
            wait_time = max(1, original_frame_time - static_cast<int>(inference_time));
        }

        // Check for key presses
        char key = waitKey(wait_time) & 0xFF;
        if (key == 27) { // ESC key
            cout << "[INFO] ESC pressed, exiting..." << endl;
            break;
        } else if (key == 9) { // Tab key
            // Switch between ROI modes: Normal -> Redlight -> Crossing Line -> Normal
            if (current_draw_mode == NORMAL_ROI_MODE) {
                current_draw_mode = REDLIGHT_ROI_MODE;
                cout << "[MODE] Switched to Redlight ROI Mode" << endl;
            } else if (current_draw_mode == REDLIGHT_ROI_MODE) {
                current_draw_mode = CROSSINGLINE_ROI_MODE;
                cout << "[MODE] Switched to Crossing Line ROI Mode" << endl;
            } else {
                current_draw_mode = NORMAL_ROI_MODE;
                cout << "[MODE] Switched to Normal ROI Mode" << endl;
            }
        }

        // Print statistics every 100 processed frames (not total frames)
        if (processed_frames % 100 == 0) {
            if (is_video_file && total_frames > 0) {
                double progress_percent = (double)frame_count / total_frames * 100;
                cout << "[INFO] Read " << frame_count << "/" << total_frames << " frames ("
                     << int(progress_percent) << "%), Processed: " << processed_frames
                     << ", Objects: " << num_objects << ", Inference time: " << inference_time << "ms" << endl;
            } else {
                cout << "[INFO] Processed " << processed_frames << " frames, Objects: " << num_objects
                     << ", Inference time: " << inference_time << "ms" << endl;
            }
        }
    }

    // Cleanup
    cap.release();
    destroyAllWindows();

    // Remove ROI
    svRemove_ROIandWall(camera_id, selected_function, 0);

    cout << "[INFO] Detection finished. Total frames processed: " << processed_frames << endl;
    cout << "[INFO] Total alert objects detected: " << total_red_box_count << endl;

    if (is_video_file) {
        cout << "[INFO] Video processing completed successfully" << endl;
    }

    return 0;
}
