#include "yolov11_dll.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

// Global variables for mouse callback
vector<Point2f> clicked_points;  // Store all clicked points
bool point_clicked = false;
bool close_polygon = false;     // Flag to close the polygon
int frame_width_global = 0;
int frame_height_global = 0;
functions current_function = functions::YOLO_COLOR;  // Store current function
int current_camera_id = 0;  // Store current camera ID

// Mouse callback function
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point2f new_point;
        new_point.x = (float)x / frame_width_global;  // Normalize to 0-1
        new_point.y = (float)y / frame_height_global; // Normalize to 0-1

        clicked_points.push_back(new_point);
        point_clicked = true;
        close_polygon = false;  // Reset close flag when adding new point

        cout << "[MOUSE] Left click - Added point " << clicked_points.size()
             << " at pixel(" << x << ", " << y << ") = normalized("
             << new_point.x << ", " << new_point.y << ")" << endl;
    }
    else if (event == EVENT_RBUTTONDOWN) {
        if (clicked_points.size() >= 3) {  // Need at least 3 points to create ROI
            close_polygon = true;

            // Remove existing ROI
            svRemove_ROIandWall(current_camera_id, current_function, 0);
            cout << "[MOUSE] Right click - Removed existing ROI" << endl;

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
                cout << "[MOUSE] Right click - Created new WALL (non-closed) for CLIMB with " << clicked_points.size() << " points:" << endl;
            } else {
                cout << "[MOUSE] Right click - Created new ROI with " << clicked_points.size() << " points:" << endl;
            }

            for (size_t i = 0; i < clicked_points.size(); i++) {
                cout << "  Point " << (i+1) << ": (" << points_x[i] << ", " << points_y[i] << ")" << endl;
            }
        } else {
            cout << "[MOUSE] Right click - Need at least 3 points to create ROI (current: "
                 << clicked_points.size() << ")" << endl;
        }
    }
    else if (event == EVENT_MBUTTONDOWN) {
        // Middle click to clear all points and remove ROI/Wall
        clicked_points.clear();
        close_polygon = false;
        point_clicked = false;

        // Remove existing ROI/Wall without creating new one
        svRemove_ROIandWall(current_camera_id, current_function, 0);
        cout << "[MOUSE] Middle click - Cleared all points and removed ROI/Wall" << endl;
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
    cout << "用法: " << program_name << " <function> <rtsp_url> <log_file>" << endl;
    cout << "\n参数:" << endl;
    cout << "  function    检测功能 (yolo|fall|climb)" << endl;
    cout << "  rtsp_url    RTSP流地址" << endl;
    cout << "  log_file    日志文件路径" << endl;
    cout << "\n示例:" << endl;
    cout << "  " << program_name << " yolo rtsp://admin:admin123@192.168.1.100:554/stream1 log/log.log" << endl;
    cout << "  " << program_name << " fall rtsp://192.168.1.100:8554/stream log/fall.log" << endl;
    cout << "  " << program_name << " climb rtsp://192.168.1.64:554/cam/realmonitor?channel=1&subtype=0 log/climb.log" << endl;
    cout << "\n交互控制:" << endl;
    cout << "  鼠标左键    点击添加点到多边形" << endl;
    cout << "  鼠标右键    创建ROI区域(需至少3个点)" << endl;
    cout << "  鼠标中键    清除所有点并重置为全屏ROI" << endl;
    cout << "  ESC键       退出程序" << endl;
}

void drawDetectionResults(Mat& frame, svObjData_t* results, int num_objects, functions function_type) {
    int frame_width = frame.cols;
    int frame_height = frame.rows;

    // Draw clicked points and lines
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

        // Display instructions
        string instruction = "Points: " + to_string(clicked_points.size()) +
                           " | L-Click: Add | R-Click: Create ROI | M-Click: Reset";
        putText(frame, instruction, Point(10, frame_height - 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
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

        // 画标签背景
        int baseline = 0;
        Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        rectangle(frame, Point(x1, y1 - text_size.height - 5),
                 Point(x1 + text_size.width, y1), color, -1);

        // 画标签文字
        putText(frame, label, Point(x1, y1 - 5), FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }
}int main(int argc, char* argv[]) {
    // Check minimum arguments
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    // Parse required arguments
    string function_str = argv[1];
    string rtsp_url = argv[2];
    string log_file_path = argv[3];

    // Parse function type
    functions selected_function = parseFunction(function_str);

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

    // Validate RTSP URL
    if (rtsp_url.find("rtsp://") != 0) {
        cerr << "[ERROR] Invalid RTSP URL format. Must start with 'rtsp://', got: " << rtsp_url << endl;
        return 1;
    }

    cout << "=== RTSP Detection System ===" << endl;
    cout << "Function: " << getFunctionName(selected_function) << endl;
    cout << "Engine 1: " << engine_path1 << endl;
    cout << "Engine 2: " << engine_path2 << endl;
    cout << "RTSP URL: " << rtsp_url << endl;
    cout << "Log File: " << log_file_path << endl;
    cout << "按ESC键退出..." << endl;

    // Open RTSP stream
    VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Failed to open RTSP stream: " << rtsp_url << endl;
        return 1;
    }

    // Set capture properties for better performance
    cap.set(CAP_PROP_BUFFERSIZE, 1);  // Minimal buffer size

    // Get first frame to determine dimensions
    Mat frame_bgr;
    if (!cap.read(frame_bgr) || frame_bgr.empty()) {
        cerr << "[ERROR] Failed to read first frame from RTSP stream" << endl;
        return 1;
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
    string window_name = "RTSP Detection - " + string(getFunctionName(selected_function));
    namedWindow(window_name, WINDOW_AUTOSIZE);

    // Set mouse callback
    setMouseCallback(window_name, onMouse, nullptr);

    Mat frame_yuv;
    svObjData_t results[MAX_OBJECTS];
    int frame_count = 0;

    while (true) {
        // Read frame from RTSP
        if (!cap.read(frame_bgr) || frame_bgr.empty()) {
            cerr << "[WARNING] Failed to read frame, attempting to reconnect..." << endl;
            cap.release();
            cap.open(rtsp_url);
            if (!cap.isOpened()) {
                cerr << "[ERROR] Failed to reconnect to RTSP stream" << endl;
                break;
            }
            cap.set(CAP_PROP_BUFFERSIZE, 1);
            continue;
        }

        frame_count++;

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

        // Add information text
        string info_text = string(getFunctionName(selected_function)) + " | Objects: " + to_string(num_objects) +
                          " | Inference: " + to_string(inference_time) + "ms";
        putText(frame_bgr, info_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(frame_bgr, "Frame: " + to_string(frame_count), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);        // Display frame
        imshow(window_name, frame_bgr);

        // Check for ESC key
        char key = waitKey(1) & 0xFF;
        if (key == 27) { // ESC key
            cout << "[INFO] ESC pressed, exiting..." << endl;
            break;
        }

        // Print statistics every 100 frames
        if (frame_count % 100 == 0) {
            cout << "[INFO] Processed " << frame_count << " frames, Objects: " << num_objects
                 << ", Inference time: " << inference_time << "ms" << endl;
        }
    }

    // Cleanup
    cap.release();
    destroyAllWindows();

    // Remove ROI
    svRemove_ROIandWall(camera_id, selected_function, 0);

    cout << "[INFO] Detection finished. Total frames processed: " << frame_count << endl;
    return 0;
}
