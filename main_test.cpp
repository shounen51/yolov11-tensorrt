#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "ColorClassifier.h"

using namespace std;
using namespace cv;

// Global variables for mouse callback
vector<Point> clicked_points;  // Store clicked points (in pixel coordinates)
bool drawing = false;
bool rectangle_complete = false;
int frame_width_global = 0;
int frame_height_global = 0;
HSVColorClassifier colorClassifier;

// Mouse callback function
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
        rectangle_complete = false;
        clicked_points.clear();
        clicked_points.push_back(Point(x, y));

        cout << "[MOUSE] Left click - Start drawing rectangle at (" << x << ", " << y << ")" << endl;
    }
    else if (event == EVENT_MOUSEMOVE && drawing) {
        // Update the second point for live rectangle drawing
        if (clicked_points.size() == 1) {
            clicked_points.push_back(Point(x, y));
        } else if (clicked_points.size() == 2) {
            clicked_points[1] = Point(x, y);
        }
    }
    else if (event == EVENT_LBUTTONUP && drawing) {
        drawing = false;
        rectangle_complete = true;

        if (clicked_points.size() >= 2) {
            cout << "[MOUSE] Rectangle completed: (" << clicked_points[0].x << ", " << clicked_points[0].y
                 << ") to (" << clicked_points[1].x << ", " << clicked_points[1].y << ")" << endl;
        }
    }
    else if (event == EVENT_RBUTTONDOWN) {
        // Right click to clear rectangle
        clicked_points.clear();
        drawing = false;
        rectangle_complete = false;
        cout << "[MOUSE] Right click - Cleared rectangle" << endl;
    }
}

// Function to analyze color in the selected rectangle
vector<string> analyzeRectangleColor(const Mat& frame, const vector<Point>& points) {
    vector<string> color_results;

    if (points.size() < 2) {
        return color_results;
    }

    // Get rectangle bounds
    int x1 = min(points[0].x, points[1].x);
    int y1 = min(points[0].y, points[1].y);
    int x2 = max(points[0].x, points[1].x);
    int y2 = max(points[0].y, points[1].y);

    // Ensure bounds are within image
    x1 = max(0, x1);
    y1 = max(0, y1);
    x2 = min(frame.cols - 1, x2);
    y2 = min(frame.rows - 1, y2);

    if (x2 <= x1 || y2 <= y1) {
        return color_results;
    }

    // Extract ROI
    Rect roi(x1, y1, x2 - x1, y2 - y1);
    Mat roi_image = frame(roi);

    // Use ColorClassifier to analyze colors (原始 BGR 圖像)
    vector<unsigned char> color_stats = colorClassifier.classifyStatistics(
        roi_image,
        1000,  // sample number
        COLOR_BGR2HSV,  // conversion code - 讓 classifyStatistics 自己轉換
        Mat(),  // no mask
        true    // use percent
    );

    // 只顯示 red, white, black 三種顏色的比例
    // 根據 ColorLabels enum: color_black=1, color_white=3, color_red=4
    vector<pair<string, int>> target_colors = {
        {"red", 4},      // color_red
        {"white", 3},    // color_white
        {"black", 1}     // color_black
    };

    for (const auto& color_pair : target_colors) {
        int color_index = color_pair.second;
        string color_name = color_pair.first;

        int percentage = 0;
        if (color_index < color_stats.size()) {
            percentage = color_stats[color_index];
        }

        string color_info = color_name + ": " + to_string(percentage) + "%";
        color_results.push_back(color_info);
    }

    return color_results;
}

void drawRectangleAndColorInfo(Mat& frame, const vector<Point>& points, bool complete) {
    if (points.size() < 2) {
        return;
    }

    // Draw rectangle
    Scalar rect_color = complete ? Scalar(0, 255, 0) : Scalar(0, 255, 255);  // Green if complete, yellow if drawing
    int thickness = complete ? 2 : 1;
    rectangle(frame, points[0], points[1], rect_color, thickness);

    // If rectangle is complete, show color analysis
    if (complete) {
        vector<string> colors = analyzeRectangleColor(frame, points);

        // Draw color information - 始終顯示三種顏色
        int text_y = points[0].y - 10;
        if (text_y < 80) text_y = points[1].y + 20;  // 調整位置以容納三行文字

        for (int i = 0; i < colors.size(); i++) {
            putText(frame, colors[i], Point(points[0].x, text_y + i * 20),
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        // Show rectangle dimensions
        int width = abs(points[1].x - points[0].x);
        int height = abs(points[1].y - points[0].y);
        string size_info = "Size: " + to_string(width) + "x" + to_string(height);
        putText(frame, size_info, Point(points[0].x, text_y - 25),
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    // Draw instructions
    string instruction = "L-Click+Drag: Draw Rectangle | R-Click: Clear | ESC: Exit";
    putText(frame, instruction, Point(10, frame.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
}

void printUsage(const char* program_name) {
    cout << "\n=== Color Analysis Test Tool ===" << endl;
    cout << "用法: " << program_name << " <video_file>" << endl;
    cout << "\n参数:" << endl;
    cout << "  video_file    视频文件路径" << endl;
    cout << "\n示例:" << endl;
    cout << "  " << program_name << " test.mp4" << endl;
    cout << "  " << program_name << " video.avi" << endl;
    cout << "\n交互控制:" << endl;
    cout << "  鼠标左键拖拽  绘制矩形区域" << endl;
    cout << "  鼠标右键      清除矩形" << endl;
    cout << "  ESC键         退出程序" << endl;
}

int main(int argc, char* argv[]) {
    // Check arguments
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    string video_path = argv[1];

    cout << "=== Color Analysis Test Tool ===" << endl;
    cout << "Video File: " << video_path << endl;
    cout << "按ESC键退出..." << endl;

    // Open video file
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Failed to open video file: " << video_path << endl;
        return 1;
    }

    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    cout << "[INFO] Video FPS: " << fps << ", Total frames: " << total_frames << endl;

    // Get first frame to determine dimensions
    Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        cerr << "[ERROR] Failed to read first frame from video file" << endl;
        return 1;
    }

    frame_width_global = frame.cols;
    frame_height_global = frame.rows;
    cout << "[INFO] Frame size: " << frame_width_global << "x" << frame_height_global << endl;

    // Initialize color classifier
    colorClassifier.setDefaultColorRange();
    cout << "[INFO] Color classifier initialized" << endl;

    // Create window
    string window_name = "Color Analysis Test Tool";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    // Set mouse callback
    setMouseCallback(window_name, onMouse, nullptr);

    int frame_count = 0;

    // Reset video to beginning
    cap.set(CAP_PROP_POS_FRAMES, 0);

    while (true) {
        // Read frame from video
        if (!cap.read(frame) || frame.empty()) {
            // Video finished, restart from beginning
            cap.set(CAP_PROP_POS_FRAMES, 0);
            if (!cap.read(frame) || frame.empty()) {
                cerr << "[ERROR] Failed to restart video" << endl;
                break;
            }
            frame_count = 0;
            cout << "[INFO] Video restarted" << endl;
        }

        frame_count++;

        // Draw rectangle and color information
        drawRectangleAndColorInfo(frame, clicked_points, rectangle_complete);

        // Add frame counter
        string frame_info = "Frame: " + to_string(frame_count) + "/" + to_string(total_frames);
        putText(frame, frame_info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // Display frame
        imshow(window_name, frame);

        // Control playback speed
        int wait_time = max(1, int(1000.0 / fps));
        char key = waitKey(wait_time) & 0xFF;

        if (key == 27) { // ESC key
            cout << "[INFO] ESC pressed, exiting..." << endl;
            break;
        }
        else if (key == 32) { // Space key to pause/resume
            cout << "[INFO] Paused. Press any key to continue..." << endl;
            waitKey(0);
        }
        else if (key == 'r' || key == 'R') { // R key to restart video
            cap.set(CAP_PROP_POS_FRAMES, 0);
            frame_count = 0;
            cout << "[INFO] Video restarted" << endl;
        }
    }

    // Cleanup
    cap.release();
    destroyAllWindows();

    cout << "[INFO] Program finished. Total frames played: " << frame_count << endl;
    return 0;
}
