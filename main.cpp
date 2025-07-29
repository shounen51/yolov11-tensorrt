#include "yolov11_dll.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

using namespace std;
using namespace cv;

// Global control for stopping threads
atomic<bool> should_stop(false);

// Thread function for YOLO_COLOR detection
void yolo_thread(const char* engine_path1, const char* engine_path2, const char* video_path,
                 const char* log_file, int camera_id, int fps) {
    const int MAX_OBJECTS = 100;
    svObjData_t results[MAX_OBJECTS];

    // Calculate delay based on FPS
    int delay_ms = 1000 / fps;
    cout << "[YOLO] Thread started for camera " << camera_id << " at " << fps << " FPS" << endl;

    // Open video file
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "[YOLO] Failed to open video: " << video_path << endl;
        return;
    }

    // Get video dimensions
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    cout << "[YOLO] Video size: " << width << "x" << height << endl;

    // API 1: Initialize the model
    svCreate_ObjectModules(functions::YOLO_COLOR, 10, engine_path1, engine_path2, 0.3f, log_file);

    // API 2: create a ROI
    float points_x[] = {0.5f, 1.0f, 1.0f, 0.5f}; // Right half of screen
    float points_y[] = {0.0f, 0.0f, 1.0f, 1.0f}; // Right half of screen
    svCreate_ROI(camera_id, functions::YOLO_COLOR, 0, width, height, points_x, points_y, 4);

    Mat frame_bgr, frame_yuv;
    uint8_t* yuv_data = nullptr;

    // Processing loop
    while (!should_stop) {
        auto loop_start = chrono::high_resolution_clock::now();

        // Read frame from video
        if (!cap.read(frame_bgr)) {
            // End of video, restart from beginning
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // Convert BGR to YUV420
        cvtColor(frame_bgr, frame_yuv, COLOR_BGR2YUV_I420);
        yuv_data = frame_yuv.data;

        // API 3: Process yuv image
        int ok = svObjectModules_inputImageYUV(functions::YOLO_COLOR, camera_id, yuv_data, width, height, 3, MAX_OBJECTS);
        if (ok == 0) {
            cerr << "[YOLO] Failed to process image." << endl;
            break;
        }

        // API 4: Get results and discard them
        int num = svObjectModules_getResult(functions::YOLO_COLOR, camera_id, results, MAX_OBJECTS, true);
        if (num == -1) {
            cerr << "[YOLO] Thread have been stopped." << endl;
            break;
        }

        auto loop_end = chrono::high_resolution_clock::now();
        auto processing_time = chrono::duration_cast<chrono::milliseconds>(loop_end - loop_start).count();
        int actual_delay = max(0, delay_ms - static_cast<int>(processing_time));
        // if (actual_delay == 0) {
        //     float actual_fps = 1000.0f / processing_time;
        //     cerr << "[YOLO] Processing took too long, skipping delay. Actual FPS: " << actual_fps << endl;
        // }

        this_thread::sleep_for(chrono::milliseconds(actual_delay));
    }

    // Cleanup
    svRemove_ROIandWall(camera_id, functions::YOLO_COLOR, 0);
    cap.release();
    cout << "\n[YOLO] Thread finished for camera " << camera_id << endl;
}

// Thread function for FALL detection
void fall_thread(const char* engine_path1, const char* engine_path2, const char* video_path,
                 const char* log_file, int camera_id, int fps) {
    const int MAX_OBJECTS = 100;
    svObjData_t results[MAX_OBJECTS];

    // Calculate delay based on FPS
    int delay_ms = 1000 / fps;
    cout << "[FALL] Thread started for camera " << camera_id << " at " << fps << " FPS" << endl;

    // Open video file
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "[FALL] Failed to open video: " << video_path << endl;
        return;
    }

    // Get video dimensions
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    cout << "[FALL] Video size: " << width << "x" << height << endl;

    // API 1: Initialize the model
    svCreate_ObjectModules(functions::FALL, 10, engine_path1, engine_path2, 0.3f, log_file);

    // API 2: create a ROI
    float points_x[] = {0.3f, 1.0f, 1.0f, 0.3f}; // Right part of screen
    float points_y[] = {0.0f, 0.0f, 1.0f, 1.0f}; // Right part of screen
    svCreate_ROI(camera_id, functions::FALL, 0, width, height, points_x, points_y, 4);

    Mat frame_bgr, frame_yuv;
    uint8_t* yuv_data = nullptr;

    // Processing loop
    while (!should_stop) {
        auto loop_start = chrono::high_resolution_clock::now();

        // Read frame from video
        if (!cap.read(frame_bgr)) {
            // End of video, restart from beginning
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // Convert BGR to YUV420
        cvtColor(frame_bgr, frame_yuv, COLOR_BGR2YUV_I420);
        yuv_data = frame_yuv.data;

        // API 3: Process yuv image
        int ok = svObjectModules_inputImageYUV(functions::FALL, camera_id, yuv_data, width, height, 3, MAX_OBJECTS);
        if (ok == 0) {
            cerr << "[FALL] Failed to process image." << endl;
            break;
        }

        // API 4: Get results and discard them
        int num = svObjectModules_getResult(functions::FALL, camera_id, results, MAX_OBJECTS, true);
        if (num == -1) {
            cerr << "[FALL] Thread have been stopped." << endl;
            break;
        }

        auto loop_end = chrono::high_resolution_clock::now();
        auto processing_time = chrono::duration_cast<chrono::milliseconds>(loop_end - loop_start).count();
        int actual_delay = max(0, delay_ms - static_cast<int>(processing_time));
        // if (actual_delay == 0) {
        //     float actual_fps = 1000.0f / processing_time;
        //     cerr << "[FALL] Processing took too long, skipping delay. Actual FPS: " << actual_fps << endl;
        // }

        this_thread::sleep_for(chrono::milliseconds(actual_delay));
    }

    // Cleanup
    svRemove_ROIandWall(camera_id, functions::FALL, 0);
    cap.release();
    cout << "\n[FALL] Thread finished for camera " << camera_id << endl;
}

// Thread function for CLIMB detection
void climb_thread(const char* engine_path1, const char* engine_path2, const char* video_path,
                  const char* log_file, int camera_id, int fps) {
    const int MAX_OBJECTS = 100;
    svObjData_t results[MAX_OBJECTS];

    // Calculate delay based on FPS
    int delay_ms = 1000 / fps;
    cout << "[CLIMB] Thread started for camera " << camera_id << " at " << fps << " FPS" << endl;

    // Open video file
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "[CLIMB] Failed to open video: " << video_path << endl;
        return;
    }

    // Get video dimensions
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // API 1: Initialize the model
    svCreate_ObjectModules(functions::CLIMB, 10, engine_path1, engine_path2, 0.3f, log_file);

    // API 2: create a wall
    float wall_points_x[] = {0.4734375f, 0.225f}; // Wall point coordinates
    float wall_points_y[] = {0.391667f, 0.610185f}; // Wall point coordinates
    svCreate_wall(camera_id, functions::CLIMB, 0, width, height, wall_points_x, wall_points_y, 2);

    Mat frame_bgr, frame_yuv;
    uint8_t* yuv_data = nullptr;

    // Processing loop
    while (!should_stop) {
        auto loop_start = chrono::high_resolution_clock::now();

        // Read frame from video
        if (!cap.read(frame_bgr)) {
            // End of video, restart from beginning
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // Convert BGR to YUV420
        cvtColor(frame_bgr, frame_yuv, COLOR_BGR2YUV_I420);
        yuv_data = frame_yuv.data;

        // API 3: Process yuv image
        int ok = svObjectModules_inputImageYUV(functions::CLIMB, camera_id, yuv_data, width, height, 3, MAX_OBJECTS);
        if (ok == 0) {
            cerr << "[CLIMB] Failed to process image." << endl;
            break;
        }

        // API 4: Get results and discard them
        int num = svObjectModules_getResult(functions::CLIMB, camera_id, results, MAX_OBJECTS, true);
        if (num == -1) {
            cerr << "[CLIMB] Thread have been stopped." << endl;
            break;
        }

        auto loop_end = chrono::high_resolution_clock::now();
        auto processing_time = chrono::duration_cast<chrono::milliseconds>(loop_end - loop_start).count();
        int actual_delay = max(0, delay_ms - static_cast<int>(processing_time));
        // if (actual_delay == 0) {
        //     float actual_fps = 1000.0f / processing_time;
        //     cerr << "[CLIMB] Processing took too long, skipping delay. Actual FPS: " << actual_fps << endl;
        // }

        this_thread::sleep_for(chrono::milliseconds(actual_delay));
    }

    // Cleanup
    svRemove_ROIandWall(camera_id, functions::CLIMB, 0);
    cap.release();
    cout << "\n[CLIMB] Thread finished for camera " << camera_id << endl;
}

int main() {
    const char* yolo_engine_path1 = "wheelchair_m_1.3.0.engine";
    const char* yolo_engine_path2 = "wheelchair_m_1.3.0.engine";
    const char* fall_engine_path1 = "wheelchair_m_1.3.0.engine";
    const char* fall_engine_path2 = "yolo-fall4-cls.engine";
    const char* climb_engine_path1 = "yolo11x-pose.engine";
    const char* climb_engine_path2 = "yolo11x-pose.engine";
    const char* yolo_video_path = "yolo.mp4";
    const char* fall_video_path = "fall.mp4";
    const char* climb_video_path = "climb.mp4";
    const char* log_file = "log/log.log";

    // FPS settings for each thread
    int yolo_fps = 5;
    int fall_fps = 5;
    int climb_fps = 5;

    cout << "YOLO engines: " << yolo_engine_path1 << ", " << yolo_engine_path2 << endl;
    cout << "FALL engines: " << fall_engine_path1 << ", " << fall_engine_path2 << endl;
    cout << "CLIMB engines: " << climb_engine_path1 << ", " << climb_engine_path2 << endl;
    cout << "YOLO video: " << yolo_video_path << " at " << yolo_fps << " FPS" << endl;
    cout << "FALL video: " << fall_video_path << " at " << fall_fps << " FPS" << endl;
    cout << "CLIMB video: " << climb_video_path << " at " << climb_fps << " FPS" << endl;
    cout << "\nStarting three detection threads..." << endl;

    // Create three threads with different camera IDs, video paths, and FPS
    thread t_yolo(yolo_thread, yolo_engine_path1, yolo_engine_path2, yolo_video_path, log_file, 1, yolo_fps);
    thread t_fall(fall_thread, fall_engine_path1, fall_engine_path2, fall_video_path, log_file, 2, fall_fps);
    thread t_climb(climb_thread, climb_engine_path1, climb_engine_path2, climb_video_path, log_file, 3, climb_fps);

    // Let threads run for some time
    cout << "\nThreads running... Press Enter to stop." << endl;
    cin.get();

    // Signal threads to stop
    should_stop = true;
    cout << "\nStopping threads..." << endl;

    // Wait for all threads to finish
    t_yolo.join();
    t_fall.join();
    t_climb.join();

    cout << "All threads finished." << endl;
    return 0;
}
