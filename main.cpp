#include "yolov11_dll.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

// usage: ./yolov11_test.exe <engine_path> <yuv_path> <width> <height> <log_file>
int main(int argc, char** argv) {
    const int MAX_OBJECTS = 100; // Maximum number of objects to detect
    svObjData_t results[MAX_OBJECTS];
    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <engine_path> <yuv_path> <width> <height> <log_file>" << endl;
        return -1;
    }
    // Parse command line arguments
    const char* engine_path = argv[1];
    const char* yuv_path = argv[2];
    int width = std::stoi(argv[3]);
    int height = std::stoi(argv[4]);
    const char* log_file = "";
    if (argc >= 6) log_file = argv[5];
    cout << "Engine path: " << engine_path << endl;
    cout << "Yuv path: " << yuv_path << endl;

    // try to find same name .jpg image
    std::string jpg_path = yuv_path;
    size_t dot_pos = jpg_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        jpg_path.replace(dot_pos, jpg_path.length() - dot_pos, ".jpg");
    } else {
        jpg_path += ".jpg";
    }
    std::ifstream jpg_file(jpg_path, std::ios::binary);
    if (!jpg_file.good()) {
        jpg_path.clear();
    }else{
        cout << "Found jpg image: " << jpg_path << endl;
    }
    jpg_file.close();
    Mat img;
    if (!jpg_path.empty()) {
        // Load jpg image
        img = imread(jpg_path);
        if (img.empty()) {
            cerr << "Failed to load jpg image: " << jpg_path << endl;
            return -1;
        }
    }

    // API 1: Initialize the model
    svCreate_ObjectModules("yolo_color", 10, engine_path, 0.3f, log_file); // 初始化功能名稱, 攝影機數量, 權重, 閾值, log
    // API 2: create a ROI
    float points_x[] = {0.5f, 1.0f, 1.0f, 0.5f}; //右半邊畫面
    float points_y[] = {0.0f, 0.0f, 1.0f, 1.0f}; //右半邊畫面
    svCreate_ROI(0, width, height, points_x, points_y, 4); // create a ROI with id 0, width, height, points_x, points_y, point_count
    // load yuv image as a uint8_t array
    std::ifstream file(yuv_path, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* frame = new uint8_t[fileSize];
    file.read(reinterpret_cast<char*>(frame), fileSize);
    file.close();

    // test mutiple times
    auto t1 = chrono::high_resolution_clock::now();
    int num = 0;
    int num_tests = 10;
    for (int i = 0; i < num_tests; ++i) {
        // API 3: Process yuv image
        int ok = svObjectModules_inputImageYUV("yolo_color", 9, 0, frame, width, height, 3, MAX_OBJECTS);
        if (ok == 0) {
            cerr << "Failed to process image." << endl;
            return -1;
        }
        // API 4: Get results
        num = svObjectModules_getResult("yolo_color", 9, results, MAX_OBJECTS, true);
        if (num == -1) {
            cerr << "Thread have been stoped." << endl;
            break;
        }
        cout << " test times left: " << num_tests - i << "   \r";
    }
    auto t2 = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "Inference time: " << duration/num_tests << " ms" << endl;

    // API 5: Remove ROI
    svRemove_ROI(0);

    cout << "Detected " << num << " objects.\n";
    for (int i = 0; i < num; ++i) {
        auto& r = results[i];
        if (r.class_id != 0) continue; // only show class "person"
        cout << "Class: " << r.class_id
                << ", Conf: " << r.confidence
                << ", BBox: [" << r.bbox_xmin << "," << r.bbox_ymin
                << "," << r.bbox_xmax << "," << r.bbox_ymax << "]"
                << ", Color_first: " << std::string(r.color_label_first)
                << ", Color_second: " << std::string(r.color_label_second)
                << ", In_ROI_ID: " << std::to_string(r.in_roi_id) << "\n";
        if (img.empty()) continue; // skip drawing if no jpg image is found
        // Draw bounding box and label on the image
        rectangle(img, Rect(Point(r.bbox_xmin*img.cols, r.bbox_ymin*img.rows),
                                    Point(r.bbox_xmax*img.cols, r.bbox_ymax*img.rows)),
                    Scalar(0, 255, 0), 2);
        putText(img, std::to_string(r.in_roi_id), Point(r.bbox_xmin*img.cols, r.bbox_ymin*img.rows - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
    if (!img.empty()) {
        imshow("Result", img);
        waitKey(0);
    }
    delete[] frame;

    return 0;
}
