#include "yolov11_dll.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

// usage: ./yolov11_test.exe <engine_path> <image_path> <width> <height> <log_file>
int main(int argc, char** argv) {
    const int MAX_OBJECTS = 100; // Maximum number of objects to detect
    svObjData_t results[MAX_OBJECTS];
    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <engine_path> <image_path> <width> <height> <log_file>" << endl;
        return -1;
    }
    // Parse command line arguments
    const char* engine_path = argv[1];
    const char* image_path = argv[2];
    int width = std::stoi(argv[3]);
    int height = std::stoi(argv[4]);
    const char* log_file = "";
    if (argc >= 6) log_file = argv[5];
    cout << "Engine path: " << engine_path << endl;
    cout << "Image path: " << image_path << endl;
    // API 1: Initialize the model
    svCreate_ObjectModules(engine_path, 0.3f, log_file); // log_file can be empty
    // load yuv image as a uint8_t array
    std::ifstream file(image_path, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* frame = new uint8_t[fileSize];
    file.read(reinterpret_cast<char*>(frame), fileSize);
    file.close();

    // test mutiple times
    auto t1 = chrono::high_resolution_clock::now();
    int num = 0;
    int num_tests = 1000;
    for (int i = 0; i < num_tests; ++i) {
        // API 2: Process yuv image
        int ok = svObjectModules_inputImageYUV(frame, width, height, 3, MAX_OBJECTS);
        if (ok == 0) {
            cerr << "Failed to process image." << endl;
            return -1;
        }
        // API 3: Get results
        num = svObjectModules_getResult(results, MAX_OBJECTS, true);
        if (num == -1) {
            cerr << "Thread have been stoped." << endl;
            break;
        }
        cout << " test times left: " << num_tests - i << "   \r";
    }
    auto t2 = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "Inference time: " << duration/num_tests << " ms" << endl;

    cout << "Detected " << num << " objects.\n";
    for (int i = 0; i < num; ++i) {
        auto& r = results[i];
        // if (r.class_id != 0) continue; // only show class "person on wheelchair"
        cout << "Class: " << r.class_id
                << ", Conf: " << r.confidence
                << ", BBox: [" << r.bbox_xmin << "," << r.bbox_ymin
                << "," << r.bbox_xmax << "," << r.bbox_ymax << "]"
                << ", Color: " << std::string(r.color_label) << "\n";
        // rectangle(frame, Rect(Point(r.bbox_xmin*frame.cols, r.bbox_ymin*frame.rows),
        //                             Point(r.bbox_xmax*frame.cols, r.bbox_ymax*frame.rows)),
        //             Scalar(0, 255, 0), 2);
        // putText(frame, std::string(r.color_label), Point(r.bbox_xmin*frame.cols, r.bbox_ymin*frame.rows - 10),
        //         FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
    // imshow("Result", frame);
    delete[] frame;

    return 0;
}
