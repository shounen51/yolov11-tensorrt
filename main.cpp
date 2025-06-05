#include "yolov11_dll.h"
#include <opencv2/opencv.hpp>
#include "ColorClassifier.h"
#include <iostream>

using namespace std;
using namespace cv;
// usage: ./yolov11_test.exe <engine_path> <video_path>
int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <engine_path> <video_path>" << endl;
        return -1;
    }
    const char* engine_path = argv[1];
    const char* video_path = argv[2];
    // API 1: Initialize the model
    svCreate_ObjectModules(engine_path, 0.3f);

    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Failed to open video: " << video_path << endl;
        return -1;
    }

    int frame_count = 0;
    double total_time_ms = 0.0;
    const int MAX_OBJECTS = 100;
    // API 2: Allocate memory for results
    svObjData_t results[MAX_OBJECTS];

    HLSColorClassifier colorClassfer;
    colorClassfer.setDefaultColorRange();
    vector<vector<unsigned char>> peopleColors;
    Mat frame;
    while (cap.read(frame)) {
        auto t1 = chrono::high_resolution_clock::now();
        int width = frame.cols;
        int height = frame.rows;

        // API 2: Process the image and get results
        int ok = svObjectModules_inputImageBGR(frame.ptr<unsigned char>(0), frame.cols, frame.rows, frame.channels(), MAX_OBJECTS);
        if (ok == 0) {
            cerr << "Failed to process image." << endl;
            break;
        }
        // API 3: Get results
        int num = svObjectModules_getResult(results, MAX_OBJECTS, true);
        if (num == -1) {
            cerr << "Thread have been stoped." << endl;
            break;
        }
        auto t2 = chrono::high_resolution_clock::now();
        double duration_ms = chrono::duration<double, milli>(t2 - t1).count();
        total_time_ms += duration_ms;
        frame_count++;

        // cout << "Detected " << num << " objects.\n";
        for (int i = 0; i < num; ++i) {
            auto& r = results[i];
            if (r.class_id != 0) continue; // only show class "person on wheelchair"
            cout << "Class: " << r.class_id
                    << ", Conf: " << r.confidence
                    << ", BBox: [" << r.bbox_xmin << "," << r.bbox_ymin
                    << "," << r.bbox_xmax << "," << r.bbox_ymax << "]\n";
            rectangle(frame, Rect(Point(r.bbox_xmin*frame.cols, r.bbox_ymin*frame.rows),
                                        Point(r.bbox_xmax*frame.cols, r.bbox_ymax*frame.rows)),
                        Scalar(0, 255, 0), 2);
            putText(frame, std::string(r.color_label), Point(r.bbox_xmin*frame.cols, r.bbox_ymin*frame.rows - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        imshow("Result", frame);
        if (waitKey(1) == 27) break;
    }
    release();

    if (frame_count > 0) {
        double avg_time = total_time_ms / frame_count;
        double avg_fps = 1000.0 / avg_time;
        cout << "Processed " << frame_count << " frames" << endl;
        cout << "Average time per frame: " << avg_time << " ms" << endl;
        cout << "Average FPS: " << avg_fps << endl;
    }
    return 0;

}
