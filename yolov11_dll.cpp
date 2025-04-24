#include "yolov11_dll.h"
#include "yolov11.h"
#include "logging.h"
#include <memory>
#include <opencv2/opencv.hpp>

static std::unique_ptr<YOLOv11> model;
static Logger logger;
static std::thread inferenceThread; // 用於執行 inference_thread 的執行緒
static bool stopThread = false;     // 用於控制執行緒的停止

struct InputData {
    unsigned char* image_data;
    int width;
    int height;
    int channels;
    int max_output;
};

static std::queue<InputData> inputQueue;
static std::queue<OutputData> outputQueue;
static std::mutex queueMutex;
static std::condition_variable queueCondition;

extern "C" {
YOLOV11_API void svCreate_ObjectModules(const char* engine_path, float conf_threshold) {
    model = std::make_unique<YOLOv11>(engine_path, conf_threshold, logger);

    // 啟動執行緒執行 infernce_thread
    stopThread = false;
    inferenceThread = std::thread([]() {
            infernce_thread();
    });
}

YOLOV11_API int svObjectModules_inputImageBGR(unsigned char* image_data, int width, int height, int channels, svObjData_t* output, int max_output) {
    if (!model || !image_data || !output) return 0;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        inputQueue.push({image_data, width, height, channels, max_output});
    }
    queueCondition.notify_one();
    return 1;
}

YOLOV11_API void infernce_thread() {
    while (!stopThread) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCondition.wait(lock, [] { return !inputQueue.empty() || stopThread; });

        if (stopThread && inputQueue.empty()) break;

        // 從 inputQueue 中取出資料
        InputData input = inputQueue.front();
        inputQueue.pop();
        lock.unlock();

        // 處理輸入資料
        cv::Mat image(input.height, input.width, CV_8UC3);
        memcpy(image.data, input.image_data, input.width * input.height * input.channels);

        std::vector<Detection> detections;
        model->preprocess(image);
        model->infer();
        model->postprocess(detections);

        int count = std::min(static_cast<int>(detections.size()), input.max_output);

        // YOLO input size（根據模型固定）
        const int INPUT_W = 640;
        const int INPUT_H = 640;

        // 計算 resize 比例與 padding
        float r = std::min(1.0f * INPUT_W / input.width, 1.0f * INPUT_H / input.height);
        int unpad_w = static_cast<int>(r * input.width);
        int unpad_h = static_cast<int>(r * input.height);
        int pad_x = (INPUT_W - unpad_w) / 2;
        int pad_y = (INPUT_H - unpad_h) / 2;

        for (int i = 0; i < count; ++i) {
            const auto& det = detections[i];

            // 模型輸出的 bbox 相對於 640x640，要先扣 padding 再除以 r
            float x1 = (det.bbox.x - pad_x) / r;
            float y1 = (det.bbox.y - pad_y) / r;
            float x2 = (det.bbox.x + det.bbox.width - pad_x) / r;
            float y2 = (det.bbox.y + det.bbox.height - pad_y) / r;

            // 正規化成 [0~1]
            x1 = std::clamp(x1 / input.width, 0.0f, 1.0f);
            y1 = std::clamp(y1 / input.height, 0.0f, 1.0f);
            x2 = std::clamp(x2 / input.width, 0.0f, 1.0f);
            y2 = std::clamp(y2 / input.height, 0.0f, 1.0f);

            input.output[i].class_id = det.class_id;
            input.output[i].confidence = det.conf;
            input.output[i].bbox_xmin = x1;
            input.output[i].bbox_ymin = y1;
            input.output[i].bbox_xmax = x2;
            input.output[i].bbox_ymax = y2;
        }

        // 將結果放入 outputQueue
        lock.lock();
        outputQueue.push({input.output, count});
        lock.unlock();
    }
}

YOLOV11_API int svObjectModules_getResult(svObjData_t* output, int max_output) {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (outputQueue.empty()) return 0;

    // 從 outputQueue 中取出結果
    OutputData result = outputQueue.front();
    outputQueue.pop();

    // 複製結果到輸出
    int count = std::min(result.count, max_output);
    memcpy(output, result.output, count * sizeof(svObjData_t));

    return count; // 返回檢測到的物件數量
}

YOLOV11_API void release() {
    stopThread = true;
    if (inferenceThread.joinable()) {
        inferenceThread.join();
    }

    model.reset();
}
}
