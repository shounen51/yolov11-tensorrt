#include "yolov11_dll.h"
#include "yolov11.h"
#include "logging.h"
#include <memory>
#include <opencv2/opencv.hpp>

static std::unique_ptr<YOLOv11> model;
static Logger logger;
static std::thread inferenceThread; // 用於執行 inference_thread 的執行緒
static bool stopThread = false;     // 用於控制執行緒的停止



static std::queue<InputData> inputQueue;
static std::queue<OutputData> outputQueue;
static std::mutex queueMutex;
static std::condition_variable inputQueueCondition;
static std::condition_variable outputQueueCondition;

YOLOV11_API void infernce_thread() {
    cout << "Inference thread started." << std::endl;
    while (!stopThread) {
        std::unique_lock<std::mutex> lock(queueMutex);

        // 如果 inputQueue 為空，則等待通知
        inputQueueCondition.wait(lock, [] { return !inputQueue.empty() || stopThread; });

        // 再次檢查條件，避免虛假喚醒
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
        svObjData_t* output = new svObjData_t[count];
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

            // 將結果放入 output
            output[i].bbox_xmin = x1;
            output[i].bbox_ymin = y1;
            output[i].bbox_xmax = x2;
            output[i].bbox_ymax = y2;
            output[i].class_id = det.class_id;
            output[i].confidence = det.conf;
        }

        // 將結果放入 outputQueue
        lock.lock();
        outputQueue.push({output, count});
        lock.unlock();
        outputQueueCondition.notify_one(); // 通知等待的執行緒有新結果可用
    }
}
extern "C" {
    YOLOV11_API void svCreate_ObjectModules(const char* engine_path, float conf_threshold) {
        model = std::make_unique<YOLOv11>(engine_path, conf_threshold, logger);

        // 啟動執行緒執行 infernce_thread
        stopThread = false;
        inferenceThread = std::thread([]() {
                infernce_thread();
        });
    }

    YOLOV11_API int svObjectModules_inputImageBGR(unsigned char* image_data, int width, int height, int channels, int max_output) {
        if (!model || !image_data) return 0;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            inputQueue.push({image_data, width, height, channels, max_output});
        }
        inputQueueCondition.notify_one();
        return 1;
    }

    YOLOV11_API int svObjectModules_getResult(svObjData_t* output, int max_output, bool wait) {
        std::unique_lock<std::mutex> lock(queueMutex);
        if (wait) {
            // 等待直到有結果可用
            outputQueueCondition.wait(lock, [] { return !outputQueue.empty() || stopThread; });
        }
        if (stopThread || outputQueue.empty()) return -1;
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
        inputQueueCondition.notify_all(); // 通知執行緒停止等待
        outputQueueCondition.notify_all(); // 通知執行緒停止等待
        if (inferenceThread.joinable()) {
            inferenceThread.join();
        }
        cout << "Inference thread stopped." << std::endl;
        model.reset();
    }
}
