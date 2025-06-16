#include "yolov11_dll.h"
#include "yolov11.h"
#include "logging.h"
#include "ColorClassifier.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include <cuda_utils.h>
#include <memory>
#include <opencv2/opencv.hpp>

static std::unique_ptr<YOLOv11> model;
static Logger logger;
static std::thread inferenceThread; // 用於執行 inference_thread 的執行緒
static bool stopThread = false;     // 用於控制執行緒的停止

static HLSColorClassifier colorClassfer;

static std::queue<InputData> inputQueue;
static std::queue<OutputData> outputQueue;
static std::mutex queueMutex;
static std::condition_variable inputQueueCondition;
static std::condition_variable outputQueueCondition;

static int default_yuv_size = 1920*1920*3/2 * sizeof(uint8_t); // Example size for 1920x1920 YUV420 image
static uint8_t* yuv_buffer_host = new uint8_t[default_yuv_size];
static uint8_t* yuv_buffer_device = nullptr;

uint8_t* GetYuvGpuBuffer(uint8_t* yuv, int width, int height) {
    int img_size = width * height * 3 / 2 * sizeof(uint8_t);
    if (img_size > default_yuv_size) {
        AILOG_INFO("YUV buffer size changed from " + std::to_string(default_yuv_size) + " to " + std::to_string(img_size));
        if (yuv_buffer_host) delete[] yuv_buffer_host;
        yuv_buffer_host = new uint8_t[img_size];
        if (yuv_buffer_device) CUDA_CHECK(cudaFree(yuv_buffer_device));
        CUDA_CHECK(cudaMalloc(&yuv_buffer_device, img_size));
        default_yuv_size = img_size;
    }
    if (yuv_buffer_device == nullptr) {
        AILOG_INFO("Allocating GPU YUV buffer of size: " + std::to_string(default_yuv_size));
        CUDA_CHECK(cudaMalloc(&yuv_buffer_device, default_yuv_size));
    }
    memcpy(yuv_buffer_host, yuv, img_size);
    // Copy YUV data to device
    CUDA_CHECK(cudaMemcpy(yuv_buffer_device, yuv_buffer_host, img_size, cudaMemcpyHostToDevice));
    return yuv_buffer_device;
}

void infernce_thread() {
    AILOG_INFO("Inference thread started.");
    colorClassfer.setDefaultColorRange();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
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
        // 取得 GPU buffers
        uint8_t* gpu_yuv_buffer = GetYuvGpuBuffer(input.image_data, input.width, input.height);
        uint8_t* gpu_rgb_buffer = model->getGpuRgbBuffer(input.width, input.height);
        // 將 yuv 轉換成 rgb
        yuv420toRGBInPlace(gpu_yuv_buffer, input.width, input.height, gpu_rgb_buffer, stream);

        std::vector<Detection> detections;

        model->preprocess(gpu_rgb_buffer, input.width, input.height, false);
        // 利用空檔時間將 GPU buffer 從 device 轉成 Mat
        std::vector<uint8_t> rgb_host(input.width * input.height * 3);
        CUDA_CHECK(cudaMemcpy(rgb_host.data(), gpu_rgb_buffer, input.width * input.height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        cv::Mat rgb_image(input.height, input.width, CV_8UC3, rgb_host.data());
        // 等待 preprocess 完成
        model->joinGPUStream();
        model->infer();
        model->postprocess(detections);

        int count = std::min(static_cast<int>(detections.size()), input.max_output);

        // YOLO input size（根據模型固定）
        const int INPUT_W = model->input_w;
        const int INPUT_H = model->input_h;

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

            // 計算顏色
            cv::Mat personCrop = rgb_image(Rect(Point(x1, y1),
                                            Point(x2, y2)));
            cv::cvtColor(personCrop, personCrop, cv::COLOR_BGR2RGB); // 將 BGR 轉換為 RGB
            vector<unsigned char> color = colorClassfer.classifyStatistics(personCrop, 500, cv::COLOR_BGR2HLS);
            int maxIndex = 0;
            int maxCount = 0;
            for (int j = 0; j < color.size(); j++) {
                if (color[j] > maxCount) {
                    maxCount = color[j];
                    maxIndex = j;
                }
            }
            string colorStr = ColorLabelsString[maxIndex];

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
            strncpy(output[i].color_label, colorStr.c_str(), sizeof(output[i].color_label) - 1);
            output[i].color_label[sizeof(output[i].color_label) - 1] = '\0'; // 確保以空字元結尾
        }

        // 將結果放入 outputQueue
        lock.lock();
        outputQueue.push({output, count});
        lock.unlock();
        outputQueueCondition.notify_one(); // 通知等待的執行緒有新結果可用
    }
}
extern "C" {
    YOLOV11_API void svCreate_ObjectModules(const char* engine_path, float conf_threshold, const char* logFilePath) {
        AILogger::init(std::string(logFilePath));
        if (std::string(logFilePath) == "") {
            AILOG_INFO("No log file path specified, using default console logging.");
            AILogger::setConsoleOnly(true); // 如果沒有指定 logFilePath，則只輸出到 console
        }
        else {
            AILOG_INFO("Logging to file: " + std::string(logFilePath));
        }
        AILOG_INFO("Initializing YOLOv11 model with engine: " + std::string(engine_path));
        model = std::make_unique<YOLOv11>(engine_path, conf_threshold, logger);

        // 啟動執行緒執行 infernce_thread
        stopThread = false;
        inferenceThread = std::thread([]() {
                infernce_thread();
        });
    }

    YOLOV11_API int svObjectModules_inputImageYUV(unsigned char* image_data, int width, int height, int channels, int max_output) {
        if (!model){
            AILOG_WARN("Model not initialized. Call svCreate_ObjectModules first.");
            return 0;
        }
        std::lock_guard<std::mutex> lock(queueMutex);
        inputQueue.push({image_data, width, height, channels, max_output});
        inputQueueCondition.notify_one();
        if (inputQueue.size() > 100)
            AILOG_WARN("Input queue size exceeds 100, input too fast.");
        return 1;
    }

    YOLOV11_API int svObjectModules_getResult(svObjData_t* output, int max_output, bool wait) {
        std::unique_lock<std::mutex> lock(queueMutex);
        if (wait) {
            // 等待直到有結果可用
            if (outputQueue.empty())
                // AILOG_INFO("Waiting for output queue to have results.");
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
        AILOG_INFO("Inference thread stopped.");
        model.reset();
    }
}
