#include "climb_thread.h"
#include "preprocess.h"
#include "logging.h"
#include "YUV420ToRGB.h"
#include "Logger.h"
#include "yolov11_dll.h"
#include <cuda_utils.h>
#include <opencv2/opencv.hpp>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int default_yuv_size = 1920*1920*3/2 * sizeof(uint8_t); // Example size for 1920x1920 YUV420 image
static uint8_t* yuv_buffer_device = nullptr;
constexpr int INPUT_W = 640;
constexpr int INPUT_H = 640;

// 線段相交判斷函數
namespace {
    // 計算向量叉積 (p1-p0) × (p2-p0)
    float crossProduct(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2) {
        return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
    }

    // 檢查點是否在線段上（假設點已經共線）
    bool onSegment(cv::Point2f p, cv::Point2f q, cv::Point2f r) {
        return (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
                q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y));
    }

    // 判斷兩條線段是否相交
    // 線段1: p1-q1, 線段2: p2-q2
    bool doIntersect(cv::Point2f p1, cv::Point2f q1, cv::Point2f p2, cv::Point2f q2) {
        float d1 = crossProduct(p2, q2, p1);
        float d2 = crossProduct(p2, q2, q1);
        float d3 = crossProduct(p1, q1, p2);
        float d4 = crossProduct(p1, q1, q2);

        // 一般情況：如果兩條線段跨越彼此
        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
            ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            return true;
        }

        // 特殊情況：點共線且重疊
        if (d1 == 0 && onSegment(p2, p1, q2)) return true;
        if (d2 == 0 && onSegment(p2, q1, q2)) return true;
        if (d3 == 0 && onSegment(p1, p2, q1)) return true;
        if (d4 == 0 && onSegment(p1, q2, q1)) return true;

        return false;
    }
}

namespace climb {
    ICudaEngine* engine;
    IExecutionContext* context;
    queue<InputData> inputQueue;
    std::mutex inputQueueMutex;
    condition_variable inputQueueCondition;
    vector<queue<OutputData>> outputQueues;
    vector<std::unique_ptr<std::mutex>> outputQueueMutexes;
    vector<std::unique_ptr<std::condition_variable>> outputQueueConditions;
    Logger logger;
    bool stopThread = false;
    std::thread inferenceThread;

    float threshold = 0.3;
    int gpu_rgb_buffer_size = 1920*1920*3*sizeof(uint8_t); //!< The size of the device buffer for RGB input
    uint8_t* gpu_rgb_buffer = nullptr;

    float input[1 * 3 * INPUT_W * INPUT_H];
    size_t output_size = 1 * 56 * 8400;
    float* output = new float[output_size]();
    void* buffers[2];

    // Helper function to read engine file
    std::vector<char> readEngineFile(const std::string& engineFile) {
        std::ifstream file(engineFile, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open engine file");
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        return buffer;
    }

    void createModelAndStartThread(const char* engine_path, int camera_amount, float conf_threshold, const char* logFilePath) {
        AILogger::init(std::string(logFilePath));
        if (std::string(logFilePath) == "") {
            AILOG_INFO("No log file path specified, using default console logging.");
            AILogger::setConsoleOnly(true); // 如果沒有指定 logFilePath，則只輸出到 console
        } else {
            AILOG_INFO("Logging to file: " + std::string(logFilePath));
        }
        AILOG_INFO("Initializing skeleton model with engine: " + std::string(engine_path));
        threshold = conf_threshold;
        auto runtime = createInferRuntime(logger);
        auto engineData = readEngineFile(engine_path);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        context = engine->createExecutionContext();
        cudaMalloc(&buffers[0], sizeof(input));
        cudaMalloc(&buffers[1], output_size * sizeof(float));

        // 初始化 outputQueues - 使用智能指針避免複製問題
        outputQueues.reserve(camera_amount);
        outputQueueMutexes.reserve(camera_amount);
        outputQueueConditions.reserve(camera_amount);

        for (int i = 0; i < camera_amount; ++i) {
            outputQueues.emplace_back();
            outputQueueMutexes.emplace_back(std::make_unique<std::mutex>());
            outputQueueConditions.emplace_back(std::make_unique<std::condition_variable>());
        }

        // 啟動執行緒執行 inference_thread
        inferenceThread = std::thread([&]() {
            inference_thread();
        });
    }

    uint8_t* GetYuvGpuBuffer(uint8_t* yuv, int width, int height) {
        int img_size = width * height * 3 / 2 * sizeof(uint8_t);
        if (img_size > default_yuv_size) {
            AILOG_INFO("YUV buffer size changed from " + std::to_string(default_yuv_size) + " to " + std::to_string(img_size));

            if (yuv_buffer_device) CUDA_CHECK(cudaFree(yuv_buffer_device));
            CUDA_CHECK(cudaMalloc(&yuv_buffer_device, img_size));
            default_yuv_size = img_size;
        }
        if (yuv_buffer_device == nullptr) {
            AILOG_INFO("Allocating GPU YUV buffer of size: " + std::to_string(default_yuv_size));
            CUDA_CHECK(cudaMalloc(&yuv_buffer_device, default_yuv_size));
        }
        // Copy YUV data to device
        CUDA_CHECK(cudaMemcpy(yuv_buffer_device, yuv, img_size, cudaMemcpyHostToDevice));
        return yuv_buffer_device;
    }

    uint8_t* getGpuRgbBuffer(int width, int height) {
        int img_size = width * height * 3 * sizeof(uint8_t);
        if (img_size > gpu_rgb_buffer_size) {
            AILOG_INFO("RGB buffer size changed from " + std::to_string(gpu_rgb_buffer_size) + " to " + std::to_string(img_size));
            if (gpu_rgb_buffer) {
                CUDA_CHECK(cudaFree(gpu_rgb_buffer));
            }
            CUDA_CHECK(cudaMalloc(&gpu_rgb_buffer, img_size));
            gpu_rgb_buffer_size = img_size;
        }
        if (gpu_rgb_buffer == nullptr) {
            AILOG_INFO("Allocating GPU RGB buffer of size: " + std::to_string(gpu_rgb_buffer_size));
            CUDA_CHECK(cudaMalloc(&gpu_rgb_buffer, gpu_rgb_buffer_size));
        }
        return gpu_rgb_buffer;
    }

    void inference_thread() {
        AILOG_INFO("Inference thread started.");
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        while (!stopThread) {
            std::unique_lock<std::mutex> lock(inputQueueMutex);

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
            uint8_t* gpu_rgb_buffer = getGpuRgbBuffer(input.width, input.height);
            // 將 yuv 轉換成 rgb
            yuv420toRGBInPlace(gpu_yuv_buffer, input.width, input.height, gpu_rgb_buffer, stream);

            cuda_preprocess(gpu_rgb_buffer, input.width, input.height,
                            static_cast<float*>(buffers[0]), INPUT_W, INPUT_H, stream);
            context->executeV2(buffers);
            cudaMemcpy(output, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost);
            vector<Rect> boxes;
            vector<int> class_ids;
            vector<float> confidences;
            vector<skeleton_points> skeletons;
            int x_index = 0, y_index = 8400, w_index = 8400*2, h_index = 8400*3, conf_index = 8400*4;
            for (int i = 0; i < 8400; ++i) {
                float conf = output[conf_index + i];
                if (conf > threshold) {
                    int class_id = static_cast<int>(output[i]);
                    int x = static_cast<int>(output[x_index + i]);
                    int y = static_cast<int>(output[y_index + i]);
                    int w = static_cast<int>(output[w_index + i]);
                    int h = static_cast<int>(output[h_index + i]);
                    boxes.emplace_back(x, y, w, h);
                    class_ids.push_back(class_id);
                    confidences.push_back(conf);
                    skeleton_points skel;
                    skel.LeftShoulder = cv::Point(static_cast<int>(output[kps_index::LeftShoulder + i]), static_cast<int>(output[kps_index::LeftShoulder + 8400 + i]));
                    skel.RightShoulder = cv::Point(static_cast<int>(output[kps_index::RightShoulder + i]), static_cast<int>(output[kps_index::RightShoulder + 8400 + i]));
                    skel.LeftHip = cv::Point(static_cast<int>(output[kps_index::LeftHip + i]), static_cast<int>(output[kps_index::LeftHip + 8400 + i]));
                    skel.RightHip = cv::Point(static_cast<int>(output[kps_index::RightHip + i]), static_cast<int>(output[kps_index::RightHip + 8400 + i]));
                    // skel.LeftWrist = cv::Point(static_cast<int>(output[kps_index::LeftWrist + i]), static_cast<int>(output[kps_index::LeftWrist + 8400 + i]));
                    // skel.RightWrist = cv::Point(static_cast<int>(output[kps_index::RightWrist + i]), static_cast<int>(output[kps_index::RightWrist + 8400 + i]));
                    // skel.LeftKnee = cv::Point(static_cast<int>(output[kps_index::LeftKnee + i]), static_cast<int>(output[kps_index::LeftKnee + 8400 + i]));
                    // skel.RightKnee = cv::Point(static_cast<int>(output[kps_index::RightKnee + i]), static_cast<int>(output[kps_index::RightKnee + 8400 + i]));
                    // skel.LeftAnkle = cv::Point(static_cast<int>(output[kps_index::LeftAnkle + i]), static_cast<int>(output[kps_index::LeftAnkle + 8400 + i]));
                    // skel.RightAnkle = cv::Point(static_cast<int>(output[kps_index::RightAnkle + i]), static_cast<int>(output[kps_index::RightAnkle + 8400 + i]));
                    skeletons.push_back(skel);
                }
            }
            vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, confidences, threshold, 0.7f, nms_result);
            int count = std::min(static_cast<int>(nms_result.size()), input.max_output);

            // 計算 resize 比例與 padding
            float r = std::min(1.0f * INPUT_W / input.width, 1.0f * INPUT_H / input.height);
            int unpad_w = static_cast<int>(r * input.width);
            int unpad_h = static_cast<int>(r * input.height);
            int pad_x = (INPUT_W - unpad_w) / 2;
            int pad_y = (INPUT_H - unpad_h) / 2;
            svObjData_t* output = new svObjData_t[count];

            // 初始化所有元素
            for (int i = 0; i < count; ++i) {
                svObjData_init(&output[i]);
            }

            // 取得 roi
            std::unordered_map<int, ROI> roi_map;
            if (camera_function_roi_map.find(input.camera_id) != camera_function_roi_map.end()) {
                if (camera_function_roi_map[input.camera_id].find(functions::CLIMB) != camera_function_roi_map[input.camera_id].end())
                    roi_map = camera_function_roi_map[input.camera_id][functions::CLIMB];
            }
            ROI* roi_ptr = nullptr;
            for (auto& roi_pair : roi_map) {
                roi_ptr = &roi_pair.second; // 取得指針，指向原始數據
                roi_ptr->alarm <<= 1; // 左移一位，丟棄最舊的警報
            }

            for (int i = 0; i < count; ++i) {
                // 模型輸出的 bbox 相對於 640x640，要先扣 padding 再除以 r
                float index = nms_result[i];
                float x1 = static_cast<float>((boxes[index].x - boxes[index].width/2 - pad_x) / r);
                float y1 = static_cast<float>((boxes[index].y - boxes[index].height/2 - pad_y) / r);
                float x2 = static_cast<float>((boxes[index].x + boxes[index].width/2 - pad_x) / r);
                float y2 = static_cast<float>((boxes[index].y + boxes[index].height/2 - pad_y) / r);
                float ShoulderX = static_cast<float>((skeletons[index].LeftShoulder.x + skeletons[index].RightShoulder.x) / 2 - pad_x) / r;
                float ShoulderY = static_cast<float>((skeletons[index].LeftShoulder.y + skeletons[index].RightShoulder.y) / 2 - pad_y) / r;
                float HipX = static_cast<float>((skeletons[index].LeftHip.x + skeletons[index].RightHip.x) / 2 - pad_x) / r;
                float HipY = static_cast<float>((skeletons[index].LeftHip.y + skeletons[index].RightHip.y) / 2 - pad_y) / r;

                // 正規化成 [0~1]
                float norm_x1 = std::clamp(x1 / input.width, 0.0f, 1.0f);
                float norm_y1 = std::clamp(y1 / input.height, 0.0f, 1.0f);
                float norm_x2 = std::clamp(x2 / input.width, 0.0f, 1.0f);
                float norm_y2 = std::clamp(y2 / input.height, 0.0f, 1.0f);
                float norm_ShoulderX = std::clamp(ShoulderX / input.width, 0.0f, 1.0f);
                float norm_ShoulderY = std::clamp(ShoulderY / input.height, 0.0f, 1.0f);
                float norm_HipX = std::clamp(HipX / input.width, 0.0f, 1.0f);
                float norm_HipY = std::clamp(HipY / input.height, 0.0f, 1.0f);

                // 將結果放入 output
                output[i].bbox_xmin = norm_x1;
                output[i].bbox_ymin = norm_y1;
                output[i].bbox_xmax = norm_x2;
                output[i].bbox_ymax = norm_y2;
                output[i].class_id = 0;
                output[i].confidence = confidences[index];

                // 計算 Shoulder 和 Hip 是否位於 bbox 內
                if (ShoulderX < x1 || ShoulderX > x2 ||
                    ShoulderY < y1 || ShoulderY > y2 ||
                    HipX < x1 || HipX > x2 ||
                    HipY < y1 || HipY > y2) {
                    AILOG_DEBUG("Shoulder or Hip is outside the bounding box, skipping climb detection");
                    continue; // 如果 Shoulder 或 Hip 不在 bbox 內，則跳過爬牆偵測
                }

                // 計算Shoulder到Hip的向量
                float vectorX = HipX - ShoulderX;
                float vectorY = HipY - ShoulderY;

                // 計算向量長度
                float vectorLength = sqrt(vectorX * vectorX + vectorY * vectorY);

                // 如果向量長度不為0，則進行延伸
                if (vectorLength > 0) {
                    // 正規化向量
                    float normalizedVectorX = vectorX / vectorLength;
                    float normalizedVectorY = vectorY / vectorLength;

                    // 向兩端延伸一倍距離
                    // 新的Shoulder點：向Shoulder方向延伸一倍距離
                    float extendedShoulderX = ShoulderX - normalizedVectorX * vectorLength;
                    float extendedShoulderY = ShoulderY - normalizedVectorY * vectorLength;

                    // 新的Hip點：向Hip方向延伸一倍距離
                    float extendedHipX = HipX + normalizedVectorX * vectorLength;
                    float extendedHipY = HipY + normalizedVectorY * vectorLength;

                    // 更新Shoulder和Hip座標
                    ShoulderX = extendedShoulderX;
                    ShoulderY = extendedShoulderY;
                    HipX = extendedHipX;
                    HipY = extendedHipY;
                }

                // 計算Shoulder-Hip線段相對於垂直方向的傾斜角度
                float dx = norm_HipX - norm_ShoulderX;  // x方向的差值
                float dy = norm_HipY - norm_ShoulderY;  // y方向的差值

                // 計算與垂直線的夾角（以垂直為0度）
                // atan2(dx, dy) 計算與垂直向下方向的角度
                float angle_rad = atan2(abs(dx), abs(dy));  // 使用絕對值，只關心傾斜程度
                float angle_deg = angle_rad * 180.0f / M_PI;  // 轉換為度數

                AILOG_DEBUG("Shoulder-Hip line angle: " + std::to_string(angle_deg) + " degrees from vertical");


                // 只有當傾斜角度大於等於30度時才進行ROI相交判斷
                if (angle_deg < 30.0f) {
                    continue; // 如果角度小於30度，則不進行爬牆偵測
                }
                for (auto& roi_pair : roi_map) {
                    ROI* roi_ptr = &roi_pair.second;
                    bool in_roi = false;

                    // 使用Shoulder-Hip線段與ROI邊界進行相交判斷
                    cv::Point2f shoulder_point(norm_ShoulderX, norm_ShoulderY);
                    cv::Point2f hip_point(norm_HipX, norm_HipY);

                    // 檢查Shoulder-Hip線段是否與ROI的任意邊界線段相交
                    const std::vector<cv::Point2f>& points = roi_ptr->points;
                    if (points.size() >= 2) {
                        for (size_t j = 0; j < points.size(); ++j) {
                            // 獲取相鄰的兩個點形成ROI邊界線段
                            cv::Point2f roi_point1 = points[j];
                            cv::Point2f roi_point2 = points[(j + 1) % points.size()]; // 環形連接，最後一個點連接第一個點

                            // 判斷Shoulder-Hip線段是否與當前ROI邊界線段相交
                            if (doIntersect(shoulder_point, hip_point, roi_point1, roi_point2)) {
                                in_roi = true;
                                AILOG_DEBUG("Climb detected: Shoulder-Hip line intersects with ROI " +
                                        std::to_string(roi_pair.first) + " edge from (" +
                                        std::to_string(roi_point1.x) + "," + std::to_string(roi_point1.y) + ") to (" +
                                        std::to_string(roi_point2.x) + "," + std::to_string(roi_point2.y) + ")");
                                break; // 找到相交就跳出循環
                            }
                        }
                    }
                    if (in_roi){
                        output[i].in_roi_id = roi_pair.first; // 設定 ROI ID
                        strncpy(output[i].climb, "climb", sizeof(output[i].climb) - 1); // 設定姿勢為 "climb"
                        output[i].climb[sizeof(output[i].climb) - 1] = '\0'; // 確保字符串結尾
                        AILOG_INFO("Climb behavior detected in ROI " + std::to_string(roi_pair.first));
                        break; // 找到一個相交的ROI就足夠了
                    }
                }
            }

            // 將結果放入 outputQueue
            int camera_id = input.camera_id;
            {
                std::lock_guard<std::mutex> lock(*outputQueueMutexes[camera_id]);
                outputQueues[camera_id].push({output, count});
            }
            outputQueueConditions[camera_id]->notify_one(); // 通知等待的執行緒有新結果可用
        }
    }
}