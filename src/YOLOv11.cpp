#include "YOLOv11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include "Logger.h"
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


YOLOv11::YOLOv11(string model_path, float conf_threshold, nvinfer1::ILogger& logger)
{
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        init(model_path, conf_threshold, logger);
    }
    // Build an engine from an onnx model
    else
    {
        AILOG_INFO("Build an engine from an onnx model");
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}


void YOLOv11::init(std::string engine_path, float conf_threshold, nvinfer1::ILogger& logger)
{
    this->conf_threshold = conf_threshold;
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
    num_classes = detection_attribute_size - 4;

    // Initialize input buffers
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&stream));


    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        AILOG_INFO("model warmup 10 times");
    }
}

YOLOv11::~YOLOv11()
{
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    if (gpu_rgb_buffer) {
        CUDA_CHECK(cudaFree(gpu_rgb_buffer));
        gpu_rgb_buffer = nullptr;
    }

    delete[] cpu_output_buffer;

    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

uint8_t* YOLOv11::getGpuRgbBuffer(int width, int height) {
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

void YOLOv11::preprocess(uint8_t* gpu_rgb_buffer, int im0_w, int im0_h, bool block) {
    // Preprocessing data on gpu
    cuda_preprocess(gpu_rgb_buffer, im0_w, im0_h, gpu_buffers[0], input_w, input_h, stream);
    if (block) CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer()
{
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    this->context->enqueueV3(this->stream);
#endif
}

void YOLOv11::postprocess(vector<Detection>& output)
{
    // Memcpy from device output buffer to host output buffer
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes_pw; // pw special
    vector<int> class_ids_pw; // pw special
    vector<float> confidences_pw; // pw special
    vector<Rect> boxes_w; // pw special
    vector<int> class_ids_w; // pw special
    vector<float> confidences_w; // pw special
    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            if ((box.width > 0.5*640 || box.height > 0.5*640) && class_id_point.y == static_cast<int>(CustomClass::PERSON)) {
                // filter out large boxes
                AILOG_INFO("Filter out large person box: " + std::to_string(box.width) + "x" + std::to_string(box.height));
                continue;
            }
            if (class_id_point.y == static_cast<int>(CustomClass::PERSON_ON_WHEELCHAIR)) { // pw special
                boxes_pw.push_back(box); // pw special
                class_ids_pw.push_back(class_id_point.y); // pw special
                confidences_pw.push_back(score); // pw special
            } // pw special
            else if (class_id_point.y == static_cast<int>(CustomClass::WHEELCHAIR)) {
                boxes_w.push_back(box);
                class_ids_w.push_back(class_id_point.y);
                confidences_w.push_back(score);
            }

            else { // pw special
                boxes.push_back(box);
                class_ids.push_back(class_id_point.y);
                confidences.push_back(score);
            } // pw special
        }
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
    vector<int> nms_result_pw; // pw special
    dnn::NMSBoxes(boxes_pw, confidences_pw, conf_threshold, nms_threshold, nms_result_pw); // pw special
    vector<int> nms_result_w; // pw special
    dnn::NMSBoxes(boxes_w, confidences_w, conf_threshold, nms_threshold, nms_result_w); // pw special

    // 收集所有人框的中心點 (包括一般人和輪椅上的人)
    vector<Point> person_centers;
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        if (class_ids[idx] == static_cast<int>(CustomClass::PERSON)) {
            Rect box = boxes[idx];
            Point center(box.x + box.width/2, box.y + box.height/2);
            person_centers.push_back(center);
        }
    }
    for (int i = 0; i < nms_result_pw.size(); i++) {
        int idx = nms_result_pw[i];
        if (class_ids_pw[idx] == static_cast<int>(CustomClass::PERSON_ON_WHEELCHAIR)) {
            Rect box = boxes_pw[idx];
            Point center(box.x + box.width/2, box.y + box.height/2);
            person_centers.push_back(center);
        }
    }

    // 過濾一般檢測框 - 移除包含超過2個人框中心的框
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Rect current_box = boxes[idx];

        // 計算當前框中包含多少個人框中心
        int person_count_in_box = 0;
        for (const Point& center : person_centers) {
            if (current_box.contains(center)) {
                person_count_in_box++;
            }
        }

        // 如果包含超過2個人框中心，則跳過這個檢測
        if (person_count_in_box > 2) {
            AILOG_INFO("Filter out box containing " + std::to_string(person_count_in_box) + " person centers: " +
                      std::to_string(current_box.width) + "x" + std::to_string(current_box.height));
            continue;
        }

        Detection result;
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }

    // 過濾輪椅上的人檢測框 - 移除包含超過2個人框中心的框
    for (int i = 0; i < nms_result_pw.size(); i++) // pw special
    {
        int idx = nms_result_pw[i];
        Rect current_box = boxes_pw[idx];

        // 計算當前框中包含多少個人框中心
        int person_count_in_box = 0;
        for (const Point& center : person_centers) {
            if (current_box.contains(center)) {
                person_count_in_box++;
            }
        }

        // 如果包含超過2個人框中心，則跳過這個檢測
        if (person_count_in_box > 2) {
            AILOG_INFO("Filter out person_on_wheelchair box containing " + std::to_string(person_count_in_box) +
                      " person centers: " + std::to_string(current_box.width) + "x" + std::to_string(current_box.height));
            continue;
        }

        Detection result;
        result.class_id = class_ids_pw[idx]; // pw special
        result.conf = confidences_pw[idx]; // pw special
        result.bbox = boxes_pw[idx]; // pw special
        output.push_back(result); // pw special
    } // pw special
    for (int i = 0; i < nms_result_w.size(); i++) // pw special
    {
        Detection result;
        int idx = nms_result_w[i]; // pw special
        result.class_id = class_ids_w[idx]; // pw special
        result.conf = confidences_w[idx]; // pw special
        result.bbox = boxes_w[idx]; // pw special
        output.push_back(result); // pw special
    } // pw special
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
        AILOG_INFO("FP16");
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    AILOG_INFO("createInferRuntime");
    runtime = createInferRuntime(logger);
    AILOG_INFO("deserializeCudaEngine");
    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    AILOG_INFO("createExecutionContext");
    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool YOLOv11::saveEngine(const std::string& onnxpath)
{
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false;
    }

    // Save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            AILOG_INFO("Create engine file" + engine_path + " failed");
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}