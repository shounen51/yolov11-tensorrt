#include "yolov11_dll.h"
#include "yolov11.h"
#include "logging.h"
#include <memory>
#include <opencv2/opencv.hpp>

static std::unique_ptr<YOLOv11> model;
static Logger logger;

extern "C" {
YOLOV11_API void svCreate_ObjectModules(const char* engine_path, float conf_threshold) {
    model = std::make_unique<YOLOv11>(engine_path, conf_threshold, logger);
}

YOLOV11_API int svObjectModules_inputImageBGR(unsigned char* image_data, int width, int height, int channels, svObjData_t* output, int max_output) {
    if (!model || !image_data || !output) return 0;

    // 建立 Mat
    cv::Mat image;
    image = cv::Mat(height, width, CV_8UC3);
	memcpy(image.data, image_data, width*height*channels);
    // cv::Mat image(height, width, CV_8UC3, image_data);

    std::vector<Detection> detections;
    model->preprocess(image);
    model->infer();
    model->postprocess(detections);

    int count = std::min(static_cast<int>(detections.size()), max_output);

    // YOLO input size（根據模型固定）
    const int INPUT_W = 640;
    const int INPUT_H = 640;

    // 計算 resize 比例與 padding
    float r = std::min(1.0f * INPUT_W / width, 1.0f * INPUT_H / height);
    int unpad_w = static_cast<int>(r * width);
    int unpad_h = static_cast<int>(r * height);
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
        x1 = std::clamp(x1 / width, 0.0f, 1.0f);
        y1 = std::clamp(y1 / height, 0.0f, 1.0f);
        x2 = std::clamp(x2 / width, 0.0f, 1.0f);
        y2 = std::clamp(y2 / height, 0.0f, 1.0f);

        output[i].class_id = det.class_id;
        output[i].confidence = det.conf;
        output[i].bbox_xmin = x1;
        output[i].bbox_ymin = y1;
        output[i].bbox_xmax = x2;
        output[i].bbox_ymax = y2;
    }
    return count;
}

YOLOV11_API void release() {
    model.reset();
}
}
