#pragma once
#include <memory>

#ifdef YOLOV11_EXPORTS
#define YOLOV11_API __declspec(dllexport)
#else
#define YOLOV11_API __declspec(dllimport)
#endif

extern "C" {

typedef struct svResultProjectObject_DataType
{
    svResultProjectObject_DataType()
        : bbox_xmin(0.0f), bbox_ymin(0.0f), bbox_xmax(0.0f), bbox_ymax(0.0f),
          confidence(0.0f), class_id(-1) {
        strncpy(color_label_first, "none", sizeof(color_label_first));
        color_label_first[sizeof(color_label_first) - 1] = '\0';
        strncpy(color_label_second, "none", sizeof(color_label_second));
        color_label_second[sizeof(color_label_second) - 1] = '\0';
    }
    float bbox_xmin;
    float bbox_ymin;
    float bbox_xmax;
    float bbox_ymax;
    float confidence;
    int class_id;
    char color_label_first[256];
    char color_label_second[256];
} svObjData_t;

typedef struct InputData {
    unsigned char* image_data;
    int width;
    int height;
    int channels;
    int max_output;
};

typedef struct OutputData {
    svObjData_t* output;
    int count;
};

/**
 * 初始化模型
 */
YOLOV11_API void svCreate_ObjectModules(const char* engine_path, float conf_threshold, const char* logFilePath = "");

/**
 * 處理影像並輸出結果
 *
 * @param image_data   指向 BGR image 資料
 * @param width        圖片寬
 * @param height       圖片高
 * @param channels     通常為 3
 * @param max_output   最大可寫入數量
 * @return             實際填入的數量
 */
YOLOV11_API int svObjectModules_inputImageYUV(unsigned char* image_data, int width, int height, int channels, int max_output);
YOLOV11_API int svObjectModules_getResult(svObjData_t* output, int max_output, bool wait=true);
/**
 * 清理資源
 */
YOLOV11_API void release();
}
