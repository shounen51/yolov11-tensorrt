#pragma once
#include <memory>
#include <unordered_map>
#include <bitset>
#include <opencv2/opencv.hpp>
#ifdef YOLOV11_EXPORTS
#define YOLOV11_API __declspec(dllexport)
#else
#define YOLOV11_API __declspec(dllimport)
#endif

struct ROI {
    cv::Mat mask;
    std::vector<cv::Point2f> points;
    int width;
    int height;
    std::bitset<5> alarm;  // 5位的位元集合，用於警報狀態
};
extern std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, ROI>>> camera_function_roi_map;

// points is 0~1
cv::Mat createROI(int camera_id, int function_id, int roi_id, int width, int height, float* points_x, float* points_y, int point_count);

extern "C" {

typedef struct svResultProjectObject_DataType
{
    float bbox_xmin;
    float bbox_ymin;
    float bbox_xmax;
    float bbox_ymax;
    float confidence;
    int class_id;
    int in_roi_id;
    char color_label_first[16];
    char color_label_second[16];
    char pose[16];
    char climb[16];
} svObjData_t;

// C 兼容的初始化函數
inline void svObjData_init(svObjData_t* obj) {
    obj->bbox_xmin = 0.0f;
    obj->bbox_ymin = 0.0f;
    obj->bbox_xmax = 0.0f;
    obj->bbox_ymax = 0.0f;
    obj->confidence = 0.0f;
    obj->class_id = -1;
    obj->in_roi_id = -1;
    strncpy(obj->color_label_first, "none", sizeof(obj->color_label_first));
    obj->color_label_first[sizeof(obj->color_label_first) - 1] = '\0';
    strncpy(obj->color_label_second, "none", sizeof(obj->color_label_second));
    obj->color_label_second[sizeof(obj->color_label_second) - 1] = '\0';
    strncpy(obj->pose, "none", sizeof(obj->pose));
    obj->pose[sizeof(obj->pose) - 1] = '\0';
    strncpy(obj->climb, "none", sizeof(obj->climb));
    obj->climb[sizeof(obj->climb) - 1] = '\0';
}

typedef struct InputData {
    int camera_id;
    // 注意：image_data 是指向 YUV420 格式的影像資料
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

enum functions {
    YOLO_COLOR = 0,
    FALL = 1,
    CLIMB = 2
};

/**
 * 初始化模型
 */
YOLOV11_API void svCreate_ObjectModules(int function, int camera_amount, const char* engine_path1, const char* engine_path2, float conf_threshold, const char* logFilePath = "");

/**
 * 處理影像並輸出結果
 *
 * @param function     函數名稱，[ "yolo_color"]
 * @param camera_id    攝影機 ID
 * @param roi_id       ROI ID，若為 -1 則不使用 ROI
 * @param image_data   指向 BGR image 資料
 * @param width        圖片寬
 * @param height       圖片高
 * @param channels     通常為 3
 * @param max_output   最大可寫入數量
 * @return             實際填入的數量
 */
YOLOV11_API int svObjectModules_inputImageYUV(int function, int camera_id, unsigned char* image_data, int width, int height, int channels, int max_output);
// 如果 wait 為 false 且 outputQueue 為空，則直接返回 -1
YOLOV11_API int svObjectModules_getResult(int function, int camera_id, svObjData_t* output, int max_output, bool wait=true);

YOLOV11_API void svCreate_ROI(int camera_id, int function_id, int roi_id, int width, int height, float* points_x, float* points_y, int point_count);
YOLOV11_API void svRemove_ROIandWall(int camera_id, int function_id, int roi_id);
/**
 * 清理資源
 */
YOLOV11_API void release();

}