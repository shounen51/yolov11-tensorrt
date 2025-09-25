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

struct MRTRedlightROI {
    cv::Mat mask;
    std::vector<cv::Point2f> points;
    cv::Point left_top; // 左上角點
    cv::Point right_bottom; // 右下角點
    int width;
    int height;
    std::bitset<3> alarm;  // 3位的位元集合，用於警報狀態
};
extern std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, MRTRedlightROI>>> MRTRedlightROI_map;

struct CrossingLineROI {
    std::vector<cv::Point2f> points;
    int width;
    int height;
    std::vector<int> in_area_direction; // 進入區域的方向
};
extern std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, CrossingLineROI>>> CrossingLineROI_map;
// points is 0~1
cv::Mat createROI(int camera_id, int function_id, int roi_id, int width, int height, float* points_x, float* points_y, int point_count);

// 線段相交判斷函數
namespace GeometryUtils {
    // 計算向量叉積 (p1-p0) × (p2-p0)
    YOLOV11_API float crossProduct(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2);

    // 檢查點是否在線段上（假設點已經共線）
    YOLOV11_API bool onSegment(cv::Point2f p, cv::Point2f q, cv::Point2f r);

    // 判斷兩條線段是否相交，並返回p1相對於q1q2線段的位置關係
    // 線段1: q1-q2, 線段2: p1-p2
    // 返回值：0=不相交, -1=p1在q1q2的A側, 1=p1在q1q2的B側
    YOLOV11_API int doIntersect(cv::Point2f q1, cv::Point2f q2, cv::Point2f p1, cv::Point2f p2);
}

// 內部使用的工具函數
bool checkFileExists(const char* file_path);

extern "C" {

typedef struct svResultProjectObject_DataType
{
    float bbox_xmin;
    float bbox_ymin;
    float bbox_xmax;
    float bbox_ymax;
    float confidence;
    int track_id;
    int class_id;
    int in_roi_id;
    int crossing_line_id; // 用於 Crossing Line ROI 的 ID
    int crossing_line_direction; // 用於 Crossing Line ROI 的方向
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
    obj->track_id = -1;
    obj->class_id = -1;
    obj->in_roi_id = -1;
    obj->crossing_line_id = -1; // 初始化 Crossing Line ID
    obj->crossing_line_direction = 0; // 初始化 Crossing Line 方向
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
    CLIMB = 2,
    CROWD = 3
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
YOLOV11_API void svCreate_MRTRedlightROI(int camera_id, int function_id, int roi_id, int width, int height, float* points_x, float* points_y, int point_count);
YOLOV11_API void svRemove_MRTRedlightROI(int camera_id, int function_id, int roi_id);
YOLOV11_API void svCreate_CrossingLine(int camera_id, int function_id, int roi_id, int width, int height, float* points_x, float* points_y, int point_count);
YOLOV11_API void svRemove_CrossingLine(int camera_id, int function_id, int roi_id);

/**
 * 清理資源
 */
YOLOV11_API void release();

}