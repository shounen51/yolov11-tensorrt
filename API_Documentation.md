# YOLOv11 TensorRT DLL API 文檔

## 概述

YOLOv11 TensorRT DLL 提供了物件檢測、跌倒檢測、攀爬檢測等功能，支援多攝像頭同時處理、ROI 區域檢測、穿越線檢測和追蹤功能。

## 版本訊息

**當前版本**：1.1.1

模型名稱：
 - detection_model
 - fall_model
 - pose_model

使用 threshold：
 - YOLO_COLOR:
    - threshold: 0.5
 - FALL:
    - threshold: 0.5
 - CLIMB:
    - threshold: 0.3
 - CROWD:
    - threshold: 0.3

更新內容：

1. 新增功能「CROWD」以偵測群聚人群，功能如下
    - 不會輸出一般的框
    - 必須繪製 roi 才可以進行偵測
    - 偵測 roi 範圍內被人框覆蓋的百分比
    - 當百分比超過 30% 時輸出一個與 roi 同大小、位置的人框，並且以 confidence 紀錄覆蓋的百分比，以 in_roi 紀錄計算的 roi_id
2. 為了未來改版時可能只改模型不改 code，模型的讀取一律改用 symlink，當重新轉換 tensorrt 檔案後務必刪除舊的 model symlink，並且以新的 engine 檔案建立 symlink，建立方式是將 engine 檔案拖曳到「create_link.bat - 捷徑」中，輸入該 engine 的對應模型名稱即可
3. FALL 功能的跌倒定義由「坐在地上+倒地」改為「倒地」，移除的姿勢包含低坐姿、低蹲姿，坐下往後撐地仍然是倒地姿勢
3. 增加 log
---
## 核心數據結構

### 函數類型枚舉
```cpp
enum functions {
    YOLO_COLOR = 0,   // 物件檢測與顏色分類
    FALL = 1,         // 跌倒檢測
    CLIMB = 2         // 攀爬檢測
};
```

### 檢測結果結構
```cpp
typedef struct svResultProjectObject_DataType {
    float bbox_xmin;              // 邊界框左上角 X 座標 (正規化 0-1)
    float bbox_ymin;              // 邊界框左上角 Y 座標 (正規化 0-1)
    float bbox_xmax;              // 邊界框右下角 X 座標 (正規化 0-1)
    float bbox_ymax;              // 邊界框右下角 Y 座標 (正規化 0-1)
    float confidence;             // 檢測信心值 (0-1)
    int track_id;                 // 追蹤 ID (-1 表示無追蹤)
    int class_id;                 // 類別 ID
    int in_roi_id;                // ROI ID (-1 表示不在任何 ROI 內)
    int crossing_line_id;         // 穿越線 ID (-1 表示未穿越)
    int crossing_line_direction;  // 穿越方向 (1=正向, -1=反向, 0=無穿越)
    char color_label_first[16];   // 第一個顏色標籤 (上半身或整體)
    char color_label_second[16];  // 第二個顏色標籤 (下半身)
    char pose[16];                // 姿勢標籤 (FALL 功能使用) [none, stand, falling, fall]
    char climb[16];               // 攀爬標籤 (CLIMB 功能使用) [none, stand, climbing, climb]
} svObjData_t;
```

### class_id 類別對應

檢測結果中的 `class_id` 已映射轉換為標準 COCO 類別編號：

```
0  - person
1  - bicycle
2  - car
3  - motorcycle
5  - bus
6  - train
13 - bench
24 - backpack
25 - umbrella
26 - handbag
28 - suitcase
39 - bottle
40 - wine_glass
56 - chair
57 - couch
75 - vase
80 - wheelchair
81 - person_on_wheelchair
```


#### 常用類別快速參考
```cpp
// 常用的 COCO class_id
const int PERSON = 0;              // 人員
const int BICYCLE = 1;             // 腳踏車
const int CAR = 2;                 // 汽車
const int MOTORCYCLE = 3;          // 機車
const int BUS = 5;                 // 公車
const int TRAIN = 6;               // 火車
const int WHEELCHAIR = 80;         // 輪椅
const int PERSON_ON_WHEELCHAIR = 81; // 坐輪椅的人
```

```
## 核心 API 函數

### 1. 模型初始化

```cpp
YOLOV11_API void svCreate_ObjectModules(
    int function,              // 功能類型 (functions 枚舉)
    int camera_amount,         // 攝像頭數量
    const char* engine_path1,  // 主模型引擎路徑
    const char* engine_path2,  // 輔助模型引擎路徑 (某些功能需要)
    float conf_threshold,      // 信心度閾值
    const char* logFilePath    // 日誌檔案路徑 (可選)
);
```

**說明**：
- **YOLO_COLOR**：需要一個 YOLOv11 物件檢測模型
- **FALL**：需要兩個模型 (物件檢測 + 跌倒分類)
- **CLIMB**：需要一個姿勢檢測模型

**範例**：
```cpp
// 初始化物件檢測
svCreate_ObjectModules(functions::YOLO_COLOR, 128,
                      "wheelchair_m_1.3.0.engine",
                      "wheelchair_m_1.3.0.engine",
                      0.3f, "log/detection.log");

// 初始化跌倒檢測
svCreate_ObjectModules(functions::FALL, 128,
                      "wheelchair_m_1.3.0.engine",
                      "yolo-fall4s-cls_1.5.0.engine",
                      0.3f, "log/fall.log");
```

### 2. 影像處理

```cpp
YOLOV11_API int svObjectModules_inputImageYUV(
    int function,              // 功能類型
    int camera_id,            // 攝像頭 ID (0 開始)
    unsigned char* image_data, // YUV420 格式影像數據
    int width,                // 影像寬度
    int height,               // 影像高度
    int channels,             // 通道數 (通常為 3)
    int max_output            // 最大輸出物件數量
);
```

**返回值**：輸入佇列大小，-1 表示失敗

**範例**：
```cpp
Mat frame_bgr, frame_yuv;
cap >> frame_bgr;
cvtColor(frame_bgr, frame_yuv, COLOR_BGR2YUV_I420);

int queue_size = svObjectModules_inputImageYUV(
    functions::YOLO_COLOR, 0, frame_yuv.data,
    width, height, 3, 100);
```

### 3. 結果獲取

```cpp
YOLOV11_API int svObjectModules_getResult(
    int function,              // 功能類型
    int camera_id,            // 攝像頭 ID
    svObjData_t* output,      // 輸出結果數組
    int max_output,           // 最大輸出數量
    bool wait = true          // 是否等待結果
);
```

**返回值**：實際檢測到的物件數量，-1 表示失敗

**範例**：
```cpp
const int MAX_OBJECTS = 100;
svObjData_t results[MAX_OBJECTS];

int num_objects = svObjectModules_getResult(
    functions::YOLO_COLOR, 0, results, MAX_OBJECTS, true);

for (int i = 0; i < num_objects; i++) {
    cout << "Object " << i << ": confidence=" << results[i].confidence
         << ", track_id=" << results[i].track_id
         << ", in_roi=" << results[i].in_roi_id << endl;
}
```

## ROI 管理 API

### 1. 創建普通 ROI

```cpp
YOLOV11_API void svCreate_ROI(
    int camera_id,            // 攝像頭 ID
    int function_id,          // 功能 ID
    int roi_id,               // ROI ID
    int width,                // 圖片寬度
    int height,               // 圖片高度
    float* points_x,          // X 座標數組 (正規化 0-1)
    float* points_y,          // Y 座標數組 (正規化 0-1)
    int point_count           // 點數量 (至少 3 個)
);
```

**範例**：
```cpp
// 創建右半邊螢幕的 ROI
float points_x[] = {0.5f, 1.0f, 1.0f, 0.5f};
float points_y[] = {0.0f, 0.0f, 1.0f, 1.0f};
svCreate_ROI(0, functions::YOLO_COLOR, 0, width, height,
             points_x, points_y, 4);
```

### 2. 創建穿越線 ROI

```cpp
YOLOV11_API void svCreate_CrossingLine(
    int camera_id,            // 攝像頭 ID
    int function_id,          // 功能 ID
    int roi_id,               // ROI ID
    int width,                // 圖片寬度
    int height,               // 圖片高度
    float* points_x,          // X 座標數組 (正規化 0-1)
    float* points_y,          // Y 座標數組 (正規化 0-1)
    int point_count           // 點數量 (至少 2 個)
);
```

**說明**：
- 支援多個線段，每對相鄰點構成一條線段
- 系統會自動計算每條線段的方向判斷基準
- 當物件的追蹤軌跡穿越線段時會觸發穿越事件

**範例**：
```cpp
// 創建 U 字型穿越線
float line_x[] = {0.2f, 0.2f, 0.8f, 0.8f};
float line_y[] = {0.2f, 0.8f, 0.8f, 0.2f};
svCreate_CrossingLine(0, functions::YOLO_COLOR, 0, width, height,
                     line_x, line_y, 4);
```

### 3. 創建 MRT 紅燈 ROI

```cpp
YOLOV11_API void svCreate_MRTRedlightROI(
    int camera_id,            // 攝像頭 ID
    int function_id,          // 功能 ID
    int roi_id,               // ROI ID
    int width,                // 圖片寬度
    int height,               // 圖片高度
    float* points_x,          // X 座標數組 (正規化 0-1)
    float* points_y,          // Y 座標數組 (正規化 0-1)
    int point_count           // 點數量 (至少 3 個)
);
```

### 4. 移除 ROI

```cpp
YOLOV11_API void svRemove_ROIandWall(int camera_id, int function_id, int roi_id);
YOLOV11_API void svRemove_CrossingLine(int camera_id, int function_id, int roi_id);
YOLOV11_API void svRemove_MRTRedlightROI(int camera_id, int function_id, int roi_id);
```

## 使用流程

### 基本檢測流程

```cpp
// 1. 初始化模型
svCreate_ObjectModules(functions::YOLO_COLOR, 128,
                      "model.engine", "model.engine", 0.3f, "log.txt");

// 2. (可選) 創建 ROI
float roi_x[] = {0.0f, 1.0f, 1.0f, 0.0f};
float roi_y[] = {0.0f, 0.0f, 1.0f, 1.0f};
svCreate_ROI(0, functions::YOLO_COLOR, 0, width, height, roi_x, roi_y, 4);

// 3. (可選) 創建穿越線
float line_x[] = {0.0f, 1.0f};
float line_y[] = {0.5f, 0.5f};
svCreate_CrossingLine(0, functions::YOLO_COLOR, 0, width, height, line_x, line_y, 2);

// 4. 主處理循環
while (running) {
    // 讀取影像並轉換為 YUV420
    cap >> frame_bgr;
    cvtColor(frame_bgr, frame_yuv, COLOR_BGR2YUV_I420);

    // 輸入影像
    svObjectModules_inputImageYUV(functions::YOLO_COLOR, 0,
                                 frame_yuv.data, width, height, 3, 100);

    // 獲取結果
    svObjData_t results[100];
    int num = svObjectModules_getResult(functions::YOLO_COLOR, 0, results, 100, true);

    // 處理結果
    for (int i = 0; i < num; i++) {
        if (results[i].crossing_line_id != -1) {
            cout << "Crossing detected! Track ID: " << results[i].track_id
                 << ", Direction: " << results[i].crossing_line_direction << endl;
        }
    }
}

// 5. 清理資源
svRemove_ROIandWall(0, functions::YOLO_COLOR, 0);
svRemove_CrossingLine(0, functions::YOLO_COLOR, 0);
release();
```

### 多攝像頭處理

```cpp
const int CAMERA_COUNT = 4;
const int MAX_OBJECTS = 100;

// 初始化
svCreate_ObjectModules(functions::YOLO_COLOR, CAMERA_COUNT,
                      "model.engine", "model.engine", 0.3f, "log.txt");

// 為每個攝像頭設置 ROI
for (int cam = 0; cam < CAMERA_COUNT; cam++) {
    svCreate_ROI(cam, functions::YOLO_COLOR, 0, width, height,
                roi_x, roi_y, 4);
}

// 處理循環
while (running) {
    // 為所有攝像頭輸入影像
    for (int cam = 0; cam < CAMERA_COUNT; cam++) {
        // 獲取攝像頭影像...
        svObjectModules_inputImageYUV(functions::YOLO_COLOR, cam,
                                     yuv_data, width, height, 3, MAX_OBJECTS);
    }

    // 獲取所有攝像頭的結果
    for (int cam = 0; cam < CAMERA_COUNT; cam++) {
        svObjData_t results[MAX_OBJECTS];
        int num = svObjectModules_getResult(functions::YOLO_COLOR, cam,
                                          results, MAX_OBJECTS, true);
        // 處理該攝像頭的結果...
    }
}
```

## 追蹤功能

DLL 內建了物件追蹤功能，會自動為檢測到的人員分配 `track_id`：

- `track_id = -1`：未分配追蹤 ID
- `track_id >= 1`：有效的追蹤 ID

## 穿越線檢測原理

1. **線段定義**：每對相鄰點構成一條線段
2. **方向計算**：系統自動計算每條線段的垂直方向作為判斷基準
3. **穿越檢測**：當物件的追蹤軌跡(前一幀和當前幀的底邊中心)與線段相交時觸發
4. **方向判斷**：比較物件移動方向與預設方向，返回 1（正向）或 -1（反向）

## 注意事項

1. **座標系統**：所有 ROI 座標都使用正規化座標 (0-1)
2. **影像格式**：輸入必須是 YUV420 格式
3. **線程安全**：API 支援多攝像頭併發處理
4. **記憶體管理**：記得調用 `release()` 清理資源
5. **模型路徑**：確保引擎檔案存在且可讀取

## 錯誤處理

- 函數返回 -1 通常表示錯誤
- 檢查日誌檔案獲取詳細錯誤信息
- ROI ID 不能為 -1
- 點數量必須符合最小要求（ROI 至少 3 點，穿越線至少 2 點）

## 清理資源

```cpp
// 移除所有 ROI
svRemove_ROIandWall(camera_id, function_id, roi_id);
svRemove_CrossingLine(camera_id, function_id, roi_id);
svRemove_MRTRedlightROI(camera_id, function_id, roi_id);

// 釋放模型資源
release();
```


## 完整範例參考

詳細的使用範例可參考：
- `main.cpp`：基本多攝像頭處理範例

    執行：`.\yolov11_dll.exe`
- `main_rtsp.cpp`：RTSP 串流處理與 ROI 互動範例

    執行：`.\main_rtsp.exe climb rtsp://root:root@192.168.31.212/cam1/h264-1 ./log/climb.log`
