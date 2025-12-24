# yolov11_dll 模組說明

> 本文件聚焦於 `src/yolov11_dll.*` 封裝的 API，協助整合者快速理解其責任範圍、輸入輸出、生命週期與最佳實務。若需要更完整的產品說明，可搭配根目錄的 `API_Documentation.md` 一同閱讀。

## 1. 模組概觀

`yolov11_dll` 是對多個 TensorRT 模型與後處理邏輯的整合層，暴露一組 C-friendly API，讓上層應用（例如 `main.cpp`、`main_rtsp.cpp` 或其他程式）能以統一的流程驅動以下功能：

| functions 枚舉值 | 功能 | 主要用途 | 需載入的 engine 路徑 |
| --- | --- | --- | --- |
| `YOLO_COLOR` (0) | 一般偵測 + 簡易顏色辨識 | 追蹤、穿越線、ROI 告警 | `detection_model` (engine_path1)、`detection_model` (engine_path2) |
| `FALL` (1) | 跌倒偵測 | 姿勢辨識 + ROI 告警 | `detection_model`、`fall_model` |
| `CLIMB` (2) | 攀爬 (MRT) 偵測 | 牆脊線觸發、紅燈 ROI | `pose_model`、`detection_model` |
| `CROWD` (3) | 群聚/覆蓋率分析 | ROI 覆蓋面積、人數統計 | `detection_model`、`detection_model` |
| `YOLO_CLOTH_COLOR` (4) | 衣著顏色偵測 | 先裁切人框，再做細緻顏色分類 | `detection_model`、`clothes_model` |

- **CUDA Stream**: 每個模型實例在各自的執行緒中建立並擁有獨立的 `cudaStream_t`，確保模型推論與預處理互不干擾。
- **執行緒模型**: 每個功能（如 YOLO_COLOR, FALL）都有獨立的 worker thread，並行運作。

### 開發者快速啟動（Handoff TL;DR）

1. **建置環境**：Windows + Visual Studio 2019/2022，CUDA 11.8，TensorRT ≥ 8.6，OpenCV 4.6（CUDA build）。以 `CMakeLists.txt` 產生 solution，並確保 `yolov11_dll` target 可在 Debug/Release 編譯。
2. **模型資產**：`weights/` 目錄下必須已有對應功能的 `.engine` 或鏈結檔（`detection_model`、`fall_model` 等）。缺檔時 `svCreate_ObjectModules` 會立即失敗。
3. **API 導入**：上層專案包含 `src/yolov11_dll.h`，連結 `yolov11_dll.dll`。生命週期流程為 `svCreate_ObjectModules → svObjectModules_inputImageYUV → svObjectModules_getResult → svRelease`。
4. **執行緒與清理**：每個 `functions` 值都會啟動 dedicated worker。退出前要停止影像來源、確保 `svRelease()` 被呼叫，並可額外 sleep 500ms 讓 CUDA/TensorRT 正常釋放。
5. **偵錯入口**：預設 log 寫入 `log/` 路徑，配合 `AILOG_DEBUG` 可看到 preprocess/infer/postprocess 耗時；若出現 `[E][TRT]` 或 CUDA runtime error，對照本文「故障排除與檢查清單」。

### 建置 / 部署需求

- **CMake / VS Project**：新增模組時別忘了同步更新 `CMakeLists.txt` 與對應 `.vcxproj`。
- **Runtime 依賴**：CUDA Driver、TensorRT DLL、OpenCV CUDA DLL 需存在於 PATH 或與應用程式同層；`weights/` 需隨附。
- **模型轉換**：若要從 ONNX 重新產生 engine，可利用 `export.py` 或 `build.ps1`；產生後透過 `convert_tensorrt.bat` / `create_link.bat` 建立固定檔名。
- **部署**：打包 `yolov11_dll.dll/.lib`、所有依賴 DLL、`weights/`、`log/`，以及必要的設定檔；確保執行環境擁有寫入 log 的權限。

## 2. 系統資料流

```mermaid
flowchart LR
    subgraph App[應用程式]
        start([svCreate_ObjectModules]) --> pipe1
        pipe1[svObjectModules_inputImageYUV (YUV420)] --> queue{{Input Queue per camera}}
        queue --> workers
        workers --> getres[svObjectModules_getResult]
        getres --> render[[使用結果：畫框 / 告警 / ROI 統計]]
        render --> rel([svRelease])
        roi[[ROI APIs]] -.-> workers
    end
    subgraph DLL[workers (DLL 內部)]
        subgraph YOLO
            Ypre[CUDA preprocess] --> Yinfer[TensorRT enqueue] --> Ypost[CPU postprocess / NMS]
        end
        subgraph FALL
            Fdetect[Detection] --> Fclass[姿勢分類]
        end
        subgraph CLIMB
            Cpose[Pose Model] --> Clogic[牆脊線/紅燈邏輯]
        end
        subgraph CROWD
            Croi[ROI coverage + 人數統計]
        end
        subgraph CLOTH
            Ccrop[cudaMemcpy2D 裁切] --> Cseg[衣著分割] --> Ccolor[顏色分類]
        end
    end
```

## 3. 公開資料結構

### 3.1 `svObjData_t`

```cpp
struct svObjData_t {
    float bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax; // 0~1 的正規化座標
    float confidence;                                 // 信心值
    int track_id;                                     // -1 表示無追蹤
    int class_id;                                     // 依 COCO ID (包含 80=wheelchair, 81=person_on_wheelchair, 82=pushchair)
    int in_roi_id;                                    // ROI 命中的 ID，無則為 -1
    int crossing_line_id;                             // 穿越線 ID，無則為 -1
    int crossing_line_direction;                      // 1=正向, -1=反向, 0=無事件
    char color_label_first[16];                       // 上半身/全身顏色
    char color_label_second[16];                      // 下半身顏色
    char pose[16];                                    // FALL: stand/falling/fall/... 其他功能為 "none"
    char climb[16];                                   // CLIMB: stand/climbing/climb/... 其他功能為 "none"
};
```

提供 `svObjData_init` 幫助初始化上述欄位。

### 3.2 ROI 資料

- `ROI`：多邊形 mask + 點位（正規化座標），附 5 bits 的警報狀態。
- `MRTRedlightROI`：MRT 用 ROI，附上左右上下界，3 bits 警報狀態。
- `CrossingLineROI`：儲存線段點列與方向資訊。

這些結構由 DLL 內部維護，外部僅透過 API `svCreate_*`、`svRemove_*` 操作。

## 4. 生命週期與必備流程

| 步驟 | API | 說明 |
| --- | --- | --- |
| 1 | `svCreate_ObjectModules(function, camera_amount, engine_path1, engine_path2, conf_threshold, logPath)` | 初始化選定功能，建立工作執行緒、載入 TensorRT engine、配置 CUDA 資源與 log 檔。`camera_amount` 會決定內部隊列/追蹤器數量。需注意相同 function 不能呼叫第二次，會引發不預期的錯誤。|
| 2 | (可選) `svCreate_ROI` / `svCreate_MRTRedlightROI` / `svCreate_CrossingLine` | 以正規化座標建立 ROI 或穿越線。可重複呼叫、覆寫同一 `roi_id`。|
| 3 | `svObjectModules_inputImageYUV(function, camera_id, yuv420_ptr, width, height, 3, max_output)` | 推送每一幀影像（須為 YUV420）。回傳值為輸入佇列長度，<0 表示失敗。建議在多攝影機時以 ring buffer 控制來源速度。|
| 4 | `svObjectModules_getResult(function, camera_id, results, max_output, wait)` | 取回結果。`wait=false` 且佇列空則回傳 -1；`wait=true` 會阻塞直到有輸出或模組停止。|
| 5 | (可選) `svRemove_*` | 在重新繪製 ROI 或關閉攝影機前，可移除特定 ROI/線。|
| 6 | `svRelease()` | 停止所有 worker thread，釋放輸入/輸出佇列、TensorRT engine/context/runtime、CUDA stream 與自訂 GPU buffer。這個函數應在所有攝影機工作完成且執行緒皆已停止後呼叫一次。|

### 時序建議

1. `svCreate_ObjectModules`
2. 對每個 camera 呼叫 `svCreate_ROI` 等設定
3. 主迴圈中：`inputImageYUV` → `getResult`
4. 收到退出指令後：停止取像、等待主迴圈結束
5. `svRemove_*` (視需求)
6. `svRelease`

## 5. ROI 與穿越線 API 摘要

| API | 主要參數 | 限制 |
| --- | --- | --- |
| `svCreate_ROI(camera_id, function_id, roi_id, width, height, points_x, points_y, point_count)` | `point_count >= 3`、點須為 0~1 的正規化座標。系統會產生遮罩供 `in_roi_id` 判斷。|
| `svCreate_MRTRedlightROI(...)` | 亦需 >=3 點；僅 CLIMB/MRT 相關流程會使用。|
| `svCreate_CrossingLine(...)` | `point_count >= 2`。每對相鄰點視為一條線；DLL 會根據 `GeometryUtils` 計算方向，用於輸出的 `crossing_line_direction`。|
| `svRemove_ROIandWall / svRemove_MRTRedlightROI / svRemove_CrossingLine` | 傳遞同樣的 `camera_id / function_id / roi_id` 即可刪除設定。|

## 6. 執行緒、記憶體與錯誤責任

- **Worker Threads**：`svCreate_ObjectModules` 會針對選定功能（YOLO/FALL...）各生成一支持續運作的執行緒。離開程式前務必呼叫 `svRelease`，以免 thread 存活導致 DLL 無法卸載。
- **Input Buffer 擁有權**：`svObjectModules_inputImageYUV` 只存指標與尺寸，不複製資料。呼叫端需確保 YUV buffer 在 worker 讀取前仍有效（若使用 OpenCV Mat，建議在輸入前呼叫 `.clone()` 或使用 ring buffer）。
- **Output Buffer 釋放**：DLL 會在 `svObjectModules_getResult` 將資料複製到呼叫端陣列後自動 `delete[]` 內部 buffer，外部不可再釋放。
- **CUDA/TensorRT 資源**：`YOLOv11` 類負責 `cudaMalloc` 出來的 RGB 與 I/O buffer 及 stream；修改 preprocess/inference 邏輯時，記得同步更新 `YOLOv11::~YOLOv11()` 的釋放流程與 `cuda_preprocess_destroy`。
- **錯誤回報**：
    - 執行中錯誤會透過 `AILOG_ERROR` / `[E][TRT]` 輸出。遇到 `cudaErrorIllegalAddress` 等錯誤時，先確認 ROI/指標沒有越界。
    - 關閉時若出現 `driver shutting down`，代表在 CUDA runtime 關閉後才釋放 TensorRT 物件。建議流程：停止讀流 → `svRelease()` → `std::this_thread::sleep_for(500ms)` → 程式結束。

## 7. 佇列與執行緒模型

- 每個功能在初始化時會建立對應的 worker 執行緒。不同功能彼此獨立，可同時存在。
- 每支攝影機會對應一組環形佇列：
  - Input queue：保存 `InputData`，由 `svObjectModules_inputImageYUV` 填入。
  - Output queue：保存 `OutputData`，worker 處理後放入，`svObjectModules_getResult` 取出後由 DLL 自動 `delete[] output`（若呼叫者將 `wait=true` 取得資料，DLL 會在複製完結果後釋放內部緩衝）。
- `wait=false` 時若沒有結果會立即返回 -1，可用於非阻塞輪詢。
- CUDA preprocess 與 TensorRT enqueue 使用同一 stream；某些功能（例如衣著顏色）會為裁切上傳使用 `cudaMemcpy2D` 避免 ROI 非連續記憶體問題。

## 8. 錯誤處理與日誌

- 模組初始化會將 log 檔寫到 `logFilePath`。內部使用 `AILOG_INFO/DEBUG/WARN/ERROR` 等級，便於追蹤 preprocess/inference/postprocess 耗時。
- 常見錯誤碼：
  - `svObjectModules_inputImageYUV` 回傳 `-1`：攝影機 ID 超出範圍、影像尺寸無效或 worker 未啟動。
  - `svObjectModules_getResult` 回傳 `-1`：`wait=false` 且無輸出，或模組已停止。
- 程式結束時若 CUDA 資源尚未完全釋放，TensorRT 可能輸出 `Error Code 1: Cuda Runtime (driver shutting down)`，可透過確保 `svRelease()` 於所有執行緒 join 後呼叫、並在退出前稍待（例如 `std::this_thread::sleep_for(500ms)`) 來避免。

> **建議**：在 Debug build 下開啟 `AILOG_DEBUG`，可看到 postprocess 各階段毫秒數，方便效能分析。

## 9. 範例流程（C++）

```cpp
const int MAX_OBJECTS = 100;
svObjData_t results[MAX_OBJECTS];

// 1. 初始化 YOLO + 顏色
svCreate_ObjectModules(functions::YOLO_COLOR, /*camera_amount=*/4,
                      "weights/detection_model", "weights/detection_model",
                      0.5f, "log/yolo.log");

// 2. 建立 ROI
float roi_x[] = {0.1f, 0.9f, 0.9f, 0.1f};
float roi_y[] = {0.1f, 0.1f, 0.9f, 0.9f};
svCreate_ROI(/*camera_id=*/0, functions::YOLO_COLOR, /*roi_id=*/0,
             frame_width, frame_height, roi_x, roi_y, 4);

while (running) {
    cap >> frame_bgr;
    cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420);

    // 3. 輸入影像
    svObjectModules_inputImageYUV(functions::YOLO_COLOR, 0,
                                  frame_yuv.data, frame_width, frame_height,
                                  /*channels=*/3, MAX_OBJECTS);

    // 4. 取回結果（阻塞等待）
    int count = svObjectModules_getResult(functions::YOLO_COLOR, 0,
                                          results, MAX_OBJECTS, true);
    if (count < 0) continue;

    for (int i = 0; i < count; ++i) {
        if (results[i].in_roi_id != -1) {
            // TODO: 達成警示、畫框…
        }
    }
}

// 5. 清理
svRemove_ROIandWall(0, functions::YOLO_COLOR, 0);
svRelease();
```

## 10. 效能建議

1. **影像格式**：務必先於 CPU 端轉成 YUV420（`cv::COLOR_BGR2YUV_I420`）。
2. **批次/幀率**：以 `svObjectModules_inputImageYUV` 拿到的回傳值監控佇列淤積，必要時可在上層做 frame skipping。
3. **ROI 數量**：大量 ROI 會增加 mask 檢查成本，建議整理成最小必要範圍。
4. **複數功能同時執行**：每種功能需獨立呼叫一次 `svCreate_ObjectModules`；注意 GPU 記憶體是否足夠。
5. **清理順序**：在呼叫 `svRelease()` 前，確保所有取像 / 顯示執行緒已停止，避免 CUDA/TensorRT 在 driver 關閉後才釋放。

## 11. 擴充與客製化指引

1. **新增 functions 枚舉**：在 `yolov11_dll.h` 的 `enum functions` 加入新的功能代號，並同步更新 `API_Documentation.md`。
2. **建立模組 thread**：參考 `src/yolo_cloth_color`、`src/crowd` 等目錄撰寫 `<feature>_thread.cpp`，負責處理 input queue、TensorRT 推論與 post-process。
3. **註冊路由**：在 `yolov11_dll.cpp` 的初始化、input、getResult、release switch-case 加入新的 branch，確保 queue 管理與資源釋放一致。
4. **CMake/VS**：將新檔案加入 `CMakeLists.txt`、`yolov11.vcxproj`，並確認 DLL 對外輸出的 symbol 一致。
5. **測試/驗證**：先用 `sample/sample.cpp` 做最小測試，再於 `main.cpp` / `main_rtsp.cpp` 加 CLI 與輸出顯示。
6. **文檔**：為新功能更新本文與 `API_Documentation.md`，並記下模型需求、閾值、特殊行為。

## 12. 故障排除與交接檢查清單

| 檢查項 | 徵兆 | 排查方式 |
| --- | --- | --- |
| 模型路徑錯誤 | `checkFileExists` / `AILOG_ERROR` 提示 Missing engine | 確認 `weights/` 中的 engine/link 存在並具讀取權限 |
| 佇列塞滿 | `svObjectModules_inputImageYUV` 回傳 -1 或 WARN | 啟用 frame skipping、拉大 queue、降低來源 FPS |
| CUDA driver shutting down | 程式結束時 `[E][TRT] driver shutting down` | 停止影像 → `svRelease()` → `sleep(500ms)`；避免重複 delete TRT 物件 |
| ROI 無效 | `in_roi_id` 永遠 -1 | 建立 ROI 時 width/height 必須是原始影像尺寸，points 為 0~1；可在 `main_rtsp` 印出點位檢查 |
| Crossing line 不觸發 | `crossing_line_id`、direction 永遠 0 | 確認 `svCreate_CrossingLine` 順序與點數；必要時使用 `GeometryUtils::doIntersect` 測試 |
| 衣著顏色錯誤 | label 和畫面不符 | 確認 `cudaMemcpy2D` pitch 與 Mat.step 相同、裁切 ROI 落在畫面內 |

**交接 Checklist**

- [ ] 確認所有 `.engine`、ONNX、轉換腳本與 `create_link.bat` 的使用方式已交付。
- [ ] 提供 Visual Studio solution / CMake 指令、第三方庫（CUDA/TensorRT/OpenCV）版本與安裝位置。
- [ ] 準備一組可重現的輸入（RTSP、影片檔、測試 YUV）與對應 log，供接手者驗證。
- [ ] 若自訂 ROI/紅燈/穿越線，附上座標或 `main_rtsp` 互動操作說明。
- [ ] 將 `svRelease` 呼叫位置、程式退出順序寫在 README 或交接筆記，避免再次觸發驅動關閉錯誤。

---

如需擴充 API（例如新增設定項、查詢內部狀態），建議以 `extern "C"` 的方式在 `yolov11_dll.h/.cpp` 定義，並在此文件新增章節描述行為，讓應用端可持續追蹤。若對特定功能有更細部需求，請同步更新 `API_Documentation.md`，保持兩份文件一致。