cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

# find_package(CUDA REQUIRED)
# Find and include OpenCV
set(OpenCV_DIR "C:/Program Files (x86)/opencv_cuda-4.6.0")
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ColorClassifier)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/YUV420ToRGB)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/fall)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/climb)

# Set TensorRT path
set(TENSORRT_DIR "C:/Program Files (x86)/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8")
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
enable_language(CUDA)

# === Source files ===
set(COMMON_SOURCES
    src/yolov11.cpp
    src/Logger.cpp
    src/preprocess.cu
    src/detection_color_thread.cpp
    src/ColorClassifier/ColorClassifier.cpp
    src/YUV420ToRGB/YUV420ToRGB.cu
    src/fall/fall_thread.cpp
    src/climb/climb_thread.cpp
)
set(COMMON_HEADERS
    src/yolov11.h
    src/Logger.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
    src/common.h
    src/detection_color_thread.h
    src/ColorClassifier/ColorClassifier.h
    src/YUV420ToRGB/YUV420ToRGB.h
    src/fall/fall_thread.h
    src/climb/climb_thread.h
)

# === DLL target ===
set(DLL_SOURCES
    src/yolov11_dll.cpp
)

add_library(yolov11 SHARED ${DLL_SOURCES} ${COMMON_SOURCES} ${COMMON_HEADERS})
target_compile_definitions(yolov11 PRIVATE YOLOV11_EXPORTS)

target_link_libraries(yolov11
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)
message(STATUS "COMMON_SOURCES: ${COMMON_SOURCES}")
# === Test executable ===
set(EXEC_SOURCES
    main.cpp
    src/Logger.cpp
    src/ColorClassifier/ColorClassifier.cpp
    src/YUV420ToRGB/YUV420ToRGB.cu
)

add_executable(yolov11_dll ${EXEC_SOURCES})
target_link_libraries(yolov11_dll yolov11 ${OpenCV_LIBS})

# === Main RTSP Detection executable ===
set(MAIN_RTSP_SOURCES
    main_rtsp.cpp
)

add_executable(main_rtsp ${MAIN_RTSP_SOURCES})
target_link_libraries(main_rtsp yolov11 ${OpenCV_LIBS})
