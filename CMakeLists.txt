﻿cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

find_package(CUDA REQUIRED)
# Find and include OpenCV
set(OpenCV_DIR "C:/Program Files (x86)/opencv_cuda-4.6.0")
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ColorClassifier)

# Set TensorRT path
set(TENSORRT_DIR "C:/Program Files (x86)/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8")
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
enable_language(CUDA)

# === Source files ===
set(COMMON_SOURCES
    src/yolov11.cpp
    src/preprocess.cu
)
set(COMMON_HEADERS
    src/yolov11.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
    src/common.h
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
    src/ColorClassifier/ColorClassifier.cpp
)

add_executable(yolov11_test ${EXEC_SOURCES})
target_link_libraries(yolov11_test yolov11 ${OpenCV_LIBS})
