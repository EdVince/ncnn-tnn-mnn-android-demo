project(styletransferncnn)

cmake_minimum_required(VERSION 3.4.1)

# opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.4-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

# ncnn
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220721-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# mnn
set(MNN_DIR ${CMAKE_SOURCE_DIR}/mnn)
include_directories(${MNN_DIR}/include/)
include_directories(${MNN_DIR}/include/MNN/)
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${MNN_DIR}/${ANDROID_ABI}/libMNN.so)

# tnn
set(TNN_DIR ${CMAKE_SOURCE_DIR}/tnn-v0.3.0-android)
include_directories(${TNN_DIR}/include/)
include_directories(${TNN_DIR}/include/tnn/)
add_library(TNN SHARED IMPORTED)
set_target_properties(TNN PROPERTIES IMPORTED_LOCATION ${TNN_DIR}/${ANDROID_ABI}/libTNN.so)


add_library(styletransferncnn SHARED styletransferncnn_jni.cpp)
target_link_libraries(styletransferncnn MNN TNN ncnn ${OpenCV_LIBS})
