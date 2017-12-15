cmake_minimum_required(VERSION 3.8)
file(GLOB inception3_src
  common.cpp
  common.h
  inception_v3.cpp
  inception_v3.h
)
add_library(inception3 STATIC ${inception3_src})
target_compile_features(inception3 PUBLIC cxx_std_11)

add_executable(inception3_basic main.cpp)
target_link_libraries(inception3_basic PRIVATE inception3 nvinfer nvparsers nvinfer_plugin cudnn cublas cudart_static nvToolsExt cudart rt dl pthread)
