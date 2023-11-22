# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.10)

# Name of the project
project(LabelAnalysis)
find_package(OpenCV 4.5.1 REQUIRED)
include_directories(../../include)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
file(GLOB SRC_FILES ./*.cpp)
add_library(LabelAnalysis ${SRC_FILES})
