# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.10)

# Name of the project
project(utils)
file(GLOB SRC_FILES ./*.cpp)
add_library(utils ${SRC_FILES})
