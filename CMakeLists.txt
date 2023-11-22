cmake_minimum_required(VERSION 3.10)

# set the project name
project(IMANTool VERSION 1.0)
find_package(OpenCV 4.5.1 REQUIRED)
find_package(PythonLibs 3.6)

# save binaries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# add the executable
include_directories(include third_party/include)
include_directories( ${OpenCV_INCLUDE_DIRS} )
#Change the following line to matplotlibcpp-cpp path in your system
include_directories( $ENV{MatPlotLibCPP} )
add_subdirectory(src/ImgAnalysis)
add_subdirectory(src/utils)
add_subdirectory(src/LabelAnalysis)

add_executable(IMANTool example/example.cpp)
target_include_directories(IMANTool PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(IMANTool ImgAnalysis LabelAnalysis utils stdc++fs "${OpenCV_LIBS}" ${PYTHON_LIBRARIES})
set_property(TARGET IMANTool PROPERTY CXX_STANDARD 17)
