# Requirments

## g++ 8 Compiler
```
$ sudo apt-get install g++-8
```
### Set environment variables
```
$ export CXX=/usr/bin/g++-8
$ export CC=/usr/bin/gcc-8
```
## cmake(version>=3.10.2)
```
$ sudo apt install cmake
```
## OpenCV C++ (version==4.5.1)
### Install minimal prerequisites

```
$ sudo apt update && sudo apt install -y cmake wget unzip
```
## matplotlibcpp

```
$ git clone https://github.com/lava/matplotlib-cpp
$ export MatPlotLibCPP = /path/to/matplotlib-cpp
```
### Download and unpack sources
```
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
$ unzip opencv.zip
```
### Create build directory
```
$ mkdir -p build && cd build
```
### Configure
```
$ cmake  ../opencv-master
```
### Build
```
$ cmake --build .
```
### Set environment variable
```
$ export OpenCV_DIR=/path/to/OpenCV/build
```
# Build the project
```
$ cd data_analysis
$ cd build
$ cmake ..
$ make
```
# Execution
```
$ cd ../bin/
$ ./IMANTool --data_path ../example/vin_data/ --format YOLO --size 224 --view_console --plot
```
