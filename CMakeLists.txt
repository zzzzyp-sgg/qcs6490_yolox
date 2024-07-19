cmake_minimum_required(VERSION 3.0)
project(test)

set(DEFAULT_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -ggdb ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(include)
set(AidSDKLib "/usr/local/lib/libaidlite.so")

# glog
find_package(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIRS})

# gflags
find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRS})

# opencv
find_package(OpenCV REQUIRED
             PATHS /usr/local/lib/aidlux_opencv/lib/cmake/opencv4
             NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(YoloxLib SHARED src/yolox.cpp)
target_link_libraries(YoloxLib ${AidSDKLib} glog::glog ${gflags_LIBRARIES} ${OpenCV_LIBS})

add_executable(test test.cpp)
target_link_libraries(test YoloxLib)