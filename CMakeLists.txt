cmake_minimum_required(VERSION 3.17)
project(gravi2d LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gravi2d main.cpp Window.hpp Body.hpp create_random_universe.hpp random.hpp Physics.hpp Physics.cu)
target_link_libraries(gravi2d ${OpenCV_LIBRARIES})

set_target_properties(
        gravi2d
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)