cmake_minimum_required(VERSION 3.17)
project(gravi2d LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gravi2d src/main.cpp src/Window.hpp src/Body.hpp src/create_random_universe.hpp src/random.hpp src/Physics.hpp src/Physics.cu)
target_link_libraries(gravi2d ${OpenCV_LIBRARIES})

set_target_properties(
        gravi2d
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)