cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(hw)

set(CMAKE_PREFIX_PATH /home/thainq/Desktop/capstone/libtorch/libtorch_lib)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} hw.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)