cmake_minimum_required(VERSION 3.10)
project(TensorAutograd CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tensor_tests tests.cpp)
add_executable(tensor_demo main.cpp)
add_executable(tensor_cli interactive.cpp)
