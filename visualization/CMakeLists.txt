cmake_minimum_required(VERSION 3.10)
project(Visualization CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(train_export train_and_export.cpp)
target_include_directories(train_export PRIVATE ../Tensor)

add_executable(train_live train_live.cpp)
target_include_directories(train_live PRIVATE ../Tensor)
