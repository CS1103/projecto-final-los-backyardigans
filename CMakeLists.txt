cmake_minimum_required(VERSION 3.16)

project(prog3_nn_final_project_v2025_01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${PROJECT_SOURCE_DIR}/src)


set(SOURCES
        src/main.cpp
        src/tensor/tensor.h
        src/Procesamiento/data_loader.h
)


add_executable(${PROJECT_NAME} ${SOURCES})



