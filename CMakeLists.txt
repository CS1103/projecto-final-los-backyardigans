cmake_minimum_required(VERSION 3.16)

project(prog3_nn_final_project_v2025_01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluir headers
include_directories(${PROJECT_SOURCE_DIR}/src)

# Ejecutable principal
add_executable(${PROJECT_NAME}
        src/main.cpp
        src/tensor/tensor.h
        src/Procesamiento/data_loader.h
)

# Ejecutable para test_tensor
add_executable(test_tensor
        test/test_tensor.cpp
        src/tensor/tensor.h
)
add_executable(test_relu
        test/test_relu.cpp
        src/tensor/tensor.h
        src/nn/nn_activation.h
)


# Puedes añadir más tests aquí si deseas
# add_executable(test_relu tests/test_relu.cpp src/nn/nn_activation.h src/tensor/tensor.h)
