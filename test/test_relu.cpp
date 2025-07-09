#include <iostream>
#include "../src/nn/nn_activation.h"
#include "../src/tensor/tensor.h"

using utec::algebra::Tensor;
using utec::neural_network::ReLU;

void test_relu_forward() {
    Tensor<float, 2> input(2, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;
    input(1, 0) = -2.5f;
    input(1, 1) = 2.5f;
    input(1, 2) = -0.1f;

    ReLU<float> relu;
    Tensor<float, 2> output = relu.forward(input);

    bool passed = true;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            float val = output(i, j);
            if (val < 0.0f) {
                passed = false;
            }
        }
    }

    if (passed)
        std::cout << "[OK] test_relu_forward\n";
    else
        std::cerr << "[FALLO] test_relu_forward\n";
}

int main() {
    test_relu_forward();
    return 0;
}

