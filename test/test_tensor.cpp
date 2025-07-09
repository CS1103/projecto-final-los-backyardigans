#include <iostream>
#include "../src/tensor/tensor.h"

using utec::algebra::Tensor;

void test_tensor_creation() {
    Tensor<int, 2> t(3, 4);
    if (t.shape()[0] != 3 || t.shape()[1] != 4) {
        std::cerr << "[FALLO] test_tensor_creation\n";
        return;
    }
    std::cout << "[OK] test_tensor_creation\n";
}

void test_tensor_fill() {
    Tensor<float, 2> t(2, 2);
    t.fill(3.14f);
    for (auto v : t) {
        if (v != 3.14f) {
            std::cerr << "[FALLO] test_tensor_fill\n";
            return;
        }
    }
    std::cout << "[OK] test_tensor_fill\n";
}

int main() {
    test_tensor_creation();
    test_tensor_fill();
    return 0;
}
