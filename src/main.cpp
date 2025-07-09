#include <iostream>
#include "tensor/tensor.h"
#include "Procesamiento/data_loader.h"

using utec::algebra::Tensor;

int main() {
    Tensor<float, 2> X = data::load_inputs("../data/digits01_inputs.csv");
    Tensor<float, 2> Y = data::load_labels("../data/digits01_labels.csv");

    std::cout << "X shape: " << X.shape()[0] << "x" << X.shape()[1] << "\n";
    std::cout << "Y shape: " << Y.shape()[0] << "x" << Y.shape()[1] << "\n";


}
