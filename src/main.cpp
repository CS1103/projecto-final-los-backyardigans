#include <iostream>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>

#include "tensor/tensor.h"
#include "Procesamiento/data_loader.h"
#include "nn/neural_network.h"
#include "nn/nn_dense.h"
#include "nn/nn_activation.h"
#include "nn/nn_loss.h"
#include "nn/nn_optimizer.h"

using utec::algebra::Tensor;

int main() {
    // Cargar datos CSV (360 muestras de dígitos 0 y 1)
    Tensor<float, 2> X = data::load_inputs("../data/digits01_inputs.csv");   // [360, 64]
    Tensor<float, 2> Y = data::load_labels("../data/digits01_labels.csv");   // [360, 1]

    std::cout << "X shape: " << X.shape()[0] << " x " << X.shape()[1] << "\n";
    std::cout << "Y shape: " << Y.shape()[0] << " x " << Y.shape()[1] << "\n";

    // Inicializador Xavier
    std::mt19937 gen(4);
    auto xavier_init = [&](auto& parameter) {
        const float limit = std::sqrt(6.0f / (parameter.shape()[0] + parameter.shape()[1]));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& v : parameter) v = dist(gen);
    };

    // Construcción de la red neuronal
    utec::neural_network::NeuralNetwork<float> net;
    net.add_layer(std::make_unique<utec::neural_network::Dense<float>>(64, 16, xavier_init, xavier_init));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<float>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<float>>(16, 1, xavier_init, xavier_init));
    net.add_layer(std::make_unique<utec::neural_network::Sigmoid<float>>());

    // Entrenamiento
    constexpr size_t epochs = 500;
    constexpr size_t batch_size = 32;
    constexpr float learning_rate = 0.1f;
    net.train<utec::neural_network::BCELoss>(X, Y, epochs, batch_size, learning_rate);

    // Predicciones
    auto preds = net.predict(X);
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < 10; ++i) {
        float p = preds(i, 0);
        std::cout << "Muestra #" << i
                  << " → Prob: " << p
                  << " → Clasificado como: " << (p >= 0.5f ? "1" : "0")
                  << " | Real: " << Y(i, 0) << "\n";
    }

    return 0;
}
