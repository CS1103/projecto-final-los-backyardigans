#include <iostream>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>  // Para std::iota

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

    // Definir tamaños
    const size_t total_samples = X.shape()[0];
    const size_t train_samples = static_cast<size_t>(0.8f * total_samples);
    const size_t test_samples = total_samples - train_samples;

    // Mezclar los índices de forma aleatoria (con semilla para reproducibilidad)
    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);  // [0, 1, ..., total_samples - 1]
    std::mt19937 rng(42);  // Semilla fija
    std::shuffle(indices.begin(), indices.end(), rng);

    // Crear tensores para train y test


    Tensor<float, 2> X_train(std::array<size_t, 2>{train_samples, 64});
    Tensor<float, 2> Y_train(std::array<size_t, 2>{train_samples, 1});
    Tensor<float, 2> X_test(std::array<size_t, 2>{test_samples, 64});
    Tensor<float, 2> Y_test(std::array<size_t, 2>{test_samples, 1});


    // Llenar X_train, Y_train
    for (size_t i = 0; i < train_samples; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < 64; ++j)
            X_train(i, j) = X(idx, j);
        Y_train(i, 0) = Y(idx, 0);
    }

    // Llenar X_test, Y_test
    for (size_t i = 0; i < test_samples; ++i) {
        size_t idx = indices[train_samples + i];
        for (size_t j = 0; j < 64; ++j)
            X_test(i, j) = X(idx, j);
        Y_test(i, 0) = Y(idx, 0);
    }

    // Inicializador Xavier


    // Agregar ruido gaussiano a todos los datos (ya mezclados)
    std::mt19937 noise_gen(123);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (size_t i = 0; i < X.shape()[0]; ++i) {
        for (size_t j = 0; j < X.shape()[1]; ++j) {
            X(i, j) += noise(noise_gen);
            if (X(i, j) < 0.0f) X(i, j) = 0.0f;
            if (X(i, j) > 1.0f) X(i, j) = 1.0f;
        }
    }

    // Inicializador Xavier
    std::mt19937 gen(4);
    auto xavier_init = [&](auto& param) {
        float limit = std::sqrt(6.0f / (param.shape()[0] + param.shape()[1]));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& v : param) v = dist(gen);
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
    net.train<utec::neural_network::BCELoss>(X_train, Y_train, epochs, batch_size, learning_rate);

    // Predicciones
    auto preds = net.predict(X_test);
    std::cout << std::fixed << std::setprecision(4);
    // Mostrar 10 muestras aleatorias del conjunto de prueba
    std::cout << "\n--- 10 muestras aleatorias del conjunto de prueba ---\n";
    std::mt19937 rng_muestra(2025);  // Semilla fija para reproducibilidad
    std::uniform_int_distribution<size_t> dist(0, test_samples - 1);

    for (size_t i = 0; i < 10; ++i) {
        size_t idx = dist(rng_muestra);  // Índice aleatorio dentro del conjunto de prueba
        float p = preds(idx, 0);
        std::cout << "Muestra #" << idx
                  << " → Prob: " << p
                  << " → Clasificado como: " << (p >= 0.5f ? "1" : "0")
                  << " | Real: " << Y_test(idx, 0) << "\n";
    }



    // Accuracy
    size_t correct = 0;
    for (size_t i = 0; i < preds.shape()[0]; ++i) {
        int predicted = (preds(i, 0) >= 0.5f) ? 1 : 0;
        int real = static_cast<int>(Y_test(i, 0));
        if (predicted == real) ++correct;
    }
    float accuracy = static_cast<float>(correct) / preds.shape()[0];
    std::cout << "\nTest Accuracy: " << accuracy * 100 << "%\n";

    return 0;
}
