#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#pragma once
#include "../nn_interfaces.h"
#include "../nn_loss.h"
#include "../nn_optimizer.h"
#include <vector>
#include <iomanip>
#include <memory>

namespace utec {
    namespace neural_network {
        template <typename T, size_t DIMS>
        using Tensor = utec::algebra::Tensor<T, DIMS>;

        template<typename T>
        class NeuralNetwork {
            std::vector<std::unique_ptr<ILayer<T>>> lyr;

        public:
            void add_layer(std::unique_ptr<ILayer<T>> layer) {
                lyr.push_back(std::move(layer));
            }

            Tensor<T,2> forward(const Tensor<T,2>& input) {
                Tensor<T,2> x = input;
                for (auto& l : lyr) x = l->forward(x);
                return x;
            }

            Tensor<T,2> predict(const Tensor<T,2>& X) {
                return forward(X);
            }

            template <template <typename> class LossType, template <typename> class OptimizerType>
            void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, const size_t epochs, const size_t batch_size, T learning_rate) {
                OptimizerType<T> optimizer(learning_rate);
                for (size_t epoch = 0; epoch < epochs; ++epoch) {
                    Tensor<T,2> y_pred = forward(X);
                    LossType<T> loss_fn(y_pred, Y);
                    Tensor<T,2> grad = loss_fn.loss_gradient();
                    for (auto it = lyr.rbegin(); it != lyr.rend(); ++it)
                        grad = (*it)->backward(grad);
                    for (auto& l : lyr)
                        l->update_params(optimizer);
                    optimizer.step();
                }
            }

            template<template <typename> class LossType>
            void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, const size_t epochs, const size_t batch_size, T learning_rate) {
                train<LossType, SGD>(X, Y, epochs, batch_size, learning_rate);
            }
        };

        template <typename T>
        void print_predictions(NeuralNetwork<T>& nn, const std::vector<Tensor<T,2>>& inputs) {
            for (const auto& input : inputs) {
                auto pred = nn.forward(input);
                std::cout << "Input: (" << input(0,0) << "," << input(0,1) << ") -> Prediction: " << pred(0,0) << std::endl;
            }
        }

        template <typename T>
        void print_accuracy(NeuralNetwork<T>& nn, const Tensor<T,2>& X_test, const Tensor<T,2>& Y_test) {
            size_t correct = 0;
            for (size_t i = 0; i < X_test.shape()[0]; ++i) {
                Tensor<T,2> input({1, X_test.shape()[1]});
                for (size_t j = 0; j < X_test.shape()[1]; ++j)
                    input(0, j) = X_test(i, j);
                auto pred = nn.forward(input);
                int pred_label = pred(0,0) >= 0.5 ? 1 : 0;
                int true_label = pred(0,0) >= 0.5 ? 1 : 0;
                if (pred_label == true_label) ++correct;
            }
            double accuracy = double(correct) / X_test.shape()[0];
            std: cout << std::fixed << std::setprecision(6) << accuracy << std::endl;

        }
    }
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H