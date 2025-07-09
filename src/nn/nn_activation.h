#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#pragma once
#include "nn/nn_interfaces.h"
#include <algorithm>
#include <cmath>

namespace utec {
    namespace neural_network {

        template<typename T, size_t DIMS>
        using Tensor = utec::algebra::Tensor<T, DIMS>;

        template<typename T>
        class ReLU final : public ILayer<T> {
            Tensor<T,2> ult_input;
        public:
            Tensor<T,2> forward(const Tensor<T,2>& z) override {
                ult_input = z;
                Tensor<T,2> out = z;
                for (auto& v : out) v = std::max(T(0), v);
                return out;
            }

            Tensor<T,2> backward(const Tensor<T,2>& g) override {
                Tensor<T,2> grad = g;
                for (size_t i = 0; i < grad.size(); ++i)
                    grad[i] = (ult_input[i] > T(0)) ? g[i] : T(0);
                return grad;
            }
        };

        template<typename T>
        class Sigmoid final : public ILayer<T> {
            Tensor<T,2> ult_input;
        public:
            Tensor<T,2> forward(const Tensor<T,2>& z) override {
                ult_input = z;
                for (auto& v : ult_input)
                    v = T(1) / (T(1) + std::exp(-v));
                return ult_input;
            }

            Tensor<T,2> backward(const Tensor<T,2>& g) override {
                Tensor<T,2> grad = g;
                for (size_t i = 0; i < grad.size(); ++i)
                    grad[i] = g[i] * ult_input[i] * (T(1) - ult_input[i]);
                return grad;
            }
        };

    } // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H