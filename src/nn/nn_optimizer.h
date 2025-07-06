#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#pragma once
#include "nn_interfaces.h"
#include <vector>
#include <cmath>

namespace utec {
    namespace neural_network {

        template<typename T, size_t DIMS>
        using Tensor = utec::algebra::Tensor<T, DIMS>;

        template<typename T>
        class SGD final : public IOptimizer<T> {
            T lrn;
        public:
            explicit SGD(T learning_rate = 0.01) : lrn(learning_rate) {}
            void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
                for (size_t i = 0; i < params.size(); ++i)
                    params[i] -= lrn * grads[i];}
        };

        template<typename T>
        class Adam final : public IOptimizer<T> {
            T lrn, beta1_, beta2_, eps_;
            std::vector<T> m_, v_;
            size_t t_ = 0;
        public:
            explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
                : lrn(learning_rate), beta1_(beta1), beta2_(beta2), eps_(epsilon) {}

            void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
                if (m_.size() != params.size()) {
                    m_.assign(params.size(), T(0));
                    v_.assign(params.size(), T(0));
                    t_ = 0;
                }

                ++t_;

                for (size_t i = 0; i < params.size(); ++i) {
                    m_[i] = beta1_ * m_[i] + (1 - beta1_) * grads[i];
                    v_[i] = beta2_ * v_[i] + (1 - beta2_) * grads[i] * grads[i];
                    T m_hat = m_[i] / (1 - std::pow(beta1_, t_));
                    T v_hat = v_[i] / (1 - std::pow(beta2_, t_));
                    params[i] -= lrn * m_hat / (std::sqrt(v_hat) + eps_);}
            }

            void step() override {}
        };

    } // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H