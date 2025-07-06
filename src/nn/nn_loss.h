#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#pragma once
#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec {
namespace neural_network {

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

template<typename T>
class MSELoss final : public ILoss<T, 2> {
    Tensor<T,2> y_pred_, y_true_;
public:
    MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true)
        : y_pred_(y_prediction), y_true_(y_true) {}

    T loss() const override {
        T sum = 0;
        for (size_t i = 0; i < y_pred_.size(); ++i)
            sum += (y_pred_[i] - y_true_[i]) * (y_pred_[i] - y_true_[i]);
        return sum / y_pred_.size();
    }

    Tensor<T,2> loss_gradient() const override {
        Tensor<T,2> grad = y_pred_;
        for (size_t i = 0; i < grad.size(); ++i)
            grad[i] = 2.0 * (y_pred_[i] - y_true_[i]) / y_pred_.size();
        return grad;
    }
};

template<typename T>
class BCELoss final : public ILoss<T, 2> {
    Tensor<T,2> y_pred_, y_true_;
    static constexpr T eps = 1e-7;
public:
    BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true)
        : y_pred_(y_prediction), y_true_(y_true) {}

    T loss() const override {
        T sum = 0;
        for (size_t i = 0; i < y_pred_.size(); ++i) {
            T p = std::clamp(y_pred_[i], eps, T(1)-eps);
            sum += -y_true_[i]*std::log(p) - (T(1)-y_true_[i])*std::log(T(1)-p);
        }
        return sum / y_pred_.size();
    }

    Tensor<T,2> loss_gradient() const override {
        Tensor<T,2> grad = y_pred_;
        for (size_t i = 0; i < grad.size(); ++i) {
            T p = std::clamp(y_pred_[i], eps, T(1)-eps);
            grad[i] = (p - y_true_[i]) / (p * (T(1) - p)) / y_pred_.size();
        }
        return grad;
    }
};

using utec::neural_network::MSELoss;
using utec::neural_network::BCELoss;

}} // namespace neural_network

namespace utec {
    using neural_network::MSELoss;
    using neural_network::BCELoss;
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H