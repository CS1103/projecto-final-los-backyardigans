#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#pragma once
#include "../nn_interfaces.h"

namespace utec {
    namespace neural_network {

        template<typename T, size_t DIMS>
        using Tensor = utec::algebra::Tensor<T, DIMS>;

        template<typename T>
        class Dense final : public ILayer<T> {
            Tensor<T,2> W_, b_;
            Tensor<T,2> last_input_;
            Tensor<T,2> grad_W_, grad_b_;
        public:
            template<typename InitWFun, typename InitBFun>
            Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
            : W_(in_f, out_f), b_(1, out_f) {
                init_w_fun(W_);
                init_b_fun(b_);
            }

            Tensor<T,2> forward(const Tensor<T,2>& x) override {
                last_input_ = x;
                Tensor<T,2> out = matrix_product(x, W_);
                for (size_t i = 0; i < out.shape()[0]; ++i)
                    for (size_t j = 0; j < out.shape()[1]; ++j)
                        out(i, j) += b_(0, j);
                return out;
            }

            Tensor<T,2> backward(const Tensor<T,2>& dZ) override {
                grad_W_ = matrix_product(transpose_2d(last_input_), dZ);
                grad_b_ = sum_rows(dZ);
                return matrix_product(dZ, transpose_2d(W_));
            }

            void update_params(IOptimizer<T>& optimizer) override {
                optimizer.update(W_, grad_W_);
                optimizer.update(b_, grad_b_);
            }

        private:
            Tensor<T,2> sum_rows(const Tensor<T,2>& t) {
                size_t fils = t.shape()[0], cols = t.shape()[1];
                Tensor<T,2> result(1, cols);
                for (size_t j = 0; j < cols; ++j) {
                    T sum = 0;
                    for (size_t i = 0; i < fils; ++i)
                        sum += t(i, j);
                    result(0, j) = sum;
                }
                return result;
            }
        };


    } // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
