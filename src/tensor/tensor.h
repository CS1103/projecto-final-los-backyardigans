#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <vector>
#include <numeric>
#include <array>
#include <stdexcept>

namespace utec {
    namespace algebra {

        template <typename T, size_t Rank>
        class Tensor {

        private:
            std::array<size_t, Rank> _shape;
            std::vector<T> _data;

            template <typename... Idxs>
            size_t get_linear_index(Idxs... idxs) const {
                std::array<size_t, Rank> indices{ static_cast<size_t>(idxs)... };

                for (size_t i = 0; i < Rank; ++i) {
                    if (indices[i] >= _shape[i]) {
                        throw std::out_of_range("Index out of range");
                    }
                }

                size_t index = 0;
                size_t multiplier = 1;
                for (int i = Rank - 1; i >= 0; --i) {
                    index += indices[i] * multiplier;
                    multiplier *= _shape[i];
                }
                return index;
            }

            void multiply_with_broadcasting(Tensor& result, const Tensor& a, const Tensor& b,
                                             std::array<size_t, Rank>& indices, size_t dim = 0) const {
                if (dim == Rank) {
                    std::array<size_t, Rank> a_indices, b_indices;
                    for (size_t i = 0; i < Rank; ++i) {
                        a_indices[i] = (a._shape[i] == 1) ? 0 : indices[i];
                        b_indices[i] = (b._shape[i] == 1) ? 0 : indices[i];
                    }
                    result(indices) = a(a_indices) * b(b_indices);
                    return;
                }
                for (size_t i = 0; i < result._shape[dim]; ++i) {
                    indices[dim] = i;
                    multiply_with_broadcasting(result, a, b, indices, dim + 1);
                }
            }

            template <typename... Dims>
            static std::array<size_t, Rank> make_dims_array(Dims... dims) {
                if (sizeof...(Dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions must match tensor rank");
                }
                return std::array<size_t, Rank>{ static_cast<size_t>(dims)... };
            }

        public:

            size_t size() const { return _data.size(); }
            T& operator[](size_t idx) { return _data[idx]; }
            const T& operator[](size_t idx) const { return _data[idx]; }


            using iterator = typename std::vector<T>::iterator;
            using const_iterator = typename std::vector<T>::const_iterator;

            iterator begin() noexcept { return _data.begin(); }
            iterator end() noexcept { return _data.end(); }
            const_iterator begin() const noexcept { return _data.begin(); }
            const_iterator end() const noexcept { return _data.end(); }
            const_iterator cbegin() const noexcept { return _data.cbegin(); }
            const_iterator cend() const noexcept { return _data.cend(); }

            Tensor() : _shape{}, _data{} {}

            Tensor(const std::array<size_t, Rank>& shape) : _shape(shape) {
                size_t total_size = 1;
                for (size_t dim : shape) total_size *= dim;
                _data.resize(total_size, T{});
            }

            template <typename... Dims>
            Tensor(Dims... dims) {
                if (sizeof...(Dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with 2");
                }
                std::array<size_t, Rank> temp;
                size_t i = 0;
                ((temp[i++] = static_cast<size_t>(dims)), ...);
                _shape = temp;
                size_t total = 1;
                for (auto d : _shape) total *= d;
                _data.resize(total, T{});
            }

            Tensor& operator=(std::initializer_list<T> list) {
                if (list.size() != _data.size()) {
                    throw std::invalid_argument("Data size does not match tensor size");
                }
                std::copy(list.begin(), list.end(), _data.begin());
                return *this;
            }

            template <typename... Idxs>
            T& operator()(Idxs... idxs) {
                return _data[get_linear_index(idxs...)];
            }

            template <typename... Idxs>
            const T& operator()(Idxs... idxs) const {
                return _data[get_linear_index(idxs...)];
            }

            T& operator()(const std::array<size_t, Rank>& idxs) {
                size_t index = 0;
                size_t p = 1;
                for (int i = Rank - 1; i >= 0; --i) {
                    if (idxs[i] >= _shape[i]) {
                        throw std::out_of_range("Index out of range");
                    }
                    index += idxs[i] * p;
                    p *= _shape[i];
                }
                return _data[index];
            }

            const T& operator()(const std::array<size_t, Rank>& idxs) const {
                size_t index = 0;
                size_t p = 1;
                for (int i = Rank - 1; i >= 0; --i) {
                    if (idxs[i] >= _shape[i]) {
                        throw std::out_of_range("Index out of range");
                    }
                    index += idxs[i] * p;
                    p *= _shape[i];
                }
                return _data[index];
            }

            const std::array<size_t, Rank>& shape() const noexcept {
                return _shape;
            }

            void reshape(const std::array<size_t, Rank>& new_shape) {
                size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ul, std::multiplies<>());
                size_t total_size = _data.size();

                if (new_size != total_size) {
                    if (new_size > total_size) {
                        _data.resize(new_size, T{});
                    }
                    _shape = new_shape;
                }
                else{
                    _shape = new_shape;
                }
            }

            template <typename... Dims>
            void reshape(Dims... dims) {
                if (sizeof...(Dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with 2");}
                std::array<size_t, Rank> temp;
                size_t i = 0;
                ((temp[i++] = static_cast<size_t>(dims)), ...);
                reshape(temp);
            }

            void fill(const T& value) {
                std::fill(_data.begin(), _data.end(), value);
            }

            Tensor operator+(const Tensor& other) const {
                std::array<size_t, Rank> array_result;
                for (size_t i = 0; i < Rank; ++i) {
                    if (_shape[i] != other._shape[i] && _shape[i] != 1 && other._shape[i] != 1) {
                        throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");}
                    array_result[i] = std::max(_shape[i], other._shape[i]);
                }
                Tensor result(array_result);
                std::array<size_t, Rank> indices{};
                auto rec = [&](auto&& self, size_t dim = 0) -> void {
                    if (dim == Rank) {
                        std::array<size_t, Rank> a_idx, b_idx;
                        for (size_t i = 0; i < Rank; ++i) {
                            a_idx[i] = (_shape[i] == 1) ? 0 : indices[i];
                            b_idx[i] = (other._shape[i] == 1) ? 0 : indices[i];}
                        result(indices) = (*this)(a_idx) + other(b_idx);
                        return;
                    }
                    for (size_t i = 0; i < array_result[dim]; ++i) {
                        indices[dim] = i;
                        self(self, dim + 1);}};
                rec(rec);
                return result;
            }

            Tensor operator-(const Tensor& other) const {
                std::array<size_t, Rank> array_result;
                for (size_t i = 0; i < Rank; ++i) {
                    if (_shape[i] != other._shape[i] && _shape[i] != 1 && other._shape[i] != 1) {
                        throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");}
                    array_result[i] = std::max(_shape[i], other._shape[i]);
                }
                Tensor result(array_result);
                std::array<size_t, Rank> indices{};
                auto rec = [&](auto&& self, size_t dim = 0) -> void {
                    if (dim == Rank) {
                        std::array<size_t, Rank> a_idx, b_idx;
                        for (size_t i = 0; i < Rank; ++i) {
                            a_idx[i] = (_shape[i] == 1) ? 0 : indices[i];
                            b_idx[i] = (other._shape[i] == 1) ? 0 : indices[i];}
                        result(indices) = (*this)(a_idx) - other(b_idx);
                        return;
                    }
                    for (size_t i = 0; i < array_result[dim]; ++i) {
                        indices[dim] = i;
                        self(self, dim + 1);}};
                rec(rec);
                return result;
            }

            Tensor operator*(const Tensor& other) const {
                std::array<size_t, Rank> array_result;
                for (size_t i = 0; i < Rank; ++i) {
                    if (_shape[i] != other._shape[i] && _shape[i] != 1 && other._shape[i] != 1) {
                        throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");}
                    array_result[i] = std::max(_shape[i], other._shape[i]);
                }
                Tensor result(array_result);
                std::array<size_t, Rank> indices{};
                auto rec = [&](auto&& self, size_t dim = 0) -> void {
                    if (dim == Rank) {
                        std::array<size_t, Rank> a_idx, b_idx;
                        for (size_t i = 0; i < Rank; ++i) {
                            a_idx[i] = (_shape[i] == 1) ? 0 : indices[i];
                            b_idx[i] = (other._shape[i] == 1) ? 0 : indices[i];}
                        result(indices) = (*this)(a_idx) * other(b_idx);
                        return;
                    }
                    for (size_t i = 0; i < array_result[dim]; ++i) {
                        indices[dim] = i;
                        self(self, dim + 1);}};
                rec(rec);
                return result;
            }

            Tensor operator+(const T& scalar) const {
                Tensor result(_shape);
                for (size_t i = 0; i < _data.size(); ++i) {
                    result._data[i] = _data[i] + scalar;
                }
                return result;
            }

            Tensor operator-(const T& scalar) const {
                Tensor result(_shape);
                for (size_t i = 0; i < _data.size(); ++i) {
                    result._data[i] = _data[i] - scalar;
                }
                return result;
            }

            Tensor operator*(const T& scalar) const {
                Tensor result(_shape);
                for (size_t i = 0; i < _data.size(); ++i) {
                    result._data[i] = _data[i] * scalar;
                }
                return result;
            }

            Tensor operator/(const T& scalar) const {
                Tensor result(_shape);
                for (size_t i = 0; i < _data.size(); ++i) {
                    result._data[i] = _data[i] / scalar;
                }
                return result;
            }

            friend Tensor operator+(const T& esc, const Tensor& t) {
                return t + esc;
            }

            friend Tensor operator-(const T& esc, const Tensor& t) {
                Tensor result(t._shape);
                for (size_t i = 0; i < t._data.size(); ++i) {
                    result._data[i] = esc - t._data[i];}
                return result;
            }

            friend Tensor operator*(const T& esc, const Tensor& t) {
                return t * esc;
            }

            friend Tensor operator/(const T& esc, const Tensor& t) {
                Tensor result(t._shape);
                for (size_t i = 0; i < t._data.size(); ++i) {
                    result._data[i] = esc / t._data[i];}
                return result;
            }

            Tensor transpose_2d() const {
                if constexpr (Rank < 2) {
                    throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");}
                std::array<size_t, Rank> new_shape = _shape;
                std::swap(new_shape[Rank - 1], new_shape[Rank - 2]);
                Tensor result(new_shape);

                std::array<size_t, Rank> idx;
                auto rec = [&](auto&& same, size_t dim) -> void {
                    if (dim == Rank) {
                        std::array<size_t, Rank> new_idx = idx;
                        std::swap(new_idx[Rank - 1], new_idx[Rank - 2]);
                        result(new_idx) = (*this)(idx);
                        return;}
                    for (size_t i = 0; i < _shape[dim]; ++i) {
                        idx[dim] = i;
                        same(same, dim + 1);}};
                rec(rec, 0);
                return result;
            }

            friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
                const auto& shape = t.shape();

                if constexpr (Rank == 4) {
                    os << "{" << std::endl;
                    for (size_t i = 0; i < shape[0]; ++i) {
                        os << "{" << std::endl;
                        for (size_t j = 0; j < shape[1]; ++j) {
                            os << "{" << std::endl;
                            for (size_t k = 0; k < shape[2]; ++k) {
                                for (size_t l = 0; l < shape[3]; ++l) {
                                    if (l > 0) os << " ";
                                    os << t(i, j, k, l);}
                                os << std::endl;}
                            os << "}" << std::endl;}
                        os << "}" << std::endl;}
                    os << "}" << std::endl;
                    return os;}

                else if constexpr (Rank >= 2) {
                    size_t out = 1;
                    for (size_t i = 0; i < Rank - 2; ++i) out *= shape[i];
                    size_t fils = shape[Rank - 2];
                    size_t cols = shape[Rank - 1];
                    os << "{" << std::endl;
                    for (size_t i = 0; i < out; ++i) {
                        if (out > 1) os << "{" << std::endl;

                        for (size_t r = 0; r < fils; ++r) {
                            for (size_t c = 0; c < cols; ++c) {
                                std::array<size_t, Rank> idx;
                                size_t tmp = i;
                                for (int d = static_cast<int>(Rank) - 3; d >= 0; --d) {
                                    size_t dim = shape[d];
                                    idx[d] = tmp % dim;
                                    tmp /= dim;}

                                idx[Rank - 2] = r;
                                idx[Rank - 1] = c;
                                size_t idx_now = 0, p = 1;
                                for (int j = Rank - 1; j >= 0; --j) {
                                    idx_now += idx[j] * p;
                                    p *= shape[j];}

                                if (c > 0) os << " ";
                                if (idx_now < t._data.size())
                                    os << t(idx);
                                else
                                    os << 0;}
                            os << std::endl;}

                        if (out > 1) os << "}" << std::endl;}
                    os << "}" << std::endl;}

                else {
                    for (size_t i = 0; i < shape[0]; ++i) {
                        if (i < t._data.size())
                            os << t._data[i];
                        else
                            os << 0;
                        if (i + 1 < shape[0]) os << " ";}}
                os << std::endl;
                return os;}
        };

        template <typename T, size_t Rank>
        Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& t) {
            if constexpr (Rank < 2) {
                throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");}
            return t.transpose_2d();
        }

        template <typename T, size_t Rank>
        Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {
            if constexpr (Rank < 2) throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
            if (A.shape()[Rank - 1] != B.shape()[Rank - 2])
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
            for (size_t i = 0; i < Rank - 2; ++i) {
                if (A.shape()[i] != B.shape()[i])
                    throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");}
            std::array<size_t, Rank> shape_result = A.shape();
            shape_result[Rank - 1] = B.shape()[Rank - 1];
            Tensor<T, Rank> result(shape_result);
            std::array<size_t, Rank> idx;
            auto rec = [&](auto&& self, size_t d) -> void {
                if (d == Rank - 2) {
                    for (size_t i = 0; i < A.shape()[Rank - 2]; ++i) {
                        for (size_t j = 0; j < B.shape()[Rank - 1]; ++j) {
                            T sum = T{};
                            for (size_t k = 0; k < A.shape()[Rank - 1]; ++k) {
                                auto a_idx = idx;
                                a_idx[Rank - 2] = i;
                                a_idx[Rank - 1] = k;
                                auto b_idx = idx;
                                b_idx[Rank - 2] = k;
                                b_idx[Rank - 1] = j;
                                sum += A(a_idx) * B(b_idx);}
                            auto r_idx = idx;
                            r_idx[Rank - 2] = i;
                            r_idx[Rank - 1] = j;
                            result(r_idx) = sum;}}
                    return;
                }
                for (size_t i = 0; i < A.shape()[d]; ++i) {
                    idx[d] = i;
                    self(self, d + 1);
                }
            };
            rec(rec, 0);
            return result;
        };

        template <typename T, size_t Rank, typename F>
        Tensor<T, Rank> apply_function(const Tensor<T, Rank>& t, F&& func) {
            Tensor<T, Rank> result(t.shape());
            for (size_t i = 0; i < t.size(); ++i) {
                result[i] = func(t[i]);
            }
            return result;
        }

    }
} // namespace utec::algebra

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
