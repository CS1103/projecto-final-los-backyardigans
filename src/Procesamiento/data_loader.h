#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "tensor/tensor.h"

namespace data {

    using utec::algebra::Tensor;

    Tensor<float, 2> load_inputs(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) throw std::runtime_error("No se pudo abrir el archivo de inputs.");

        std::vector<std::vector<float>> data;
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<float> row;
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stof(value));
            }
            data.push_back(row);
        }

        size_t rows = data.size();
        size_t cols = data[0].size();
        Tensor<float, 2> tensor(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                tensor(i, j) = data[i][j];

        return tensor;
    }

    Tensor<float, 2> load_labels(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) throw std::runtime_error("No se pudo abrir el archivo de labels.");

        std::string line;
        std::vector<float> labels;
        while (std::getline(file, line)) {
            labels.push_back(std::stof(line));
        }

        Tensor<float, 2> tensor(labels.size(), 1);
        for (size_t i = 0; i < labels.size(); ++i)
            tensor(i, 0) = labels[i];

        return tensor;
    }

}

#endif // DATA_LOADER_H
