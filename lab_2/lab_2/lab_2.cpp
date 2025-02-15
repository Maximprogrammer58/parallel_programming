﻿#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <filesystem>
#include <string>
#include <chrono>

#include <omp.h>


void write_matrix_to_file(const std::string& filename, const std::vector<std::vector<int>>& matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error when opening the file for recording!" << std::endl;
        return;
    }
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    file << rows << " " << cols << std::endl;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << matrix[i][j] << " ";
        }
        file << std::endl;
    }
    file.close();
}


std::vector<std::vector<int>> read_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error when opening the file for reading!" << std::endl;
        exit(1);
    }
    size_t rows, cols;
    file >> rows >> cols;
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }
    file.close();
    return matrix;
}


std::vector<std::vector<int>> generate_random_matrix(size_t rows, size_t cols, int min = 0, int max = 99) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);

    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}


std::chrono::duration<double> mul_matrix(const std::vector<std::vector<int>>& matrix_1, const std::vector<std::vector<int>>& matrix_2, std::vector<std::vector<int>>& matrix_result) {
    size_t rows_1 = matrix_1.size();
    size_t cols_1 = matrix_1[0].size();
    size_t rows_2 = matrix_2.size();
    size_t cols_2 = matrix_2[0].size();

    if (cols_1 != rows_2) {
        std::cerr << "The number of columns of the first matrix is not equal to the number of rows of the second matrix!" << std::endl;
        return std::chrono::duration<double>::zero();
    }

    matrix_result.resize(rows_1, std::vector<int>(cols_2, 0));

    auto start = std::chrono::high_resolution_clock::now();
    int i, j, k;

#pragma omp parallel for shared(matrix_1, matrix_2, matrix_result) private(i, j, k)
    for (i = 0; i < rows_1; ++i) {
        for (j = 0; j < cols_2; ++j) {
            for (k = 0; k < cols_1; ++k) {
                matrix_result[i][j] += matrix_1[i][k] * matrix_2[k][j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}


int main() {
    int num_threads = 12;

    std::vector<size_t> matrix_sizes = { 100, 200, 300, 400, 500, 1000, 1500, 2000 };

    std::filesystem::create_directory("results");

    std::string results_filename = "results/results_" + std::to_string(num_threads) + "_process.txt";
    std::ofstream results_file(results_filename);

    for (size_t size : matrix_sizes) {
        std::string folder_name = std::to_string(size);

        std::vector<std::vector<int>> matrix_1 = read_matrix_from_file(folder_name + "/matrix_1.txt");
        std::vector<std::vector<int>> matrix_2 = read_matrix_from_file(folder_name + "/matrix_2.txt");
        std::vector<std::vector<int>> result;

        omp_set_num_threads(num_threads);

        auto duration = mul_matrix(matrix_1, matrix_2, result);

        results_file << "SIZE: " << size << "x" << size << ", ";
        results_file << "Threads: " << omp_get_max_threads() << ", ";
        results_file << "Time: " << duration.count() << " seconds" << std::endl;

        write_matrix_to_file(folder_name + "/result.txt", result);
    }

    results_file.close();

    return 0;
}