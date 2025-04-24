#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <string>
#include <mpi.h>

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

std::chrono::duration<double> parallel_matrix_multiply(
    const std::vector<std::vector<int>>& matrix_1,
    const std::vector<std::vector<int>>& matrix_2,
    std::vector<std::vector<int>>& matrix_result,
    int rank, int size) {

    size_t n = matrix_1.size();
    matrix_result.resize(n, std::vector<int>(n, 0));

    auto start = std::chrono::high_resolution_clock::now();

    size_t rows_per_process = n / size;
    size_t start_row = rank * rows_per_process;
    size_t end_row = (rank == size - 1) ? n : start_row + rows_per_process;

    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                matrix_result[i][j] += matrix_1[i][k] * matrix_2[k][j];
            }
        }
    }

    if (rank != 0) {
        for (size_t i = start_row; i < end_row; ++i) {
            MPI_Send(matrix_result[i].data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    else {
        for (int proc = 1; proc < size; ++proc) {
            size_t proc_start = proc * rows_per_process;
            size_t proc_end = (proc == size - 1) ? n : proc_start + rows_per_process;

            for (size_t i = proc_start; i < proc_end; ++i) {
                MPI_Recv(matrix_result[i].data(), n, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<size_t> matrix_sizes = {100, 200, 300, 400, 500, 1000, 1500, 2000};

    if (rank == 0) {
        std::filesystem::create_directory("results");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t matrix_size : matrix_sizes) {
        std::vector<std::vector<int>> matrix_1, matrix_2, result;

        if (rank == 0) {
            std::string folder_name = std::to_string(matrix_size);
            std::filesystem::create_directory(folder_name);

            matrix_1 = generate_random_matrix(matrix_size, matrix_size);
            matrix_2 = generate_random_matrix(matrix_size, matrix_size);

            write_matrix_to_file(folder_name + "/matrix_1.txt", matrix_1);
            write_matrix_to_file(folder_name + "/matrix_2.txt", matrix_2);
        }

        if (rank == 0) {
            for (int proc = 1; proc < size; ++proc) {
                for (const auto& row : matrix_1) {
                    MPI_Send(row.data(), matrix_size, MPI_INT, proc, 0, MPI_COMM_WORLD);
                }
                for (const auto& row : matrix_2) {
                    MPI_Send(row.data(), matrix_size, MPI_INT, proc, 1, MPI_COMM_WORLD);
                }
            }
        }
        else {
            matrix_1.resize(matrix_size, std::vector<int>(matrix_size));
            matrix_2.resize(matrix_size, std::vector<int>(matrix_size));

            for (size_t i = 0; i < matrix_size; ++i) {
                MPI_Recv(matrix_1[i].data(), matrix_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(matrix_2[i].data(), matrix_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        auto duration = parallel_matrix_multiply(matrix_1, matrix_2, result, rank, size);

        if (rank == 0) {
            std::string folder_name = std::to_string(matrix_size);
            std::string result_filename = "results/results_" + std::to_string(size) + "_processes.txt";

            write_matrix_to_file(folder_name + "/result.txt", result);

            std::ofstream results_file(result_filename, std::ios::app);
            results_file << "SIZE: " << matrix_size << "x" << matrix_size << ", ";
            results_file << "Processes: " << size << ", ";
            results_file << "Time: " << duration.count() << " seconds" << std::endl;
            results_file.close();
        }
    }

    MPI_Finalize();
    return 0;
}