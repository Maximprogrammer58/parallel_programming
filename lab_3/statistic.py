import numpy as np
import matplotlib.pyplot as plt
import glob


def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        rows, cols = map(int, lines[0].split())
        matrix = np.array([list(map(int, line.split())) for line in lines[1:]])
    return matrix


def check_matrix_multiplication(size):
    folder_name = str(size)
    matrix_1 = read_matrix_from_file(f"{folder_name}/matrix_1.txt")
    matrix_2 = read_matrix_from_file(f"{folder_name}/matrix_2.txt")
    result_numpy = np.dot(matrix_1, matrix_2)
    result_file = read_matrix_from_file(f"{folder_name}/result.txt")
    return np.array_equal(result_numpy, result_file)


def write_results_to_file(results, filename):
    with open(filename, 'w') as file:
        for size, is_correct in results:
            file.write(f"SIZE: {size}x{size}: {'Correct' if is_correct else 'Error'}\n")


def plot_results(folder_name):
    plt.figure(figsize=(10, 6))

    result_files = glob.glob(f"{folder_name}/results_*.txt")

    data = {}

    for result_file in result_files:
        sizes, times = [], []
        threads = None

        with open(result_file, 'r') as file:
            for line in file:
                parts = line.split(", ")
                size_part = parts[0].split(": ")[1]
                time_part = parts[2].split(": ")[1].replace(" seconds", "")

                if threads is None:
                    threads = int(parts[1].split(": ")[1])

                sizes.append(int(size_part.split("x")[0]))
                times.append(float(time_part))

        if threads not in data:
            data[threads] = (sizes, times)

    sorted_threads = sorted(data.keys())

    for threads in sorted_threads:
        sizes, times = data[threads]
        plt.plot(sizes, times, marker='o', label=f'Threads: {threads}')

    plt.title("Зависимость времени от размера матрицы")
    plt.xlabel("Размер матрицы (n x n)")
    plt.ylabel("Время (секунды)")
    plt.grid()
    plt.legend()
    plt.savefig("matrix_multiplication_time.png")
    plt.show()


if __name__ == "__main__":
    matrix_sizes = [100, 200, 300, 400, 500, 1000, 1500, 2000]
    results = []

    for size in matrix_sizes:
        is_correct = check_matrix_multiplication(size)
        results.append((size, is_correct))

    write_results_to_file(results, "comparison_results.txt")

    plot_results("results")