import subprocess
import os


def run_mpi_exec(process_counts, executable):
    for n in process_counts:
        command = f"mpiexec -n {n} {executable}"
        print(f"Запуск с {n} процессами: {command}")
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Выполнение с {n} процессами завершено успешно")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при выполнении с {n} процессами: {e}")


if __name__ == "__main__":
    process_counts = [1, 2, 4, 6, 8, 10]
    run_mpi_exec(process_counts, "cmake-build-debug/MPI_Project.exe")
    print("Все запуски завершены")