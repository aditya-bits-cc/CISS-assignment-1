import numpy as np
import time
from functools import wraps
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# decorator to find the execution time
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration
    return wrapper

# Multiply rows in parallel execution
def multiply_row(args):
    row, B = args
    n = B.shape[1]
    result_row = np.zeros(n)
    for j in range(n):
        for k in range(B.shape[0]):
            result_row[j] += row[k] * B[k, j]
    return result_row

# Sequential Matrix Multiplication
@timing_decorator
def sequential_matrix_multiply(A, B):
    n = A.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i, j] += A[i, k] * B[k, j]
    return result

# Parallel Matrix Multiplication
@timing_decorator
def parallel_matrix_multiply(A, B, num_processes):
    with Pool(processes=num_processes) as pool:
        result_rows = pool.map(multiply_row, [(A[i, :], B) for i in range(A.shape[0])])
    return np.array(result_rows)

if __name__ == "__main__":
    n = 300  # matrix size
    print(f"\n=== Matrix Multiplication ({n}x{n}): Sequential vs Parallel ===")

    # Generate random matrices
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    print("\n--- Sequential Execution ---")
    _, t_seq = sequential_matrix_multiply(A, B)
    print(f"Sequential execution time: {t_seq:.4f} s")

    process_counts = [i for i in range(2, cpu_count())]
    times = []
    speedups = []
    efficiencies = []

    for p in process_counts:
        print(f"\n--- Parallel Execution with {p} process(es) ---")
        _, t_par = parallel_matrix_multiply(A, B, p)
        times.append(t_par)
        sp = t_seq / t_par
        eff = sp / p
        speedups.append(sp)
        efficiencies.append(eff)
        print(f"Execution time: {t_par:.4f} s, Speedup: {sp:.2f}, Efficiency: {eff:.2f}")

    print("\n=== Execution summary ===")
    print(f"{'Processes':<12}{'Time (s)':<15}{'Speedup':<12}{'Efficiency':<12}")
    print(f"{'Seq (1 core)':<12}{t_seq:<15.4f}{1.0:<12.2f}{1.0:<12.2f}")
    for p, t, s, e in zip(process_counts, times, speedups, efficiencies):
        print(f"{p:<12}{t:<15.4f}{s:<12.2f}{e:<12.2f}")


    plt.figure(figsize=(12,5))

    # Execution Time Plot
    plt.subplot(1,2,1)
    plt.plot([1]+process_counts, [t_seq]+times, marker='o', color='blue', label='Execution Time')
    plt.title("Execution Time: Sequential vs Parallel")
    plt.xlabel("Number of Processes")
    plt.ylabel("Time (s)")
    plt.xticks([1]+process_counts)
    plt.grid(True)
    plt.legend()

    # Speedup & Efficiency Plot
    plt.subplot(1,2,2)
    plt.plot(process_counts, speedups, marker='o', color='green', label='Speedup')
    plt.plot(process_counts, efficiencies, marker='s', color='red', label='Efficiency')
    plt.title("Speedup & Efficiency vs Processes")
    plt.xlabel("Number of Processes")
    plt.ylabel("Value")
    plt.xticks(process_counts)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
