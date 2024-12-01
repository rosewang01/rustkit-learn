#!python
import sklbench as bench
import numpy as np
from tqdm import tqdm
import rustkit

def benchmark_pca(X, n_components=2, n_iterations=100):
    def fit_pca(X):
        pca = rustkit.PCA()
        return pca.fit_transform(X, n_components)
    
    params = bench.parse_args()  # Leverage Intel benchmark params.
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        fit_time, _ = bench.measure_function_time(fit_pca, X)
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"PCA Average Time: {average_time:.4f}s")
    return average_time


def benchmark_standard_scaler(X, n_iterations=100):
    def fit_standard_scaler(X):
        scaler = rustkit.StandardScaler()
        return scaler.fit_transform(X)
    
    params = bench.parse_args()  # Leverage Intel benchmark params.
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        fit_time, _ = bench.measure_function_time(fit_standard_scaler, X)
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"Standard Scaler Average Time: {average_time:.4f}s")
    return average_time


def benchmark_ridge(X, y, alpha=1.0, n_iterations=100):
    def fit_ridge(X, y):
        ridge = rustkit.RidgeRegression(alpha, True)
        ridge.fit(X, y)
        return ridge
    
    params = bench.parse_args()
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        fit_time, _ = bench.measure_function_time(fit_ridge, X, y)
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"Ridge Regression Average Time: {average_time:.4f}s")
    return average_time


def benchmark_r2(y_true, y_pred, n_iterations=100):
    def compute_r2(y_true, y_pred):
        return rustkit.R2Score.compute(y_true, y_pred)
    
    params = bench.parse_args()
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        fit_time, _ = bench.measure_function_time(compute_r2, y_true, y_pred)
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"R² Score Average Time: {average_time:.4f}s")
    return average_time


def benchmark_kmeans(X, n_clusters=3, n_iterations=100):
    def fit_kmeans(X):
        kmeans = rustkit.KMeans(n_clusters, "random", 200, 10)
        kmeans.fit(X)
        return kmeans
    
    params = bench.parse_args()
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        fit_time, _ = bench.measure_function_time(fit_kmeans, X)
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"KMeans Average Time: {average_time:.4f}s")
    return average_time


def main():
    X = np.random.rand(1000, 100)
    y = np.random.rand(1000)
    y_true = np.random.rand(1000)
    y_pred = np.random.rand(1000)
    
    benchmark_pca(X)
    benchmark_standard_scaler(X)
    benchmark_ridge(X, y)
    benchmark_r2(y_true, y_pred)
    benchmark_kmeans(X)