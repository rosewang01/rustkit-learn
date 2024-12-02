import numpy as np
import time
from tqdm import tqdm
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.linear_model import Ridge as SklearnRidgeRegression
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs

def benchmark_pca(X, n_components=2, n_iterations=10):
    def fit_pca(X):
        pca = SklearnPCA(n_components)
        return pca.fit_transform(X)
    
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        start_time = time.time()  # Start timing
        fit_pca(X)  # Call the PCA fitting function
        fit_time = time.time() - start_time  # Calculate the time taken
        total_time += fit_time  # Add to the total time
    
    average_time = total_time / n_iterations
    print(f"PCA Average Time: {average_time:.4f}s")
    return average_time


def benchmark_standard_scaler(X, n_iterations=10):
    def fit_standard_scaler(X):
        scaler = SklearnStandardScaler()
        return scaler.fit_transform(X)
    
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        start_time = time.time()
        fit_standard_scaler(X)
        fit_time = time.time() - start_time
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"Standard Scaler Average Time: {average_time:.4f}s")
    return average_time


def benchmark_ridge(X, y, alpha=1.0, n_iterations=10):
    def fit_ridge(X, y):
        ridge = SklearnRidgeRegression(alpha=1.0, fit_intercept=True)
        ridge.fit(X, y)
        return ridge
    
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        start_time = time.time()
        fit_ridge(X, y)
        fit_time = time.time() - start_time
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"Ridge Regression Average Time: {average_time:.4f}s")
    return average_time


def benchmark_r2(y_true, y_pred, n_iterations=10):
    def compute_r2(y_true, y_pred):
        return r2_score(y_true, y_pred)
    
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        start_time = time.time()
        compute_r2(y_true, y_pred)
        fit_time = time.time() - start_time
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"R² Score Average Time: {average_time:.4f}s")
    return average_time

def benchmark_mse(y_true, y_pred, n_iterations=10):
    def compute_mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        start_time = time.time()
        compute_mse(y_true, y_pred)
        fit_time = time.time() - start_time
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"MSE Average Time: {average_time:.4f}s")
    return average_time


def benchmark_kmeans_random(X, n_clusters=10, n_iterations=10):
    def fit_kmeans(X):
        kmeans = SklearnKMeans(n_clusters=n_clusters, init="random")
        kmeans.fit(X)
        return kmeans
    
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        start_time = time.time()
        fit_kmeans(X)
        fit_time = time.time() - start_time
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"KMeans - Random Init Average Time: {average_time:.4f}s")
    return average_time

def benchmark_kmeans(X, n_clusters=10, n_iterations=10):
    def fit_kmeans(X):
        kmeans = SklearnKMeans(n_clusters)
        kmeans.fit(X)
        return kmeans
    
    total_time = 0
    
    for _ in tqdm(range(n_iterations)):
        start_time = time.time()
        fit_kmeans(X)
        fit_time = time.time() - start_time
        total_time += fit_time
    
    average_time = total_time / n_iterations
    print(f"KMeans Average Time: {average_time:.4f}s")
    return average_time

def run_benchmark(nrows, ncols, filename):
    X = np.random.rand(nrows, ncols)
    X_clustered = make_blobs(n_samples=nrows, n_features=10, centers=3, random_state=42)[0]
    y = np.random.rand(nrows)
    y_true = np.random.rand(nrows)
    y_pred = np.random.rand(nrows)
    
    pca_time = benchmark_pca(X, n_iterations=50)
    standard_scaler_time = benchmark_standard_scaler(X, n_iterations=50)
    ridge_time = benchmark_ridge(X, y, n_iterations=50)
    r2_time = benchmark_r2(y_true, y_pred, n_iterations=50)
    mse_time = benchmark_mse(y_true, y_pred, n_iterations=50)
    kmeans_time = benchmark_kmeans(X_clustered, n_clusters=3, n_iterations=50)
    kmeans_random_time = benchmark_kmeans_random(X_clustered, n_clusters=3, n_iterations=50)
    
    with open(filename, "a") as f:
        f.write(f"PCA::fit_transform,{nrows},{ncols},{pca_time}\n")
        f.write(f"StandardScaler::fit_transform,{nrows},{ncols},{standard_scaler_time}\n")
        f.write(f"RidgeRegression::fit,{nrows},{ncols},{ridge_time}\n")
        f.write(f"R2Score::compute,{1},{ncols},{r2_time}\n")
        f.write(f"MSE::compute,{1},{ncols},{mse_time}\n")
        f.write(f"KMeans::fit,{nrows},{10},{kmeans_time}\n")
        f.write(f"KMeans(Random)::fit,{nrows},{10},{kmeans_random_time}\n")    


def main():
    nrows = [10, 50, 100, 250, 500, 750, 1000]
    ncols = [10, 50, 100, 250, 500, 750, 1000]
    filename = "sklearn_benchmarking.csv"
    for i in range(len(nrows)):
        run_benchmark(nrows[i], ncols[i], filename)
    

if __name__ == "__main__":
    main()
