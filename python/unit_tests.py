import numpy as np
import rustkit
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.linear_model import Ridge as SklearnRidgeRegression
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs

def get_pca_equality(sklearn_result, rustkit_result):
    # check col wise if values are equal or -1 * the other col
    for i in range(sklearn_result.shape[1]):
        if (np.allclose(sklearn_result[:, i], rustkit_result[:, i], atol=1e-5)):
            continue
        elif (np.allclose(sklearn_result[:, i], -rustkit_result[:, i], atol=1e-5)):
            continue
        else:
            return False
    return True

def test_pca_correctness(X):
    n_components = np.min(X.shape)
    sklearn_pca = SklearnPCA(n_components)
    rustkit_pca = rustkit.PCA()
    
    sklearn_result = sklearn_pca.fit_transform(X)
    rustkit_result = rustkit_pca.fit_transform(X, n_components)

    if (get_pca_equality(sklearn_result, rustkit_result)):
        print("PCA correctness test passed!")
    else:
        print("PCA correctness test failed!")
        print("Sklearn PCA result:")
        print(sklearn_result)
        print("Rustkit PCA result:")
        print(rustkit_result)
    
    # assert np.allclose(sklearn_result, rustkit_result, atol=1e-5), "PCA results differ!"
    # print("PCA correctness test passed!")

def get_kmeans_equality(sklearn_result, rustkit_result, n_clusters):
    sklearn_dict = {
        i: np.where(sklearn_result == i)[0] for i in range(n_clusters)
    }
    rustkit_dict = {
        i: np.where(rustkit_result == i)[0] for i in range(n_clusters)
    }

    for i in sklearn_dict.keys():
        for j in rustkit_dict.keys():
            if (len(sklearn_dict[i]) != len(rustkit_dict[j])):
                continue
            elif (np.allclose(sklearn_dict[i], rustkit_dict[j])):
                rustkit_dict.pop(j)
                break
            else:
                continue
    if (len(rustkit_dict) > 0):
        return False
    return True

def test_kmeans_correctness(X):
    # FLAG: seems like there's an error where if the number of cols in the input matrix is neq to the number of clusters, the test will fail
    n_clusters = X.shape[1]
    sklearn_kmeans_random = SklearnKMeans(n_clusters=n_clusters, init="random")
    sklearn_kmeans = SklearnKMeans(n_clusters)
    rustkit_kmeans_random = rustkit.KMeans(n_clusters, "random", 200, 10)
    rustkit_kmeans = rustkit.KMeans(n_clusters, "kmeans++", 200, 10)
    
    sklearn_result_random = sklearn_kmeans_random.fit_predict(X)
    sklearn_result = sklearn_kmeans.fit_predict(X)
    rustkit_result_random = rustkit_kmeans_random.fit_predict(X)
    rustkit_result = rustkit_kmeans.fit_predict(X)
    
    if (get_kmeans_equality(sklearn_result, rustkit_result, n_clusters)):
        print("KMeans - KMeans++ correctness test passed!")
    else:
        print("KMeans - KMeans++ correctness test failed!")
        print("Sklearn KMeans result:")
        print(sklearn_result)
        print("Rustkit KMeans result:")
        print(rustkit_result)
    
    if (get_kmeans_equality(sklearn_result_random, rustkit_result_random, n_clusters)):
        print("KMeans - Random correctness test passed!")
    else:
        print("KMeans - Random correctness test failed!")
        print("Sklearn KMeans result:")
        print(sklearn_result)
        print("Rustkit KMeans result:")
        print(rustkit_result_random)
    # assert np.allclose(sklearn_result, rustkit_result), "KMeans results differ!"
    # print("KMeans correctness test passed!")

def test_kmeans_inertia(X):
    # FLAG: change this back when the bug is fixed
    n_clusters = X.shape[1]
    kmeans = rustkit.KMeans(n_clusters, "random", 200, 10)
    kmeans.fit(X)
    
    labels = kmeans.predict(X)
    inertia = kmeans.compute_inertia(X, labels)
    
    # Compute inertia manually
    computed_inertia = sum(
        np.linalg.norm(X[i] - kmeans.centroids[labels[i]])**2 for i in range(len(X))
    )

    if (abs(inertia - computed_inertia) < 1e-5):
        print("KMeans inertia test passed!")
    else:
        print("KMeans inertia test failed!")
        print("Inertia:", inertia)
        print("Computed Inertia:", computed_inertia)
    
    # assert abs(inertia - computed_inertia) < 1e-5, "Inertia computation mismatch!"
    # print("KMeans inertia test passed!")

def test_standard_scaler_correctness(X):
    sklearn_scaler = SklearnStandardScaler()
    rustkit_scaler = rustkit.StandardScaler()
    
    sklearn_result = sklearn_scaler.fit_transform(X)
    rustkit_result = rustkit_scaler.fit_transform(X)

    if (np.allclose(sklearn_result, rustkit_result, atol=1e-5)):
        print("Standard Scaler correctness test passed!")
    else:
        print("Standard Scaler correctness test failed!")
        print("Sklearn Standard Scaler result:")
        print(sklearn_result)
        print("Rustkit Standard Scaler result:")
        print(rustkit_result)
    
    # assert np.allclose(sklearn_result, rustkit_result, atol=1e-5), "Standard Scaler results differ!"
    # print("Standard Scaler correctness test passed!")

def test_ridge_correctness(X, y):
    sklearn_ridge = SklearnRidgeRegression(alpha=1.0, fit_intercept=True)
    rustkit_ridge = rustkit.RidgeRegression(1.0, True)
    
    sklearn_ridge.fit(X, y)
    rustkit_ridge.fit(X, y)

    if (np.allclose(sklearn_ridge.coef_, rustkit_ridge.weights, atol=1e-5)):
        print("Ridge Regression correctness test passed!")
    else:
        print("Ridge Regression correctness test failed!")
        print("Sklearn Ridge Regression result:")
        print(sklearn_ridge.coef_)
        print("Rustkit Ridge Regression result:")
        print(rustkit_ridge.weights)
    
    # assert np.allclose(sklearn_ridge.coef_, rustkit_ridge.weights, atol=1e-5), "Ridge Regression results differ!"
    # print("Ridge Regression correctness test passed!")

def test_r2_correctness(y_true, y_pred):    
    sklearn_r2 = r2_score(y_true, y_pred)
    rustkit_r2 = rustkit.R2Score.compute(y_true, y_pred)

    if (abs(sklearn_r2 - rustkit_r2) < 1e-5):
        print("R² correctness test passed!")
    else:
        print("R² correctness test failed!")
        print("Sklearn R² Score:", sklearn_r2)
        print("Rustkit R² Score:", rustkit_r2)
    
    # assert abs(sklearn_r2 - rustkit_r2) < 1e-5, "R² results differ!"
    # print("R² correctness test passed!")

def test_mse_correctness(y_true, y_pred):
    sklearn_mse = np.mean((y_true - y_pred)**2)
    rustkit_mse = rustkit.MSE.compute(y_true, y_pred)

    if (abs(sklearn_mse - rustkit_mse) < 1e-5):
        print("MSE correctness test passed!")
    else:
        print("MSE correctness test failed!")
        print("Sklearn MSE:", sklearn_mse)
        print("Rustkit MSE:", rustkit_mse)
    
    # assert abs(sklearn_mse - rustkit_mse) < 1e-5, "MSE results differ!"
    # print("MSE correctness test passed!")


def test_empty_input():
    print("EMPTY INPUT")
    # Test empty input
    X = np.array([[]])
    y = np.array([])
    y_true = np.array([])
    y_pred = np.array([])
    
    test_standard_scaler_correctness(X)
    test_ridge_correctness(X, y)
    test_r2_correctness(y_pred, y_true)
    test_mse_correctness(y_pred, y_true)
    test_pca_correctness(X)
    test_kmeans_correctness(X)
    test_kmeans_inertia(X)

def test_square_input():
    print("SQUARE INPUT")
    # Test square input
    X = np.random.rand(10, 10)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    X_clustered, _ = make_blobs(n_samples=10, centers=3, n_features=3, random_state=0)
    
    test_standard_scaler_correctness(X)
    test_ridge_correctness(X, y)
    test_pca_correctness(X)
    test_kmeans_correctness(X_clustered)
    test_kmeans_inertia(X_clustered)

def test_single_input():
    print("SINGLE INPUT")
    # Test single input
    X = np.random.rand(1, 1)
    y = np.array([1.0])
    y_pred = np.array([2.0])

    
    test_standard_scaler_correctness(X)
    test_ridge_correctness(X, y)
    test_r2_correctness(y_pred, y)
    test_mse_correctness(y_pred, y)
    test_pca_correctness(X)
    test_kmeans_correctness(X)
    test_kmeans_inertia(X)

def test_large_input():
    print("LARGE INPUT")
    # Test large input
    X = np.random.rand(1000, 100)
    y = np.random.rand(1000)
    y_true = np.random.rand(1000)
    y_pred = np.random.rand(1000)
    X_clustered, _ = make_blobs(n_samples=1000, centers=10, n_features=10, random_state=0)
    
    test_standard_scaler_correctness(X)
    test_ridge_correctness(X, y)
    test_r2_correctness(y_pred, y_true)
    test_mse_correctness(y_pred, y_true)
    test_pca_correctness(X)
    test_kmeans_correctness(X_clustered)
    test_kmeans_inertia(X_clustered)

def test_negative_input():
    print("NEGATIVE INPUT")
    # Test negative input
    X = np.random.rand(10, 10) - 0.5
    X_clustered = make_blobs(n_samples=10, centers=3, n_features=3, random_state=0)[0] - 0.5
    y = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0])
    y_true = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0])
    y_pred = np.array([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0])
    
    test_standard_scaler_correctness(X)
    test_ridge_correctness(X, y)
    test_r2_correctness(y_pred, y_true)
    test_mse_correctness(y_pred, y_true)
    test_pca_correctness(X)
    test_kmeans_correctness(X_clustered)
    test_kmeans_inertia(X_clustered)

def test_mixed_input():
    print("MIXED INPUT")
    # Test mixed input
    X = np.random.rand(10, 10)
    X_clustered_1 = make_blobs(n_samples=10, centers=2, n_features=4, random_state=0)[0] - 0.5
    X_clustered_2 = make_blobs(n_samples=10, centers=2, n_features=4, random_state=0)[0]
    X_clustered = np.concatenate((X_clustered_1, X_clustered_2), axis=0)
    y = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0])
    y_true = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0])
    y_pred = np.array([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0])
    
    test_standard_scaler_correctness(X)
    test_ridge_correctness(X, y)
    test_r2_correctness(y_pred, y_true)
    test_mse_correctness(y_pred, y_true)
    test_pca_correctness(X)
    test_kmeans_correctness(X_clustered)
    test_kmeans_inertia(X_clustered)

def main():
    # test_empty_input()
    # print("\n\n")
    test_square_input()
    print("\n\n")
    test_single_input()
    print("\n\n")
    test_large_input()
    print("\n\n")
    test_negative_input()
    print("\n\n")
    test_mixed_input()

if __name__ == "__main__":
    main()
