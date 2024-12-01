import numpy as np
import rustkit
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.linear_model import Ridge as SklearnRidgeRegression
from sklearn.cluster import KMeans as SklearnKMeans

def test_pca_correctness():
    X = np.random.rand(10, 5)
    sklearn_pca = SklearnPCA(n_components=2)
    rustkit_pca = rustkit.PCA()
    
    sklearn_result = sklearn_pca.fit_transform(X)
    rustkit_result = rustkit_pca.fit_transform(X, 2)
    
    assert np.allclose(sklearn_result, rustkit_result, atol=1e-5), "PCA results differ!"
    print("PCA correctness test passed!")

def test_kmeans_correctness():
    X = np.random.rand(10, 5)
    # FLAG: seems like there's an error where if the number of cols in the input matrix is neq to the number of clusters, the test will fail
    sklearn_kmeans = SklearnKMeans(n_clusters=5)
    rustkit_kmeans = rustkit.KMeans(5, "random", 200, 10)
    
    sklearn_result = sklearn_kmeans.fit_predict(X)
    rustkit_result = rustkit_kmeans.fit_predict(X)
    
    assert np.allclose(sklearn_result, rustkit_result), "KMeans results differ!"
    print("KMeans correctness test passed!")

def test_standard_scaler_correctness():
    X = np.random.rand(10, 5)
    sklearn_scaler = SklearnStandardScaler()
    rustkit_scaler = rustkit.StandardScaler()
    
    sklearn_result = sklearn_scaler.fit_transform(X)
    rustkit_result = rustkit_scaler.fit_transform(X)
    
    assert np.allclose(sklearn_result, rustkit_result, atol=1e-5), "Standard Scaler results differ!"
    print("Standard Scaler correctness test passed!")

def test_ridge_correctness():
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    sklearn_ridge = SklearnRidgeRegression(alpha=1.0, fit_intercept=True)
    rustkit_ridge = rustkit.RidgeRegression(1.0, True)
    
    sklearn_ridge.fit(X, y)
    rustkit_ridge.fit(X, y)
    
    assert np.allclose(sklearn_ridge.coef_, rustkit_ridge.weights, atol=1e-5), "Ridge Regression results differ!"
    print("Ridge Regression correctness test passed!")

def test_r2_correctness():
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)
    
    sklearn_r2 = r2_score(y_true, y_pred)
    rustkit_r2 = rustkit.R2Score.compute(y_true, y_pred)
    
    assert abs(sklearn_r2 - rustkit_r2) < 1e-5, "R² results differ!"
    print("R² correctness test passed!")

def test_kmeans_inertia():
    X = np.random.rand(100, 2)
    kmeans = rustkit.KMeans(3, "random", 200, 10)
    kmeans.fit(X)
    
    labels = kmeans.predict(X)
    inertia = kmeans.compute_inertia(X, labels)
    
    # Compute inertia manually
    computed_inertia = sum(
        np.linalg.norm(X[i] - kmeans.centroids[labels[i]])**2 for i in range(len(X))
    )
    
    assert abs(inertia - computed_inertia) < 1e-5, "Inertia computation mismatch!"
    print("KMeans inertia test passed!")


if __name__ == "__main__":
    test_standard_scaler_correctness() # pass
    test_ridge_correctness() # pass
    test_r2_correctness() # pass
    test_pca_correctness()
    test_kmeans_correctness()
    test_kmeans_inertia()