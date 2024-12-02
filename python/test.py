import rustkit
import numpy as np

def test_converter_vector():
    print("=" * 77)
    print("VECTOR TEST")
    print("=" * 77)
    input_vector = np.array([1.0, 2.0, 3.0, 4.0])
    result = rustkit.converter_vector_test(input_vector)
    
    result_vector = np.array(result)
    
    print("Vector test")
    print("Input vector:")
    print(input_vector)
    print("Result vector:")
    print(result_vector)
    assert np.array_equal(input_vector, result_vector), "Test failed! Input and output vectors are not equal."
    print("Vector test passed!")

def test_converter_matrix():
    print("=" * 77)
    print("MATRIX TEST")
    print("=" * 77)
    input_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    result = rustkit.converter_matrix_test(input_matrix)
    
    result_matrix = np.array(result)
    
    print("Input matrix:")
    print(input_matrix)
    print("Result matrix:")
    print(result_matrix)
    assert np.array_equal(input_matrix, result_matrix), "Test failed! Input and output matrices are not equal."
    print("Matrix test passed!")

def test_converter_opt_matrix():
    print("=" * 77)
    print("NULL VAL MATRIX TEST")
    print("=" * 77)
    input_matrix = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    
    result = rustkit.converter_matrix_opt_test(input_matrix)
    
    result_matrix = np.array(result)
    
    print("Input matrix:")
    print(input_matrix)
    print("Result matrix:")
    print(result_matrix)
    
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            if np.isnan(input_matrix[i][j]):
                assert np.isnan(result_matrix[i][j]), "Test failed! NaN values are not equal."
            else:
                assert input_matrix[i][j] == result_matrix[i][j], "Test failed! Values are not equal."
    print("Null val matrix test passed!")

def sample_scaler():
    print("=" * 77)
    print("STANDARD SCALER EXAMPLE")
    print("=" * 77)
    
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    
    scaler = rustkit.StandardScaler()
    standardized_data = scaler.fit_transform(data)
    print("Standardized Data:")
    print(standardized_data)
    
    original_data = scaler.inverse_transform(standardized_data)
    print("Original Data (after inverse transform):")
    print(original_data)

def sample_pca():
    print("=" * 77)
    print("PCA EXAMPLE")
    print("=" * 77)
    
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    
    pca = rustkit.PCA()
    transformed_data = pca.fit_transform(data, 2)
    print("Original Data:")
    print(data)
    print("Transformed Data:")
    print(transformed_data)
    print("Principal Components:")
    print(pca.components)
    print("Explained Variance:")
    print(pca.explained_variance)
    
    original_data = pca.inverse_transform(transformed_data)
    print("Reconstructed Data (after inverse transform):")
    print(original_data)

def sample_ridge():
    print("=" * 77)
    print("RIDGE REGRESSION EXAMPLE")
    print("=" * 77)
    
    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    
    ridge_with_bias = rustkit.RidgeRegression(1.0, True)
    ridge_with_bias.fit(x, y)
    print("With Bias - Weights:")
    print(ridge_with_bias.weights)
    print("With Bias - Intercept:", ridge_with_bias.intercept)
    print("With Bias - Predictions:")
    print(ridge_with_bias.predict(x))
    
    ridge_no_bias = rustkit.RidgeRegression(1.0, False)
    ridge_no_bias.fit(x, y)
    print("No Bias - Weights:")
    print(ridge_no_bias.weights)
    print("No Bias - Intercept:", ridge_no_bias.intercept)
    print("No Bias - Predictions:")
    print(ridge_no_bias.predict(x))

def sample_r2():
    print("=" * 77)
    print("R2-SCORE & MSE EXAMPLE")
    print("=" * 77)
    
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    r2_score = rustkit.R2Score.compute(y_true, y_pred)
    mse = rustkit.MSE.compute(y_true, y_pred)
    print("R² Score:", r2_score)
    print("MSE:", mse)

def sample_kmeans():
    print("=" * 77)
    print("KMEANS EXAMPLE")
    print("=" * 77)
    
    data = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.9, 1.9, 2.9],
        [8.0, 9.0, 10.0],
        [8.1, 9.1, 10.1],
        [7.9, 8.9, 9.9],
        [4.0, 5.0, 6.0],
        [4.1, 5.1, 6.1],
        [3.9, 4.9, 5.9],
        [4.0, 5.0, 6.0]
    ])
    
    kmeans = rustkit.KMeans(3, "random", 200, 10)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    inertia = kmeans.compute_inertia(data, labels)
    
    print("Results with Random Initialization:")
    print("Labels:")
    print(labels)
    print("Centroids:")
    print(kmeans.centroids)
    print("Total Inertia:", inertia)

if __name__ == "__main__":
    test_converter_vector()
    print("\n\n")
    test_converter_matrix()
    print("\n\n")
    test_converter_opt_matrix()
    print("\n\n")
    sample_scaler()
    print("\n\n")
    sample_pca()
    print("\n\n")
    sample_ridge()
    print("\n\n")
    sample_r2()
    print("\n\n")
    sample_kmeans()