import rustkit
import numpy as np

def print_vector(vector):
    print("[", end=" ")
    for i in range(len(vector)):
        print(f"{vector[i]:.4f}", end=" " if i < len(vector) - 1 else "")
    print("]")

def print_matrix(matrix):
    print("[")
    for row in matrix:
        print(" [", end=" ")
        for element in row:
            print(f"{element:.4f}", end=" ")
        print("]")
    print("]")

def test_converter_vector():
    input_vector = np.array([1.0, 2.0, 3.0, 4.0])
    result = rustkit.converter_vector_test(input_vector)
    
    result_vector = np.array(result)
    
    print("Vector test")
    print("Input vector:")
    print_vector(input_vector)
    print("Result vector:")
    print_vector(result_vector)
    assert np.array_equal(input_vector, result_vector), "Test failed! Input and output vectors are not equal."
    print("Vector test passed!")

def test_converter_matrix():
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    result = rustkit.converter_matrix_test(input_matrix)
    
    result_matrix = np.array(result)
    
    print("Matrix test")
    print("Input matrix:")
    print_matrix(input_matrix)
    print("Result matrix:")
    print_matrix(result_matrix)
    assert np.array_equal(input_matrix, result_matrix), "Test failed! Input and output matrices are not equal."
    print("Matrix test passed!")

if __name__ == "__main__":
    test_converter_vector()
    test_converter_matrix()