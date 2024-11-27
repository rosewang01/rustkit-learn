use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn python_to_rust_dynamic_vector(array: &PyReadonlyArray1<f64>) -> DVector<f64> {
    // Convert a NumPy array to nalgebra::DVector
    let vector = array.as_array();
    let elements = vector.iter().cloned().collect::<Vec<_>>();
    DVector::from_row_slice(&elements)
}

pub fn python_to_rust_dynamic_matrix(array: &PyReadonlyArray2<f64>) -> DMatrix<f64> {
    // Convert a NumPy array to nalgebra::DMatrix
    let matrix = array.as_array();
    let shape = matrix.shape();
    let rows = shape[0];
    let cols = shape[1];
    let elements = matrix.iter().cloned().collect::<Vec<_>>();
    DMatrix::from_row_slice(rows, cols, &elements)
}

pub fn rust_to_python_dynamic_vector(
    py: Python,
    vector: DVector<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    // Convert a nalgebra::DVector (constructed from elements) to a NumPy array
    let array = Array1::from_vec(vector.data.into());
    Ok(PyArray1::from_array(py, &array).into())
}

pub fn rust_to_python_dynamic_matrix(
    py: Python,
    matrix: DMatrix<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    // Convert a nalgebra::DMatrix (constructed from rows, cols, and elements) to a NumPy array
    let shape = matrix.shape();
    let rows = shape.0;
    let cols = shape.1;
    // transpose matrix to be like numpy layout (row major)
    let transposed_matrix = matrix.transpose();
    let array = Array2::from_shape_vec((cols, rows), transposed_matrix.data.into());
    let numpy_array = match array {
        Ok(arr) => PyArray2::from_array(py, &arr),
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Array creation failed: {}",
                e
            )))
        }
    };
    Ok(numpy_array.to_owned())
}

#[pyfunction]
pub fn converter_vector_test(
    py: Python,
    vector: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let rust_vector = python_to_rust_dynamic_vector(&vector);
    rust_to_python_dynamic_vector(py, rust_vector)
}

#[pyfunction]
pub fn converter_matrix_test(
    py: Python,
    matrix: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let rust_matrix = python_to_rust_dynamic_matrix(&matrix);
    rust_to_python_dynamic_matrix(py, rust_matrix)
}
