use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn rust_to_python_dynamic_vector(
    py: Python,
    elements: Vec<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    // Convert a nalgebra::DVector (constructed from elements) to a NumPy array
    let array = Array1::from(elements);
    Ok(PyArray1::from_array(py, &array).into())
}

#[pyfunction]
pub fn rust_to_python_dynamic_matrix(
    py: Python,
    rows: usize,
    cols: usize,
    elements: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    // Convert a nalgebra::DMatrix (constructed from rows, cols, and elements) to a NumPy array
    let array = Array2::from_shape_vec((rows, cols), elements).unwrap();
    Ok(PyArray2::from_array(py, &array).into())
}

pub fn python_to_rust_dynamic_vector(array: PyReadonlyArray1<f64>) -> DVector<f64> {
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
