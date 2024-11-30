use pyo3::prelude::*;

mod preprocessing;
use preprocessing::simple_imputer::Imputer;
use preprocessing::standard_scaler::StandardScaler;

mod supervised;
use supervised::ridge_regression::RidgeRegression;

mod testing;
use testing::regression_metrics::R2Score;
use testing::regression_metrics::MSE;

pub mod converters;
use converters::{converter_matrix_test, converter_vector_test};

#[pymodule]
fn rustkit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<StandardScaler>()?;
    m.add_class::<Imputer>()?;
    m.add_class::<RidgeRegression>()?;
    m.add_class::<R2Score>()?;
    m.add_class::<MSE>()?;

    m.add_function(wrap_pyfunction!(converter_vector_test, m)?)?;
    m.add_function(wrap_pyfunction!(converter_matrix_test, m)?)?;

    Ok(())
}
