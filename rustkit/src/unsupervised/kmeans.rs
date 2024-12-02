use crate::benchmarking::log_function_time;
use crate::converters::{
    python_to_rust_dynamic_matrix, python_to_rust_dynamic_vector, rust_to_python_dynamic_matrix,
    rust_to_python_dynamic_vector,
};
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFloat;
use pyo3::Python;
use rand::seq::SliceRandom;
use rand::Rng;
use std::f64;

#[derive(Debug, Clone, Copy)]
pub enum InitMethod {
    KMeansPlusPlus,
    Random,
}

#[pyclass]
pub struct KMeans {
    k: usize,                        // Number of clusters
    max_iter: usize,                 // Maximum number of iterations for a single run
    n_init: usize, // Number of runs of the algo (we pick the best, particularly relevant for random initialization)
    init_method: InitMethod, // Initialization method (KMeans++ or Random)
    centroids: Option<DMatrix<f64>>, // Centroids of the clusters after fitting
    labels: Option<DVector<usize>>, // Cluster labels for each data point
}

// Implementation of Lloyd's KMeans clustering algorithm with KMeans++ or random initialization
// IMPORTANT: Centroids are stored as a (d x k) matrix where d is the dimension of your data.
//  in particular, this means that the centroids are stored as the *columns* of the centroids matrix
#[pymethods]
impl KMeans {
    // Creates a new KMeans instance with specified parameters
    #[new]
    pub fn new(
        k: usize,
        init_method_str: &str,
        max_iter: Option<usize>,
        n_init: Option<usize>,
    ) -> Self {
        let init_method = match init_method_str {
            "kmeans++" => InitMethod::KMeansPlusPlus,
            "random" => InitMethod::Random,
            _ => panic!("Invalid initialization method"),
        };
        let max_iter = max_iter.unwrap_or(200);
        let n_init = n_init.unwrap_or_else(|| match init_method {
            InitMethod::KMeansPlusPlus => 1,
            InitMethod::Random => 10,
        });

        KMeans {
            k,
            max_iter,
            n_init,
            init_method,
            centroids: None,
            labels: None,
        }
    }

    pub fn fit(&mut self, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let data = python_to_rust_dynamic_matrix(&data);
        let result = log_function_time(|| self.fit_helper(&data), "KMeans::fit");
        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    pub fn fit_predict(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray1<usize>>> {
        let data = python_to_rust_dynamic_matrix(&data);
        let labels =
            log_function_time(|| self.fit_predict_helper(&data), "KMeans::fit_predict").unwrap();
        rust_to_python_dynamic_vector(py, labels)
    }

    pub fn compute_inertia(
        &self,
        py: Python,
        data: PyReadonlyArray2<f64>,
        labels: PyReadonlyArray1<usize>,
    ) -> PyResult<Py<PyFloat>> {
        let data = python_to_rust_dynamic_matrix(&data);
        let labels = python_to_rust_dynamic_vector(&labels);
        let float = log_function_time(
            || self.compute_inertia_helper(&data, &labels, self.centroids.as_ref().unwrap()),
            "KMeans::compute_inertia",
        )
        .unwrap();
        Ok(PyFloat::new(py, float).into())
    }

    pub fn predict(
        &self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray1<usize>>> {
        let data = python_to_rust_dynamic_matrix(&data);
        let labels = log_function_time(|| self.predict_helper(&data), "KMeans::predict").unwrap();
        match labels {
            Some(labels) => rust_to_python_dynamic_vector(py, labels),
            None => Err(PyValueError::new_err("Model has not been fitted")),
        }
    }

    #[getter]
    pub fn centroids(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let centroids_opt = self.get_centroids_helper();
        match centroids_opt {
            Some(centroids) => rust_to_python_dynamic_matrix(py, centroids.clone()),
            None => Err(PyValueError::new_err("Centroids have not been computed")),
        }
    }
}

impl KMeans {
    // Fits the KMeans model to the data
    pub fn fit_helper(&mut self, data: &DMatrix<f64>) {
        let mut best_inertia = f64::MAX;
        let mut best_centroids = None;
        let mut best_labels = None;

        // Run the algorithm n_init times and keep the best result
        for _ in 0..self.n_init {
            let (centroids, labels, inertia) = self.run_single(data);
            if inertia < best_inertia {
                best_inertia = inertia;
                best_centroids = Some(centroids);
                best_labels = Some(labels);
            }
        }

        self.centroids = best_centroids;
        self.labels = best_labels;
    }

    // Combines fit and predict into a single function
    pub fn fit_predict_helper(&mut self, data: &DMatrix<f64>) -> DVector<usize> {
        self.fit_helper(data);
        self.predict_helper(data).unwrap()
    }

    // Runs a single iteration of KMeans, initializing centroids and iteratively updating them
    fn run_single(&self, data: &DMatrix<f64>) -> (DMatrix<f64>, DVector<usize>, f64) {
        let mut centroids = match self.init_method {
            InitMethod::KMeansPlusPlus => self.kmeans_plus_plus(data),
            InitMethod::Random => self.random_init(data),
        };

        let mut labels = DVector::from_element(data.nrows(), 0);
        let mut inertia = f64::MAX;

        // Iterate to refine centroids and labels
        for _ in 0..self.max_iter {
            labels = self.assign_labels(data, &centroids);
            let new_centroids = self.update_centroids(data, &labels);

            let new_inertia = self.compute_inertia_helper(data, &labels, &new_centroids);
            if (inertia - new_inertia).abs() < 1e-4 {
                break; // Stop if the improvement in inertia is negligible
            }

            centroids = new_centroids;
            inertia = new_inertia;
        }

        (centroids, labels, inertia)
    }

    // Randomly selects k data points as initial centroids
    fn random_init(&self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..data.nrows()).collect();
        indices.shuffle(&mut rng);
        let selected = indices.iter().take(self.k).copied().collect::<Vec<usize>>();
        DMatrix::from_rows(&selected.iter().map(|&i| data.row(i)).collect::<Vec<_>>())
    }

    // Implements KMeans++ initialization to choose centroids
    fn kmeans_plus_plus(&self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::new();
        centroids.push(data.row(rng.gen_range(0..data.nrows())).transpose());

        for _ in 1..self.k {
            let distances: Vec<f64> = data
                .row_iter()
                .map(|row| {
                    centroids
                        .iter()
                        .map(|centroid| (row - centroid.transpose()).norm_squared())
                        .fold(f64::MAX, f64::min)
                })
                .collect();

            let cumulative_distances: Vec<f64> = distances
                .iter()
                .scan(0.0, |acc, &dist| {
                    *acc += dist;
                    Some(*acc)
                })
                .collect();

            let total_distance = *cumulative_distances.last().unwrap();
            let rand_distance = rng.gen_range(0.0..total_distance);

            let next_idx = cumulative_distances
                .iter()
                .position(|&d| d >= rand_distance)
                .unwrap();

            centroids.push(data.row(next_idx).transpose());
        }

        DMatrix::from_columns(&centroids)
    }

    // Assigns each data point to the nearest centroid
    fn assign_labels(&self, data: &DMatrix<f64>, centroids: &DMatrix<f64>) -> DVector<usize> {
        DVector::from_iterator(
            data.nrows(),
            data.row_iter().map(|row| {
                centroids
                    .column_iter()
                    .enumerate()
                    .map(|(i, centroid)| (i, (row - centroid.transpose()).norm_squared()))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
            }),
        )
    }

    // Updates centroids based on the mean of assigned points
    fn update_centroids(&self, data: &DMatrix<f64>, labels: &DVector<usize>) -> DMatrix<f64> {
        let mut centroids = vec![DVector::zeros(data.ncols()); self.k];
        let mut counts = vec![0; self.k];

        for (i, label) in labels.iter().enumerate() {
            centroids[*label] += data.row(i).transpose();
            counts[*label] += 1;
        }

        for (centroid, &count) in centroids.iter_mut().zip(&counts) {
            if count > 0 {
                *centroid /= count as f64;
            }
        }

        DMatrix::from_columns(&centroids)
    }

    // Computes the inertia (sum of squared distances to nearest centroids)
    pub fn compute_inertia_helper(
        &self,
        data: &DMatrix<f64>,
        labels: &DVector<usize>,
        centroids: &DMatrix<f64>,
    ) -> f64 {
        data.row_iter()
            .enumerate()
            .map(|(i, row)| (row - centroids.column(labels[i]).transpose()).norm_squared())
            .sum()
    }

    // Predicts cluster labels for new data points
    pub fn predict_helper(&self, data: &DMatrix<f64>) -> Option<DVector<usize>> {
        self.centroids
            .as_ref()
            .map(|centroids| self.assign_labels(data, centroids))
    }

    //
    pub fn get_centroids_helper(&self) -> Option<&DMatrix<f64>> {
        self.centroids.as_ref()
    }
}
