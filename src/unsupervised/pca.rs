use nalgebra::{linalg::SVD, DMatrix, DVector, RowDVector};

pub struct PCA {
    components: DMatrix<f64>,
    explained_variance: DVector<f64>,
    mean: RowDVector<f64>,
}

// TODO: make a fix to use partial SVD decomposition for efficiency
impl PCA {
    /// Creates a new PCA instance.
    pub fn new() -> Self {
        PCA {
            components: DMatrix::zeros(0, 0),
            explained_variance: DVector::zeros(0),
            mean: RowDVector::zeros(0),
        }
    }

    /// Fits the PCA model to the data and computes the principal components.
    pub fn fit(&mut self, data: &DMatrix<f64>, n_components: usize) {
        let (n_samples, n_features) = data.shape();
        assert!(
            n_components <= n_features,
            "n_components must be <= number of features"
        );

        // Compute the mean of each feature
        self.mean = data.row_mean();

        // Center the data by subtracting the mean
        let centered_data = data - DMatrix::from_rows(&vec![self.mean.clone(); n_samples]);

        // Compute the covariance matrix
        let covariance_matrix =
            &centered_data.transpose() * &centered_data / (n_samples as f64 - 1.0);

        // Perform Singular Value Decomposition
        let svd = SVD::new(covariance_matrix, true, true);

        // Extract the top n_components
        self.components = svd.v_t.unwrap().rows(0, n_components).transpose();
        self.explained_variance = svd
            .singular_values
            .rows(0, n_components)
            .map(|s| s * s / (n_samples as f64 - 1.0));
    }

    /// Transforms the data to the principal component space.
    pub fn transform(&self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let n_samples = data.nrows();
        let centered_data = data - DMatrix::from_rows(&vec![self.mean.clone(); n_samples]);
        &centered_data * &self.components
    }

    /// Fits the PCA model and transforms the data.
    pub fn fit_transform(&mut self, data: &DMatrix<f64>, n_components: usize) -> DMatrix<f64> {
        self.fit(data, n_components);
        self.transform(data)
    }

    /// Inversely transforms the data back to the original feature space.
    pub fn inverse_transform(&self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let n_samples = data.nrows();
        let reconstructed_data = data * self.components.transpose();
        reconstructed_data + DMatrix::from_rows(&vec![self.mean.clone(); n_samples])
    }

    /// Returns the principal components.
    pub fn components(&self) -> &DMatrix<f64> {
        &self.components
    }

    /// Returns the amount of variance explained by each of the selected components.
    pub fn explained_variance(&self) -> &DVector<f64> {
        &self.explained_variance
    }
}
