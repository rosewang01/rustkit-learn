use nalgebra::DVector;

/// A class to compute the R² score between two vectors.
pub struct R2Score;

impl R2Score {
    /// Computes the R² score between the true values and predictions.
    ///
    /// # Arguments
    /// - `y_true`: The vector of true values.
    /// - `y_pred`: The vector of predicted values.
    ///
    /// # Returns
    /// - The R² score (coefficient of determination).
    ///
    /// # Panics
    /// - If `y_true` and `y_pred` have different lengths.
    pub fn compute(y_true: &DVector<f64>, y_pred: &DVector<f64>) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have the same length."
        );

        let mean_y_true = y_true.mean();
        let total_variance: f64 = y_true.iter().map(|&y| (y - mean_y_true).powi(2)).sum();

        let residual_sum_of_squares: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
            .sum();

        if total_variance == 0.0 {
            if residual_sum_of_squares == 0.0 {
                return 1.0; // Perfect prediction for a constant true vector
            } else {
                return f64::NEG_INFINITY; // Undefined for non-zero residuals
            }
        }

        1.0 - residual_sum_of_squares / total_variance
    }
}

pub struct MSE;

impl MSE {
    /// Computes the R² score between the true values and predictions.
    ///
    /// # Arguments
    /// - `y_true`: The vector of true values.
    /// - `y_pred`: The vector of predicted values.
    ///
    /// # Returns
    /// - The R² score (coefficient of determination).
    ///
    /// # Panics
    /// - If `y_true` and `y_pred` have different lengths.
    pub fn compute(y_true: &DVector<f64>, y_pred: &DVector<f64>) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have the same length."
        );

        let residual_sum_of_squares: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
            .sum();

        residual_sum_of_squares / (y_pred.len() as f64)
    }
}
