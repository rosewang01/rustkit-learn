use nalgebra::{DMatrix, DVector};
mod testing;
mod preprocessing;
mod supervised;
mod unsupervised;

use testing::r2_score::R2Score;
use preprocessing::standard_scaler::StandardScaler;
use unsupervised::pca::PCA;
use supervised::ridge_regression::RidgeRegression;

fn main() {
    sample_scaler();
    print!("\n \n");
    sample_pca();
    print!("\n \n");
    sample_ridge();
    print!("\n \n");
    sample_r2();
}

fn sample_ridge() {
    // Training data: 4 samples, 2 features
    let x = DMatrix::from_row_slice(4, 2, &[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ]);
    let y = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);

    // Ridge regression with bias (default behavior)
    let mut ridge_with_bias = RidgeRegression::new(1.0, true);
    ridge_with_bias.fit(&x, &y);

    println!("With Bias - Weights: {}", ridge_with_bias.weights());
    println!("With Bias - Intercept: {:?}", ridge_with_bias.intercept());
    println!("With Bias - Predictions: {}", ridge_with_bias.predict(&x));

    // Ridge regression without bias
    let mut ridge_no_bias = RidgeRegression::new(1.0, false);
    ridge_no_bias.fit(&x, &y);

    println!("No Bias - Weights: {}", ridge_no_bias.weights());
    println!("No Bias - Intercept: {:?}", ridge_no_bias.intercept());
    println!("No Bias - Predictions: {}", ridge_no_bias.predict(&x));
}

fn sample_pca() {
    // Sample data: 4 samples, 3 features
    let data = DMatrix::from_row_slice(4, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ]);

    let mut pca = PCA::new();
    let transformed_data = pca.fit_transform(&data, 2);

    println!("Transformed Data:\n{}", transformed_data);
    println!("Principal Components:\n{}", pca.components());
    println!("Explained Variance:\n{}", pca.explained_variance());

    let original_data = pca.inverse_transform(&transformed_data);
    println!("Reconstructed Data (after inverse transform):\n{}", original_data);
}


fn sample_scaler() {
    let data = DMatrix::from_row_slice(4, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ]);

    let mut scaler = StandardScaler::new();
    let standardized_data = scaler.fit_transform(& data);

    println!("Standardized Data:\n{}", standardized_data);

    let original_data = scaler.inverse_transform(& standardized_data);
    println!("Original Data (after inverse transform):\n{}", original_data);
}

fn sample_r2() {
    // True values
    let y_true = DVector::from_row_slice(&[3.0, -0.5, 2.0, 7.0]);

    // Predicted values
    let y_pred = DVector::from_row_slice(&[2.5, 0.0, 2.0, 8.0]);

    // Compute the R² score
    let r2_score = R2Score::compute(&y_true, &y_pred);

    println!("R² Score: {}", r2_score);

    // Test case for a constant true vector
    let y_true_constant = DVector::from_row_slice(&[1.0, 1.0, 1.0, 1.0]);
    let y_pred_constant = DVector::from_row_slice(&[1.0, 1.0, 1.0, 1.0]);

    let r2_score_constant = R2Score::compute(&y_true_constant, &y_pred_constant);
    println!("R² Score (constant): {}", r2_score_constant);

    // Example where R² is negative (bad model)
    let y_pred_bad = DVector::from_row_slice(&[10.0, 10.0, 10.0, 10.0]);
    let r2_score_bad = R2Score::compute(&y_true, &y_pred_bad);
    println!("R² Score (bad model): {}", r2_score_bad);
}