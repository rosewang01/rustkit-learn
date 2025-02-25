use nalgebra::{DMatrix, DVector};
pub mod benchmarking;
pub mod converters;

mod preprocessing;
mod supervised;
mod testing;
mod unsupervised;

use preprocessing::simple_imputer::Imputer;
use preprocessing::standard_scaler::StandardScaler;
use supervised::ridge_regression::RidgeRegression;
use testing::regression_metrics::{R2Score, MSE};
use unsupervised::kmeans::KMeans;
use unsupervised::pca::PCA;

fn main() {
    sample_scaler();
    print!("\n \n");
    sample_pca();
    print!("\n \n");
    sample_ridge();
    print!("\n \n");
    sample_r2();

    print!("\n \n");
    sample_kmeans();

    print!("\n \n");
    sample_imputer();
}

fn sample_ridge() {
    println!("=============================================================================");
    println!("RIDGE REGRESSION EXAMPLE");
    println!("=============================================================================");
    // Training data: 4 samples, 2 features
    let x = DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);

    // Ridge regression with bias (default behavior)
    let mut ridge_with_bias = RidgeRegression::new(1.0, true);
    ridge_with_bias.fit_helper(&x, &y);

    println!("With Bias - Weights: {}", ridge_with_bias.weights_helper());
    println!(
        "With Bias - Intercept: {:?}",
        ridge_with_bias.intercept_helper()
    );
    println!(
        "With Bias - Predictions: {}",
        ridge_with_bias.predict_helper(&x)
    );

    // Ridge regression without bias
    let mut ridge_no_bias = RidgeRegression::new(1.0, false);
    ridge_no_bias.fit_helper(&x, &y);

    println!("No Bias - Weights: {}", ridge_no_bias.weights_helper());
    println!(
        "No Bias - Intercept: {:?}",
        ridge_no_bias.intercept_helper()
    );
    println!(
        "No Bias - Predictions: {}",
        ridge_no_bias.predict_helper(&x)
    );
}

fn sample_pca() {
    println!("=============================================================================");
    println!("PCA EXAMPLE");
    println!("=============================================================================");
    // Sample data: 4 samples, 3 features
    let data = DMatrix::from_row_slice(
        4,
        3,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let mut pca = PCA::new();
    let transformed_data = pca.fit_transform_helper(&data, 2);
    println!("Original Data:\n{}", data);
    println!("Transformed Data:\n{}", transformed_data);
    println!("Principal Components:\n{}", pca.components_helper());
    println!("Explained Variance:\n{}", pca.explained_variance_helper());

    let original_data = pca.inverse_transform_helper(&transformed_data);
    println!(
        "Reconstructed Data (after inverse transform):\n{}",
        original_data
    );
}

fn sample_scaler() {
    println!("=============================================================================");
    println!("STANDARD SCALER EXAMPLE");
    println!("=============================================================================");

    let data = DMatrix::from_row_slice(
        4,
        3,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let mut scaler = StandardScaler::new();
    let standardized_data = scaler.fit_transform_helper(&data);

    println!("Standardized Data:\n{}", standardized_data);

    let original_data = scaler.inverse_transform_helper(&standardized_data);
    println!(
        "Original Data (after inverse transform):\n{}",
        original_data
    );
}

fn sample_r2() {
    // True values
    println!("=============================================================================");
    println!("R2-SCORE & MSE EXAMPLE");
    println!("=============================================================================");

    let y_true = DVector::from_row_slice(&[3.0, -0.5, 2.0, 7.0]);

    // Predicted values
    let y_pred = DVector::from_row_slice(&[2.5, 0.0, 2.0, 8.0]);

    // Compute the scores
    let r2_score = R2Score::compute_helper(&y_true, &y_pred);
    let mse = MSE::compute_helper(&y_true, &y_pred);

    println!("R² Score: {}", r2_score);
    println!("MSE: {}", mse);

    // Test case for a constant true vector
    let y_true_constant = DVector::from_row_slice(&[1.0, 1.0, 1.0, 1.0]);
    let y_pred_constant = DVector::from_row_slice(&[1.0, 1.0, 1.0, 1.0]);

    let r2_score_constant = R2Score::compute_helper(&y_true_constant, &y_pred_constant);
    let mse_constant = MSE::compute_helper(&y_true_constant, &y_pred_constant);
    println!("R² Score (constant): {}", r2_score_constant);
    println!("MSE (constant): {}", mse_constant);

    // Example where R² is negative (bad model)
    let y_pred_bad = DVector::from_row_slice(&[10.0, 10.0, 10.0, 10.0]);
    let r2_score_bad = R2Score::compute_helper(&y_true, &y_pred_bad);
    let mse_bad = MSE::compute_helper(&y_true, &y_pred_bad);
    println!("R² Score (bad model): {}", r2_score_bad);
    println!("MSE (bad model): {}", mse_bad);
}

fn sample_imputer() {
    println!("=============================================================================");
    println!("IMPUTER EXAMPLE");
    println!("=============================================================================");
    let data = DMatrix::from_row_slice(
        3,
        2,
        &[
            Some(1.0),
            None,
            None,
            Some(4.0),
            Some(5.0),
            None,
            None,
            Some(8.0),
            None,
        ],
    );

    let mut imputer_mean = Imputer::new("mean", None);
    let mut imputer_cons = Imputer::new("constant", Some(-1.0));
    match imputer_mean.fit_helper(&data) {
        Ok(()) => {
            println!("Original data:\n{:?}", data);
            println!(
                "Mean imputed data:\n{}",
                imputer_mean.transform_helper(&data)
            );
        }
        Err(e) => eprintln!("Mean imputation error: {}", e),
    }
    match imputer_cons.fit_transform_helper(&data) {
        Ok(imputed_data) => {
            println!("Cons imputed data:\n{}", imputed_data);
        }
        Err(e) => eprintln!("Cons imputation error: {}", e),
    }
    let test_data = DMatrix::from_row_slice(
        5,
        2,
        &[
            None,
            Some(1.0),
            Some(1.0),
            None,
            None,
            Some(1.0),
            Some(1.0),
            None,
            None,
            Some(1.0),
        ],
    );
    println!("Original test data:\n{:?}", test_data);
    println!(
        "Mean imputed test data (fit on original data above):\n{}",
        imputer_mean.transform_helper(&test_data)
    );
}

fn sample_kmeans() {
    println!("=============================================================================");
    println!("R2-SCORE & MSE EXAMPLE");
    println!("=============================================================================");
    let data = DMatrix::from_row_slice(
        10,
        3,
        &[
            1.0, 2.0, 3.0, // Point 1
            1.1, 2.1, 3.1, // Point 2
            0.9, 1.9, 2.9, // Point 3
            8.0, 9.0, 10.0, // Point 4
            8.1, 9.1, 10.1, // Point 5
            7.9, 8.9, 9.9, // Point 6
            4.0, 5.0, 6.0, // Point 7
            4.1, 5.1, 6.1, // Point 8
            3.9, 4.9, 5.9, // Point 9
            4.0, 5.0, 6.0, // Point 10
        ],
    );

    // Number of clusters
    let k = 2;
    println!("{}", data.row(1));

    // Run KMeans with Random initialization
    let mut kmeans_random = KMeans::new(k, "random", Some(200), Some(10));
    kmeans_random.fit_helper(&data);
    let labels_random = kmeans_random.predict_helper(&data).unwrap();
    let inertia_random = kmeans_random.compute_inertia_helper(
        &data,
        &labels_random,
        kmeans_random.get_centroids_helper().unwrap(),
    );

    // Print results for Random initialization
    println!("Results with Random Initialization:");
    println!("Labels: {}", labels_random.transpose());
    println!(
        "Centroids: {}",
        kmeans_random.get_centroids_helper().unwrap()
    );
    println!("Total Inertia: {:.4}", inertia_random);

    // Run KMeans with KMeans++ initialization
    let mut kmeans_plus_plus = KMeans::new(k, "kmeans++", None, None);
    let labels_plus_plus = kmeans_plus_plus.fit_predict_helper(&data);
    let inertia_plus_plus = kmeans_plus_plus.compute_inertia_helper(
        &data,
        &labels_plus_plus,
        kmeans_plus_plus.get_centroids_helper().unwrap(),
    );

    // Print results for KMeans++ initialization
    println!("\nResults with KMeans++ Initialization:");
    println!("Labels: {}", labels_plus_plus.transpose());
    println!(
        "Centroids: {}",
        kmeans_plus_plus.get_centroids_helper().unwrap()
    );
    println!("Total Inertia: {:.4}", inertia_plus_plus);
}
