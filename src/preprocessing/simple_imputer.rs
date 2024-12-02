use nalgebra::DMatrix;
use::std::fmt;
use std::collections::HashMap;

#[derive(Debug)]
pub struct ImputerError {
    column_index: usize,
}

impl fmt::Display for ImputerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Column {} has no non-missing values to compute the mean.", self.column_index)
    }
}

impl std::error::Error for ImputerError {}

#[derive(Debug)]
pub enum ImputationType {
    Mean,
    Constant(f64),
}

pub struct Imputer {
    strategy: ImputationType,
    impute_values: Option<HashMap<usize, f64>>,
}

impl Imputer {
    pub fn new(strategy: ImputationType) -> Self {
        Imputer {
            strategy,
            impute_values: None,
        }
    }

    pub fn fit_transform(&mut self, data: &DMatrix<Option<f64>>) -> Result<DMatrix<f64>, ImputerError> {
        // Fit the imputer to compute imputation values
        self.fit(data)?;

        // Transform the data using the computed imputation values
        self.transform(data)
    }


    /// Fits the imputer to the data, computing imputation values for each column.
    pub fn fit(&mut self, data: &DMatrix<Option<f64>>) -> Result<(), ImputerError> {
        let (_, ncols) = data.shape();
        let mut impute_values = HashMap::new();

        for j in 0..ncols {
            let column = data.column(j);

            // Compute the imputation value based on the strategy
            let impute_value = match &self.strategy {
                ImputationType::Mean => {
                    let mut non_missing_values = Vec::new();

                    for i in 0..data.nrows() {
                        if let Some(value) = column[i] {
                            non_missing_values.push(value);
                        }
                    }

                    if non_missing_values.is_empty() {
                        return Err(ImputerError { column_index: j }); // Error if column is all None
                    }

                    let sum: f64 = non_missing_values.iter().sum();
                    sum / non_missing_values.len() as f64
                }
                ImputationType::Constant(val) => *val,
            };

            impute_values.insert(j, impute_value);
        }

        self.impute_values = Some(impute_values);
        Ok(())
    }

    /// Transforms the data using the computed imputation values.
    pub fn transform(&self, data: &DMatrix<Option<f64>>) -> Result<DMatrix<f64>, ImputerError> {
        let (nrows, ncols) = data.shape();
        let mut result = DMatrix::zeros(nrows, ncols);

        if self.impute_values.is_none() {
            return Err(ImputerError {
                column_index: usize::MAX, // Indicate fitting wasn't done
            });
        }

        let impute_values = self.impute_values.as_ref().unwrap();

        for j in 0..ncols {
            let column = data.column(j);

            let impute_value = impute_values.get(&j).ok_or(ImputerError { column_index: j })?;

            for i in 0..nrows {
                result[(i, j)] = column[i].unwrap_or(*impute_value);
            }
        }

        Ok(result)
    }
}
