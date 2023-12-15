use std::error::Error;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use rand::seq::SliceRandom;

mod csv_reader;
mod data_processing;

use csv_reader::read_csv_file;
use data_processing::{clean_data, calculate_average_rating, extract_features_and_target, z_score_normalization, calculate_mse, calculate_mae};


fn main() -> Result<(), Box<dyn Error>> {
    let path = "Books_Data_Clean.csv";
    let data = read_csv_file(path)?;

    // Clean data
    let mut data_cleaned = clean_data(data);

    // Calculate publisher rating and append it to each entry in the dataset
    let publisher_average_ratings = calculate_average_rating(&data_cleaned);

    for record in &mut data_cleaned {
        let publisher = record.get("Publisher").unwrap();
        let average_rating = publisher_average_ratings.get(publisher).unwrap();
        record.insert("Publisher_rating".to_string(), average_rating.to_string());
    }

    // Print the first record for inspection
    println!("First Record:");
    let first_record = data_cleaned.get(0).unwrap();
    for (key, value) in first_record.iter() {
        println!("{}: {}", key, value);
    }
    println!();

    // Shuffle the records because it is ordered by sales rank
    let mut rng = rand::thread_rng();
    data_cleaned.shuffle(&mut rng);

    // Extract features and target variable for regression
    let (features, target) = extract_features_and_target(&data_cleaned);

    // Print the first few records for inspection
    println!("Features and Target value of three random records:");
    for (f, t) in features.iter().zip(&target).take(3) {
        println!("Features : {:?}, Target (units sold): {}", f, t);

    }
    println!();

    // Normalize features
    let features_normalized = z_score_normalization(features);

    // Convert features and target to nalgebra DMatrix and DVector
    let x = DMatrix::from_row_slice(features_normalized.len(), features_normalized[0].len(), &features_normalized.concat());
    let y = DMatrix::from_column_slice(target.len(), 1, &target);
    
    // Add a column of ones to X for the intercept term
    let x_with_intercept = x.clone().insert_column(5, 1.0);

    // Fit linear regression model on the data
    let coefficients = (x_with_intercept.transpose() * &x_with_intercept)
        .try_inverse()
        .unwrap()
        * &x_with_intercept.transpose()
        * &y;

    // Extract coefficients and intercept
    let coeff = coefficients.rows(0, 6);
    let intercept = coefficients[(5, 0)];

    println!("Coefficients: {}", coeff);
    println!("Intercept: {}", intercept);

    // Calculate predictions
    let predictions = &x_with_intercept * &coeff + &DMatrix::repeat(x_with_intercept.nrows(), 1, intercept);

    // Convert predictions and target to DVector so it can used in calculation
    let predictions_vector = DVector::from_column_slice(predictions.column(0).as_slice());
    let target_vector = DVector::from_column_slice(target.as_slice());

    // Calculate MSE and MAE
    let mse = calculate_mse(&predictions_vector, &target_vector);
    let mae = calculate_mae(&predictions_vector, &target_vector);

    println!("Mean Squared Error (MSE): {}", mse);
    println!("Mean Absolute Error (MAE): {}", mae);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_average_rating() {
        // This test checks to see if it properly calculates the average book rating for each unique publisher
        let mut data = Vec::new();

        let mut testdata1 = HashMap::new();
        testdata1.insert("Book_average_rating".to_string(), "5.0".to_string());
        testdata1.insert("Publisher".to_string(), "PublisherA".to_string());
    
        let mut testdata2 = HashMap::new();
        testdata2.insert("Book_average_rating".to_string(), "3.0".to_string());
        testdata2.insert("Publisher".to_string(), "PublisherA".to_string());
    
        let mut testdata3 = HashMap::new();
        testdata3.insert("Book_average_rating".to_string(), "2.0".to_string());
        testdata3.insert("Publisher".to_string(), "PublisherB".to_string());

    
        data.push(testdata1);
        data.push(testdata2);
        data.push(testdata3);

        let result = calculate_average_rating(&data);

        assert_eq!(result.len(), 2);
        assert_eq!(result["PublisherA"], 4.0);
        assert_eq!(result["PublisherB"], 2.0);
    }

    #[test]
    fn test_calculate_mse() {
        // this tests to see if the calculation for mse is correct
        let predictions = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = DVector::from_vec(vec![1.5, 2.5, 3.5]);

        let result = calculate_mse(&predictions, &targets);

        assert!(result.abs() == 0.25); 
    }

    #[test]
    fn test_calculate_mae() {
        // this tests to see if the calculation for mae is correct
        let predictions = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = DVector::from_vec(vec![1.5, 2.5, 3.5]);

        let result = calculate_mae(&predictions, &targets);

        assert!(result.abs() == 0.5); 
    }
}
