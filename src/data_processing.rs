use std::collections::HashMap;
use nalgebra::DVector;

// Trim leading/trailing whitespace from keys in dataset and
// remove records with missing values in relevant fields
pub fn clean_data(data: Vec<HashMap<String, String>>) -> Vec<HashMap<String, String>> {
    let cleaned_data: Vec<HashMap<String, String>> = data
        .into_iter().map(|record| {
            let cleaned_data: HashMap<String, String> = record
                .into_iter().map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
                .collect();

            cleaned_data
        })
        .filter(|record| {
            !record["Publishing Year"].is_empty()
                && !record["Book_average_rating"].is_empty()
                && !record["gross sales"].is_empty()
                && !record["sale price"].is_empty()
                && !record["sales rank"].is_empty()
                && !record["Publisher"].is_empty()
                && !record["units sold"].is_empty()
        })
        .collect();
    cleaned_data
}

// Calculate the average rating of each Publisher
pub fn calculate_average_rating(data: &[HashMap<String, String>]) -> HashMap<String, f64> {
    let mut publisher_ratings_sum: HashMap<String, f64> = HashMap::new();
    let mut publisher_ratings_count: HashMap<String, usize> = HashMap::new();

    // Iterate over each record in the dataset
    for record in data {
        let (publisher, rating_str) = (record.get("Publisher").unwrap(), record.get("Book_average_rating").unwrap());
        let rating = rating_str.parse::<f64>().unwrap();
        let entry = publisher_ratings_sum.entry(publisher.clone()).or_insert(0.0);
        *entry += rating;

        let count_entry = publisher_ratings_count.entry(publisher.clone()).or_insert(0);
        *count_entry += 1;
    }

    let mut publisher_average_ratings: HashMap<String, f64> = HashMap::new();
    for (publisher, sum) in publisher_ratings_sum {
        let count = publisher_ratings_count[&publisher];
        let average = sum / count as f64;
        publisher_average_ratings.insert(publisher, average);
    }

    publisher_average_ratings
}

// extract the features I deemed relevant and target variable
pub fn extract_features_and_target(data: &[HashMap<String, String>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let features: Vec<Vec<f64>> = data
        .iter().map(|record| { vec![
                record["Publishing Year"].parse::<f64>().unwrap(),
                record["Book_average_rating"].parse::<f64>().unwrap(),
                record["gross sales"].parse::<f64>().unwrap(),
                record["sale price"].parse::<f64>().unwrap(),
                record["Publisher_rating"].parse::<f64>().unwrap(),
            ]
        })
        .collect();

    let target: Vec<f64> = data
        .iter().map(|record| record["units sold"].parse::<f64>().unwrap())
        .collect();

    (features, target)
}

pub fn z_score_normalization(features: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let num_features = features[0].len();

    // Calculate means and standard deviations for each feature
    let means: Vec<f64> = (0..num_features)
        .map(|i| features.iter().map(|row| row[i]).sum::<f64>() / features.len() as f64)
        .collect();

    let s_deviation: Vec<f64> = (0..num_features)
        .map(|i| {
            let variance = features
                .iter().map(|row| (row[i] - means[i]).powi(2))
                .sum::<f64>()
                / features.len() as f64;
            variance.sqrt()
        })
        .collect();

    let normalized_features: Vec<Vec<f64>> = features
        .iter().map(|row| {
            row.iter().enumerate().map(|(i, &x)| (x - means[i]) / s_deviation[i])
            .collect()
        })
        .collect();

    normalized_features
}

// Calculate Mean Squared Error
pub fn calculate_mse(predictions: &DVector<f64>, targets: &DVector<f64>) -> f64 {
    let errors = predictions - targets;
    let squared_errors = errors.component_mul(&errors);
    let mse = squared_errors.sum() / predictions.nrows() as f64;
    mse
}

// Calculate Mean Absolute Error
pub fn calculate_mae(predictions: &DVector<f64>, targets: &DVector<f64>) -> f64 {
    let errors = predictions - targets;
    let absolute_errors = errors.map(|x| x.abs());
    let mae = absolute_errors.sum() / predictions.nrows() as f64;
    mae
}