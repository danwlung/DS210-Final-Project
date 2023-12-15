use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;

extern crate csv;

// read a CSV file and return a vector of hash maps
pub fn read_csv_file(path: &str) -> Result<Vec<HashMap<String, String>>, Box<dyn Error>> {
    let mut result: Vec<HashMap<String, String>> = Vec::new();
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Get the headers from the CSV file
    let headers = rdr.headers()?.clone();

    for record in rdr.records() {
        let line_str = record?;
        let mut record_map = HashMap::new();

        for (header, value) in headers.iter().zip(line_str.iter()) {
            record_map.insert(header.to_owned(), value.to_owned());
        }

        result.push(record_map);
    }

    Ok(result)
}