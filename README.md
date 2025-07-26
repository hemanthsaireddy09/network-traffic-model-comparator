# Network Traffic Analysis Application

This application allows you to analyze network traffic data using various machine learning algorithms. It provides a step-by-step workflow for data preprocessing, model training, and model comparison.

## Features

- Upload and analyze network traffic datasets
- Preprocess data with different methods (Standard Scaling, Min-Max Scaling, Normalization)
- Train multiple machine learning models (KNN, SVM, Naive Bayes, Decision Tree)
- Compare model performance with visualizations
- Flexible target column selection
- Detailed error handling and recommendations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/hemanthsaireddy09/network-traffic-model-comparator.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`



## Using the Application

1. **Upload Data**: Click the "Upload Data" button to select a CSV file containing network traffic data.

2. **Select Target Column**: Choose the column you want to predict (classification target).

3. **Choose Preprocessing Method**: Select how to preprocess the numerical features.

4. **Run Models**: Click on the model buttons to train and evaluate different algorithms.

5. **Compare Results**: Click the "Show Comparison" button to visualize the performance of all trained models.

## Dataset Format

The application expects CSV files with the following format:
- One row per network traffic record
- Multiple columns representing features
- A target column for classification

Example columns for network traffic data:
- Protocol type (TCP, UDP, ICMP)
- Service (HTTP, FTP, SSH)
- Source/destination bytes
- Connection duration
- Number of failed login attempts
- And more...

## Troubleshooting

If you encounter issues with specific models:

- **Naive Bayes**: Try using 'minmax' preprocessing instead of 'standard'
- **Decision Tree**: Check for class imbalance in your dataset
- **All Models**: Ensure your data doesn't contain missing values or non-numeric features


## Acknowledgments

- Built with Flask web framework
- Uses scikit-learn for machine learning models
- Styled with Bootstrap 5 
