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
git clone https://github.com/yourusername/network_traffic_web.git
cd network_traffic_web
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

## Downloading Datasets

The application includes a script to download and prepare network traffic datasets:

```bash
python download_dataset.py
```

This script will download:
- NSL-KDD dataset (full and sample versions)
- Create a sample of the CIC-IDS2017 dataset

The datasets will be saved in the `data` directory.

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Flask web framework
- Uses scikit-learn for machine learning models
- Styled with Bootstrap 5 