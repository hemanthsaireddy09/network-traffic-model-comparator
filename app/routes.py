from flask import render_template, request, jsonify, send_file
from app import app
from app.ml_models import process_data, train_model, compare_models
import os
import pandas as pd
import io
import matplotlib.pyplot as plt
import base64
import json
import numpy as np
import traceback

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_dataset(dataset, target_column):
    """Validate the uploaded dataset and return error messages if any."""
    errors = []
    
    # Check if dataset is empty
    if dataset.empty:
        errors.append("The uploaded file is empty")
        return errors
    
    # Check for required target column
    if target_column not in dataset.columns:
        errors.append(f"Missing required '{target_column}' column for classification")
    
    # Check for minimum number of features
    if len(dataset.columns) < 2:
        errors.append("Dataset must have at least one feature column and one target column")
    
    # Check for missing values
    missing_cols = dataset.columns[dataset.isnull().any()].tolist()
    if missing_cols:
        errors.append(f"The following columns contain missing values (will be automatically handled): {', '.join(missing_cols)}")
    
    # Check for minimum number of samples
    if len(dataset) < 10:
        errors.append("Dataset must have at least 10 samples for meaningful analysis")
    
    # Check for class balance
    if target_column in dataset.columns:
        class_counts = dataset[target_column].value_counts()
        if len(class_counts) < 2:
            errors.append("Dataset must have at least 2 different classes")
        elif class_counts.min() < 2:
            errors.append("Each class must have at least 2 samples")
    
    return errors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Read the dataset
            dataset = pd.read_csv(file)
            
            # Get column information for target selection
            columns = []
            for column in dataset.columns:
                info = {
                    'name': column,
                    'type': str(dataset[column].dtype),
                    'unique_values': len(dataset[column].unique()),
                    'sample_values': dataset[column].unique()[:5].tolist()
                }
                columns.append(info)
            
            # Suggest target column (prefer categorical columns with fewer unique values)
            suggested_target = None
            min_unique_ratio = float('inf')
            
            for col in columns:
                unique_ratio = col['unique_values'] / len(dataset)
                if unique_ratio < min_unique_ratio and col['unique_values'] >= 2:
                    min_unique_ratio = unique_ratio
                    suggested_target = col['name']
            
            return jsonify({
                'success': True,
                'columns': columns,
                'suggested_target': suggested_target,
                'total_records': len(dataset),
                'dataset': dataset.to_json(orient='records')
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'})

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        data = request.get_json()
        dataset = pd.read_json(data['dataset'])
        target_column = data['target_column']
        preprocessing_method = data.get('preprocessing_method', 'standard')
        
        # Validate the dataset
        validation_errors = validate_dataset(dataset, target_column)
        if validation_errors:
            return jsonify({
                'warnings': validation_errors,
                'error': None  # We'll proceed with warnings
            })
        
        # Process the data
        X_train, X_test, y_train, y_test = process_data(
            dataset, 
            target_column=target_column,
            preprocessing_method=preprocessing_method
        )
        
        return jsonify({
            'success': True,
            'X_train': X_train.tolist(),
            'X_test': X_test.tolist(),
            'y_train': y_train.tolist(),
            'y_test': y_test.tolist(),
            'feature_names': [col for col in dataset.columns if col != target_column],
            'class_names': sorted(dataset[target_column].unique().tolist())
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/run_model/<model_name>', methods=['POST'])
def run_model(model_name):
    try:
        data = request.get_json()
        X_train = np.array(data['X_train'])
        X_test = np.array(data['X_test'])
        y_train = np.array(data['y_train'])
        y_test = np.array(data['y_test'])
        model_params = data.get('model_params', None)
        
        # Validate model name
        valid_models = ['knn', 'svm', 'rf', 'nb', 'xgboost']
        if model_name not in valid_models:
            return jsonify({'error': f"Invalid model name: {model_name}. Valid models are: {', '.join(valid_models)}"})
        
        # Model-specific parameter adjustments
        if model_name == 'rf':
            # For Random Forest, adjust parameters based on data size
            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            
            # Suggest parameters based on data characteristics
            suggested_params = {
                'n_estimators': 100,
                'max_depth': min(20, max(10, int(np.log2(n_samples)))),
                'min_samples_split': max(2, int(n_samples * 0.01)),
                'random_state': 42,
                'n_jobs': -1  # Use all available CPU cores
            }
            
            # If no parameters provided, use suggested ones
            if model_params is None:
                model_params = suggested_params
            else:
                # Merge with suggested parameters, keeping user-provided ones if they exist
                for key, value in suggested_params.items():
                    if key not in model_params:
                        model_params[key] = value
        
        elif model_name == 'xgboost':
            # For XGBoost, adjust parameters based on data size
            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            
            # Suggest parameters based on data characteristics
            suggested_params = {
                'n_estimators': 100,
                'max_depth': min(10, max(5, int(np.log2(n_samples)))),
                'learning_rate': 0.1,
                'random_state': 42
            }
            
            # If no parameters provided, use suggested ones
            if model_params is None:
                model_params = suggested_params
            else:
                # Merge with suggested parameters, keeping user-provided ones if they exist
                for key, value in suggested_params.items():
                    if key not in model_params:
                        model_params[key] = value
        
        # Train the specified model
        results = train_model(model_name, X_train, X_test, y_train, y_test, model_params)
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        error_message = str(e)
        traceback_info = traceback.format_exc()
        print(f"Error in run_model for {model_name}: {error_message}")
        print(traceback_info)
        
        # Provide more user-friendly error messages
        if "Input contains NaN values" in error_message:
            return jsonify({'error': "The dataset contains NaN values. Please check your data and try again."})
        elif "Empty input data" in error_message:
            return jsonify({'error': "The dataset is empty or invalid. Please check your data and try again."})
        elif "Not enough classes" in error_message:
            return jsonify({'error': "The dataset doesn't have enough classes for classification. Please check your target column."})
        elif "Features must be numeric" in error_message:
            return jsonify({'error': "Some features are not numeric. Please check your data and try again."})
        else:
            return jsonify({'error': f"An error occurred while training the {model_name} model: {error_message}"})

@app.route('/compare_models', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        model_results = data['results']
        
        # Create comparison plot
        plot_html = compare_models(model_results)
        
        return jsonify({
            'success': True,
            'plot': plot_html
        })
    except Exception as e:
        return jsonify({'error': str(e)})