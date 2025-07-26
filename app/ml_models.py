import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import traceback
import matplotlib.pyplot as plt

def preprocess_features(X, preprocessing_method='standard'):
    """
    Preprocess features using the specified method
    
    Parameters:
    -----------
    X : numpy.ndarray
        The feature matrix to preprocess
    preprocessing_method : str
        Method to preprocess features ('standard', 'minmax', 'normalize', or 'none')
    """
    # Check if X is empty
    if X.size == 0:
        raise ValueError("Empty feature matrix")
    
    # Check for non-numeric values
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"Features must be numeric, got {X.dtype}")
    
    # Check for infinite values
    if np.any(np.isinf(X)):
        print("Warning: Data contains infinite values, replacing with large finite values")
        X = np.nan_to_num(X, posinf=1e10, neginf=-1e10)
    
    # Apply the specified preprocessing method
    if preprocessing_method == 'standard':
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        print(f"Applied StandardScaler: mean={np.mean(X_processed):.2f}, std={np.std(X_processed):.2f}")
    elif preprocessing_method == 'minmax':
        scaler = MinMaxScaler()
        X_processed = scaler.fit_transform(X)
        print(f"Applied MinMaxScaler: min={np.min(X_processed):.2f}, max={np.max(X_processed):.2f}")
    elif preprocessing_method == 'normalize':
        X_processed = normalize(X)
        print(f"Applied normalization: min={np.min(X_processed):.2f}, max={np.max(X_processed):.2f}")
    elif preprocessing_method == 'none':
        X_processed = X
        print("No preprocessing applied")
    else:
        raise ValueError(f"Unknown preprocessing method: {preprocessing_method}")
    
    # Final check for NaN values
    if np.isnan(X_processed).any():
        print("Warning: NaN values detected after preprocessing, replacing with zeros")
        X_processed = np.nan_to_num(X_processed, nan=0.0)
    
    return X_processed

def process_data(dataset, target_column, preprocessing_method='standard', test_size=0.2):
    """
    Process the dataset with configurable options
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input dataset
    target_column : str
        The name of the target column for classification
    preprocessing_method : str
        Method to preprocess features ('standard', 'minmax', 'normalize', or 'none')
    test_size : float
        Proportion of the dataset to include in the test split
    """
    try:
        # Validate dataset
        if dataset.empty:
            raise ValueError("Empty dataset")
        
        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Make a copy to avoid modifying the original
        df = dataset.copy()
        
        # Print dataset information for debugging
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types: {df.dtypes}")
        
        # Handle missing values based on data type
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in numeric column '{col}' with median: {median_val}")
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in categorical column '{col}' with mode: {mode_val}")
        
        # Encode categorical features
        le_dict = {}
        for column in df.columns:
            if column != target_column and df[column].dtype == 'object':
                try:
                    # Try to convert to numeric first
                    df[column] = pd.to_numeric(df[column], errors='ignore')
                    if df[column].dtype == 'object':
                        # If still object, use label encoding
                        le = LabelEncoder()
                        df[column] = le.fit_transform(df[column].astype(str))
                        le_dict[column] = le
                        print(f"Encoded categorical column '{column}' using LabelEncoder")
                except Exception as e:
                    print(f"Error processing column '{column}': {str(e)}")
                    raise
        
        # Encode target column
        try:
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column].astype(str))
            print(f"Encoded target column '{target_column}' using LabelEncoder")
        except Exception as e:
            print(f"Error encoding target column: {str(e)}")
            raise
        
        # Check for class balance
        class_counts = df[target_column].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # Check for class imbalance
        if len(class_counts) < 2:
            raise ValueError(f"Not enough classes in the dataset. Found only {len(class_counts)} class.")
        
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        if min_class_count < 5:
            print(f"Warning: Some classes have very few samples (minimum: {min_class_count})")
        
        # Split features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Check for non-numeric values in features
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"Features must be numeric after preprocessing, got {X.dtype}")
        
        # Check for constant features
        constant_features = np.where(np.std(X, axis=0) == 0)[0]
        if len(constant_features) > 0:
            print(f"Warning: Found {len(constant_features)} constant features. They will be removed.")
            X = np.delete(X, constant_features, axis=1)
        
        # Preprocess features
        X = preprocess_features(X, preprocessing_method)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        print(f"Data split: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        print(traceback.format_exc())
        raise

def train_model(model_name, X_train, X_test, y_train, y_test, model_params=None):
    """
    Train a model with configurable parameters
    """
    try:
        # Validate input data
        if X_train.size == 0 or X_test.size == 0 or y_train.size == 0 or y_test.size == 0:
            raise ValueError("Empty input data")
            
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("Input contains NaN values")
        
        # Print data information for debugging
        print(f"Training {model_name} with data shapes: X_train {X_train.shape}, X_test {X_test.shape}")
        print(f"Feature ranges: min={X_train.min()}, max={X_train.max()}")
        print(f"Class distribution in training data: {np.bincount(y_train)}")
        
        # Set default parameters if none provided
        if model_params is None:
            model_params = {}
            
        # Initialize the appropriate model with parameters
        if model_name == 'knn':
            # For KNN, adjust parameters based on data size
            n_samples = X_train.shape[0]
            default_params = {
                'n_neighbors': min(5, max(3, int(np.sqrt(n_samples)))),
                'weights': 'distance',
                'n_jobs': -1
            }
            params = {**default_params, **model_params}
            model = KNeighborsClassifier(**params)
            
        elif model_name == 'nb':
            # For Naive Bayes, ensure data is suitable
            if np.any(X_train < 0) or np.any(X_test < 0):
                print("Warning: Data contains negative values, applying MinMaxScaler for Naive Bayes")
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Check class distribution
            class_counts = np.bincount(y_train)
            if len(class_counts) < 2:
                raise ValueError(f"Not enough classes in the dataset. Found only {len(class_counts)} class.")
            
            # Check for class imbalance
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            if min_class_count < 5:
                print(f"Warning: Some classes have very few samples (minimum: {min_class_count})")
            
            # Use default parameters for Naive Bayes
            model = GaussianNB(**model_params)
            
        elif model_name == 'rf':
            # For Random Forest, adjust parameters based on data characteristics
            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            
            # Suggest parameters based on data characteristics
            suggested_params = {
                'n_estimators': 100,
                'max_depth': min(20, max(10, int(np.log2(n_samples)))),
                'min_samples_split': max(2, int(n_samples * 0.01)),
                'min_samples_leaf': max(1, int(n_samples * 0.005)),
                'random_state': 42,
                'n_jobs': -1,  # Use all available CPU cores
                'class_weight': 'balanced'  # Handle class imbalance
            }
            
            # If no parameters provided, use suggested ones
            if model_params is None:
                model_params = suggested_params
            else:
                # Merge with suggested parameters, keeping user-provided ones if they exist
                for key, value in suggested_params.items():
                    if key not in model_params:
                        model_params[key] = value
                
            model = RandomForestClassifier(**model_params)
            
        elif model_name == 'svm':
            # For SVM, adjust parameters based on data size
            n_samples = X_train.shape[0]
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42,
                'class_weight': 'balanced'  # Handle class imbalance
            }
            params = {**default_params, **model_params}
            model = SVC(**params)
            
        elif model_name == 'xgboost':
            # For XGBoost, adjust parameters based on data characteristics
            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            
            # Suggest parameters based on data characteristics
            suggested_params = {
                'n_estimators': 100,
                'max_depth': min(10, max(5, int(np.log2(n_samples)))),
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,  # Use all available CPU cores
                'scale_pos_weight': 1,  # Handle class imbalance
                'tree_method': 'hist'  # Use histogram-based algorithm for better performance
            }
            
            # If no parameters provided, use suggested ones
            if model_params is None:
                model_params = suggested_params
            else:
                # Merge with suggested parameters, keeping user-provided ones if they exist
                for key, value in suggested_params.items():
                    if key not in model_params:
                        model_params[key] = value
                
            model = xgb.XGBClassifier(**model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train the model
        print(f"Training {model_name} with parameters: {model_params}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions) * 100,
            'precision': precision_score(y_test, predictions, average='macro', zero_division=0) * 100,
            'recall': recall_score(y_test, predictions, average='macro', zero_division=0) * 100,
            'f1': f1_score(y_test, predictions, average='macro', zero_division=0) * 100
        }
        
        print(f"{model_name} results: {metrics}")
        return metrics
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        print(traceback.format_exc())
        raise

def compare_models(model_results):
    """
    Create comparison visualizations for model results using Matplotlib
    """
    try:
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle('Model Performance Comparison', fontsize=16, y=0.95)
        
        algorithms = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # First subplot: Accuracy comparison
        accuracies = [model_results[alg]['accuracy'] for alg in algorithms]
        bars = ax1.bar(algorithms, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Second subplot: All metrics comparison
        colors = ['blue', 'red', 'green', 'purple']
        markers = ['o', 's', '^', 'D']
        for i, metric in enumerate(metrics):
            values = [model_results[alg][metric] for alg in algorithms]
            ax2.plot(algorithms, values, 
                    marker=markers[i], 
                    color=colors[i], 
                    label=metric.capitalize(),
                    linewidth=2,
                    markersize=8)
        
        ax2.set_title('All Metrics Comparison')
        ax2.set_ylabel('Score (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Third subplot: Model performance heatmap
        metrics_matrix = np.zeros((len(algorithms), len(metrics)))
        for i, alg in enumerate(algorithms):
            for j, metric in enumerate(metrics):
                metrics_matrix[i, j] = model_results[alg][metric]
        
        im = ax3.imshow(metrics_matrix, cmap='YlGnBu', aspect='auto')
        ax3.set_title('Model Performance Heatmap')
        
        # Set ticks and labels
        ax3.set_xticks(np.arange(len(metrics)))
        ax3.set_yticks(np.arange(len(algorithms)))
        ax3.set_xticklabels([m.capitalize() for m in metrics])
        ax3.set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(metrics)):
                text = ax3.text(j, i, f'{metrics_matrix[i, j]:.1f}%',
                              ha="center", va="center", color="black")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label('Score (%)')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plot_data
        
    except Exception as e:
        print(f"Error in compare_models: {str(e)}")
        print(traceback.format_exc())
        raise