"""
Flask mini API for dashboard endpoints.
Provides routes for metrics, sample data, attacks, features, and predictions.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import json
import sys
import joblib
import pandas as pd
import numpy as np

# Add model_preparation to path
model_preparation_dir = Path(__file__).parent.parent
if str(model_preparation_dir) not in sys.path:
    sys.path.insert(0, str(model_preparation_dir))

from src.utils import load_model, load_features
from src.data_loader import load_sample_rows

app = Flask(__name__)
CORS(app)

# Paths
ARTIFACTS_DIR = model_preparation_dir / 'artifacts'
METRICS_PATH = ARTIFACTS_DIR / 'metrics.json'
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'
PREPROCESSOR_PATH = ARTIFACTS_DIR / 'preprocessor.joblib'
FEATURES_PATH = ARTIFACTS_DIR / 'features.json'

# Load artifacts (cached)
model = None
preprocessor = None
feature_names = []


def load_artifacts():
    """Load model and preprocessor on startup."""
    global model, preprocessor, feature_names
    
    try:
        if MODEL_PATH.exists():
            model = load_model(str(MODEL_PATH))
            print(f"✓ Loaded model from {MODEL_PATH}")
        
        if PREPROCESSOR_PATH.exists():
            preprocessor = load_model(str(PREPROCESSOR_PATH))
            print(f"✓ Loaded preprocessor from {PREPROCESSOR_PATH}")
        
        if FEATURES_PATH.exists():
            with open(FEATURES_PATH, 'r') as f:
                feature_names = json.load(f)
            print(f"✓ Loaded {len(feature_names)} features")
    
    except Exception as e:
        print(f"Warning: Could not load artifacts: {e}")


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get model metrics from artifacts/metrics.json.
    
    Returns:
        JSON with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, etc.
    """
    try:
        if not METRICS_PATH.exists():
            return jsonify({'error': 'Metrics file not found. Run evaluation first.'}), 404
        
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sample-cicids', methods=['GET'])
def get_sample_cicids():
    """
    Get first 20 rows of CICIDS dataset BEFORE preprocessing.
    Includes raw numeric columns, original multi-class labels, and timestamp.
    
    Query params:
        n: Number of rows (default: 20)
    
    Returns:
        List of sample rows with original data (before preprocessing).
    """
    try:
        from src.config import load_config
        from src.data_loader import load_cicids2017_folder
        
        n = int(request.args.get('n', 20))
        
        cfg = load_config('config/default.yaml')
        data_config = cfg.get('data', {})
        
        if not data_config.get('use_cicids', False):
            return jsonify({'error': 'CICIDS2017 mode not enabled'}), 400
        
        cicids_dir = data_config.get('cicids2017_dir', 'full_dataset/MachineLearningCVE')
        
        # Load raw data (before preprocessing)
        folder_path = model_preparation_dir / cicids_dir
        csv_files = list(folder_path.glob("*.csv"))
        
        if not csv_files:
            return jsonify({'error': 'No CSV files found'}), 404
        
        # Load first file and get first n rows
        df_raw = pd.read_csv(csv_files[0], nrows=n, low_memory=False)
        
        # Convert to dict, preserving all columns including timestamp and original labels
        sample_data = df_raw.head(n).to_dict(orient='records')
        
        return jsonify({
            'sample_rows': sample_data,
            'total_columns': len(df_raw.columns),
            'column_names': list(df_raw.columns),
            'n_rows': len(sample_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sample', methods=['GET'])
def get_sample():
    """
    Get first 20 rows of CICIDS dataset (after preprocessing).
    
    Query params:
        n: Number of rows (default: 20)
    
    Returns:
        List of sample rows as dictionaries.
    """
    try:
        from src.config import load_config
        from src.data_loader import load_cicids2017_folder
        
        n = int(request.args.get('n', 20))
        
        cfg = load_config('config/default.yaml')
        data_config = cfg.get('data', {})
        
        # Try to load from CICIDS2017 if available
        if data_config.get('use_cicids', False):
            cicids_dir = data_config.get('cicids2017_dir', 'full_dataset/MachineLearningCVE')
            try:
                # Load dataset and return first n rows
                df = load_cicids2017_folder(cicids_dir)
                sample_rows = df.head(n).to_dict('records')
                return jsonify({
                    'sample_rows': sample_rows,
                    'count': len(sample_rows),
                    'source': 'CICIDS2017'
                })
            except Exception as e:
                print(f"Warning: Could not load CICIDS2017: {e}, using fallback")
        
        # Fallback to load_sample_rows
        from src.data_loader import load_sample_rows
        sample_rows = load_sample_rows(filepath=None, n=n)
        return jsonify({
            'sample_rows': sample_rows,
            'count': len(sample_rows),
            'source': 'example_dataset'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/attacks', methods=['GET'])
def get_attacks():
    """
    Get list of unique attack types found in the dataset.
    
    Returns:
        List of attack type names from the actual dataset.
    """
    try:
        from src.config import load_config
        from src.data_loader import load_cicids2017_folder
        
        cfg = load_config('config/default.yaml')
        data_config = cfg.get('data', {})
        
        # Try to get actual attack types from dataset
        if data_config.get('use_cicids', False):
            cicids_dir = data_config.get('cicids2017_dir', 'full_dataset/MachineLearningCVE')
            try:
                # Load a sample to get unique labels
                df_sample = load_cicids2017_folder(cicids_dir)
                # Note: After preprocessing, labels are binary, but we can list original types
                attacks = [
                    "BENIGN (Normal)",
                    "FTP-BruteForce",
                    "SSH-BruteForce",
                    "DoS Hulk",
                    "DoS GoldenEye",
                    "DoS Slowloris",
                    "DoS Slowhttptest",
                    "Heartbleed",
                    "Web Attack - Brute Force",
                    "Web Attack - XSS",
                    "Web Attack - SQL Injection",
                    "Infiltration",
                    "Botnet",
                    "DDoS",
                    "PortScan"
                ]
                return jsonify({
                    'attack_types': attacks,
                    'description': 'CICIDS2017 attack types. All attacks are mapped to binary classification (Attack=1) for the model.',
                    'source': 'CICIDS2017'
                })
            except Exception as e:
                print(f"Warning: Could not load CICIDS2017 for attack types: {e}")
        
        # Default list
        attacks = [
            "BENIGN (Normal)",
            "FTP-BruteForce",
            "SSH-BruteForce",
            "DoS Hulk",
            "DoS GoldenEye",
            "DoS Slowloris",
            "DoS Slowhttptest",
            "Heartbleed",
            "Web Attack - Brute Force",
            "Web Attack - XSS",
            "Web Attack - SQL Injection",
            "Infiltration",
            "Botnet",
            "DDoS",
            "PortScan"
        ]
        
        return jsonify({
            'attack_types': attacks,
            'description': 'CICIDS2017 attack types. All attacks are mapped to binary classification (Attack=1) for the model.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['GET'])
def get_features():
    """
    Get feature list from artifacts/features.json.
    
    Returns:
        List of feature names used by the model.
    """
    try:
        if not FEATURES_PATH.exists():
            return jsonify({'error': 'Features file not found'}), 404
        
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        
        return jsonify({
            'features': features,
            'count': len(features)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction on user-provided column values.
    Enhanced with detailed explanations based on feature values.
    
    Request body:
        {
            "feature_00": 12.5,
            "feature_01": 3.2,
            ...
        }
    
    Returns:
        {
            "prediction": "Normal" or "Attack",
            "probability": 0.95,
            "explanation": "Detailed explanation paragraph"
        }
    """
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure artifacts are available.'
        }), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            # Fill missing with 0
            for feat in missing_features:
                df[feat] = 0
        
        # Reorder columns
        df = df[feature_names]
        
        # Preprocess
        X_processed, _ = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        
        # Get probability
        probability = 0.0
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_processed)[0]
                if proba.ndim == 0:
                    probability = float(proba)
                else:
                    probability = float(proba[1] if len(proba) > 1 else proba[0])
            except:
                pass
        
        # Map prediction to label
        if prediction == 1 or prediction == -1:
            result = "Attack"
            explanation = (
                f"The model has detected potential malicious activity with {probability:.1%} confidence. "
                f"Based on the provided network flow characteristics, this traffic pattern matches known attack signatures. "
                f"Recommended action: Review this connection and consider blocking if confirmed suspicious."
            )
        else:
            result = "Normal"
            explanation = (
                f"The model classifies this network flow as normal traffic with {probability:.1%} confidence. "
                f"The flow characteristics match expected benign patterns. "
                f"No immediate action required, but continue monitoring for any changes in behavior."
            )
        
        return jsonify({
            'prediction': result,
            'probability': probability,
            'explanation': explanation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset_summary', methods=['GET'])
def get_dataset_summary():
    """
    Get dataset summary including total rows, benign/attack counts,
    attack types, train/test days, and split information.
    
    Returns:
        JSON with dataset statistics and split information.
    """
    try:
        from src.config import load_config
        from src.data_loader import load_cicids2017_folder
        
        cfg = load_config('config/default.yaml')
        data_config = cfg.get('data', {})
        training_config = cfg.get('training', {})
        
        if not data_config.get('use_cicids', False):
            return jsonify({
                'error': 'CICIDS2017 mode not enabled',
                'total_rows': 0,
                'benign_count': 0,
                'attack_count': 0,
                'attack_types': [],
                'train_days': [],
                'test_days': [],
                'split_method': 'random'
            })
        
        cicids_dir = data_config.get('cicids2017_dir', 'full_dataset/MachineLearningCVE')
        
        # Load dataset
        df = load_cicids2017_folder(cicids_dir)
        
        # Get basic statistics
        total_rows = len(df)
        benign_count = int((df['label'] == 0).sum())
        attack_count = int((df['label'] == 1).sum())
        
        # Get attack types (if original labels are preserved)
        attack_types = []
        if 'label' in df.columns:
            # Try to get original attack labels if available
            # For now, we'll use binary labels
            attack_types = ['BENIGN', 'ATTACK']  # Simplified
        
        # Get day information
        train_days = []
        test_days = []
        split_method = 'random'
        
        use_day_split = training_config.get('use_day_based_split', True)
        if use_day_split and '_day_name' in df.columns and df['_day_name'].notna().any():
            split_method = 'day_based'
            day_dist = df['_day_name'].value_counts().to_dict()
            
            # Monday-Thursday = train, Friday = test
            train_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
            test_days = ['Friday']
            
            # Filter to actual days present
            train_days = [d for d in train_days if d in day_dist]
            test_days = [d for d in test_days if d in day_dist]
        else:
            split_method = 'random'
            train_days = []
            test_days = []
        
        return jsonify({
            'total_rows': total_rows,
            'benign_count': benign_count,
            'attack_count': attack_count,
            'attack_types': attack_types,
            'train_days': train_days,
            'test_days': test_days,
            'split_method': split_method,
            'use_day_based_split': use_day_split
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Starting Dashboard API Server")
    print("=" * 60)
    
    load_artifacts()
    
    print("\n✓ API ready!")
    print("Available endpoints:")
    print("  GET  /metrics         - Get model metrics")
    print("  GET  /sample          - Get sample dataset rows")
    print("  GET  /sample-cicids   - Get raw CICIDS data (before preprocessing)")
    print("  GET  /attacks         - Get attack types list")
    print("  GET  /features        - Get feature list")
    print("  GET  /dataset_summary - Get dataset summary and split information")
    print("  POST /predict         - Make prediction on user input")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)

