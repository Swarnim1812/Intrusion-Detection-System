"""
Serving script for Jharkhand-IDS.

Provides a Streamlit web interface for model predictions.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import logging

from src.config import load_config
from src.utils import load_model, load_features, setup_logging
from src.preprocessing import create_preprocessor

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Jharkhand-IDS",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Jharkhand-IDS: Intrusion Detection System")
st.markdown("Upload network flow data or enter features manually to detect intrusions.")


@st.cache_resource
def load_artifacts(config_path: str = 'config/default.yaml'):
    """
    Load model, preprocessor, and features (cached).
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Tuple of (model, preprocessor, feature_names, config).
    """
    try:
        config = load_config(config_path)
        paths_config = config.get('paths', {})
        
        model_path = paths_config.get('model_file', 'artifacts/model.joblib')
        preprocessor_path = paths_config.get('preprocessor_file', 'artifacts/preprocessor.joblib')
        features_path = paths_config.get('features_file', 'artifacts/features.json')
        
        model = load_model(model_path)
        preprocessor = load_model(preprocessor_path)
        feature_names = load_features(features_path)
        
        return model, preprocessor, feature_names, config
    
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.info("Please ensure you have trained a model first using: `python -m src.train`")
        return None, None, None, None


def predict_single(model, preprocessor, feature_names, input_data: dict) -> dict:
    """
    Make prediction for a single sample.
    
    Args:
        model: Trained model.
        preprocessor: Fitted preprocessor.
        feature_names: List of feature names.
        input_data: Dictionary of feature values.
        
    Returns:
        Dictionary with prediction and confidence.
    """
    try:
        # Create DataFrame with single row
        df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0  # Fill missing with 0
        
        # Select only required features in correct order
        df = df[feature_names]
        
        # Preprocess
        X_processed, _ = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        
        # Get probabilities if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_processed)[0]
                confidence = float(np.max(proba))
            except:
                pass
        
        # Map prediction to label
        if hasattr(preprocessor, 'label_encoder'):
            label = preprocessor.label_encoder.inverse_transform([prediction])[0]
        else:
            label = "Anomaly" if prediction == -1 else "Normal"
        
        return {
            'prediction': int(prediction),
            'label': str(label),
            'confidence': confidence
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {'error': str(e)}


def predict_user_input(
    model,
    preprocessor,
    feature_names: List[str],
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make prediction on user-entered column values with explanation.
    
    Args:
        model: Trained model.
        preprocessor: Fitted preprocessor.
        feature_names: List of feature names.
        input_data: Dictionary of user-entered feature values.
        
    Returns:
        Dictionary with prediction, probability, and explanation.
    """
    try:
        # Create DataFrame with single row
        df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0  # Fill missing with 0
        
        # Select only required features in correct order
        df = df[feature_names]
        
        # Preprocess
        X_processed, _ = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        
        # Get probabilities if available
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
        if isinstance(prediction, (np.integer, int)):
            is_attack = (prediction == 1 or prediction == -1)
        else:
            is_attack = (prediction != 0)
        
        if is_attack:
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
        
        return {
            'result': result,
            'prob': probability,
            'explanation': explanation
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {'error': str(e)}


def predict_batch(model, preprocessor, feature_names, df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions for a batch of samples.
    
    Args:
        model: Trained model.
        preprocessor: Fitted preprocessor.
        feature_names: List of feature names.
        df: DataFrame with features.
        
    Returns:
        DataFrame with predictions added.
    """
    try:
        # Ensure all required features are present
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        # Select only required features
        df_features = df[feature_names]
        
        # Preprocess
        X_processed, _ = preprocessor.transform(df_features)
        
        # Predict
        predictions = model.predict(X_processed)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_processed)
                df['confidence'] = np.max(proba, axis=1)
            except:
                pass
        
        # Map predictions to labels
        if hasattr(preprocessor, 'label_encoder'):
            labels = preprocessor.label_encoder.inverse_transform(predictions)
        else:
            labels = ["Anomaly" if p == -1 else "Normal" for p in predictions]
        
        df['prediction'] = predictions
        df['label'] = labels
        
        return df
    
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return df


# Load artifacts
model, preprocessor, feature_names, config = load_artifacts()

if model is None:
    st.stop()

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.info(f"Model loaded: {config.get('model', {}).get('name', 'Unknown')}")
st.sidebar.info(f"Features: {len(feature_names)}")

# Main interface
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Single Sample Prediction")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Enter Feature Values")
        
        # Create input fields for first 10 features (for demo)
        input_data = {}
        n_features_show = min(10, len(feature_names))
        
        cols = st.columns(2)
        for i, feat in enumerate(feature_names[:n_features_show]):
            with cols[i % 2]:
                input_data[feat] = st.number_input(
                    feat,
                    value=0.0,
                    step=0.1,
                    key=f"input_{feat}"
                )
        
        if len(feature_names) > n_features_show:
            st.info(f"Note: Only showing first {n_features_show} features. Remaining will be set to 0.")
        
        submitted = st.form_submit_button("Predict", type="primary")
        
        if submitted:
            # Fill remaining features with 0
            for feat in feature_names[n_features_show:]:
                input_data[feat] = 0.0
            
            result = predict_single(model, preprocessor, feature_names, input_data)
            
            if 'error' not in result:
                st.success("Prediction completed!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", result['label'])
                with col2:
                    st.metric("Class", result['prediction'])
                if result['confidence'] is not None:
                    with col3:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
            else:
                st.error(f"Error: {result['error']}")

with tab2:
    st.header("Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with features",
        type=['csv'],
        help="CSV file should contain feature columns matching the model"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} samples")
            
            st.subheader("Preview")
            st.dataframe(df.head())
            
            if st.button("Run Predictions", type="primary"):
                with st.spinner("Processing..."):
                    df_results = predict_batch(model, preprocessor, feature_names, df.copy())
                    
                    st.success("Predictions completed!")
                    st.subheader("Results")
                    st.dataframe(df_results)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    if 'label' in df_results.columns:
                        st.subheader("Summary")
                        label_counts = df_results['label'].value_counts()
                        st.bar_chart(label_counts)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("**Jharkhand-IDS** - Intrusion Detection System ML Pipeline")

