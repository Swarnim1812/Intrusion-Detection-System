"""
Utility functions for logging, metrics, and helper functions.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.feature_utils import normalize_feature_name


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        format_string: Custom format string for log messages.
        log_file: Optional path to log file. If None, logs to console only.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )


def save_features(feature_names: list, filepath: str) -> None:
    """
    Save feature names to a JSON file.
    
    Args:
        feature_names: List of feature names.
        filepath: Path to save the JSON file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    cleaned = []
    seen = set()
    for name in feature_names or []:
        normalized = normalize_feature_name(name)
        if not normalized or normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2)
    
    logging.info(f"Saved {len(cleaned)} features to {filepath}")


def load_features(filepath: str) -> list:
    """
    Load feature names from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        List of feature names.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    return [normalize_feature_name(f) for f in features]


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model using joblib.
    
    Args:
        model: Trained model object.
        filepath: Path to save the model.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    logging.info(f"Saved model to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the model file.
        
    Returns:
        Loaded model object.
    """
    model = joblib.load(filepath)
    logging.info(f"Loaded model from {filepath}")
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC).
        
    Returns:
        Dictionary of metric names and values.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    if y_pred_proba is not None:
        try:
            # Handle binary and multiclass cases
            if y_pred_proba.ndim == 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='weighted'
                )
        except Exception as e:
            logging.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None
    
    return metrics


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate a text classification report.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Classification report as string.
    """
    return classification_report(y_true, y_pred, zero_division=0)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Confusion matrix as numpy array.
    """
    return confusion_matrix(y_true, y_pred)

