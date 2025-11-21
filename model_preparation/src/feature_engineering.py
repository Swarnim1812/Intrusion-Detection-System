"""
Feature engineering module for Jharkhand-IDS.

Handles feature selection, variance thresholding, and feature transformations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Any
import logging
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

logger = logging.getLogger(__name__)


def remove_low_variance_features(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.01,
    selector: Optional[VarianceThreshold] = None
) -> Tuple[np.ndarray, List[str], VarianceThreshold]:
    """
    Remove features with low variance.
    
    Args:
        X: Feature array.
        feature_names: List of feature names.
        threshold: Variance threshold.
        selector: Optional pre-fitted VarianceThreshold selector (for evaluation).
        
    Returns:
        Tuple of (filtered_X, filtered_feature_names, fitted_selector).
    """
    if threshold <= 0:
        return X, feature_names, None
    
    if selector is None:
        # Fit on training data
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
    else:
        # Use pre-fitted selector for evaluation
        X_filtered = selector.transform(X)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    filtered_names = [feature_names[i] for i in selected_indices]
    
    n_removed = len(feature_names) - len(filtered_names)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} low-variance features (threshold={threshold})")
    
    return X_filtered, filtered_names, selector


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: int = 50,
    score_func: callable = f_classif
) -> Tuple[np.ndarray, List[str]]:
    """
    Select top K features using univariate statistical tests.
    
    Args:
        X: Feature array.
        y: Target array.
        feature_names: List of feature names.
        n_features: Number of features to select.
        score_func: Scoring function for feature selection.
        
    Returns:
        Tuple of (selected_X, selected_feature_names).
    """
    n_features = min(n_features, X.shape[1])
    
    selector = SelectKBest(score_func=score_func, k=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_names = [feature_names[i] for i in selected_indices]
    
    logger.info(f"Selected top {n_features} features from {len(feature_names)} original features")
    
    return X_selected, selected_names


def engineer_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: dict,
    variance_selector: Optional[VarianceThreshold] = None,
    feature_selector: Optional[Any] = None
) -> Tuple[np.ndarray, List[str], Optional[VarianceThreshold], Optional[Any]]:
    """
    Apply feature engineering pipeline.
    
    Args:
        X: Feature array.
        y: Target array.
        feature_names: List of feature names.
        config: Feature engineering configuration.
        variance_selector: Optional pre-fitted VarianceThreshold (for evaluation).
        feature_selector: Optional pre-fitted feature selector (for evaluation).
        
    Returns:
        Tuple of (engineered_X, engineered_feature_names, variance_selector, feature_selector).
    """
    X_engineered = X
    names_engineered = feature_names
    
    # Remove low variance features
    if config.get('remove_low_variance', True):
        variance_threshold = config.get('variance_threshold', 0.01)
        X_engineered, names_engineered, variance_selector = remove_low_variance_features(
            X_engineered, names_engineered, threshold=variance_threshold, selector=variance_selector
        )
    
    # Feature selection
    if config.get('feature_selection', False):
        n_features = config.get('n_features', 50)
        if feature_selector is None:
            X_engineered, names_engineered = select_features(
                X_engineered, y, names_engineered, n_features=n_features
            )
            # Note: select_features doesn't return selector yet, would need to modify
        else:
            # Use pre-fitted selector
            X_engineered = feature_selector.transform(X_engineered)
            selected_indices = feature_selector.get_support(indices=True)
            names_engineered = [names_engineered[i] for i in selected_indices]
    
    logger.info(f"Feature engineering complete: {len(names_engineered)} features")
    
    return X_engineered, names_engineered, variance_selector, feature_selector

