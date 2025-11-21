"""
Model definitions for Jharkhand-IDS.

Supports DecisionTree, RandomForest, and IsolationForest for intrusion detection.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, cross_val_score

logger = logging.getLogger(__name__)


def create_decision_tree(config: dict) -> DecisionTreeClassifier:
    """
    Create a DecisionTree classifier from configuration.
    
    Args:
        config: Model configuration dictionary.
        
    Returns:
        Configured DecisionTreeClassifier.
    """
    dt_config = config.get('decision_tree', {})
    
    model = DecisionTreeClassifier(
        max_depth=dt_config.get('max_depth', 20),
        min_samples_split=dt_config.get('min_samples_split', 5),
        min_samples_leaf=dt_config.get('min_samples_leaf', 2),
        criterion=dt_config.get('criterion', 'gini'),
        random_state=config.get('random_state', 42)
    )
    
    logger.info("Created DecisionTree classifier")
    return model


def create_random_forest(config: dict) -> RandomForestClassifier:
    """
    Create a RandomForest classifier from configuration.
    
    Args:
        config: Model configuration dictionary.
        
    Returns:
        Configured RandomForestClassifier.
    """
    rf_config = config.get('random_forest', {})
    
    model = RandomForestClassifier(
        n_estimators=rf_config.get('n_estimators', 100),
        max_depth=rf_config.get('max_depth', 20),
        min_samples_split=rf_config.get('min_samples_split', 5),
        min_samples_leaf=rf_config.get('min_samples_leaf', 2),
        criterion=rf_config.get('criterion', 'gini'),
        max_features=rf_config.get('max_features', 'sqrt'),
        n_jobs=rf_config.get('n_jobs', -1),
        random_state=config.get('random_state', 42)
    )
    
    logger.info("Created RandomForest classifier")
    return model


def create_isolation_forest(config: dict) -> IsolationForest:
    """
    Create an IsolationForest for anomaly detection.
    
    Args:
        config: Model configuration dictionary.
        
    Returns:
        Configured IsolationForest.
    """
    if_config = config.get('isolation_forest', {})
    
    model = IsolationForest(
        n_estimators=if_config.get('n_estimators', 100),
        contamination=if_config.get('contamination', 0.1),
        max_features=if_config.get('max_features', 1.0),
        n_jobs=if_config.get('n_jobs', -1),
        random_state=config.get('random_state', 42)
    )
    
    logger.info("Created IsolationForest for anomaly detection")
    return model


def create_model(model_name: str, config: dict):
    """
    Create a model instance based on the model name.
    
    Args:
        model_name: Name of the model ('DecisionTree', 'RandomForest', 'IsolationForest').
        config: Model configuration dictionary.
        
    Returns:
        Model instance.
        
    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = model_name.lower()
    
    if model_name == 'decisiontree':
        return create_decision_tree(config)
    elif model_name == 'randomforest':
        return create_random_forest(config)
    elif model_name == 'isolationforest':
        return create_isolation_forest(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_hyperparameter_grid(model_name: str, fast_mode: bool = False) -> Dict[str, list]:
    """
    Get hyperparameter grid for grid search.
    
    Args:
        model_name: Name of the model.
        fast_mode: If True, return smaller grid for faster search.
        
    Returns:
        Dictionary of hyperparameter grids.
    """
    model_name = model_name.lower()
    
    if model_name == 'decisiontree':
        if fast_mode:
            return {
                'max_depth': [10, 20],
                'min_samples_split': [5, 10]
            }
        return {
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif model_name == 'randomforest':
        if fast_mode:
            return {
                'n_estimators': [50, 100],
                'max_depth': [10, 20]
            }
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    
    elif model_name == 'isolationforest':
        if fast_mode:
            return {
                'n_estimators': [50, 100],
                'contamination': [0.05, 0.1]
            }
        return {
            'n_estimators': [50, 100, 200],
            'contamination': [0.05, 0.1, 0.15]
        }
    
    return {}


def perform_grid_search(
    model,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, list],
    cv: int = 5,
    scoring: str = 'f1_weighted',
    n_jobs: int = -1
) -> Any:
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        model: Model instance.
        X: Training features.
        y: Training labels.
        param_grid: Hyperparameter grid.
        cv: Number of cross-validation folds.
        scoring: Scoring metric.
        n_jobs: Number of parallel jobs.
        
    Returns:
        Best model from grid search.
    """
    if not param_grid:
        logger.info("No hyperparameter grid provided, skipping grid search")
        return model
    
    logger.info(f"Performing grid search with {cv}-fold CV")
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'f1_weighted'
) -> np.ndarray:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model instance.
        X: Training features.
        y: Training labels.
        cv: Number of cross-validation folds.
        scoring: Scoring metric.
        
    Returns:
        Array of cross-validation scores.
    """
    logger.info(f"Performing {cv}-fold cross-validation")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    logger.info(f"CV {scoring} scores: {scores}")
    logger.info(f"Mean CV {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores

