"""
Unit tests for models module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models import (
    create_model, create_decision_tree, create_random_forest,
    create_isolation_forest, get_hyperparameter_grid
)


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y


def test_create_decision_tree():
    """Test DecisionTree creation."""
    config = {
        'decision_tree': {
            'max_depth': 10,
            'min_samples_split': 5
        },
        'random_state': 42
    }
    
    model = create_decision_tree(config)
    
    assert model.max_depth == 10
    assert model.min_samples_split == 5
    assert model.random_state == 42


def test_create_random_forest():
    """Test RandomForest creation."""
    config = {
        'random_forest': {
            'n_estimators': 50,
            'max_depth': 15
        },
        'random_state': 42
    }
    
    model = create_random_forest(config)
    
    assert model.n_estimators == 50
    assert model.max_depth == 15
    assert model.random_state == 42


def test_create_isolation_forest():
    """Test IsolationForest creation."""
    config = {
        'isolation_forest': {
            'n_estimators': 50,
            'contamination': 0.1
        },
        'random_state': 42
    }
    
    model = create_isolation_forest(config)
    
    assert model.n_estimators == 50
    assert model.contamination == 0.1
    assert model.random_state == 42


def test_create_model_decision_tree():
    """Test create_model with DecisionTree."""
    config = {
        'decision_tree': {},
        'random_state': 42
    }
    
    model = create_model('DecisionTree', config)
    
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_create_model_random_forest():
    """Test create_model with RandomForest."""
    config = {
        'random_forest': {},
        'random_state': 42
    }
    
    model = create_model('RandomForest', config)
    
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_create_model_isolation_forest():
    """Test create_model with IsolationForest."""
    config = {
        'isolation_forest': {},
        'random_state': 42
    }
    
    model = create_model('IsolationForest', config)
    
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_create_model_invalid():
    """Test create_model with invalid model name."""
    config = {'random_state': 42}
    
    with pytest.raises(ValueError):
        create_model('InvalidModel', config)


def test_model_training(sample_data):
    """Test that models can be trained."""
    X, y = sample_data
    
    config = {
        'random_forest': {'n_estimators': 10},
        'random_state': 42
    }
    
    model = create_model('RandomForest', config)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    assert len(predictions) == len(y)
    assert predictions.dtype in [np.int32, np.int64] or predictions.dtype == float


def test_get_hyperparameter_grid():
    """Test hyperparameter grid generation."""
    # DecisionTree
    grid = get_hyperparameter_grid('DecisionTree', fast_mode=False)
    assert 'max_depth' in grid
    assert 'min_samples_split' in grid
    
    # RandomForest
    grid = get_hyperparameter_grid('RandomForest', fast_mode=True)
    assert 'n_estimators' in grid
    assert len(grid['n_estimators']) <= 2  # Fast mode should have fewer options
    
    # IsolationForest
    grid = get_hyperparameter_grid('IsolationForest', fast_mode=False)
    assert 'n_estimators' in grid
    assert 'contamination' in grid

