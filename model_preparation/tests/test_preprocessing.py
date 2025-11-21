"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor, create_preprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = DataPreprocessor(
        drop_na_threshold=0.5,
        fill_na_strategy="median",
        remove_duplicates=True,
        normalize=True
    )
    
    assert preprocessor.drop_na_threshold == 0.5
    assert preprocessor.fill_na_strategy == "median"
    assert preprocessor.remove_duplicates is True
    assert preprocessor.normalize is True


def test_preprocessor_fit_transform():
    """Test fit_transform method."""
    # Create sample data
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [10, 20, 30, 40, 50],
        'feature_3': [100, 200, 300, 400, 500]
    })
    y = pd.Series([0, 0, 1, 1, 0])
    
    preprocessor = DataPreprocessor(normalize=True)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] == 3
    assert len(y_transformed) == 5
    assert preprocessor.is_fitted is True


def test_preprocessor_missing_values():
    """Test handling of missing values."""
    X = pd.DataFrame({
        'feature_1': [1, 2, np.nan, 4, 5],
        'feature_2': [10, 20, 30, np.nan, 50],
        'feature_3': [100, 200, 300, 400, 500]
    })
    y = pd.Series([0, 0, 1, 1, 0])
    
    preprocessor = DataPreprocessor(fill_na_strategy="mean", normalize=False)
    X_transformed, _ = preprocessor.fit_transform(X, y)
    
    # Check that no NaN values remain
    assert not np.isnan(X_transformed).any()


def test_preprocessor_drop_high_na_columns():
    """Test dropping columns with high missing value ratio."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [np.nan] * 5,  # 100% missing
        'feature_3': [100, 200, 300, 400, 500]
    })
    y = pd.Series([0, 0, 1, 1, 0])
    
    preprocessor = DataPreprocessor(drop_na_threshold=0.5, normalize=False)
    X_transformed, _ = preprocessor.fit_transform(X, y)
    
    # feature_2 should be dropped
    assert X_transformed.shape[1] == 2


def test_preprocessor_remove_duplicates():
    """Test duplicate removal."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 2, 4, 5],
        'feature_2': [10, 20, 20, 40, 50]
    })
    y = pd.Series([0, 0, 0, 1, 0])
    
    preprocessor = DataPreprocessor(remove_duplicates=True, normalize=False)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Should have fewer rows after removing duplicates
    assert X_transformed.shape[0] <= X.shape[0]


def test_create_preprocessor():
    """Test preprocessor creation from config."""
    config = {
        'drop_na_threshold': 0.3,
        'fill_na_strategy': 'median',
        'remove_duplicates': True,
        'normalize': True
    }
    
    preprocessor = create_preprocessor(config)
    
    assert preprocessor.drop_na_threshold == 0.3
    assert preprocessor.fill_na_strategy == 'median'

