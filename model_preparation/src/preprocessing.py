"""
Data preprocessing module for Jharkhand-IDS.

Handles data cleaning, missing value imputation, and normalization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.feature_utils import normalize_feature_name

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessor that handles cleaning, imputation, and scaling.
    """
    
    def __init__(
        self,
        drop_na_threshold: float = 0.5,
        fill_na_strategy: str = "median",
        remove_duplicates: bool = True,
        normalize: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            drop_na_threshold: Drop columns with >threshold missing values.
            fill_na_strategy: Strategy for filling missing values (mean, median, mode, zero).
            remove_duplicates: Whether to remove duplicate rows.
            normalize: Whether to apply StandardScaler.
        """
        self.drop_na_threshold = drop_na_threshold
        self.fill_na_strategy = fill_na_strategy
        self.remove_duplicates = remove_duplicates
        self.normalize = normalize
        
        self.scaler = StandardScaler() if normalize else None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature DataFrame.
            y: Optional target Series.
            
        Returns:
            Self for method chaining.
        """
        X = X.copy()
        X.columns = [normalize_feature_name(col) for col in X.columns]
        
        # Store feature and target column info
        if y is not None:
            self.target_column = y.name if hasattr(y, 'name') else 'target'
        
        # Drop columns with too many missing values
        missing_ratio = X.isnull().sum() / len(X)
        cols_to_drop = missing_ratio[missing_ratio > self.drop_na_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{self.drop_na_threshold*100}% missing values")
            X = X.drop(columns=cols_to_drop)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Fit scaler if normalization is enabled
        if self.normalize and self.scaler is not None:
            self.scaler.fit(X)
            logger.info("Fitted StandardScaler")
        
        # Fit label encoder if target is provided
        if y is not None:
            self.label_encoder.fit(y)
            logger.info(f"Fitted LabelEncoder for {len(self.label_encoder.classes_)} classes")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame.
            y: Optional target Series.
            
        Returns:
            Tuple of (X_transformed, y_transformed).
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = X.copy()
        X.columns = [normalize_feature_name(col) for col in X.columns]
        
        # Select only the features that were used during fitting
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                logger.warning(f"Missing columns in transform: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0  # Fill with zeros
            
            X = X[self.feature_columns]
        
        # Fill missing values
        X = self._fill_missing_values(X)
        
        # Remove duplicates if enabled - but keep X and y aligned
        if self.remove_duplicates:
            n_before = len(X)
            # Create a combined dataframe to drop duplicates while maintaining alignment
            if y is not None:
                # Combine X and y to drop duplicates together
                combined = X.copy()
                combined['_temp_y'] = y.values if hasattr(y, 'values') else y
                combined = combined.drop_duplicates()
                # Separate back
                y_values = combined['_temp_y'].values
                combined = combined.drop(columns=['_temp_y'])
                X = combined
                if len(X) < n_before:
                    logger.info(f"Removed {n_before - len(X)} duplicate rows")
                    # Update y to match X
                    y = pd.Series(y_values, index=X.index[:len(y_values)])
            else:
                X = X.drop_duplicates()
                if len(X) < n_before:
                    logger.info(f"Removed {n_before - len(X)} duplicate rows")
        
        # Normalize if enabled
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        else:
            X = X.values
        
        # Transform target if provided - ensure alignment
        y_transformed = None
        if y is not None:
            # Ensure y has same length as X
            if len(y) != len(X):
                logger.warning(f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows. Truncating to match.")
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]
            y_transformed = self.label_encoder.transform(y)
        
        return X, y_transformed
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature DataFrame.
            y: Optional target Series.
            
        Returns:
            Tuple of (X_transformed, y_transformed).
        """
        return self.fit(X, y).transform(X, y)
    
    def _fill_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using the specified strategy.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            DataFrame with missing values filled.
        """
        if X.isnull().sum().sum() == 0:
            return X
        
        if self.fill_na_strategy == "mean":
            X = X.fillna(X.mean())
        elif self.fill_na_strategy == "median":
            X = X.fillna(X.median())
        elif self.fill_na_strategy == "mode":
            X = X.fillna(X.mode().iloc[0] if len(X.mode()) > 0 else 0)
        elif self.fill_na_strategy == "zero":
            X = X.fillna(0)
        else:
            logger.warning(f"Unknown fill strategy: {self.fill_na_strategy}, using zero")
            X = X.fillna(0)
        
        logger.info(f"Filled missing values using {self.fill_na_strategy} strategy")
        return X


def create_preprocessor(config: dict) -> DataPreprocessor:
    """
    Create a preprocessor from configuration.
    
    Args:
        config: Preprocessing configuration dictionary.
        
    Returns:
        Configured DataPreprocessor instance.
    """
    return DataPreprocessor(
        drop_na_threshold=config.get('drop_na_threshold', 0.5),
        fill_na_strategy=config.get('fill_na_strategy', 'median'),
        remove_duplicates=config.get('remove_duplicates', True),
        normalize=config.get('normalize', True)
    )

