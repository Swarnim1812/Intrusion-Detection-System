"""
Training script for Jharkhand-IDS.

CLI entrypoint for training models with config-driven approach.
"""

import click
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data_loader import load_data, load_cicids2017_folder
from src.preprocessing import create_preprocessor
from src.feature_engineering import engineer_features
from src.models import create_model, perform_grid_search, get_hyperparameter_grid, cross_validate_model
from src.utils import (
    setup_logging, save_model, save_features
)
from src.feature_utils import normalize_feature_name
from sklearn.utils import resample

logger = logging.getLogger(__name__)


@click.command()
@click.option('--config', default='config/default.yaml', help='Path to configuration file')
@click.option('--fast', is_flag=True, help='Fast mode: reduced dataset and model complexity')
def train(config: str, fast: bool):
    """
    Train an intrusion detection model.
    
    Example:
        python -m src.train --config config/default.yaml
        python -m src.train --config config/default.yaml --fast
    """
    # Load configuration
    cfg = load_config(config)
    
    # Setup logging
    log_config = cfg.get('logging', {})
    setup_logging(
        level=log_config.get('level', 'INFO'),
        format_string=log_config.get('format'),
        log_file=log_config.get('file')
    )
    
    logger.info("=" * 60)
    logger.info("Starting Jharkhand-IDS Training Pipeline")
    logger.info("=" * 60)
    
    # Override fast mode if flag is set
    if fast:
        cfg['training']['fast_mode'] = True
        logger.info("Fast mode enabled")
    
    # Load data
    data_config = cfg.get('data', {})
    use_cicids = data_config.get('use_cicids', False)
    
    if use_cicids:
        # CICIDS2017 mode: load from folder
        cicids_dir = data_config.get('cicids2017_dir', 'full_dataset/MachineLearningCVE')
        logger.info(f"CICIDS2017 mode enabled, loading from {cicids_dir}")
        
        # Load full CICIDS2017 dataset
        full_df = load_cicids2017_folder(cicids_dir)
        
        # Show class distribution BEFORE balancing
        logger.info("Class distribution BEFORE balancing:")
        class_dist_before = full_df['label'].value_counts().to_dict()
        for label, count in class_dist_before.items():
            logger.info(f"  Label {label}: {count:,} samples ({count/len(full_df)*100:.2f}%)")
        
        # DAY-BASED SPLIT (prevents data leakage)
        # IMPORTANT: Do NOT shuffle CICIDS2017 data randomly - it causes data leakage
        # Instead, split by days: Monday-Thursday (train) vs Friday (test)
        training_config = cfg.get('training', {})
        use_day_split = training_config.get('use_day_based_split', True)
        
        if use_day_split and '_day_name' in full_df.columns and full_df['_day_name'].notna().any():
            logger.info("=" * 60)
            logger.info("Using DAY-BASED SPLIT (prevents data leakage)")
            logger.info("=" * 60)
            
            # Get day distribution
            day_dist = full_df['_day_name'].value_counts().to_dict()
            logger.info(f"Day distribution: {day_dist}")
            
            # Split: Monday-Thursday = train, Friday = test
            # dayofweek: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
            train_mask = full_df['_day_of_week'].isin([0, 1, 2, 3])  # Mon-Thu
            test_mask = full_df['_day_of_week'] == 4  # Friday
            
            train_df = full_df[train_mask].copy()
            test_df = full_df[test_mask].copy()
            
            # Remove day columns from train/test (they're metadata, not features)
            for col in ['_day_of_week', '_date', '_day_name', '_source_file']:
                if col in train_df.columns:
                    train_df = train_df.drop(columns=[col])
                if col in test_df.columns:
                    test_df = test_df.drop(columns=[col])
            
            logger.info(f"Day-based split:")
            logger.info(f"  Train (Mon-Thu): {len(train_df):,} samples")
            logger.info(f"  Test (Friday): {len(test_df):,} samples")
            logger.info(f"  Train class distribution: {train_df['label'].value_counts().to_dict()}")
            logger.info(f"  Test class distribution: {test_df['label'].value_counts().to_dict()}")
        else:
            # Fallback to random split (NOT recommended for CICIDS2017)
            logger.warning("=" * 60)
            logger.warning("WARNING: Using RANDOM SPLIT (may cause data leakage)")
            logger.warning("For realistic evaluation, use day-based split (use_day_based_split: true)")
            logger.warning("=" * 60)
            model_config = cfg.get('model', {})
            full_df = full_df.sample(frac=1, random_state=model_config.get('random_state', 42)).reset_index(drop=True)
            
            # Remove day columns if present
            for col in ['_day_of_week', '_date', '_day_name', '_source_file']:
                if col in full_df.columns:
                    full_df = full_df.drop(columns=[col])
            
            test_size = model_config.get('test_size', 0.2)
            train_df, test_df = train_test_split(
                full_df,
                test_size=test_size,
                random_state=model_config.get('random_state', 42),
                stratify=full_df['label']
            )
            logger.info(f"Random split: {len(train_df):,} train, {len(test_df):,} test")
        
        # Apply class balancing strategy
        balance_strategy = training_config.get('balance_strategy', 'none')
        # Fallback to old balance_classes if balance_strategy not set
        if balance_strategy == 'none' and training_config.get('balance_classes', False):
            balance_strategy = 'undersample'  # Default behavior of old code
            logger.warning("Using deprecated 'balance_classes' config. Set 'balance_strategy' instead.")
        
        if balance_strategy != 'none':
            logger.info(f"Applying class balancing strategy: {balance_strategy}")
            df_normal = train_df[train_df['label'] == 0].copy()
            df_attack = train_df[train_df['label'] == 1].copy()
            
            if balance_strategy == 'undersample':
                # Undersample majority class to match minority
                min_samples = min(len(df_normal), len(df_attack))
                if len(df_normal) > len(df_attack):
                    df_normal = resample(df_normal, n_samples=min_samples, random_state=42)
                    logger.info(f"Undersampled normal class to {min_samples:,} samples")
                else:
                    df_attack = resample(df_attack, n_samples=min_samples, random_state=42)
                    logger.info(f"Undersampled attack class to {min_samples:,} samples")
                train_df = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
            
            elif balance_strategy == 'oversample':
                # Oversample minority class to match majority
                max_samples = max(len(df_normal), len(df_attack))
                if len(df_normal) < len(df_attack):
                    df_normal = resample(df_normal, n_samples=max_samples, random_state=42, replace=True)
                    logger.info(f"Oversampled normal class to {max_samples:,} samples")
                else:
                    df_attack = resample(df_attack, n_samples=max_samples, random_state=42, replace=True)
                    logger.info(f"Oversampled attack class to {max_samples:,} samples")
                train_df = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
            
            elif balance_strategy == 'smote':
                try:
                    from imblearn.over_sampling import SMOTE
                    logger.info("Using SMOTE for class balancing")
                    X_train_bal = train_df.drop(columns=['label'])
                    y_train_bal = train_df['label']
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X_train_bal, y_train_bal)
                    train_df = pd.DataFrame(X_resampled, columns=X_train_bal.columns)
                    train_df['label'] = y_resampled
                    logger.info(f"SMOTE resampling: {len(train_df):,} samples after balancing")
                except ImportError:
                    logger.warning("imblearn not installed. Falling back to oversample.")
                    balance_strategy = 'oversample'
                    # Retry with oversample
                    max_samples = max(len(df_normal), len(df_attack))
                    if len(df_normal) < len(df_attack):
                        df_normal = resample(df_normal, n_samples=max_samples, random_state=42, replace=True)
                    else:
                        df_attack = resample(df_attack, n_samples=max_samples, random_state=42, replace=True)
                    train_df = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info("Class distribution AFTER balancing:")
            class_dist_after = train_df['label'].value_counts().to_dict()
            for label, count in class_dist_after.items():
                logger.info(f"  Label {label}: {count:,} samples ({count/len(train_df)*100:.2f}%)")
    else:
        # Normal mode: use existing load_data function
        train_df, test_df = load_data(
            train_path=data_config.get('train_path'),
            test_path=data_config.get('test_path'),
            example_dataset=data_config.get('example_dataset', False),
            fast_mode=cfg.get('training', {}).get('fast_mode', False)
        )
    
    # Separate features and target
    if 'label' not in train_df.columns:
        raise ValueError("Data must contain a 'label' column")
    
    # Ensure label column exists and is properly aligned
    logger.info(f"Dataset shape before separation: {train_df.shape}")
    logger.info(f"Label column present: {'label' in train_df.columns}")
    
    # Check for any NaN values in label that might cause issues
    if train_df['label'].isnull().any():
        logger.warning(f"Found {train_df['label'].isnull().sum()} missing labels, filling with 0")
        train_df['label'] = train_df['label'].fillna(0)
    
    X_train = train_df.drop(columns=['label']).copy()
    y_train = train_df['label'].copy()
    
    # Verify alignment
    if len(X_train) != len(y_train):
        logger.error(f"Mismatch detected: X_train has {len(X_train)} rows, y_train has {len(y_train)} rows")
        # Reset indices to ensure alignment
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        logger.info("Reset indices to ensure alignment")
    
    logger.info(f"Training set: {len(X_train)} samples, {len(X_train.columns)} features")
    logger.info(f"Label set: {len(y_train)} samples")
    
    # Preprocessing
    logger.info("Step 1: Preprocessing")
    preprocessor = create_preprocessor(cfg.get('preprocessing', {}))
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Feature engineering
    logger.info("Step 2: Feature Engineering")
    feature_names = [normalize_feature_name(f) for f in preprocessor.feature_columns]
    
    # Verify alignment before feature engineering
    if len(X_train_processed) != len(y_train_processed):
        logger.error(f"Alignment issue before feature engineering: X has {len(X_train_processed)} rows, y has {len(y_train_processed)} rows")
        raise ValueError("Feature and label counts don't match after preprocessing")
    
    X_train_engineered, feature_names_engineered, variance_selector, feature_selector = engineer_features(
        X_train_processed, y_train_processed, feature_names, cfg.get('feature_engineering', {})
    )
    feature_names_engineered = [normalize_feature_name(f) for f in feature_names_engineered]
    
    # Verify alignment after feature engineering
    if len(X_train_engineered) != len(y_train_processed):
        logger.error(f"Alignment issue after feature engineering: X has {len(X_train_engineered)} rows, y has {len(y_train_processed)} rows")
        # Try to fix by ensuring same length
        min_len = min(len(X_train_engineered), len(y_train_processed))
        if len(X_train_engineered) > min_len:
            X_train_engineered = X_train_engineered[:min_len]
        if len(y_train_processed) > min_len:
            y_train_processed = y_train_processed[:min_len]
        logger.warning(f"Truncated to {min_len} samples to ensure alignment")
    
    # Train/validation split
    model_config = cfg.get('model', {})
    test_size = model_config.get('validation_size', 0.1)
    
    if test_size > 0:
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_engineered, y_train_processed,
            test_size=test_size,
            random_state=model_config.get('random_state', 42),
            stratify=y_train_processed
        )
        logger.info(f"Split: {len(X_train_final)} train, {len(X_val)} validation")
    else:
        X_train_final = X_train_engineered
        y_train_final = y_train_processed
        X_val = None
        y_val = None
    
    # Create model
    logger.info("Step 3: Model Creation")
    model_name = model_config.get('name', 'RandomForest')
    model = create_model(model_name, model_config)
    
    # Optional: Grid search for hyperparameter tuning
    # Skip grid search for very large datasets to avoid memory issues
    dataset_size = len(X_train_final)
    skip_grid_search = fast or dataset_size > 500000  # Skip for datasets > 500k samples
    
    if not skip_grid_search:
        param_grid = get_hyperparameter_grid(model_name, fast_mode=fast)
        if param_grid:
            logger.info("Step 4: Hyperparameter Tuning")
            # Use fewer parallel jobs for large datasets to reduce memory usage
            n_jobs = 1 if dataset_size > 200000 else -1
            if dataset_size > 200000:
                logger.info(f"Large dataset detected ({dataset_size:,} samples), using n_jobs=1 to reduce memory usage")
            model = perform_grid_search(
                model,
                X_train_final,
                y_train_final,
                param_grid,
                cv=model_config.get('cv_folds', 5),
                n_jobs=n_jobs
            )
    else:
        if dataset_size > 500000:
            logger.info(f"Skipping grid search due to large dataset size ({dataset_size:,} samples). Using default hyperparameters.")
        else:
            logger.info("Skipping grid search (fast mode enabled)")
    
    # Optional: Cross-validation
    # Skip cross-validation for very large datasets to save time and memory
    if not fast and dataset_size <= 500000:
        logger.info("Step 5: Cross-Validation")
        cv_scores = cross_validate_model(
            model,
            X_train_final,
            y_train_final,
            cv=model_config.get('cv_folds', 5)
        )
    elif dataset_size > 500000:
        logger.info(f"Skipping cross-validation due to large dataset size ({dataset_size:,} samples)")
    
    # Train final model
    logger.info("Step 6: Training Final Model")
    model.fit(X_train_final, y_train_final)
    
    # Evaluate on validation set if available
    if X_val is not None and y_val is not None:
        from src.utils import calculate_metrics
        y_val_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_val_pred)
        logger.info(f"Validation metrics: {metrics}")
    
    # Save artifacts
    training_config = cfg.get('training', {})
    paths_config = cfg.get('paths', {})
    
    if training_config.get('save_model', True):
        model_path = paths_config.get('model_file', 'artifacts/model.joblib')
        save_model(model, model_path)
    
    if training_config.get('save_preprocessor', True):
        preprocessor_path = paths_config.get('preprocessor_file', 'artifacts/preprocessor.joblib')
        from src.utils import save_model as save_preprocessor
        save_preprocessor(preprocessor, preprocessor_path)
    
    if training_config.get('save_features', True):
        features_path = paths_config.get('features_file', 'artifacts/features.json')
        save_features(feature_names_engineered, features_path)
    
    # Save feature engineering selectors for evaluation
    if variance_selector is not None:
        variance_selector_path = paths_config.get('variance_selector_file', 'artifacts/variance_selector.joblib')
        from src.utils import save_model as save_selector
        save_selector(variance_selector, variance_selector_path)
        logger.info(f"Saved variance selector to {variance_selector_path}")
    
    if feature_selector is not None:
        feature_selector_path = paths_config.get('feature_selector_file', 'artifacts/feature_selector.joblib')
        from src.utils import save_model as save_selector
        save_selector(feature_selector, feature_selector_path)
        logger.info(f"Saved feature selector to {feature_selector_path}")
    
    # Save test dataset for reproducibility (CICIDS2017)
    if use_cicids and test_df is not None:
        test_dataset_path = Path(paths_config.get('artifacts_dir', 'artifacts')) / 'cicids_test_actual.csv'
        test_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Save only first 100k rows to avoid huge files (or all if smaller)
            test_df_to_save = test_df.head(100000) if len(test_df) > 100000 else test_df
            test_df_to_save.to_csv(test_dataset_path, index=False)
            logger.info(f"Saved test dataset ({len(test_df_to_save):,} rows) to {test_dataset_path}")
        except Exception as e:
            logger.warning(f"Could not save test dataset: {e}")
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    train()

