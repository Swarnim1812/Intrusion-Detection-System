"""
Data loading module for Jharkhand-IDS.

Supports loading CSV files, gzipped CSV files, and generating example datasets.
Includes first-class support for CICIDS2017 dataset with automatic column detection
and multi-class to binary label mapping.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import gzip
import json

logger = logging.getLogger(__name__)

# CICIDS2017 Dataset Information
# The CICIDS2017 dataset contains network flow data collected over 5 days:
# - Monday: Normal traffic only
# - Tuesday: FTP-BruteForce, SSH-BruteForce attacks
# - Wednesday: DoS, Heartbleed attacks
# - Thursday: Web Attack (Brute Force, XSS, SQL Injection), Infiltration attacks
# - Friday: Botnet, DDoS, PortScan attacks
# 
# The dataset includes features like:
# - Flow ID, Source IP, Destination IP, Source Port, Destination Port
# - Protocol, Flow Duration, Total Fwd Packets, Total Backward Packets
# - Total Length of Fwd Packets, Total Length of Bwd Packets
# - Label: Normal or specific attack type (e.g., "BENIGN", "FTP-Patator", "DoS Hulk", etc.)
#
# For binary classification, we map:
# - "BENIGN" or "Normal" → 0 (Normal)
# - All attack types → 1 (Attack)


def is_cicids2017_dataset(df: pd.DataFrame) -> bool:
    """
    Detect if the dataset is CICIDS2017 format.
    
    CICIDS2017 has specific column names like "Flow ID", "Label", etc.
    
    Args:
        df: DataFrame to check.
        
    Returns:
        True if dataset appears to be CICIDS2017 format.
    """
    cicids_indicators = [
        'Flow ID', 'FlowID', 'flow_id',
        'Source IP', 'SourceIP', 'src_ip',
        'Destination IP', 'DestinationIP', 'dst_ip',
        'Label', 'label'
    ]
    
    df_columns_lower = [col.lower() for col in df.columns]
    matches = sum(1 for indicator in cicids_indicators 
                  if indicator.lower() in df_columns_lower)
    
    return matches >= 2


def preprocess_cicids2017(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess CICIDS2017 dataset for binary classification.
    
    - Drops non-numeric columns except Label
    - Maps multi-class labels to binary (Normal=0, Attacks=1)
    - Handles missing values
    - Handles skewed numeric features (optional log1p transformation)
    
    Args:
        df: Raw CICIDS2017 DataFrame.
        
    Returns:
        Preprocessed DataFrame with 'label' column (0=Normal, 1=Attack).
    """
    df = df.copy()
    logger.info("Preprocessing CICIDS2017 dataset...")
    
    # Find label column (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.lower() in ['label', 'labels', 'target', 'class']:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in CICIDS2017 dataset")
    
    logger.info(f"Found label column: {label_col}")
    
    # Store labels before processing
    labels = df[label_col].copy()
    
    # Drop non-numeric columns except the label column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col not in numeric_cols:
        numeric_cols.append(label_col)
    
    # Also drop common non-numeric CICIDS columns
    cols_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        if col not in numeric_cols and col != label_col:
            # Drop Flow ID, IP addresses, timestamps, etc.
            if any(x in col_lower for x in ['flow id', 'flowid', 'ip', 'timestamp', 'time']):
                cols_to_drop.append(col)
    
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} non-numeric columns: {cols_to_drop[:5]}...")
        df = df.drop(columns=cols_to_drop)
    
    # Map multi-class labels to binary
    # Normal/BENIGN → 0, all attacks → 1
    label_mapping = {}
    unique_labels = labels.unique()
    logger.info(f"Found {len(unique_labels)} unique labels in dataset")
    
    for label in unique_labels:
        label_str = str(label).strip().lower()
        if label_str in ['benign', 'normal', '0']:
            label_mapping[label] = 0
        else:
            # All attack types map to 1
            label_mapping[label] = 1
    
    # Apply mapping
    df['label'] = labels.map(label_mapping)
    
    # Drop original label column if different
    if label_col != 'label':
        df = df.drop(columns=[label_col])
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"Found {missing_before} missing values, filling with median")
        numeric_cols_except_label = [c for c in df.columns if c != 'label' and pd.api.types.is_numeric_dtype(df[c])]
        df[numeric_cols_except_label] = df[numeric_cols_except_label].fillna(df[numeric_cols_except_label].median())
    
    # Handle skewed features (optional - can be enabled via config)
    # For very large values, apply log1p transformation
    numeric_cols_except_label = [c for c in df.columns if c != 'label' and pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols_except_label:
        if df[col].min() >= 0:  # Only apply to non-negative values
            # Check if values are very large (potential skew)
            if df[col].max() > 10000:
                # Apply log1p to reduce skew
                df[col] = np.log1p(df[col])
                logger.debug(f"Applied log1p transformation to {col}")
    
    logger.info(f"CICIDS2017 preprocessing complete. Shape: {df.shape}, Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def load_csv(filepath: str, detect_cicids: bool = True, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file (regular or gzipped).
    
    Args:
        filepath: Path to the CSV file.
        **kwargs: Additional arguments to pass to pd.read_csv.
        
    Returns:
        Loaded DataFrame.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    try:
        # For very large files, use chunking
        chunksize = kwargs.pop('chunksize', None)
        
        if filepath.suffix == '.gz' or str(filepath).endswith('.csv.gz'):
            logger.info(f"Loading gzipped CSV from {filepath}")
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                if chunksize:
                    # Read in chunks for very large files
                    chunks = []
                    for chunk in pd.read_csv(f, chunksize=chunksize, **kwargs):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(f, **kwargs)
        else:
            logger.info(f"Loading CSV from {filepath}")
            if chunksize:
                chunks = []
                for chunk in pd.read_csv(filepath, chunksize=chunksize, **kwargs):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(filepath, **kwargs)
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Auto-detect and preprocess CICIDS2017 if enabled
        if detect_cicids and is_cicids2017_dataset(df):
            logger.info("Detected CICIDS2017 dataset format, applying preprocessing...")
            df = preprocess_cicids2017(df)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        raise


def generate_example_dataset(
    n_samples: int = 10000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic example dataset for testing/demo purposes.
    
    This is a placeholder function. In production, replace with actual
    NSL-KDD or CICIDS dataset loading logic.
    
    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features (excluding target).
        n_classes: Number of classes for the target variable.
        random_state: Random seed for reproducibility.
        
    Returns:
        DataFrame with features and a 'label' column.
    """
    logger.info(f"Generating example dataset: {n_samples} samples, {n_features} features")
    
    np.random.seed(random_state)
    
    # Generate feature names (simulating network flow features)
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    
    # Generate synthetic data
    data = np.random.randn(n_samples, n_features)
    
    # Add some structure to make it more realistic
    # Create correlations and patterns
    data[:, 0] = np.abs(data[:, 0]) * 10  # Simulate packet count
    data[:, 1] = np.abs(data[:, 1]) * 5   # Simulate duration
    
    # Generate labels (binary classification: 0=normal, 1=anomaly)
    # Make anomalies slightly different
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['label'] = labels
    
    logger.info(f"Generated example dataset with {n_samples} samples")
    logger.warning(
        "This is a placeholder dataset. For production use, replace with "
        "actual NSL-KDD or CICIDS dataset loading."
    )
    
    return df


def load_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    example_dataset: bool = False,
    fast_mode: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Main data loading function.
    
    Args:
        train_path: Path to training CSV file (or gzipped CSV).
        test_path: Optional path to test CSV file.
        example_dataset: If True, generate example dataset instead.
        fast_mode: If True, reduce dataset size for quick testing.
        
    Returns:
        Tuple of (train_df, test_df). test_df is None if test_path is not provided.
    """
    if example_dataset or train_path is None:
        logger.info("Using example dataset")
        n_samples = 1000 if fast_mode else 10000
        train_df = generate_example_dataset(n_samples=n_samples)
        test_df = None
        
        if test_path is None:
            # Split train into train/test if no test path provided
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                train_df, test_size=0.2, random_state=42, stratify=train_df['label']
            )
    else:
        train_df = load_csv(train_path)
        
        if test_path:
            test_df = load_csv(test_path)
        else:
            test_df = None
    
    if fast_mode and train_df is not None:
        # Reduce dataset size for fast mode
        original_size = len(train_df)
        train_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)
        logger.info(f"Fast mode: reduced training set from {original_size} to {len(train_df)} samples")
    
    return train_df, test_df


def load_cicids2017_folder(folder_path: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from CICIDS2017 folder.
    
    This function loads all CSV files from the specified folder (typically
    full_dataset/MachineLearningCVE), concatenates them, and applies CICIDS2017
    preprocessing. Designed to handle large datasets efficiently.
    
    IMPORTANT: Data Leakage Prevention
    -----------------------------------
    Random shuffling of CICIDS2017 data inflates accuracy because it mixes
    data from different days. CICIDS2017 is collected over 5 days (July 3-7, 2017):
    
    - Monday (July 3): BENIGN traffic only
    - Tuesday (July 4): BENIGN + FTP-BruteForce, SSH-BruteForce
    - Wednesday (July 5): BENIGN + DoS (Hulk, GoldenEye, Slowloris, SlowHTTPTest)
    - Thursday (July 6): BENIGN + Web Attacks (BruteForce, XSS, SQL Injection) + Infiltration
    - Friday (July 7): BENIGN + DDoS + PortScan
    
    Day-based splitting (Monday-Thursday for training, Friday for testing) prevents
    data leakage and provides realistic IDS performance evaluation. The model must
    generalize to unseen attack patterns from Friday.
    
    This function extracts day information from the Timestamp column and adds
    '_day_of_week', '_date', and '_day_name' columns for day-based splitting.
    
    Args:
        folder_path: Path to folder containing CICIDS2017 CSV files.
        
    Returns:
        Concatenated and preprocessed DataFrame with binary labels and day information.
        The DataFrame includes '_day_of_week', '_date', and '_day_name' columns
        extracted from the Timestamp column for day-based train-test splitting.
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"CICIDS2017 folder not found: {folder_path}")
    
    logger.info(f"Loading CICIDS2017 dataset from {folder_path}")
    
    # Find all CSV files in the folder
    csv_files = list(folder_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Load and concatenate all files
    # IMPORTANT: Extract day from filename for day-based splitting
    dataframes = []
    total_rows = 0
    
    # Day mapping from filename
    day_mapping = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4
    }
    
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}...")
        try:
            # Extract day from filename
            filename_lower = csv_file.name.lower()
            day_of_week = None
            day_name = None
            
            for day_key, day_num in day_mapping.items():
                if day_key in filename_lower:
                    day_of_week = day_num
                    day_name = day_key.capitalize()
                    break
            
            if day_of_week is None:
                logger.warning(f"Could not determine day from filename: {csv_file.name}")
                day_of_week = None
                day_name = None
            
            # Load in chunks for very large files
            chunk_list = []
            chunk_size = 50000  # Process 50k rows at a time
            
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
                # Add day information from filename
                chunk['_day_of_week'] = day_of_week
                chunk['_day_name'] = day_name
                chunk['_source_file'] = csv_file.name
                chunk_list.append(chunk)
            
            if chunk_list:
                df_file = pd.concat(chunk_list, ignore_index=True)
                dataframes.append(df_file)
                total_rows += len(df_file)
                logger.info(f"  Loaded {len(df_file):,} rows from {csv_file.name} (Day: {day_name})")
        
        except Exception as e:
            logger.warning(f"Error loading {csv_file.name}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No data loaded from any CSV files")
    
    # Concatenate all dataframes
    logger.info(f"Concatenating {len(dataframes)} dataframes...")
    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total rows after concatenation: {len(df):,}, columns: {len(df.columns)}")
    
    # Clean invalid values first (before preprocessing)
    logger.info("Cleaning invalid values (Infinity, NaN, etc.)...")
    
    # Replace infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Clean string representations of NaN
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(['NaN', 'Na/NaN', 'NAN', 'nan', 'Infinity', 'infinity'], np.nan)
    
    # Find label column - CICIDS2017 uses "Label" (capital L)
    # Check exact matches first, then case-insensitive
    label_col = None
    
    # Try exact matches first (common CICIDS2017 column names)
    possible_label_names = ['Label', 'label', 'LABEL']
    for col_name in possible_label_names:
        if col_name in df.columns:
            label_col = col_name
            logger.info(f"Found label column (exact match): {col_name}")
            break
    
    # If not found, try case-insensitive match with stripped whitespace
    if label_col is None:
        for col in df.columns:
            col_stripped = col.strip()
            if col_stripped.lower() == 'label':
                label_col = col
                logger.info(f"Found label column (case-insensitive): {col}")
                break
    
    # If still not found, try any column containing 'label' or 'class'
    if label_col is None:
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                label_col = col
                logger.info(f"Found label column (contains 'label'/'class'): {col}")
                break
    
    if label_col is None:
        # Show available columns for debugging
        available_cols = list(df.columns)
        logger.error(f"Available columns: {available_cols[:20]}...")
        raise ValueError(f"Could not find label column. Available columns: {available_cols[:20]}...")
    
    # Rename label column to 'label' for consistency (handle whitespace)
    if label_col.strip() != 'label':
        df = df.rename(columns={label_col: 'label'})
        logger.info(f"Renamed '{label_col}' to 'label'")
    
    # IMPORTANT: Day information extracted from filename during loading
    # Check if day columns were added from filename
    if '_day_of_week' not in df.columns or df['_day_of_week'].isna().all():
        # Try to extract from timestamp column if available (fallback)
        timestamp_col = None
        timestamp_cols_candidates = ['Timestamp', 'timestamp', 'Time', 'time']
        for col in timestamp_cols_candidates:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            for col in df.columns:
                if 'timestamp' in col.lower() or ('time' in col.lower() and 'label' not in col.lower()):
                    timestamp_col = col
                    break
        
        if timestamp_col:
            try:
                logger.info(f"Found timestamp column: {timestamp_col}, extracting day information")
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                df['_day_of_week'] = df[timestamp_col].dt.dayofweek
                df['_date'] = df[timestamp_col].dt.date
                df['_day_name'] = df[timestamp_col].dt.day_name()
                logger.info(f"Extracted day information from timestamp. Day distribution: {df['_day_name'].value_counts().to_dict()}")
            except Exception as e:
                logger.warning(f"Could not extract day from timestamp: {e}")
                df['_day_of_week'] = None
                df['_date'] = None
                df['_day_name'] = None
        else:
            logger.warning("No timestamp column found and day not extracted from filename. Day-based splitting will not be available.")
            if '_day_of_week' not in df.columns:
                df['_day_of_week'] = None
                df['_date'] = None
                df['_day_name'] = None
    else:
        # Day information already extracted from filename
        logger.info(f"Day information from filename. Day distribution: {df['_day_name'].value_counts().to_dict()}")
    
    # Drop identifier columns (but keep _day_of_week, _date, _day_name, _source_file for splitting)
    id_cols = ['Flow ID', 'FlowID', 'flow_id', 'Source IP', 'SourceIP', 'src_ip',
               'Destination IP', 'DestinationIP', 'dst_ip']
    # Check for timestamp column to drop
    timestamp_cols_to_drop = [col for col in df.columns if ('timestamp' in col.lower() or 'time' in col.lower()) and col not in ['_day_of_week', '_day_name'] and 'label' not in col.lower()]
    id_cols.extend(timestamp_cols_to_drop)
    cols_to_drop = [col for col in id_cols if col in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping identifier columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Apply CICIDS2017 preprocessing (label mapping) BEFORE dropping timestamp columns
    # This ensures label column is properly mapped and we don't lose rows
    df = preprocess_cicids2017(df)
    
    # Convert numeric columns to float32 to reduce RAM usage (after label is processed)
    logger.info("Converting numeric columns to float32...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def load_sample_rows(filepath: Optional[str] = None, n: int = 20) -> List[Dict]:
    """
    Load sample rows from dataset for dashboard preview.
    
    This helper function loads the first n rows of a dataset and returns
    them as a list of dictionaries. Used by the dashboard to show dataset preview.
    
    Args:
        filepath: Path to CSV file. If None, uses example dataset.
        n: Number of rows to return (default: 20).
        
    Returns:
        List of dictionaries, each representing a row.
    """
    try:
        if filepath:
            df = load_csv(filepath, nrows=n)
        else:
            df = generate_example_dataset(n_samples=n, random_state=42)
        
        # Convert to list of dictionaries
        sample_rows = df.head(n).to_dict('records')
        
        logger.info(f"Loaded {len(sample_rows)} sample rows")
        return sample_rows
    
    except Exception as e:
        logger.error(f"Error loading sample rows: {e}")
        # Return empty list on error
        return []

