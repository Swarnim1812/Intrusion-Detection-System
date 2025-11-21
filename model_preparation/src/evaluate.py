"""
Evaluation script for Jharkhand-IDS.

Generates evaluation metrics, confusion matrix, and classification reports.
"""

import click
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from sklearn.metrics import roc_curve, auc

from src.config import load_config
from src.data_loader import load_data, load_sample_rows
from src.preprocessing import create_preprocessor
from src.utils import (
    setup_logging, load_model, load_features, calculate_metrics,
    get_classification_report, get_confusion_matrix, save_model
)

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix"
) -> np.ndarray:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save the plot.
        title: Plot title.
        
    Returns:
        Confusion matrix as numpy array.
    """
    cm = get_confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")
    return cm


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str,
    title: str = "ROC Curve"
) -> Dict[str, float]:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities (for positive class).
        save_path: Path to save the plot.
        title: Plot title.
        
    Returns:
        Dictionary with 'fpr', 'tpr', 'auc' values.
    """
    # Handle binary and multiclass cases
    if y_pred_proba.ndim == 1:
        y_scores = y_pred_proba
    else:
        # Use probability of positive class (index 1) for binary classification
        if y_pred_proba.shape[1] > 1:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba[:, 0]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curve to {save_path} (AUC = {roc_auc:.4f})")
    
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': float(roc_auc),
        'thresholds': thresholds.tolist()
    }


def generate_html_report(
    metrics: dict,
    classification_report: str,
    confusion_matrix_path: str,
    save_path: str
) -> None:
    """
    Generate an HTML evaluation report.
    
    Args:
        metrics: Dictionary of metrics.
        classification_report: Text classification report.
        confusion_matrix_path: Path to confusion matrix image.
        save_path: Path to save HTML report.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jharkhand-IDS Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            .metrics {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .metric {{ margin: 5px 0; }}
            .metric-name {{ font-weight: bold; color: #2c3e50; }}
            pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Jharkhand-IDS Evaluation Report</h1>
        
        <h2>Metrics</h2>
        <div class="metrics">
    """
    
    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            html_content += f"""
            <div class="metric">
                <span class="metric-name">{metric_name.capitalize()}:</span> {metric_value:.4f}
            </div>
            """
    
    html_content += """
        </div>
        
        <h2>Confusion Matrix</h2>
        <img src="confusion_matrix.png" alt="Confusion Matrix">
        
        <h2>Classification Report</h2>
        <pre>
    """
    
    html_content += classification_report
    html_content += """
        </pre>
    </body>
    </html>
    """
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Saved HTML report to {save_path}")


@click.command()
@click.option('--config', default='config/default.yaml', help='Path to configuration file')
@click.option('--model', default=None, help='Path to model file (overrides config)')
@click.option('--data', default=None, help='Path to test data (overrides config)')
def evaluate(config: str, model: Optional[str], data: Optional[str]) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Example:
        python -m src.evaluate --config config/default.yaml
        python -m src.evaluate --config config/default.yaml --model artifacts/model.joblib --data test.csv
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
    logger.info("Starting Jharkhand-IDS Evaluation")
    logger.info("=" * 60)
    
    # Load model
    paths_config = cfg.get('paths', {})
    model_path = model or paths_config.get('model_file', 'artifacts/model.joblib')
    model = load_model(model_path)
    
    # Load preprocessor
    preprocessor_path = paths_config.get('preprocessor_file', 'artifacts/preprocessor.joblib')
    preprocessor = load_model(preprocessor_path)
    
    # Load features
    features_path = paths_config.get('features_file', 'artifacts/features.json')
    feature_names = load_features(features_path)
    
    # Load feature engineering selectors if available
    variance_selector = None
    feature_selector = None
    try:
        variance_selector_path = paths_config.get('variance_selector_file', 'artifacts/variance_selector.joblib')
        if Path(variance_selector_path).exists():
            variance_selector = load_model(variance_selector_path)
            logger.info("Loaded variance selector for feature engineering")
    except Exception as e:
        logger.warning(f"Could not load variance selector: {e}")
    
    try:
        feature_selector_path = paths_config.get('feature_selector_file', 'artifacts/feature_selector.joblib')
        if Path(feature_selector_path).exists():
            feature_selector = load_model(feature_selector_path)
            logger.info("Loaded feature selector for feature engineering")
    except Exception as e:
        logger.warning(f"Could not load feature selector: {e}")
    
    # Load test data
    if data:
        test_df, _ = load_data(train_path=data, example_dataset=False)
    else:
        data_config = cfg.get('data', {})
        use_cicids = data_config.get('use_cicids', False)
        
        if use_cicids:
            # For CICIDS2017, use the same day-based split as training
            logger.info("CICIDS2017 mode: Loading test data...")
            
            # First, try to load saved test dataset from training
            test_dataset_path = Path(paths_config.get('artifacts_dir', 'artifacts')) / 'cicids_test_actual.csv'
            if test_dataset_path.exists():
                logger.info(f"Loading saved test dataset from {test_dataset_path}")
                try:
                    test_df = pd.read_csv(test_dataset_path, low_memory=False)
                    logger.info(f"Loaded {len(test_df):,} test samples from saved file")
                except Exception as e:
                    logger.warning(f"Could not load saved test dataset: {e}. Regenerating with day-based split...")
                    test_df = None
            else:
                test_df = None
            
            # If saved test dataset not available, regenerate using day-based split
            if test_df is None:
                from src.data_loader import load_cicids2017_folder
                training_config = cfg.get('training', {})
                use_day_split = training_config.get('use_day_based_split', True)
                
                cicids_dir = data_config.get('cicids2017_dir', 'full_dataset/MachineLearningCVE')
                full_df = load_cicids2017_folder(cicids_dir)
                
                if use_day_split and '_day_name' in full_df.columns and full_df['_day_name'].notna().any():
                    logger.info("=" * 60)
                    logger.info("Using DAY-BASED SPLIT for evaluation (Friday only)")
                    logger.info("=" * 60)
                    
                    # Use Friday (day 4) as test set, matching training logic
                    test_mask = full_df['_day_of_week'] == 4  # Friday
                    test_df = full_df[test_mask].copy()
                    
                    # Remove day columns (metadata, not features)
                    for col in ['_day_of_week', '_date', '_day_name', '_source_file']:
                        if col in test_df.columns:
                            test_df = test_df.drop(columns=[col])
                    
                    logger.info(f"Day-based test set (Friday): {len(test_df):,} samples")
                    logger.info(f"Test class distribution: {test_df['label'].value_counts().to_dict()}")
                else:
                    # Fallback to random split (NOT recommended)
                    logger.warning("=" * 60)
                    logger.warning("WARNING: Using RANDOM SPLIT for evaluation (may cause data leakage)")
                    logger.warning("=" * 60)
                    model_config = cfg.get('model', {})
                    full_df = full_df.sample(frac=1, random_state=model_config.get('random_state', 42)).reset_index(drop=True)
                    
                    # Remove day columns if present
                    for col in ['_day_of_week', '_date', '_day_name', '_source_file']:
                        if col in full_df.columns:
                            full_df = full_df.drop(columns=[col])
                    
                    _, test_df = train_test_split(
                        full_df,
                        test_size=model_config.get('test_size', 0.2),
                        random_state=model_config.get('random_state', 42),
                        stratify=full_df['label']
                    )
                    logger.info(f"Random split test set: {len(test_df):,} samples")
            
            logger.info(f"Loaded {len(test_df):,} test samples from CICIDS2017")
        else:
            _, test_df = load_data(
                train_path=data_config.get('train_path'),
                test_path=data_config.get('test_path'),
                example_dataset=data_config.get('example_dataset', False)
            )
            
            if test_df is None:
                logger.warning("No test data provided. Using example dataset for evaluation.")
                from src.data_loader import generate_example_dataset
                test_df = generate_example_dataset(n_samples=2000, random_state=42)
    
    if 'label' not in test_df.columns:
        raise ValueError("Test data must contain a 'label' column")
    
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    
    logger.info(f"Test set: {len(X_test)} samples, {len(X_test.columns)} features")
    
    # Preprocess test data
    logger.info("Preprocessing test data")
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # Apply feature engineering (match training features)
    # The saved feature_names are the FINAL features after engineering
    # So we need to apply the same feature engineering to test data
    from src.feature_engineering import engineer_features
    feature_engineering_config = cfg.get('feature_engineering', {})
    
    # Get the original feature names from preprocessor (before engineering)
    original_feature_names = preprocessor.feature_columns
    
    # Apply the same feature engineering that was used during training
    logger.info("Applying feature engineering to test data")
    logger.info(f"Input features: {X_test_processed.shape[1]}, Original feature names: {len(original_feature_names)}")
    
    X_test_engineered, engineered_feature_names, _, _ = engineer_features(
        X_test_processed,
        y_test_processed,
        original_feature_names,
        feature_engineering_config,
        variance_selector=variance_selector,
        feature_selector=feature_selector
    )
    
    # Verify feature count matches
    if X_test_engineered.shape[1] != len(feature_names):
        logger.warning(
            f"Feature count mismatch: Test has {X_test_engineered.shape[1]} features, "
            f"but model expects {len(feature_names)} features. "
            f"Engineered features: {len(engineered_feature_names)}, Saved features: {len(feature_names)}"
        )
        # Try to align features
        if len(engineered_feature_names) == len(feature_names):
            # Features match by count, assume order is correct
            logger.info("Feature counts match, proceeding with prediction")
        else:
            # Select only the features that match
            common_features = [f for f in engineered_feature_names if f in feature_names]
            if len(common_features) == len(feature_names):
                # Reorder to match saved feature order
                feature_indices = [engineered_feature_names.index(f) for f in feature_names]
                X_test_engineered = X_test_engineered[:, feature_indices]
                logger.info(f"Reordered features to match saved feature order")
            else:
                raise ValueError(
                    f"Cannot align features. Model expects {len(feature_names)} features: {feature_names[:5]}..., "
                    f"but got {len(engineered_feature_names)} features: {engineered_feature_names[:5]}..."
                )
    
    logger.info(f"Test features after engineering: {X_test_engineered.shape[1]} (model expects: {len(feature_names)})")
    
    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(X_test_engineered)
    
    # Get prediction probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test_engineered)
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
    
    # Calculate metrics
    logger.info("Calculating metrics")
    metrics = calculate_metrics(y_test_processed, y_pred, y_pred_proba)
    
    logger.info("Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Generate classification report
    classification_report = get_classification_report(y_test_processed, y_pred)
    logger.info("\nClassification Report:\n" + classification_report)
    
    # Save plots and report
    eval_config = cfg.get('evaluation', {})
    artifacts_dir = Path(paths_config.get('artifacts_dir', 'artifacts'))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save confusion matrix
    cm_array = None
    if eval_config.get('save_plots', True):
        cm_path = artifacts_dir / 'confusion_matrix.png'
        cm_array = plot_confusion_matrix(y_test_processed, y_pred, str(cm_path))
        cm_array = cm_array.tolist()  # Convert to list for JSON serialization
    
    # Save ROC curve if probabilities available
    roc_data = None
    roc_path = None
    if y_pred_proba is not None and eval_config.get('save_plots', True):
        roc_path = artifacts_dir / 'roc_curve.png'
        try:
            roc_data = plot_roc_curve(y_test_processed, y_pred_proba, str(roc_path))
            roc_path = str(roc_path)
        except Exception as e:
            logger.warning(f"Could not generate ROC curve: {e}")
            roc_data = None
    
    # Get sample predictions (20 rows)
    sample_indices = np.random.choice(len(y_test_processed), size=min(20, len(y_test_processed)), replace=False)
    sample_predictions = []
    sample_rows_data = []
    
    for idx in sample_indices:
        prob = 0.0
        if y_pred_proba is not None:
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                prob = float(y_pred_proba[idx][1])
            else:
                prob = float(y_pred_proba[idx]) if y_pred_proba.ndim == 1 else float(y_pred_proba[idx][0])
        
        sample_predictions.append({
            'true_label': int(y_test_processed[idx]),
            'predicted_label': int(y_pred[idx]),
            'probability': prob
        })
        
        # Get corresponding row from original test data
        original_idx = idx
        if original_idx < len(X_test):
            row_dict = X_test.iloc[original_idx].to_dict()
            row_dict['true_label'] = int(y_test.iloc[original_idx])
            sample_rows_data.append(row_dict)
    
    # Also get first 20 rows of test data for sample_rows
    sample_rows_df = test_df.head(20)
    
    # Calculate per-class metrics and macro-averaged metrics
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    per_class_metrics = {}
    macro_metrics = {}
    
    try:
        # Get per-class precision, recall, f1
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_test_processed, y_pred, average=None, zero_division=0
        )
        
        # Calculate macro-averaged metrics
        macro_precision = float(np.mean(precision_per_class))
        macro_recall = float(np.mean(recall_per_class))
        macro_f1 = float(np.mean(f1_per_class))
        
        macro_metrics = {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
        
        # Per-class metrics
        unique_labels = sorted(np.unique(np.concatenate([y_test_processed, y_pred])))
        for i, label in enumerate(unique_labels):
            if i < len(precision_per_class):
                per_class_metrics[int(label)] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
        
        logger.info(f"Macro-averaged metrics: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate per-class metrics: {e}")
    
    # Calculate normalized confusion matrix
    cm_normalized = None
    if cm_array is not None:
        try:
            cm_np = np.array(cm_array)
            cm_normalized = (cm_np.astype('float') / cm_np.sum(axis=1)[:, np.newaxis]).tolist()
        except Exception as e:
            logger.warning(f"Could not normalize confusion matrix: {e}")
    
    # Detect if day-based split was used (check config or test data characteristics)
    use_day_split = cfg.get('training', {}).get('use_day_based_split', True)
    use_cicids = cfg.get('data', {}).get('use_cicids', False)
    day_based_split = use_cicids and use_day_split
    
    # Save metrics to JSON
    metrics_dict = {
        'metrics': {
            'accuracy': float(metrics.get('accuracy', 0.0)),
            'precision': float(metrics.get('precision', 0.0)),
            'recall': float(metrics.get('recall', 0.0)),
            'f1': float(metrics.get('f1', 0.0)),
            'roc_auc': float(metrics.get('roc_auc', 0.0)) if metrics.get('roc_auc') is not None else None,
        },
        'macro_metrics': macro_metrics,
        'per_class': per_class_metrics,
        'confusion_matrix': cm_array if cm_array is not None else [],
        'confusion_matrix_normalized': cm_normalized,
        'day_based_split': day_based_split,
        'roc_curve_path': roc_path,
        'roc_curve_data': roc_data,
        'sample_predictions': sample_predictions,
        'sample_rows': sample_rows_df.to_dict(orient='records') if 'sample_rows_df' in locals() else sample_rows_data[:10],
        'classification_report': classification_report,
    }
    
    metrics_json_path = artifacts_dir / 'metrics.json'
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    
    logger.info(f"Saved metrics to {metrics_json_path}")
    
    # Save HTML report
    if eval_config.get('save_report', True):
        report_path = paths_config.get('report_file', 'artifacts/report.html')
        cm_image_path = str(artifacts_dir / 'confusion_matrix.png')
        generate_html_report(metrics, classification_report, cm_image_path, report_path)
    
    logger.info("=" * 60)
    logger.info("Evaluation completed successfully!")
    logger.info(f"Saved outputs to {artifacts_dir}/")
    logger.info("=" * 60)
    
    # Return structured dict for programmatic use
    return metrics_dict


if __name__ == '__main__':
    evaluate()

