"""
Utility to parse metrics from artifacts/report.html and extract evaluation data
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup


def parse_report_html(report_path: Path) -> Dict[str, Any]:
    """
    Parse metrics from HTML report file.
    
    Args:
        report_path: Path to report.html
        
    Returns:
        Dictionary with parsed metrics
    """
    if not report_path.exists():
        return {}
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        metrics = {}
        
        # Extract metrics from div.metrics
        metrics_div = soup.find('div', class_='metrics')
        if metrics_div:
            for metric_div in metrics_div.find_all('div', class_='metric'):
                metric_name = metric_div.find('span', class_='metric-name')
                if metric_name:
                    name = metric_name.text.strip().replace(':', '').lower()
                    value_text = metric_div.text.replace(metric_name.text, '').strip()
                    try:
                        metrics[name] = float(value_text)
                    except ValueError:
                        pass
        
        # Extract confusion matrix from classification report
        pre_tag = soup.find('pre')
        if pre_tag:
            report_text = pre_tag.text
            # Parse confusion matrix values from classification report
            # This is a simplified parser - in production, you'd want more robust parsing
            lines = report_text.split('\n')
            for line in lines:
                if 'accuracy' in line.lower():
                    try:
                        acc_match = re.search(r'(\d+\.\d+)', line)
                        if acc_match:
                            metrics['accuracy'] = float(acc_match.group(1))
                    except:
                        pass
        
        return metrics
    
    except Exception as e:
        print(f"Error parsing report.html: {e}")
        return {}


def extract_confusion_matrix_from_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Extract confusion matrix values from image (if available).
    For now, returns None - would need image processing library.
    """
    # In production, you could use PIL/OpenCV to extract values
    # For now, we'll calculate it from test data if available
    return None


def generate_metrics_json(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Generate comprehensive metrics JSON from artifacts.
    
    Args:
        artifacts_dir: Path to artifacts directory
        
    Returns:
        Dictionary with all metrics
    """
    report_path = artifacts_dir / 'report.html'
    metrics = parse_report_html(report_path)
    
    # Default values if not found in report
    default_metrics = {
        'accuracy': metrics.get('accuracy', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'f1': metrics.get('f1', 0.0),
        'roc_auc': metrics.get('roc_auc', None),
    }
    
    return default_metrics

