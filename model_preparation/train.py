"""
Simple entry point for training - sets up Python path automatically.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import and run the actual training script
if __name__ == '__main__':
    from src.train import train
    train()

