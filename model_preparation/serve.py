"""
Simple entry point for Streamlit web interface - sets up Python path automatically.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now run Streamlit with the serve script
if __name__ == '__main__':
    import subprocess
    import os
    
    # Get the path to the serve.py file
    serve_script = project_root / 'src' / 'serve.py'
    
    # Run streamlit
    os.chdir(project_root)
    subprocess.run(['streamlit', 'run', str(serve_script)])

