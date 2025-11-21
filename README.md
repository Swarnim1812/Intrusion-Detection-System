# Jharkhand-IDS: Intrusion Detection System

A production-quality, end-to-end machine learning pipeline for network intrusion detection with a complete web application (React frontend + Flask backend).

## 🎯 Features

### ML Pipeline

- **Modular Architecture**: Separate modules for data loading, preprocessing, feature engineering, modeling, training, and evaluation
- **Config-Driven Design**: YAML-based configuration for easy hyperparameter tuning
- **Multiple Algorithms**: Supports DecisionTree, RandomForest, and IsolationForest
- **CICIDS2017 Support**: First-class support for CICIDS2017 dataset with automatic preprocessing
- **Large Dataset Handling**: Efficient memory management with float32 conversion and chunked loading
- **Class Balancing**: Optional undersampling/oversampling for imbalanced datasets
- **Comprehensive Evaluation**: Generates metrics, confusion matrix, ROC curves, and detailed reports

### Web Application

- **React Frontend**: Modern, responsive UI with 5 complete pages
- **Flask Backend**: RESTful API with CORS support
- **Interactive Dashboard**: Real-time metrics visualization with Chart.js
- **CSV Upload & Prediction**: Upload network flow data and get real-time predictions
- **User Input Prediction**: Manual feature input with detailed explanations
- **Model Metrics Display**: Dynamic loading of accuracy, precision, recall, F1, ROC-AUC

## 📁 Project Structure

```
Jharkhand-IDS Project/
├── model_preparation/     # ML pipeline and model-related files
│   ├── src/              # ML pipeline source code
│   │   ├── config.py     # Configuration loader
│   │   ├── data_loader.py # Data loading (supports CICIDS2017)
│   │   ├── preprocessing.py # Data preprocessing
│   │   ├── feature_engineering.py # Feature selection
│   │   ├── models.py     # Model definitions
│   │   ├── train.py      # Training script
│   │   ├── evaluate.py   # Evaluation script
│   │   ├── serve.py      # Streamlit interface
│   │   └── utils.py      # Utility functions
│   ├── config/           # Configuration files
│   │   └── default.yaml  # Default configuration
│   ├── artifacts/        # Model artifacts (generated)
│   │   ├── model.joblib
│   │   ├── preprocessor.joblib
│   │   ├── features.json
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   ├── web/              # Dashboard support files
│   │   ├── dashboard_config.json
│   │   ├── api_samples.py
│   │   └── dashboard_notes.md
│   ├── tests/            # Unit tests
│   ├── full_dataset/     # CICIDS2017 dataset
│   │   └── MachineLearningCVE/
│   ├── sample_data/      # Sample CSV files
│   ├── train.py          # Training entry point
│   ├── evaluate.py       # Evaluation entry point
│   ├── serve.py          # Streamlit entry point
│   └── requirements.txt  # ML pipeline dependencies
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/   # Reusable components
│   │   ├── pages/         # 5 main pages
│   │   ├── services/      # API client
│   │   └── utils/         # Utilities
│   └── package.json
├── backend/              # Flask backend
│   ├── app.py           # Main API server
│   └── requirements.txt
├── docker-compose.yml    # Docker Compose configuration
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (for ML pipeline and backend)
- **Node.js 18+** and **npm** (for frontend)
- **Git** (optional, for cloning)

### Step 1: Install ML Pipeline Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Navigate to model_preparation directory
cd model_preparation

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train the Model

**Option A: Using CICIDS2017 Dataset (Recommended)**

1. Update `model_preparation/config/default.yaml`:

   ```yaml
   data:
     use_cicids: true
     cicids2017_dir: "full_dataset/MachineLearningCVE"
   ```

2. Train the model:

   ```bash
   cd model_preparation
   python train.py --config config/default.yaml
   ```

**Option B: Using Custom CSV**

```bash
cd model_preparation
python train.py --config config/default.yaml
```

The training script will:

- Load and preprocess the dataset
- Perform 80/20 train-test split (for CICIDS2017)
- Apply optional class balancing
- Train the selected model
- Save artifacts to `artifacts/` directory

### Step 3: Evaluate the Model

```bash
cd model_preparation
python evaluate.py --config config/default.yaml
```

This generates:

- `artifacts/metrics.json` - All metrics in JSON format
- `artifacts/confusion_matrix.png` - Confusion matrix visualization
- `artifacts/roc_curve.png` - ROC curve plot
- `artifacts/report.html` - HTML report

### Step 4: Start Backend API

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend runs on `http://localhost:5000`

### Step 5: Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

### Step 6: Access Application

Open browser to `http://localhost:3000`

## 📊 CICIDS2017 Dataset Support

The pipeline includes first-class support for the CICIDS2017 dataset:

### Features

- **Automatic Detection**: Auto-detects CICIDS2017 format by column names
- **Multi-file Loading**: Loads and concatenates all CSV files from a folder
- **Automatic Preprocessing**:
  - Drops non-numeric columns (Flow ID, IP addresses, timestamps)
  - Maps multi-class labels to binary (BENIGN→0, Attacks→1)
  - Handles missing values and invalid entries
  - Converts numeric columns to float32 for memory efficiency
- **Memory Efficient**: Chunked loading for large files
- **Class Balancing**: Optional undersampling/oversampling

### Usage

```python
from src.data_loader import load_cicids2017_folder

# Load all CSV files from CICIDS2017 folder
df = load_cicids2017_folder("full_dataset/MachineLearningCVE")
```

### Configuration

In `model_preparation/config/default.yaml`:

```yaml
data:
  use_cicids: true
  cicids2017_dir: "full_dataset/MachineLearningCVE"

training:
  balance_classes: false  # Set to true for balanced dataset
```

## 🔧 Configuration

Edit `config/default.yaml` to customize:

### Model Selection

```yaml
model:
  name: "RandomForest"  # Options: DecisionTree, RandomForest, IsolationForest
  random_forest:
    n_estimators: 100
    max_depth: 20
```

### Preprocessing

```yaml
preprocessing:
  normalize: true
  fill_na_strategy: "median"
  remove_duplicates: true
```

### Feature Engineering

```yaml
feature_engineering:
  remove_low_variance: true
  variance_threshold: 0.01
  feature_selection: true
  k_best: 50
```

### CICIDS2017 Settings

```yaml
data:
  use_cicids: true
  cicids2017_dir: "full_dataset/MachineLearningCVE"

training:
  balance_classes: false
```

## 📡 API Endpoints

### Backend API (`http://localhost:5000`)

#### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

#### `POST /predict`

Make predictions on network flow data.

**Request:**

```json
{
  "rows": [
    {"feature_00": 12.5, "feature_01": 3.2, ...}
  ],
  "mode": "batch"
}
```

**Response:**

```json
{
  "predictions": [
    {
      "id": 1,
      "label": 0,
      "category": "normal",
      "score": 0.02
    }
  ],
  "summary": {
    "n": 10,
    "attacks": 3,
    "normal": 7
  }
}
```

#### `GET /metrics`

Get model metrics from `artifacts/metrics.json`.

**Response:**

```json
{
  "accuracy": 0.9876,
  "precision": 0.9854,
  "recall": 0.9876,
  "f1": 0.9865,
  "roc_auc": 0.9987,
  "confusion_matrix": [[...], [...]],
  "sample_predictions": [...]
}
```

#### `GET /sample-data`

Get sample rows (raw and processed).

#### `POST /predict_user_input`

Make prediction from user-entered feature values.

**Request:**

```json
{
  "feature_00": 12.5,
  "feature_01": 3.2,
  ...
}
```

**Response:**

```json
{
  "result": "Attack",
  "prob": 0.95,
  "explanation": "The model has detected potential malicious activity..."
}
```

#### `GET /model-stats`

Get model statistics and training metrics.

## 📖 Evaluation Guide

### Running Evaluation

```bash
cd model_preparation
python evaluate.py --config config/default.yaml
```

### Generated Outputs

1. **`artifacts/metrics.json`**: Complete metrics in JSON format
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion matrix (as array)
   - Classification report
   - Sample predictions (20 rows)

2. **`artifacts/confusion_matrix.png`**: Confusion matrix visualization

3. **`artifacts/roc_curve.png`**: ROC curve plot

4. **`artifacts/report.html`**: HTML report with all metrics

### Understanding Metrics

- **Accuracy > 0.95**: Excellent model performance
- **Precision > 0.90**: Low false positive rate (few false alarms)
- **Recall > 0.90**: High detection rate (catches most attacks)
- **F1-Score > 0.90**: Good balance between precision and recall
- **ROC-AUC > 0.95**: Excellent class separation

## 🎨 Frontend Pages

1. **Home** (`/`) - Hero section, key benefits, animated stats
2. **About** (`/about`) - E-governance information and security measures
3. **Threats** (`/threats`) - Threat analysis and case studies
4. **IDS** (`/ids`) - Main interaction page:
   - CSV upload and predictions
   - Model metrics viewer
   - Sample data viewer
5. **Dashboard** (`/dashboard`) - Analytics with interactive charts and filters

## 🧪 Testing

### ML Pipeline Tests

```bash
cd model_preparation
pytest tests/
```

### Backend Tests

```bash
cd backend
pytest tests/backend/
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
docker-compose up --build
```

### Individual Services

**Backend:**

```bash
cd backend
docker build -t jharkhand-ids-backend .
docker run -p 5000:5000 \
  -v $(pwd)/../artifacts:/app/artifacts \
  jharkhand-ids-backend
```

**Frontend:**

```bash
cd frontend
docker build -t jharkhand-ids-frontend .
docker run -p 80:80 jharkhand-ids-frontend
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'src'**
   - Ensure you're running from project root
   - Or use: `python -m src.train --config config/default.yaml`

2. **Model Not Loaded (Backend)**
   - Ensure artifacts exist: `ls artifacts/`
   - Train model first: `python train.py --config config/default.yaml`

3. **Memory Error During Training**
   - Use `--fast` flag for reduced dataset
   - Enable `use_cicids: false` and use smaller dataset
   - Reduce `n_estimators` in config

4. **CORS Errors (Frontend)**
   - Ensure backend is running on correct port
   - Check `VITE_API_URL` in frontend `.env` file

5. **Port Conflicts**
   - Frontend: Change port in `vite.config.js`
   - Backend: Change port in `backend/app.py`

## 📚 Data Format

### Expected CSV Format

- Feature columns (numeric values)
- A `label` column (integer: 0=normal, 1=anomaly/intrusion)

### Supported Data Sources

1. **CSV files**: Regular CSV files
2. **Gzipped CSV**: Files with `.csv.gz` extension
3. **CICIDS2017**: Automatic detection and preprocessing
4. **Example dataset**: Use `--example-dataset` flag

## 🔒 Security Notes

- File upload size limited to 10MB
- Input validation on backend
- CORS configured for frontend origin
- No code execution from uploaded files

## 📦 Requirements

### ML Pipeline

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0
- pyyaml >= 6.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- streamlit >= 1.28.0

### Backend

- Flask >= 2.3.0
- Flask-CORS >= 4.0.0
- joblib >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0

### Frontend

- Node.js 18+
- React 18
- Vite
- Tailwind CSS
- Chart.js

## 🤝 Contributing

1. Follow PEP 8 style guidelines (Python)
2. Add docstrings to all functions and classes
3. Include type hints
4. Write unit tests for new features
5. Update README.md with new functionality

## 📄 License

This project is part of the Jharkhand-IDS initiative.

## 🆘 Support

For questions or issues:

- Check the documentation in `web/dashboard_notes.md` for dashboard usage
- Review `config/default.yaml` for configuration options
- Check `artifacts/report.html` for model evaluation details

## 🎯 Future Enhancements

- [ ] Deep learning models (Neural Networks, Autoencoders)
- [ ] Real-time streaming support
- [ ] Advanced feature engineering (time-series features)
- [ ] Model versioning and experiment tracking
- [ ] Authentication/authorization
- [ ] Export to PDF functionality
- [ ] Advanced filtering and search

---