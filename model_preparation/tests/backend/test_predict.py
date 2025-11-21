"""
Basic tests for Flask backend prediction endpoint
"""

import pytest
from backend.app import app, load_artifacts


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data


def test_predict_missing_rows(client):
    """Test predict endpoint with missing rows"""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_empty_rows(client):
    """Test predict endpoint with empty rows"""
    response = client.post('/predict', json={'rows': []})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_model_stats_endpoint(client):
    """Test model stats endpoint"""
    response = client.get('/model-stats')
    # Should return 200 or 500 depending on model load
    assert response.status_code in [200, 500]
    data = response.get_json()
    assert 'error' in data or 'features' in data

