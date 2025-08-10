"""Basic API tests for Spectra AI"""
import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test the root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "Spectra AI" in data["service"]


def test_health_endpoint(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_status_endpoint(client: TestClient):
    """Test the status endpoint."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ai_provider" in data
    assert "model" in data


def test_models_endpoint(client: TestClient):
    """Test the models list endpoint."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "available" in data
    assert "timestamp" in data


def test_metrics_endpoint(client: TestClient):
    """Test the metrics endpoint."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "request_count" in data


def test_invalid_endpoint(client: TestClient):
    """Test that invalid endpoints return 404."""
    response = client.get("/invalid/endpoint")
    assert response.status_code == 404
