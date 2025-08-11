"""Basic API tests for Spectra AI (Responder ASGI app)."""
import pytest_asyncio
import httpx
from app.main import api  # api is the Responder API instance


@pytest_asyncio.fixture(scope="module")
async def client():
    transport = httpx.ASGITransport(app=api)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


async def test_root_endpoint(client):
    """Test the root endpoint returns service info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "Spectra AI" in data["service"]


async def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


async def test_status_endpoint(client):
    """Test the status endpoint."""
    response = await client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ai_provider" in data
    assert "model" in data


async def test_models_endpoint(client):
    """Test the models list endpoint."""
    response = await client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "available" in data
    assert "timestamp" in data


async def test_metrics_endpoint(client):
    """Test the metrics endpoint."""
    response = await client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "request_count" in data


async def test_invalid_endpoint(client):
    """Test that invalid endpoints return 404."""
    response = await client.get("/invalid/endpoint")
    assert response.status_code == 404
