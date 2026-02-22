from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/health/")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "details" in data  # Should include details by default


def test_metrics_endpoint(client: TestClient):
    """Test metrics endpoint."""
    response = client.get("/api/metrics/")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "app_info" in response.text


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "service" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"
