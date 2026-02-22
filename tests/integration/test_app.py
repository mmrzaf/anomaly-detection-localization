from fastapi.testclient import TestClient


def test_full_application_flow(client: TestClient):
    """Test complete application flow."""
    # Test root endpoint
    root_response = client.get("/")
    assert root_response.status_code == 200

    # Test health check
    health_response = client.get("/api/health/")
    assert health_response.status_code == 200

    # Test metrics
    metrics_response = client.get("/api/metrics/")
    assert metrics_response.status_code == 200

    # Test OpenAPI docs accessibility
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
