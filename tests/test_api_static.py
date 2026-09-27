"""
Unified Enterprise RAG System — Static and SPA Routing Tests.

Tests that root ('/'), SPA client paths, and static assets are properly routed,
and that non-existent API routes ('/api/*') return 404 JSON instead of HTML.
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from config.settings import Config


@pytest.fixture
def client():
    """Provides a TestClient instance for the FastAPI application."""
    return TestClient(app)


def test_root_endpoint_returns_ok(client):
    """Verifies that GET / returns a 200 OK (HTML or JSON status)."""
    response = client.get("/")
    assert response.status_code == 200


def test_nonexistent_api_route_returns_404_json(client):
    """Verifies that non-existent /api/* routes return structured 404 JSON."""
    response = client.get("/api/v1/unknown_resource_path")
    assert response.status_code == 404
    assert response.headers.get("content-type", "").startswith("application/json")
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"


def test_spa_serving_with_mock_dist(tmp_path, monkeypatch):
    """
    Verifies that when a built frontend dist directory is present,
    GET / and client paths return index.html, while static assets are served.
    """
    # Create a mock dist structure
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)

    index_html = dist_dir / "index.html"
    index_html.write_text("<!DOCTYPE html><html><body><div id='root'>Mock React SPA</div></body></html>")

    mock_js = assets_dir / "index-abc123.js"
    mock_js.write_text("console.log('mock bundle');")

    # Set FRONTEND_DIST_DIR in main
    monkeypatch.setattr("api.main.FRONTEND_DIST_DIR", dist_dir)

    client = TestClient(app)

    # 1. GET / returns mock index.html
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Mock React SPA" in res_root.text

    # 2. Client-side route (e.g. /dashboard) returns index.html
    res_client = client.get("/dashboard")
    assert res_client.status_code == 200
    assert "Mock React SPA" in res_client.text

    # 3. Direct static file in dist
    res_index_direct = client.get("/index.html")
    assert res_index_direct.status_code == 200
    assert "Mock React SPA" in res_index_direct.text

    # 4. API routes still return 404 JSON, not index.html
    res_api_404 = client.get("/api/v1/not_found_endpoint")
    assert res_api_404.status_code == 404
    assert res_api_404.headers.get("content-type", "").startswith("application/json")
    assert "Mock React SPA" not in res_api_404.text
