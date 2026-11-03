"""Verifica que la API arranca en modo MOCK sin torch/transformers instalados."""
from fastapi.testclient import TestClient


def test_api_en_modo_mock(monkeypatch):
    monkeypatch.setenv("USE_MOCK", "true")
    from src.api import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["backend"] == "mock"
        assert health["model_loaded"] is True

        r = client.post("/v1/chat", json={"historial": [{"role": "user", "content": "Hola"}]})
        assert r.status_code == 200
        texto = r.json()["respuesta"]
        assert "**[MOCK]**" in texto
        assert "**Hola**" in texto


def test_metricas_en_modo_mock(monkeypatch):
    monkeypatch.setenv("USE_MOCK", "true")
    from src.api import app

    with TestClient(app) as client:
        r = client.post("/v1/chat", json={"historial": [{"role": "user", "content": "Hola"}]})
        assert r.status_code == 200
        metricas = r.json()["metricas"]
        assert isinstance(metricas["latencia_s"], (int, float))
