"""Verifica que el proveedor mock cumple el contrato (docs/contrato-proveedores.md)."""
from fastapi.testclient import TestClient

from src.providers.mock.app import app

URL = "/v1/chat/completions"
MENSAJES = [
    {"role": "system", "content": "Sos un asistente."},
    {"role": "user", "content": "Hola mundo"},
]

# ---------- Pruebas ----------
def test_forma_completa_de_la_respuesta():
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "mock-x", "messages": MENSAJES})
        assert r.status_code == 200
        body = r.json()
        # Verifica que la respuesta tiene la forma completa que exige el contrato.
        assert body["id"].startswith("chatcmpl-")
        assert body["object"] == "chat.completion"
        assert isinstance(body["created"], int)
        assert body["model"] == "mock-x"
        # Verifica que choices tiene exactamente un elemento con la forma correcta.
        assert len(body["choices"]) == 1
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert "**Hola mundo**" in choice["message"]["content"]
        assert choice["finish_reason"] == "stop"
        # Verifica que usage y x_metrics tienen la forma correcta, y que no hay claves vram_*.
        usage = body["usage"]
        assert usage["prompt_tokens"] == 5  # "Sos un asistente." + "Hola mundo"
        assert usage["completion_tokens"] == len(choice["message"]["content"].split())
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        # Verifica que x_metrics tiene la forma correcta, y que no hay claves vram_*.
        metrics = body["x_metrics"]
        assert isinstance(metrics["latency_s"], (int, float))
        assert not any(k.startswith("vram_") for k in metrics)
        # tokens_per_s se omite solo si latency_s redondeada es 0; no depende de la velocidad de la máquina.
        assert ("tokens_per_s" in metrics) == (metrics["latency_s"] > 0)
        # Verifica que x_debug tiene la forma correcta, y que contiene los mensajes originales.
        assert body["x_debug"]["messages"] == MENSAJES


def test_parametro_extra_se_ignora():
    with TestClient(app) as client:
        r = client.post(URL, json={"messages": MENSAJES, "top_p": 0.9})
        assert r.status_code == 200


def test_stream_devuelve_501():
    with TestClient(app) as client:
        r = client.post(URL, json={"messages": MENSAJES, "stream": True})
        assert r.status_code == 501
        error = r.json()["error"]
        assert error["code"] == "streaming_not_supported"
        assert error["type"] == "provider_error"

        # Decisión de diseño: las peticiones con stream=true también quedan registradas.
        assert client.get("/debug/requests").json()["count"] == 1


def test_falta_messages_da_400():
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "mock"})
        assert r.status_code == 400
        error = r.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "messages"


def test_health_ready():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ready"}


def test_debug_requests_respeta_history_size(monkeypatch):
    monkeypatch.setenv("MOCK_HISTORY_SIZE", "2")
    with TestClient(app) as client:
        for texto in ("uno", "dos", "tres"):
            client.post(URL, json={"messages": [{"role": "user", "content": texto}]})

        body = client.get("/debug/requests").json()
        assert body["count"] == 2
        assert body["max"] == 2
        contenidos = [e["request"]["messages"][0]["content"] for e in body["requests"]]
        assert contenidos == ["dos", "tres"]  # se perdió la primera; la más nueva al final
        assert "received_at" in body["requests"][0]


def test_determinismo():
    with TestClient(app) as client:
        a = client.post(URL, json={"messages": MENSAJES}).json()
        b = client.post(URL, json={"messages": MENSAJES}).json()
        assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]


def test_mock_latency_se_refleja_en_metricas(monkeypatch):
    monkeypatch.setenv("MOCK_LATENCY", "0.05")
    with TestClient(app) as client:
        r = client.post(URL, json={"messages": MENSAJES})
        metrics = r.json()["x_metrics"]
        assert metrics["latency_s"] >= 0.05
        # Con latencia > 0 garantizada, tokens_per_s tiene que estar presente.
        assert metrics["tokens_per_s"] > 0


def test_json_mal_formado_da_400():
    with TestClient(app) as client:
        r = client.post(URL, content=b'{"messages": [', headers={"Content-Type": "application/json"})
        assert r.status_code == 400
        error = r.json()["error"]
        assert error["code"] == "invalid_request"
        # Sin campo identificable: param no debe ser la posición del cuerpo donde falló el parseo.
        assert error["param"] is None


def test_ruta_inexistente_usa_formato_de_error():
    with TestClient(app) as client:
        r = client.get("/no-existe")
        assert r.status_code == 404
        error = r.json()["error"]
        assert set(error) == {"message", "type", "param", "code"}
        assert error["code"] is None
