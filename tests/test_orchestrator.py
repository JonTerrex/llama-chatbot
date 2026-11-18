"""Verifica el orquestador con un proveedor simulado (httpx2.MockTransport), sin red ni servicios reales."""
import json
import logging

import httpx2
import pytest
from fastapi.testclient import TestClient

import src.orchestrator.app as orq
from src.orchestrator.app import app

URL = "/v1/chat/completions"
SYSTEM_DE_PRUEBA = "Sos el system prompt de prueba."
# Host reconocible para comprobar que nunca se filtra en los mensajes de error.
HOST_PROVEEDOR = "proveedor-secreto.internal"
MENSAJES = [{"role": "user", "content": "Hola"}]

RESPUESTA_PROVEEDOR = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1762170000,
    "model": "mock",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hola!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    "x_metrics": {"latency_s": 0.123, "tokens_per_s": 8.13},
    "x_debug": {"messages": [], "params": {}},
}


class ProveedorFalso:
    """Handler del MockTransport: guarda cada body recibido y responde con `responder` (cada test lo cambia)."""

    def __init__(self):
        self.recibidos: list[dict] = []
        self.responder = lambda request: httpx2.Response(200, json=RESPUESTA_PROVEEDOR)

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.recibidos.append(json.loads(request.content))
        return self.responder(request)


@pytest.fixture
def proveedor(tmp_path, monkeypatch):
    """Registro y prompt de prueba en tmp_path, y crear_cliente sustituido por uno con MockTransport."""
    registro = tmp_path / "providers.yaml"
    registro.write_text(
        "providers:\n"
        "  mock:\n"
        f"    url: http://{HOST_PROVEEDOR}:9000\n"
        "  local:\n"
        "    url: http://otro.internal:9001\n",
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(SYSTEM_DE_PRUEBA, encoding="utf-8")

    monkeypatch.setenv("PROVIDERS_FILE", str(registro))
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(prompt))
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("PROVIDER_TIMEOUT_S", raising=False)
    # Memoria deshabilitada por defecto, aunque el entorno real la tenga definida.
    for variable in ("MEMORY_URL", "DEV_USER_ID", "MEMORY_HISTORY_LIMIT", "MEMORY_TIMEOUT_S"):
        monkeypatch.delenv(variable, raising=False)

    falso = ProveedorFalso()
    monkeypatch.setattr(
        orq, "crear_cliente", lambda config: httpx2.AsyncClient(transport=httpx2.MockTransport(falso.handler))
    )
    return falso


def assert_forma_de_error(body):
    assert set(body["error"]) >= {"message", "type", "param", "code"}


# ---------- Reenvío ----------

def test_camino_feliz(proveedor):
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "mock", "messages": MENSAJES})
        assert r.status_code == 200
        # Pass-through: idéntica a la del proveedor, extensiones x_ incluidas.
        assert r.json() == RESPUESTA_PROVEEDOR

    assert len(proveedor.recibidos) == 1
    enviado = proveedor.recibidos[0]
    assert enviado["model"] == "mock"
    assert enviado["stream"] is False
    assert enviado["messages"] == [{"role": "system", "content": SYSTEM_DE_PRUEBA}] + MENSAJES


def test_system_del_cliente_va_despues_del_nuestro(proveedor):
    mensajes = [{"role": "system", "content": "System del cliente"}, {"role": "user", "content": "Hola"}]
    with TestClient(app) as client:
        assert client.post(URL, json={"model": "mock", "messages": mensajes}).status_code == 200

    assert proveedor.recibidos[0]["messages"] == [{"role": "system", "content": SYSTEM_DE_PRUEBA}] + mensajes


# ---------- Resolución del modelo ----------

def test_sin_model_usa_default_model(proveedor, monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "local")
    with TestClient(app) as client:
        assert client.post(URL, json={"messages": MENSAJES}).status_code == 200
    assert proveedor.recibidos[0]["model"] == "local"


def test_sin_model_ni_default_da_400(proveedor):
    with TestClient(app) as client:
        r = client.post(URL, json={"messages": MENSAJES})
        assert r.status_code == 400
        error = r.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["param"] == "model"
    assert proveedor.recibidos == []


def test_model_vacio_equivale_a_ausente(proveedor, monkeypatch):
    # Sin DEFAULT_MODEL: mismo 400 que si faltara model.
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "", "messages": MENSAJES})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "model"

    # Con DEFAULT_MODEL: se usa el default (la config se relee en el lifespan del nuevo TestClient).
    monkeypatch.setenv("DEFAULT_MODEL", "local")
    with TestClient(app) as client:
        assert client.post(URL, json={"model": "", "messages": MENSAJES}).status_code == 200
    assert proveedor.recibidos[0]["model"] == "local"


def test_modelo_desconocido_da_404(proveedor):
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "gpt-99", "messages": MENSAJES})
        assert r.status_code == 404
        error = r.json()["error"]
        assert error["code"] == "model_not_found"
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "model"
    assert proveedor.recibidos == []


# ---------- Qué se reenvía ----------

def test_stream_da_501_sin_llamar_al_proveedor(proveedor):
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "mock", "messages": MENSAJES, "stream": True})
        assert r.status_code == 501
        error = r.json()["error"]
        assert error["code"] == "streaming_not_supported"
        assert error["param"] == "stream"
    assert proveedor.recibidos == []


def test_parametro_extra_no_se_reenvia_y_se_loguea(proveedor, caplog):
    with caplog.at_level(logging.WARNING, logger="orchestrator"):
        with TestClient(app) as client:
            r = client.post(URL, json={"model": "mock", "messages": MENSAJES, "top_p": 0.9})
            assert r.status_code == 200

    assert "top_p" not in proveedor.recibidos[0]
    assert any("top_p" in rec.getMessage() for rec in caplog.records if rec.name == "orchestrator")


def test_temperature_y_max_tokens_solo_si_vienen(proveedor):
    with TestClient(app) as client:
        client.post(URL, json={"model": "mock", "messages": MENSAJES, "temperature": 0.2, "max_tokens": 64})
        client.post(URL, json={"model": "mock", "messages": MENSAJES})

    con, sin = proveedor.recibidos
    assert con["temperature"] == 0.2
    assert con["max_tokens"] == 64
    # Omitidos, no enviados como null: el proveedor aplica sus propios defaults.
    assert "temperature" not in sin
    assert "max_tokens" not in sin
    assert set(sin) == {"model", "messages", "stream"}


# ---------- Fallos del proveedor ----------

def _lanza(excepcion):
    """Responder que simula un fallo de transporte; el texto incluye la URL para detectar si se filtra."""
    def responder(request):
        raise excepcion(f"fallo hablando con {request.url}", request=request)
    return responder


@pytest.mark.parametrize(
    "responder, status, code",
    [
        (_lanza(httpx2.ConnectError), 502, "provider_error"),
        # Connect timeout = no aceptó la conexión: falla de conexión, no timeout de generación.
        (_lanza(httpx2.ConnectTimeout), 502, "provider_error"),
        (_lanza(httpx2.ReadTimeout), 504, "provider_timeout"),
        (lambda request: httpx2.Response(500, text=f"Internal error en {request.url}"), 502, "provider_error"),
        (lambda request: httpx2.Response(200, text=f"<html>{request.url}</html>"), 502, "provider_error"),
    ],
    ids=["connect_error", "connect_timeout", "read_timeout", "status_500", "cuerpo_no_json"],
)
def test_fallos_del_proveedor(proveedor, responder, status, code):
    proveedor.responder = responder
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "mock", "messages": MENSAJES})

    assert r.status_code == status
    body = r.json()
    assert_forma_de_error(body)
    error = body["error"]
    assert error["code"] == code
    assert error["type"] == "provider_error"
    assert error["x_provider"]["name"] == "mock"
    # La topología interna no debe llegar al cliente.
    assert HOST_PROVEEDOR not in error["message"]
    assert "http://" not in error["message"]


# ---------- Endpoints auxiliares ----------

def test_models_en_orden_del_registro(proveedor):
    with TestClient(app) as client:
        body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["mock", "local"]  # orden del YAML, no alfabético
    for m in body["data"]:
        assert m["object"] == "model"
        assert m["owned_by"] == "llama-chatbot"
        assert isinstance(m["created"], int)


def test_health(proveedor):
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


# ---------- Peticiones inválidas ----------

def test_falta_messages_da_400(proveedor):
    with TestClient(app) as client:
        r = client.post(URL, json={"model": "mock"})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "messages"


def test_json_mal_formado_da_400(proveedor):
    with TestClient(app) as client:
        r = client.post(URL, content=b'{"messages": [', headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] is None


def test_ruta_inexistente_usa_formato_de_error(proveedor):
    with TestClient(app) as client:
        r = client.get("/no-existe")
    assert r.status_code == 404
    assert_forma_de_error(r.json())
    assert r.json()["error"]["code"] is None


# ---------- x_conversation_id (contrato de memoria, 4.1 y 4.4) ----------

CONV = "c-123"


@pytest.fixture
def memoria_habilitada(proveedor, monkeypatch):
    """MEMORY_URL definida. En el bloque 3a el orquestador todavía no llama a la memoria."""
    monkeypatch.setenv("MEMORY_URL", "http://memoria-secreta.internal:8000")
    monkeypatch.setenv("DEV_USER_ID", "dev-user")
    return proveedor


def pedir(client, **campos):
    body = {"model": "mock", "messages": MENSAJES}
    body.update(campos)
    return client.post(URL, json=body)


def assert_error(r, status, code, param, type_="invalid_request_error"):
    assert r.status_code == status
    body = r.json()
    assert_forma_de_error(body)
    assert body["error"]["code"] == code
    assert body["error"]["param"] == param
    assert body["error"]["type"] == type_
    return body["error"]


def test_sin_x_conversation_id_con_memoria_habilitada_es_igual_que_hoy(memoria_habilitada):
    # Regresión: con memoria configurada pero sin el campo, mismo reenvío que sin memoria.
    with TestClient(app) as client:
        r = pedir(client)
    assert r.status_code == 200
    assert r.json() == RESPUESTA_PROVEEDOR
    assert memoria_habilitada.recibidos[0]["messages"] == [{"role": "system", "content": SYSTEM_DE_PRUEBA}] + MENSAJES


@pytest.mark.parametrize(
    "valor",
    ["", "a" * 65, "con espacio", "a/b", "abc\n"],
    ids=["vacio", "65_chars", "espacio", "barra", "salto_de_linea_final"],
)
def test_x_conversation_id_con_formato_invalido_da_400(memoria_habilitada, valor):
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=valor)
    assert_error(r, 400, "invalid_request", "x_conversation_id")
    assert memoria_habilitada.recibidos == []


@pytest.mark.parametrize("valor", [123, ["c-123"]], ids=["numero", "lista"])
def test_x_conversation_id_no_string_da_400_de_validacion(memoria_habilitada, valor):
    # Lo resuelve el handler de validación de pydantic (observado al ejecutar, no supuesto).
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=valor)
    error = assert_error(r, 400, "invalid_request", "x_conversation_id")
    assert error["message"] == "Input should be a valid string"
    assert memoria_habilitada.recibidos == []


@pytest.mark.parametrize(
    "mensajes",
    [
        [{"role": "system", "content": "Sos..."}],
        [{"role": "assistant", "content": "Hola"}],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [{"role": "system", "content": "Sos..."}, {"role": "user", "content": "Hola"}],
        [{"role": "assistant", "content": "x"}, {"role": "user", "content": "Hola"}],
        [],
    ],
    ids=["solo_system", "solo_assistant", "dos_user", "system_y_user", "assistant_y_user", "vacio"],
)
def test_con_x_conversation_id_messages_debe_ser_un_solo_user(memoria_habilitada, mensajes):
    # [] lo corta pydantic (min_length=1), con el mismo 400 y param.
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV, messages=mensajes)
    assert_error(r, 400, "invalid_request", "messages")
    assert memoria_habilitada.recibidos == []


def test_memoria_deshabilitada_da_503_sin_llamar_al_proveedor(proveedor, caplog):
    with caplog.at_level(logging.WARNING, logger="orchestrator"):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV)
    assert_error(r, 503, "memory_unavailable", None, type_="memory_error")
    assert proveedor.recibidos == []
    assert any(CONV in rec.getMessage() for rec in caplog.records if rec.name == "orchestrator")


@pytest.mark.parametrize(
    "campos, status, code, param",
    [
        # Formato de x_conversation_id y regla de messages antes que memoria deshabilitada.
        ({"x_conversation_id": "a/b"}, 400, "invalid_request", "x_conversation_id"),
        ({"x_conversation_id": CONV, "messages": MENSAJES * 2}, 400, "invalid_request", "messages"),
        # Stream (y modelo) antes que todo lo de memoria.
        ({"x_conversation_id": "a/b", "stream": True}, 501, "streaming_not_supported", "stream"),
        ({"x_conversation_id": CONV, "stream": True}, 501, "streaming_not_supported", "stream"),
        ({"x_conversation_id": "a/b", "model": "gpt-99"}, 404, "model_not_found", "model"),
        # Pydantic antes que todo: tipo inválido con stream sigue siendo 400 de validación.
        ({"x_conversation_id": 123, "stream": True}, 400, "invalid_request", "x_conversation_id"),
    ],
    ids=[
        "id_invalido_antes_que_503", "dos_user_antes_que_503", "stream_antes_que_id_invalido",
        "stream_antes_que_503", "modelo_antes_que_id_invalido", "pydantic_antes_que_stream",
    ],
)
def test_orden_de_evaluacion_con_memoria_deshabilitada(proveedor, campos, status, code, param):
    # Orden: pydantic -> modelo (400/404) -> stream (501) -> formato/messages (400) -> memoria (503).
    with TestClient(app) as client:
        r = pedir(client, **campos)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert r.json()["error"]["param"] == param
    assert proveedor.recibidos == []


def test_memoria_habilitada_acepta_el_campo_y_no_lo_reenvia(memoria_habilitada):
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV)
    assert r.status_code == 200
    assert r.json() == RESPUESTA_PROVEEDOR
    (enviado,) = memoria_habilitada.recibidos
    assert "x_conversation_id" not in enviado
    assert set(enviado) == {"model", "messages", "stream"}


def test_x_conversation_id_no_aparece_en_el_log_de_extras(memoria_habilitada, caplog):
    with caplog.at_level(logging.WARNING, logger="orchestrator"):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV, top_p=0.9)
    assert r.status_code == 200
    extras = [rec.getMessage() for rec in caplog.records if "no soportados" in rec.getMessage()]
    assert len(extras) == 1
    assert "top_p" in extras[0]
    assert "x_conversation_id" not in extras[0]
