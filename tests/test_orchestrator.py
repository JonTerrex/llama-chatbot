"""Verifica el orquestador con un proveedor simulado (httpx2.MockTransport), sin red ni servicios reales."""
import asyncio
import json
import logging
import uuid

import httpx2
import pytest
from fastapi.testclient import TestClient

import src.orchestrator.app as orq
from src.orchestrator.app import app
from src.orchestrator.logging_config import LOGGERS_SILENCIADOS, configurar_logging
from tests.test_memory_client import MemoriaFalsa

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


@pytest.fixture(autouse=True)
def niveles_de_loggers_http():
    """
    El lifespan fija httpx2/httpcore2 en WARNING para todo el proceso. Cada test arranca
    como un proceso nuevo (sin nivel propio, NOTSET) y al final se restaura el original,
    para no afectar a otros tests ni hacer pasar un test por un nivel que dejó otro.
    """
    originales = {n: logging.getLogger(n).level for n in LOGGERS_SILENCIADOS}
    for nombre in LOGGERS_SILENCIADOS:
        logging.getLogger(nombre).setLevel(logging.NOTSET)
    yield
    for nombre, nivel in originales.items():
        logging.getLogger(nombre).setLevel(nivel)


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


# ---------- Errores del proveedor que se propagan ----------

# Texto del proveedor que no debe llegar al cliente ni al log, venga en el campo que venga.
SENTINELA_PROVEEDOR = "SENTINELA-DEL-PROVEEDOR-77"


def error_del_proveedor(status, code, type_="provider_error"):
    """Responder con un error en el formato del proveedor, con el sentinela en todos los textos libres."""
    def responder(request):
        return httpx2.Response(status, json={"error": {
            "message": f"detalle interno {SENTINELA_PROVEEDOR}", "type": type_, "param": "messages",
            "code": code, "x_provider": {"name": "mock", "state": f"loading {SENTINELA_PROVEEDOR}"},
        }})
    return responder


def textos_de_log(caplog):
    return [f"{rec.getMessage()} {rec.args!r}" for rec in caplog.records]


@pytest.mark.parametrize(
    "status_prov, code_prov, status, code, type_, param",
    [
        (400, "context_length_exceeded", 400, "context_length_exceeded", "invalid_request_error", "messages"),
        (503, "insufficient_vram", 503, "insufficient_vram", "provider_error", None),
        (503, "provider_unavailable", 503, "provider_unavailable", "provider_error", None),
        (500, "inference_error", 502, "provider_error", "provider_error", None),
        # Cualquier otro no-2xx, aunque el code sea conocido: 502 genérico.
        (501, "streaming_not_supported", 502, "provider_error", "provider_error", None),
        (503, "context_length_exceeded", 502, "provider_error", "provider_error", None),
        (400, "insufficient_vram", 502, "provider_error", "provider_error", None),
        (429, None, 502, "provider_error", "provider_error", None),
        # Code fuera de la lista blanca.
        (400, "codigo_inventado_xyz", 502, "provider_error", "provider_error", None),
        (503, "codigo_inventado_xyz", 502, "provider_error", "provider_error", None),
    ],
    ids=["400_contexto", "503_vram", "503_no_disponible", "500_inferencia", "501_otro", "503_code_de_400",
         "400_code_de_503", "429_sin_code", "400_code_desconocido", "503_code_desconocido"],
)
def test_errores_del_proveedor_se_traducen(proveedor, caplog, status_prov, code_prov, status, code, type_, param):
    proveedor.responder = error_del_proveedor(status_prov, code_prov)
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            r = client.post(URL, json={"model": "mock", "messages": MENSAJES})

    error = assert_error(r, status, code, param, type_)
    # x_provider solo con el nombre: el state del proveedor (texto libre) no se copia.
    assert error["x_provider"] == {"name": "mock"}
    assert set(error) == {"message", "type", "param", "code", "x_provider"}
    assert SENTINELA_PROVEEDOR not in r.text
    assert HOST_PROVEEDOR not in r.text
    assert all(SENTINELA_PROVEEDOR not in t for t in textos_de_log(caplog))
    # Un code desconocido tampoco se loguea: es texto que manda el proveedor.
    assert all("codigo_inventado_xyz" not in t for t in textos_de_log(caplog))
    if status_prov == 500:
        assert "inference_error" in error["message"]  # el code va en el mensaje, armado acá
    # Una sola línea WARNING del orquestador, con el modelo y el status del proveedor.
    (aviso,) = [rec for rec in caplog.records if rec.name == "orchestrator" and rec.levelno == logging.WARNING]
    assert "mock" in aviso.getMessage() and str(status_prov) in aviso.getMessage()


@pytest.mark.parametrize(
    "cuerpo",
    [
        f"<html>{SENTINELA_PROVEEDOR}</html>",
        json.dumps({"error": {"message": SENTINELA_PROVEEDOR}}),  # JSON sin code
        json.dumps({"error": f"insufficient_vram {SENTINELA_PROVEEDOR}"}),  # error no es objeto
        json.dumps(["insufficient_vram"]),  # el cuerpo no es objeto
        json.dumps({"error": {"code": ["insufficient_vram"]}}),  # code no es str
    ],
    ids=["no_json", "sin_code", "error_no_objeto", "cuerpo_lista", "code_no_str"],
)
def test_error_del_proveedor_sin_code_valido_da_502_generico(proveedor, caplog, cuerpo):
    proveedor.responder = lambda request: httpx2.Response(
        503, content=cuerpo.encode(), headers={"content-type": "application/json"}
    )
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            r = client.post(URL, json={"model": "mock", "messages": MENSAJES})

    error = assert_error(r, 502, "provider_error", None, "provider_error")
    assert SENTINELA_PROVEEDOR not in r.text
    assert all(SENTINELA_PROVEEDOR not in t for t in textos_de_log(caplog))


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
HOST_MEMORIA = "memoria-secreta.internal"
DEV_USER = "dev-user"
# Lo que devuelve la memoria falsa por defecto al leer: incluye created_at, que no debe llegar al proveedor.
HISTORIAL_GUARDADO = [
    {"role": "user", "content": "Me llamo Ana", "created_at": 1794305797},
    {"role": "assistant", "content": "Hola, Ana.", "created_at": 1794305798},
]


def responder_memoria_por_defecto(request: httpx2.Request) -> httpx2.Response:
    """GET: historial de prueba. POST: turno guardado (201)."""
    if request.method == "GET":
        return httpx2.Response(200, json={"conversation_id": CONV, "messages": HISTORIAL_GUARDADO})
    return httpx2.Response(201, json={"conversation_id": CONV, "stored": 2})


@pytest.fixture
def memoria(monkeypatch):
    """
    Memoria falsa (la de test_memory_client) inyectada sustituyendo crear_cliente_memoria,
    igual que el fixture `proveedor` sustituye crear_cliente. Registra cada request recibido.
    """
    falsa = MemoriaFalsa()
    falsa.responder = responder_memoria_por_defecto
    falsa.clientes = []  # los httpx2.AsyncClient creados, para comprobar que se cierran

    def crear(config):
        cliente = httpx2.AsyncClient(transport=httpx2.MockTransport(falsa.handler))
        falsa.clientes.append(cliente)
        return cliente

    monkeypatch.setattr(orq, "crear_cliente_memoria", crear)
    return falsa


@pytest.fixture
def memoria_habilitada(proveedor, memoria, monkeypatch):
    """MEMORY_URL definida, con la memoria falsa detrás. Devuelve el proveedor (como en el bloque 3a)."""
    monkeypatch.setenv("MEMORY_URL", f"http://{HOST_MEMORIA}:8000")
    monkeypatch.setenv("DEV_USER_ID", DEV_USER)
    return proveedor


def posts_a_memoria(memoria):
    return [r for r in memoria.recibidos if r.method == "POST"]


def system_mas(*mensajes):
    return [{"role": "system", "content": SYSTEM_DE_PRUEBA}, *mensajes]


HISTORIAL_PROYECTADO = [{"role": m["role"], "content": m["content"]} for m in HISTORIAL_GUARDADO]


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
    assert r.json() == {**RESPUESTA_PROVEEDOR, "x_memory": {"status": "saved"}}
    (enviado,) = memoria_habilitada.recibidos
    assert "x_conversation_id" not in enviado
    assert set(enviado) == {"model", "messages", "stream"}
    # Con memoria, el cuerpo lleva el historial entre nuestro system y el mensaje nuevo.
    assert enviado["messages"] == system_mas(*HISTORIAL_PROYECTADO, *MENSAJES)


def test_x_conversation_id_no_aparece_en_el_log_de_extras(memoria_habilitada, caplog):
    with caplog.at_level(logging.WARNING, logger="orchestrator"):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV, top_p=0.9)
    assert r.status_code == 200
    extras = [rec.getMessage() for rec in caplog.records if "no soportados" in rec.getMessage()]
    assert len(extras) == 1
    assert "top_p" in extras[0]
    assert "x_conversation_id" not in extras[0]


# ---------- Flujo con memoria (contrato de memoria, 4.2 a 4.4) ----------

CONTENIDO_SECRETO = "CONTENIDO-PRIVADO-DEL-USUARIO-42"


def respuesta_con_contenido(contenido):
    """Copia de RESPUESTA_PROVEEDOR con otro choices[0].message.content (puede ser inválido a propósito)."""
    datos = json.loads(json.dumps(RESPUESTA_PROVEEDOR))
    datos["choices"][0]["message"]["content"] = contenido
    return datos


def registros_propios(caplog):
    # Solo loggers del orquestador: los INFO de httpx2 (con URL) son del bloque 3c.
    return [rec for rec in caplog.records if rec.name.startswith("orchestrator")]


def test_camino_feliz_con_historial(memoria_habilitada, memoria):
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV, messages=[{"role": "user", "content": "¿Cómo me llamo?"}])

    assert r.status_code == 200
    body = r.json()
    assert body.pop("x_memory") == {"status": "saved"}
    assert body == RESPUESTA_PROVEEDOR  # el resto es el JSON del proveedor tal cual

    # Proveedor: [system] + historial sin created_at + nuevo, y sin x_conversation_id.
    (enviado,) = memoria_habilitada.recibidos
    assert enviado["messages"] == system_mas(*HISTORIAL_PROYECTADO, {"role": "user", "content": "¿Cómo me llamo?"})
    assert "x_conversation_id" not in enviado

    # Memoria: un GET y un POST, en ese orden.
    get, post = memoria.recibidos
    assert get.method == "GET"
    assert dict(get.url.params) == {"user_id": DEV_USER, "limit": "20"}
    assert post.method == "POST"
    guardado = json.loads(post.content)
    assert set(guardado) == {"user_id", "turn_id", "messages"}
    assert guardado["user_id"] == DEV_USER
    assert uuid.UUID(guardado["turn_id"]).version == 4
    assert guardado["messages"] == [
        {"role": "user", "content": "¿Cómo me llamo?"},
        {"role": "assistant", "content": "Hola!"},
    ]


def test_historial_vacio(memoria_habilitada, memoria):
    memoria.responder = lambda r: (
        httpx2.Response(200, json={"conversation_id": CONV, "messages": []}) if r.method == "GET"
        else httpx2.Response(201, json={"conversation_id": CONV, "stored": 2})
    )
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV)
    assert r.status_code == 200
    assert r.json()["x_memory"] == {"status": "saved"}
    assert memoria_habilitada.recibidos[0]["messages"] == system_mas(*MENSAJES)
    assert len(posts_a_memoria(memoria)) == 1


async def _memoria_lenta(request):
    await asyncio.sleep(2)
    return httpx2.Response(200, json={"conversation_id": CONV, "messages": []})


def _memoria_rechaza_conexion(request):
    raise httpx2.ConnectError(f"Connection refused: {request.url}", request=request)


@pytest.mark.parametrize(
    "responder",
    [
        _memoria_rechaza_conexion,
        _memoria_lenta,
        lambda r: httpx2.Response(503, json={"error": {"message": "x"}}),
        lambda r: httpx2.Response(200, text="<html>no soy json</html>"),
    ],
    ids=["conexion", "timeout", "estado_http", "respuesta_invalida"],
)
def test_lectura_fallida_da_503_sin_llamar_al_proveedor(memoria_habilitada, memoria, monkeypatch, responder):
    monkeypatch.setenv("MEMORY_TIMEOUT_S", "0.1")  # para el caso timeout; los demás no esperan
    memoria.responder = responder
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV)

    error = assert_error(r, 503, "memory_unavailable", None, type_="memory_error")
    # La causa interna no llega al cliente.
    for detalle in ("conexion", "timeout", "estado_http", "respuesta_invalida", HOST_MEMORIA):
        assert detalle not in error["message"]
    assert memoria_habilitada.recibidos == []
    assert posts_a_memoria(memoria) == []


@pytest.mark.parametrize(
    "responder, status, code",
    [
        (_lanza(httpx2.ConnectError), 502, "provider_error"),
        (_lanza(httpx2.ReadTimeout), 504, "provider_timeout"),
        (lambda request: httpx2.Response(500, text="boom"), 502, "provider_error"),
    ],
    ids=["connect_error_502", "read_timeout_504", "status_500_502"],
)
def test_proveedor_falla_no_se_guarda_nada(memoria_habilitada, memoria, responder, status, code):
    memoria_habilitada.responder = responder
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV)

    # Mismo error de hoy: sin x_memory ni nada agregado.
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["type"] == "provider_error"
    assert body["error"]["x_provider"] == {"name": "mock"}
    assert posts_a_memoria(memoria) == []
    assert [r.method for r in memoria.recibidos] == ["GET"]  # sí se leyó el historial antes


@pytest.mark.parametrize(
    "status_prov, code, status",
    [(400, "context_length_exceeded", 400), (503, "insufficient_vram", 503), (503, "provider_unavailable", 503)],
    ids=["400_contexto", "503_vram", "503_no_disponible"],
)
def test_error_propagado_del_proveedor_no_guarda_el_turno(memoria_habilitada, memoria, status_prov, code, status):
    memoria_habilitada.responder = error_del_proveedor(status_prov, code)
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV)

    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"}  # sin x_memory
    assert body["error"]["code"] == code
    assert posts_a_memoria(memoria) == []  # memoria 4.2 paso 4: ni el mensaje del usuario
    assert [m.method for m in memoria.recibidos] == ["GET"]


def test_guardado_fallido_da_200_con_write_failed(memoria_habilitada, memoria, caplog):
    memoria.responder = lambda r: (
        responder_memoria_por_defecto(r) if r.method == "GET"
        else httpx2.Response(503, text=f"base caída {CONTENIDO_SECRETO}")
    )
    memoria_habilitada.responder = lambda r: httpx2.Response(200, json=respuesta_con_contenido(CONTENIDO_SECRETO))

    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV, messages=[{"role": "user", "content": CONTENIDO_SECRETO}])

    assert r.status_code == 200
    body = r.json()
    assert body.pop("x_memory") == {"status": "write_failed"}
    assert body == respuesta_con_contenido(CONTENIDO_SECRETO)

    (post,) = posts_a_memoria(memoria)
    turn_id = json.loads(post.content)["turn_id"]
    propios = registros_propios(caplog)
    assert any(CONV in rec.getMessage() and turn_id in rec.getMessage() for rec in propios)
    for rec in propios:
        texto = rec.getMessage() + repr(rec.args)
        assert CONTENIDO_SECRETO not in texto
        assert HOST_MEMORIA not in texto
        assert DEV_USER not in texto


@pytest.mark.parametrize(
    "respuesta",
    [
        {k: v for k, v in RESPUESTA_PROVEEDOR.items() if k != "choices"},
        {**RESPUESTA_PROVEEDOR, "choices": []},
        {**RESPUESTA_PROVEEDOR, "choices": [{"index": 0, "finish_reason": "stop"}]},
        {**RESPUESTA_PROVEEDOR, "choices": "no es lista"},
        respuesta_con_contenido(None),
        respuesta_con_contenido(42),
    ],
    ids=["sin_choices", "choices_vacio", "sin_message", "choices_no_lista", "content_null", "content_no_str"],
)
def test_respuesta_del_proveedor_sin_content_valido(memoria_habilitada, memoria, caplog, respuesta):
    memoria_habilitada.responder = lambda r: httpx2.Response(200, json=respuesta)
    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV)

    assert r.status_code == 200
    body = r.json()
    assert body.pop("x_memory") == {"status": "write_failed"}
    assert body == respuesta
    assert posts_a_memoria(memoria) == []
    errores = [rec for rec in registros_propios(caplog) if rec.levelno == logging.ERROR]
    assert len(errores) == 1
    assert CONV in errores[0].getMessage()


def test_content_vacio_se_guarda(memoria_habilitada, memoria):
    # La memoria acepta content "" (Message.content: str sin largo mínimo): se guarda igual.
    memoria_habilitada.responder = lambda r: httpx2.Response(200, json=respuesta_con_contenido(""))
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV)
    assert r.json()["x_memory"] == {"status": "saved"}
    (post,) = posts_a_memoria(memoria)
    assert json.loads(post.content)["messages"][1] == {"role": "assistant", "content": ""}


def test_respuesta_del_proveedor_que_no_es_objeto(memoria_habilitada, memoria, caplog):
    # Sin un objeto no hay dónde poner x_memory: se devuelve tal cual, sin guardar, con ERROR.
    memoria_habilitada.responder = lambda r: httpx2.Response(200, json=["no", "soy", "objeto"])
    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV)
    assert r.status_code == 200
    assert r.json() == ["no", "soy", "objeto"]
    assert posts_a_memoria(memoria) == []
    errores = [rec for rec in registros_propios(caplog) if rec.levelno == logging.ERROR]
    assert len(errores) == 1 and CONV in errores[0].getMessage()


def test_turn_id_distinto_por_pedido_y_unico_dentro_del_pedido(memoria_habilitada, memoria):
    with TestClient(app) as client:
        assert pedir(client, x_conversation_id=CONV).status_code == 200
        assert pedir(client, x_conversation_id=CONV).status_code == 200

    posts = posts_a_memoria(memoria)
    assert len(posts) == 2  # un solo POST por pedido: un solo turn_id cada uno
    turn_ids = [json.loads(p.content)["turn_id"] for p in posts]
    assert turn_ids[0] != turn_ids[1]
    assert all(uuid.UUID(t).version == 4 for t in turn_ids)


def test_sin_x_conversation_id_no_toca_la_memoria(memoria_habilitada, memoria):
    with TestClient(app) as client:
        r = pedir(client)
    assert r.status_code == 200
    assert "x_memory" not in r.json()
    assert memoria.recibidos == []


def test_cliente_de_memoria_se_cierra_al_apagar(memoria_habilitada, memoria):
    with TestClient(app) as client:
        (cliente_memoria,) = memoria.clientes
        assert not cliente_memoria.is_closed
        pedir(client, x_conversation_id=CONV)
    assert cliente_memoria.is_closed


def test_sin_memory_url_no_se_crea_cliente_de_memoria(proveedor, memoria):
    with TestClient(app) as client:
        assert client.app.state.memory_client is None
    assert memoria.clientes == []


# ---------- Logging: httpx2 no escribe URLs (bloque 3c) ----------

def test_configurar_logging_fija_warning_y_es_idempotente():
    configurar_logging()
    assert all(logging.getLogger(n).level == logging.WARNING for n in LOGGERS_SILENCIADOS)
    configurar_logging()
    assert all(logging.getLogger(n).level == logging.WARNING for n in LOGGERS_SILENCIADOS)


def _memoria_get_503(request):
    return httpx2.Response(503, json={"error": {"message": "x"}})


def _memoria_post_503(request):
    if request.method == "GET":
        return responder_memoria_por_defecto(request)
    return httpx2.Response(503, json={"error": {"message": "x"}})


@pytest.mark.parametrize(
    "responder, status, x_memory",
    [
        (responder_memoria_por_defecto, 200, {"status": "saved"}),
        (_memoria_get_503, 503, None),
        (_memoria_post_503, 200, {"status": "write_failed"}),
    ],
    ids=["camino_feliz", "lectura_falla", "guardado_falla"],
)
def test_ningun_log_contiene_urls_internas_con_root_en_info(memoria_habilitada, memoria, caplog, responder, status, x_memory):
    # Root en INFO: es justo el caso que esta protección cubre. No se filtra por logger:
    # ningún registro, de ninguna librería, puede traer el host de la memoria ni del proveedor.
    memoria.responder = responder
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV)

    assert r.status_code == status
    assert r.json().get("x_memory") == x_memory
    for rec in caplog.records:
        texto = f"{rec.name} {rec.getMessage()} {rec.args!r}"
        for secreto in (HOST_MEMORIA, f"http://{HOST_MEMORIA}:8000", HOST_PROVEEDOR, f"http://{HOST_PROVEEDOR}:9000"):
            assert secreto not in texto, f"{rec.name} filtró una URL interna"


# ---------- Presupuesto de tokens del historial (context_tokens) ----------
# Cuentas a mano con el estimador (ceil(len/3) + 8 por mensaje, OVERHEAD_BASE = 16):
#   system de prueba (31 caracteres) = 19; cada mensaje de 30 caracteres = 18.
#   fijos (system + nuevo) = 37; historial de 6 mensajes = 108; total = 145.
#   Con max_tokens = 100: disponible = tope - 100 - 16 = tope - 116.

SENTINELA_HISTORIAL = "SENT-HIST-91"


def texto30(etiqueta):
    texto = f"{etiqueta}-{SENTINELA_HISTORIAL}".ljust(30, ".")
    assert len(texto) == 30
    return texto


HIST_LARGO = [
    {"role": rol, "content": texto30(f"m{i}"), "created_at": 1794305700 + i}
    for i, rol in enumerate(["user", "assistant"] * 3)
]
HIST_LARGO_PROYECTADO = [{"role": m["role"], "content": m["content"]} for m in HIST_LARGO]
NUEVO = {"role": "user", "content": texto30("nuevo")}


def escribir_registro_con_tope(tmp_path, tope, modelo="mock"):
    """Reescribe el registro del fixture proveedor (mismo tmp_path) con context_tokens en `modelo`."""
    lineas = ["providers:"]
    for nombre, url in (("mock", f"http://{HOST_PROVEEDOR}:9000"), ("local", "http://otro.internal:9001")):
        lineas += [f"  {nombre}:", f"    url: {url}"]
        if nombre == modelo:
            lineas.append(f"    context_tokens: {tope}")
    (tmp_path / "providers.yaml").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def memoria_con_historial(memoria, historial):
    memoria.responder = lambda r: (
        httpx2.Response(200, json={"conversation_id": CONV, "messages": historial}) if r.method == "GET"
        else httpx2.Response(201, json={"conversation_id": CONV, "stored": 2})
    )


def registros_de_recorte(caplog):
    return [rec for rec in caplog.records if rec.name == "orchestrator" and "Historial recortado" in rec.getMessage()]


@pytest.mark.parametrize(
    "tope, esperado",
    [
        (261, HIST_LARGO_PROYECTADO),       # disponible 145: entra justo, completo
        (225, HIST_LARGO_PROYECTADO[2:]),   # disponible 109: hay que sacar el primer par entero
        (260, HIST_LARGO_PROYECTADO[2:]),   # disponible 144: alcanza con sacar m0; m1 (assistant) queda partido
        (189, HIST_LARGO_PROYECTADO[4:]),   # disponible 73: sacar 4
        (153, []),                          # disponible 37: solo system + nuevo
    ],
    ids=["entra_completo", "descarta_un_par", "par_partido", "descarta_dos_pares", "sin_historial"],
)
def test_recorte_del_historial_por_tokens(memoria_habilitada, memoria, tmp_path, tope, esperado):
    escribir_registro_con_tope(tmp_path, tope)
    memoria_con_historial(memoria, HIST_LARGO)
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV, messages=[NUEVO], max_tokens=100)

    assert r.status_code == 200
    assert r.json()["x_memory"] == {"status": "saved"}
    # El proveedor recibe exactamente system + lo que quedó (en orden, sin editar) + el nuevo.
    (enviado,) = memoria_habilitada.recibidos
    assert enviado["messages"] == system_mas(*esperado, NUEVO)
    assert "x_conversation_id" not in enviado
    assert enviado["max_tokens"] == 100
    if esperado:
        assert esperado[0]["role"] == "user"
    # Se guarda solo el turno nuevo, igual que sin recorte.
    (post,) = posts_a_memoria(memoria)
    assert json.loads(post.content)["messages"] == [NUEVO, {"role": "assistant", "content": "Hola!"}]


def test_sin_max_tokens_el_presupuesto_usa_el_default_del_proveedor(memoria_habilitada, memoria, tmp_path):
    # disponible = tope - 256 - 16. Con 417 entra justo (145); con 416 se descarta el primer par.
    memoria_con_historial(memoria, HIST_LARGO)
    for tope, esperado in ((417, HIST_LARGO_PROYECTADO), (416, HIST_LARGO_PROYECTADO[2:])):
        escribir_registro_con_tope(tmp_path, tope)
        memoria_habilitada.recibidos.clear()
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV, messages=[NUEVO])
        assert r.status_code == 200
        (enviado,) = memoria_habilitada.recibidos
        assert enviado["messages"] == system_mas(*esperado, NUEVO)
        assert "max_tokens" not in enviado  # el orquestador no inventa el parámetro


def test_proveedor_sin_context_tokens_no_recorta(memoria_habilitada, memoria, tmp_path):
    # Tope chico en "local", pero el pedido va a "mock", que no declara context_tokens.
    escribir_registro_con_tope(tmp_path, 1, modelo="local")
    memoria_con_historial(memoria, HIST_LARGO)
    with TestClient(app) as client:
        r = pedir(client, x_conversation_id=CONV, messages=[NUEVO], max_tokens=100000)
    assert r.status_code == 200
    (enviado,) = memoria_habilitada.recibidos
    assert enviado["messages"] == system_mas(*HIST_LARGO_PROYECTADO, NUEVO)


def test_sin_memoria_no_recorta_aunque_haya_tope(memoria_habilitada, memoria, tmp_path):
    # El cliente manda su propio historial: se reenvía entero aunque supere el tope.
    escribir_registro_con_tope(tmp_path, 153)
    mensajes = HIST_LARGO_PROYECTADO + [NUEVO]
    with TestClient(app) as client:
        r = pedir(client, messages=mensajes, max_tokens=100)
    assert r.status_code == 200
    (enviado,) = memoria_habilitada.recibidos
    assert enviado["messages"] == system_mas(*mensajes)
    assert memoria.recibidos == []


@pytest.mark.parametrize(
    "tope, max_tokens", [(152, 100), (1800, 2000)], ids=["disponible_36_de_37", "max_tokens_mayor_que_el_tope"]
)
def test_si_system_y_nuevo_no_entran_da_400_sin_llamar_al_proveedor(
    memoria_habilitada, memoria, tmp_path, caplog, tope, max_tokens
):
    escribir_registro_con_tope(tmp_path, tope)
    memoria_con_historial(memoria, HIST_LARGO)
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV, messages=[NUEVO], max_tokens=max_tokens)

    error = assert_error(r, 400, "context_length_exceeded", "messages")
    assert set(r.json()) == {"error"}  # sin x_memory
    assert "aunque se descarte todo el historial" in error["message"]
    assert SENTINELA_HISTORIAL not in error["message"]
    assert memoria_habilitada.recibidos == []  # ningún POST al proveedor
    assert posts_a_memoria(memoria) == []      # nada guardado
    assert [m.method for m in memoria.recibidos] == ["GET"]
    assert registros_de_recorte(caplog) == []
    for texto in textos_de_log(caplog):
        assert SENTINELA_HISTORIAL not in texto


def test_el_recorte_loguea_solo_numeros(memoria_habilitada, memoria, tmp_path, caplog):
    escribir_registro_con_tope(tmp_path, 225)
    memoria_con_historial(memoria, HIST_LARGO)
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            r = pedir(client, x_conversation_id=CONV, messages=[NUEVO], max_tokens=100)
    assert r.status_code == 200

    (rec,) = registros_de_recorte(caplog)
    assert rec.levelno == logging.INFO
    assert rec.args == ("mock", 2, 145, 109, 109, 225)  # modelo, descartados, antes, después, disponible, tope
    # Los mensajes descartados (y los que quedaron) llevan el sentinela: ningún registro lo trae.
    for texto in textos_de_log(caplog):
        assert SENTINELA_HISTORIAL not in texto


def test_sin_descartes_no_hay_log_de_recorte(memoria_habilitada, memoria, tmp_path, caplog):
    escribir_registro_con_tope(tmp_path, 261)
    memoria_con_historial(memoria, HIST_LARGO)
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            assert pedir(client, x_conversation_id=CONV, messages=[NUEVO], max_tokens=100).status_code == 200
    assert registros_de_recorte(caplog) == []
