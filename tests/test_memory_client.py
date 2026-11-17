"""Verifica el cliente de memoria del orquestador con httpx2.MockTransport, sin red ni servicios reales."""
import asyncio
import json
import logging
import time
from uuid import uuid4

import httpx2
import pytest

from src.orchestrator.config import Config
from src.orchestrator.memory_client import (
    MemoryClient,
    MemoryUnavailable,
    MensajeGuardado,
    crear_cliente_memoria,
)

# Host y contenido reconocibles para comprobar que nunca aparecen en logs ni excepciones.
HOST_MEMORIA = "memoria-secreta.internal"
URL_MEMORIA = f"http://{HOST_MEMORIA}:8000"
CONTENIDO_SECRETO = "CONTENIDO-PRIVADO-DEL-USUARIO-42"
CONV = "c-123"

HISTORIAL = {
    "conversation_id": CONV,
    "messages": [
        {"role": "user", "content": "Hola", "created_at": 1794305797},
        {"role": "assistant", "content": "¡Hola! ¿En qué te ayudo?", "created_at": 1794305798},
    ],
}


def config(**cambios) -> Config:
    base = dict(
        providers={}, default_model=None, timeout_s=120.0, system_prompt="",
        memory_url=URL_MEMORIA, dev_user_id="dev-user", memory_history_limit=20, memory_timeout_s=5.0,
    )
    base.update(cambios)
    return Config(**base)


class MemoriaFalsa:
    """Handler del MockTransport: guarda cada request y responde con `responder` (cada test lo cambia)."""

    def __init__(self):
        self.recibidos: list[httpx2.Request] = []
        self.responder = lambda request: httpx2.Response(200, json=HISTORIAL)

    def handler(self, request: httpx2.Request):
        self.recibidos.append(request)
        return self.responder(request)


@pytest.fixture
def memoria():
    return MemoriaFalsa()


def correr(memoria: MemoriaFalsa, operacion, **cambios_config):
    """Crea el cliente con el MockTransport y corre `operacion(cliente_memoria)` en un event loop propio."""
    async def principal():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(memoria.handler)) as http:
            return await operacion(MemoryClient(http, config(**cambios_config)))
    return asyncio.run(principal())


def leer(conversation_id=CONV):
    return lambda mc: mc.leer_historial(conversation_id)


def guardar(conversation_id=CONV, turn_id=None, user="Hola", assistant="¡Hola!"):
    return lambda mc: mc.guardar_turno(conversation_id, turn_id or uuid4(), user, assistant)


def assert_sin_secretos(texto: str):
    for secreto in (HOST_MEMORIA, URL_MEMORIA, CONTENIDO_SECRETO):
        assert secreto not in texto


# ---------- Construcción ----------

def test_sin_memory_url_da_value_error():
    with pytest.raises(ValueError, match="memory_url"):
        MemoryClient(httpx2.AsyncClient(), config(memory_url=None, dev_user_id=None))


def test_crear_cliente_memoria_usa_memory_timeout_s():
    cliente = crear_cliente_memoria(config(memory_timeout_s=3.5))
    assert cliente.timeout == httpx2.Timeout(3.5)


def test_causa_desconocida_da_value_error():
    with pytest.raises(ValueError):
        MemoryUnavailable("otra")


# ---------- Lectura ----------

def test_lectura_exitosa(memoria):
    mensajes = correr(memoria, leer())
    assert mensajes == [
        MensajeGuardado("user", "Hola", 1794305797),
        MensajeGuardado("assistant", "¡Hola! ¿En qué te ayudo?", 1794305798),
    ]


def test_lectura_lleva_user_id_y_limit_en_la_url(memoria):
    correr(memoria, leer(), dev_user_id="ana", memory_history_limit=7)
    (request,) = memoria.recibidos
    assert request.method == "GET"
    assert request.url.path == f"/v1/conversations/{CONV}/messages"
    assert dict(request.url.params) == {"user_id": "ana", "limit": "7"}


def test_conversacion_vacia_es_lista_vacia(memoria):
    memoria.responder = lambda r: httpx2.Response(200, json={"conversation_id": CONV, "messages": []})
    assert correr(memoria, leer()) == []


def test_conversation_id_con_caracteres_raros_queda_escapado(memoria):
    correr(memoria, leer("a/b?c=d#e f"))
    (request,) = memoria.recibidos
    # Un solo segmento de ruta y sin query extra: "/", "?", "#" y " " viajan escapados.
    assert request.url.raw_path == b"/v1/conversations/a%2Fb%3Fc%3Dd%23e%20f/messages?user_id=dev-user&limit=20"
    assert dict(request.url.params) == {"user_id": "dev-user", "limit": "20"}


@pytest.mark.parametrize(
    "cuerpo",
    [
        {"conversation_id": CONV},
        {"messages": "no es lista"},
        [HISTORIAL["messages"]],
        {"messages": [{"role": "system", "content": "x", "created_at": 1}]},
        {"messages": [{"role": "user", "content": 42, "created_at": 1}]},
        {"messages": [{"role": "user", "content": "x", "created_at": "1"}]},
        {"messages": [{"role": "user", "content": "x", "created_at": True}]},
        {"messages": [{"role": "user", "content": "x"}]},
        {"messages": ["no es objeto"]},
    ],
    ids=[
        "sin_messages", "messages_no_lista", "cuerpo_lista", "role_system", "content_no_str",
        "created_at_str", "created_at_bool", "sin_created_at", "item_no_objeto",
    ],
)
def test_lectura_con_forma_invalida(memoria, cuerpo):
    memoria.responder = lambda r: httpx2.Response(200, json=cuerpo)
    with pytest.raises(MemoryUnavailable) as exc:
        correr(memoria, leer())
    assert exc.value.causa == "respuesta_invalida"


# ---------- Guardado ----------

def test_guardado_exitoso_con_201(memoria):
    memoria.responder = lambda r: httpx2.Response(201, json={"conversation_id": CONV, "stored": 2})
    assert correr(memoria, guardar()) is None


def test_guardado_200_con_stored_0_es_exito(memoria):
    memoria.responder = lambda r: httpx2.Response(200, json={"conversation_id": CONV, "stored": 0})
    assert correr(memoria, guardar()) is None


def test_body_del_post_tiene_solo_user_y_assistant(memoria):
    memoria.responder = lambda r: httpx2.Response(201, json={"conversation_id": CONV, "stored": 2})
    turn_id = uuid4()
    correr(memoria, guardar(turn_id=turn_id, user="pregunta", assistant="respuesta"), dev_user_id="ana")
    (request,) = memoria.recibidos
    assert request.method == "POST"
    assert request.url.path == f"/v1/conversations/{CONV}/messages"
    assert request.url.query == b""  # user_id va en el body, no en la URL
    assert json.loads(request.content) == {
        "user_id": "ana",
        "turn_id": str(turn_id),
        "messages": [{"role": "user", "content": "pregunta"}, {"role": "assistant", "content": "respuesta"}],
    }


@pytest.mark.parametrize(
    "cuerpo", [{"conversation_id": CONV}, {"stored": "2"}, {"stored": True}, [2]],
    ids=["sin_stored", "stored_str", "stored_bool", "cuerpo_lista"],
)
def test_guardado_con_forma_invalida(memoria, cuerpo):
    memoria.responder = lambda r: httpx2.Response(201, json=cuerpo)
    with pytest.raises(MemoryUnavailable) as exc:
        correr(memoria, guardar())
    assert exc.value.causa == "respuesta_invalida"


# ---------- Fallos comunes a lectura y guardado ----------

def lanzar(excepcion):
    def responder(request):
        raise excepcion
    return responder


OPERACIONES = pytest.mark.parametrize("operacion", [leer(), guardar()], ids=["read", "write"])


@OPERACIONES
def test_503_de_la_memoria(memoria, operacion):
    memoria.responder = lambda r: httpx2.Response(503, json={"error": {"message": "x"}})
    with pytest.raises(MemoryUnavailable) as exc:
        correr(memoria, operacion)
    assert exc.value.causa == "estado_http"


@OPERACIONES
@pytest.mark.parametrize(
    "excepcion, causa",
    [
        (httpx2.ConnectError(f"Connection refused: {URL_MEMORIA}"), "conexion"),
        # ConnectTimeout hereda de TimeoutException, pero no conectar es "conexion".
        (httpx2.ConnectTimeout(f"connect timeout: {URL_MEMORIA}"), "conexion"),
        (httpx2.ReadTimeout(f"read timeout: {URL_MEMORIA}"), "timeout"),
        (httpx2.RemoteProtocolError(f"cortó: {URL_MEMORIA}"), "conexion"),
        # Estas dos heredan de RequestError pero no de TransportError.
        (httpx2.DecodingError(f"gzip roto: {URL_MEMORIA}"), "respuesta_invalida"),
        (httpx2.RequestError(f"otro error: {URL_MEMORIA}"), "conexion"),
    ],
    ids=[
        "conexion_rechazada", "connect_timeout", "read_timeout", "otro_transport_error",
        "decoding_error", "request_error_generico",
    ],
)
def test_errores_de_transporte(memoria, operacion, excepcion, causa):
    memoria.responder = lanzar(excepcion)
    with pytest.raises(MemoryUnavailable) as exc:
        correr(memoria, operacion)
    assert exc.value.causa == causa
    assert exc.value.__cause__ is None  # la excepción original (con la URL) no viaja encadenada


@OPERACIONES
def test_timeout_total(memoria, operacion):
    # MockTransport ignora los timeouts de httpx2: lo corta el asyncio.timeout del cliente.
    async def responder_tarde(request):
        await asyncio.sleep(2)
        return httpx2.Response(200, json=HISTORIAL)

    memoria.responder = responder_tarde
    inicio = time.perf_counter()
    with pytest.raises(MemoryUnavailable) as exc:
        correr(memoria, operacion, memory_timeout_s=0.1)
    assert time.perf_counter() - inicio < 1.0
    assert exc.value.causa == "timeout"


@OPERACIONES
def test_cuerpo_no_json(memoria, operacion):
    memoria.responder = lambda r: httpx2.Response(200, content=b"<html>no soy json</html>")
    with pytest.raises(MemoryUnavailable) as exc:
        correr(memoria, operacion)
    assert exc.value.causa == "respuesta_invalida"


# ---------- Logs y excepciones sin datos sensibles ----------

@pytest.mark.parametrize(
    "responder, causa, nivel",
    [
        (lanzar(httpx2.ConnectError(f"refused {URL_MEMORIA}")), "conexion", logging.WARNING),
        (lambda r: httpx2.Response(500, text=f"boom {CONTENIDO_SECRETO} {URL_MEMORIA}"), "estado_http", logging.WARNING),
        (lambda r: httpx2.Response(200, text=f"no json {CONTENIDO_SECRETO}"), "respuesta_invalida", logging.ERROR),
        (lambda r: httpx2.Response(200, json={"echo": CONTENIDO_SECRETO}), "respuesta_invalida", logging.ERROR),
    ],
    ids=["conexion", "estado_http", "no_json", "forma"],
)
@pytest.mark.parametrize("op", ["read", "write"])
def test_logs_y_excepcion_sin_url_ni_contenido(memoria, caplog, op, responder, causa, nivel):
    memoria.responder = responder
    # Al guardar, el contenido secreto viaja en el body: no debe volver en logs ni excepción.
    operacion = leer() if op == "read" else guardar(user=CONTENIDO_SECRETO, assistant=CONTENIDO_SECRETO)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(MemoryUnavailable) as exc:
            correr(memoria, operacion)

    assert exc.value.causa == causa
    assert_sin_secretos(str(exc.value))
    assert_sin_secretos(repr(exc.value))

    # Solo los registros propios: los de httpx2 (INFO con la URL) se cubren en el paso de
    # cableado de app.py, porque dependen del logging global y un setLevel allá volvería
    # este test dependiente del orden de ejecución.
    propios = [rec for rec in caplog.records if rec.name == "orchestrator.memory"]
    assert len(propios) == 1
    assert propios[0].levelno == nivel
    mensaje = propios[0].getMessage()
    assert_sin_secretos(mensaje)
    assert_sin_secretos(repr(propios[0].args))
    assert CONV in mensaje and f"causa={causa}" in mensaje and f"op={op}" in mensaje
