"""
Verifica la API HTTP del servicio de memoria (contrato de memoria, 3.1-3.5).

Dos grupos: con PostgreSQL real (se saltan sin TEST_DATABASE_URL) y con la base
caída (DATABASE_URL a un puerto cerrado), que cubre errores y validación sin Docker.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

import src.memory.app as mem_app
import src.memory.db as mem_db
from src.memory.app import app
from tests.test_memory_db import verificar_base_de_test

# psycopg async no funciona con el ProactorEventLoop (default en Windows); en Linux no cambia nada.
BACKEND_OPTIONS = {"loop_factory": asyncio.SelectorEventLoop}

USER = "dev-user"
CONV = "c-123"
URL_MENSAJES = f"/v1/conversations/{CONV}/messages"

# Datos reconocibles para comprobar que nunca aparecen en respuestas ni logs.
HOST_CAIDO, PUERTO_CAIDO = "127.0.0.1", "5439"
USUARIO_DB, CLAVE_DB = "usuario-db-prueba", "clave-super-secreta"
CONTENIDO_SECRETO = "CONTENIDO-PRIVADO-DEL-USUARIO-42"


def cliente():
    return TestClient(app, backend_options=BACKEND_OPTIONS)


def turno(pregunta="Hola", respuesta="¡Hola!"):
    return [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta}]


def cuerpo(**cambios):
    base = {"user_id": USER, "turn_id": str(uuid4()), "messages": turno()}
    base.update(cambios)
    return base


@pytest.fixture
def base_real(monkeypatch):
    """Base descartable con tablas vacías; misma guarda "_test" que test_memory_db.py."""
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL no definida: se saltan los tests contra PostgreSQL.")
    verificar_base_de_test(url)
    # Conexión síncrona: no depende del event loop. Crea el esquema si falta y vacía.
    with psycopg.connect(url) as conn:
        conn.execute(mem_db.SCHEMA_FILE.read_text(encoding="utf-8"))
        conn.execute("TRUNCATE conversations CASCADE")
    monkeypatch.setenv("DATABASE_URL", url)


@pytest.fixture
def base_caida(monkeypatch):
    """Puerto cerrado y timeouts cortos: cada intento y el cierre del pool tardan 0.5 s en vez de 5 s."""
    monkeypatch.setenv(
        "DATABASE_URL", f"postgresql://{USUARIO_DB}:{CLAVE_DB}@{HOST_CAIDO}:{PUERTO_CAIDO}/memory_test"
    )
    monkeypatch.setattr(mem_db, "POOL_TIMEOUT_S", 0.5)
    monkeypatch.setattr(mem_app, "POOL_CLOSE_TIMEOUT_S", 0.5)


def assert_error(r, status, code, param="__no_comprobar__"):
    assert r.status_code == status
    error = r.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["code"] == code
    if param != "__no_comprobar__":
        assert error["param"] == param
    return error


def assert_sin_datos_de_conexion(texto):
    for secreto in (HOST_CAIDO, PUERTO_CAIDO, USUARIO_DB, CLAVE_DB):
        assert secreto not in texto


# ==================== a) Con base real ====================

def test_ida_y_vuelta(base_real):
    with cliente() as c:
        r = c.post(URL_MENSAJES, json=cuerpo(messages=turno("Hola", "¡Hola! ¿En qué te ayudo?")))
        assert r.status_code == 201
        assert r.json() == {"conversation_id": CONV, "stored": 2}

        r = c.get(URL_MENSAJES, params={"user_id": USER, "limit": 20})
        assert r.status_code == 200
        body = r.json()
    assert body["conversation_id"] == CONV
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "Hola"), ("assistant", "¡Hola! ¿En qué te ayudo?"),
    ]
    assert all(type(m["created_at"]) is int for m in body["messages"])


def test_turno_duplicado_da_200_con_stored_0(base_real):
    with cliente() as c:
        b = cuerpo()
        assert c.post(URL_MENSAJES, json=b).status_code == 201
        r = c.post(URL_MENSAJES, json=b)
        assert r.status_code == 200
        assert r.json() == {"conversation_id": CONV, "stored": 0}
        assert len(c.get(URL_MENSAJES, params={"user_id": USER, "limit": 20}).json()["messages"]) == 2


def test_conversacion_inexistente_da_lista_vacia(base_real):
    with cliente() as c:
        r = c.get("/v1/conversations/no-existe/messages", params={"user_id": USER, "limit": 20})
    assert r.status_code == 200
    assert r.json() == {"conversation_id": "no-existe", "messages": []}


def test_otro_usuario_no_ve_la_conversacion(base_real):
    with cliente() as c:
        assert c.post(URL_MENSAJES, json=cuerpo(user_id="ana")).status_code == 201
        r = c.get(URL_MENSAJES, params={"user_id": "beto", "limit": 20})
    # Ajena e inexistente son indistinguibles (contrato, 3.1).
    assert r.status_code == 200
    assert r.json()["messages"] == []


def test_health_ready_con_base(base_real):
    with cliente() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


# ==================== b) Con la base caída ====================

def test_post_con_base_caida_da_503(base_caida):
    with cliente() as c:
        r = c.post(URL_MENSAJES, json=cuerpo())
    error = assert_error(r, 503, "storage_unavailable")
    assert error["type"] == "memory_error"
    assert_sin_datos_de_conexion(r.text)


def test_get_con_base_caida_da_503(base_caida):
    with cliente() as c:
        r = c.get(URL_MENSAJES, params={"user_id": USER, "limit": 20})
    error = assert_error(r, 503, "storage_unavailable")
    assert error["type"] == "memory_error"
    assert_sin_datos_de_conexion(r.text)


def test_health_loading_con_base_caida(base_caida):
    with cliente() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "loading"}


class PoolLento:
    """Pool falso cuya connection() tarda más que el tope de /health: simula una base colgada."""

    async def open(self, wait=False):
        pass

    async def close(self, timeout=None):
        pass

    @asynccontextmanager
    async def connection(self):
        await asyncio.sleep(5)
        yield None


def test_health_respeta_su_timeout_aunque_la_base_no_responda(base_caida, monkeypatch, caplog):
    async def esquema_ok(pool):
        return None

    # El esquema "ya está" para que el lifespan no toque el pool; la demora queda en el SELECT 1.
    monkeypatch.setattr(mem_app, "crear_pool", lambda config: PoolLento())
    monkeypatch.setattr(mem_app, "aplicar_esquema", esquema_ok)
    monkeypatch.setattr(mem_app, "HEALTH_TIMEOUT_S", 0.1)
    with caplog.at_level(logging.WARNING):
        with cliente() as c:
            inicio = time.perf_counter()
            r = c.get("/health")
            duracion = time.perf_counter() - inicio

    assert r.status_code == 200
    assert r.json() == {"status": "loading"}
    assert duracion < 0.5
    assert any("TimeoutError" in rec.getMessage() for rec in caplog.records if rec.name == "memory")


def test_logs_sin_contenido_ni_datos_de_conexion_cuando_falla_el_guardado(base_caida, caplog):
    with caplog.at_level(logging.WARNING):
        with cliente() as c:
            r = c.post(URL_MENSAJES, json=cuerpo(messages=turno(CONTENIDO_SECRETO, CONTENIDO_SECRETO)))
    assert r.status_code == 503
    assert CONTENIDO_SECRETO not in caplog.text
    propios = [rec for rec in caplog.records if rec.name == "memory"]
    assert any("op=write" in rec.getMessage() and CONV in rec.getMessage() for rec in propios)
    for rec in propios:
        assert_sin_datos_de_conexion(rec.getMessage())


def test_error_inesperado_da_500_generico(base_caida, monkeypatch, caplog):
    async def esquema_ok(pool):
        return None

    async def guardar_que_explota(*args, **kwargs):
        raise RuntimeError(f"detalle interno con {CONTENIDO_SECRETO}")

    # El esquema "se aplica" para llegar a guardar_turno, que falla de forma inesperada.
    monkeypatch.setattr(mem_app, "aplicar_esquema", esquema_ok)
    monkeypatch.setattr(mem_app, "guardar_turno", guardar_que_explota)
    with caplog.at_level(logging.WARNING):
        with cliente() as c:
            r = c.post(URL_MENSAJES, json=cuerpo())
    error = assert_error(r, 500, "internal_error")
    assert error["type"] == "memory_error"
    assert "detalle interno" not in error["message"]
    assert CONTENIDO_SECRETO not in r.text
    assert CONTENIDO_SECRETO not in caplog.text
    assert any("RuntimeError" in rec.getMessage() for rec in caplog.records if rec.name == "memory")


# ---------- Validación (no toca la base: con la base caída, un 400 prueba que no la tocó) ----------

@pytest.mark.parametrize(
    "ruta, body, param",
    [
        ("/v1/conversations/c.123/messages", cuerpo(), "conversation_id"),
        (f"/v1/conversations/{'a' * 65}/messages", cuerpo(), "conversation_id"),
        (URL_MENSAJES, cuerpo(user_id="usuario con espacios"), "user_id"),
        (URL_MENSAJES, cuerpo(turn_id="no-es-un-uuid"), "turn_id"),
        (URL_MENSAJES, cuerpo(messages=[{"role": "system", "content": "Sos..."}]), "messages"),
        (URL_MENSAJES, cuerpo(messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]), "messages"),
        (URL_MENSAJES, cuerpo(messages=[]), "messages"),
        (URL_MENSAJES, cuerpo(messages=turno() + [{"role": "user", "content": "c"}]), "messages"),
    ],
    ids=[
        "conv_id_con_punto", "conv_id_65_chars", "user_id_invalido", "turn_id_no_uuid",
        "role_system", "roles_repetidos", "messages_vacio", "messages_3_items",
    ],
)
def test_post_invalido_da_400(base_caida, ruta, body, param):
    with cliente() as c:
        r = c.post(ruta, json=body)
    error = assert_error(r, 400, "invalid_request", param)
    assert error["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "params, param",
    [
        ({"user_id": USER, "limit": 0}, "limit"),
        ({"user_id": USER, "limit": 201}, "limit"),
        ({"user_id": USER, "limit": "muchos"}, "limit"),
        ({"user_id": USER}, "limit"),
        ({"limit": 20}, "user_id"),
        ({"user_id": "x/y", "limit": 20}, "user_id"),
    ],
    ids=["limit_0", "limit_201", "limit_no_numerico", "limit_ausente", "user_id_ausente", "user_id_invalido"],
)
def test_get_invalido_da_400(base_caida, params, param):
    with cliente() as c:
        r = c.get(URL_MENSAJES, params=params)
    assert_error(r, 400, "invalid_request", param)


def test_json_mal_formado_da_400_con_param_null(base_caida):
    with cliente() as c:
        r = c.post(URL_MENSAJES, content=b'{"user_id": "dev-user", "messages": [', headers={"Content-Type": "application/json"})
    assert_error(r, 400, "invalid_request", None)


def test_ruta_inexistente_da_404_con_code_null(base_caida):
    with cliente() as c:
        r = c.get("/no-existe")
    assert_error(r, 404, None)


def test_metodo_no_permitido_da_405_con_code_null(base_caida):
    with cliente() as c:
        r = c.delete(URL_MENSAJES)
    assert_error(r, 405, None)
