"""
Verifica la capa de datos de memoria contra PostgreSQL REAL.

Requiere TEST_DATABASE_URL; sin ella todos los tests que usan la base se saltan,
para que `uv run pytest` siga andando sin Docker. Los tests vacían las tablas,
así que solo corren contra una base cuyo nombre termine en "_test".
"""
import asyncio
import os
from uuid import uuid4

import pytest
from psycopg import errors
from psycopg.conninfo import conninfo_to_dict

from src.memory.config import Config
from src.memory.db import aplicar_esquema, crear_pool, guardar_turno, leer_historial

USER = "dev-user"
CONV = "c-123"


def turno(pregunta, respuesta):
    return [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta}]


def verificar_base_de_test(url: str) -> None:
    """Falla (no salta) si la base no es descartable: un salto silencioso ocultaría el riesgo."""
    dbname = conninfo_to_dict(url).get("dbname") or ""
    if not dbname.endswith("_test"):
        pytest.fail(
            f"TEST_DATABASE_URL apunta a la base '{dbname}', que no termina en '_test'. "
            "Estos tests vacían las tablas: me niego a correr contra una base real."
        )


@pytest.fixture
def db_url():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL no definida: se saltan los tests contra PostgreSQL.")
    verificar_base_de_test(url)
    return url


def correr(db_url, prueba):
    """
    Ejecuta `prueba(pool)` con un pool nuevo, el esquema aplicado y las tablas vacías.

    SelectorEventLoop explícito: psycopg async no funciona con el ProactorEventLoop,
    que es el default en Windows (en Linux ya es el default, así que no cambia nada).
    """
    async def envoltorio():
        pool = crear_pool(Config(database_url=db_url, pool_max_size=5))
        await pool.open(wait=True, timeout=10)
        try:
            await aplicar_esquema(pool)
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE conversations CASCADE")
            return await prueba(pool)
        finally:
            await pool.close()

    return asyncio.run(envoltorio(), loop_factory=asyncio.SelectorEventLoop)


async def seqs(pool, user_id=USER, conversation_id=CONV):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT seq FROM messages WHERE user_id = %s AND conversation_id = %s ORDER BY seq",
            (user_id, conversation_id),
        )
        return [fila[0] for fila in await cur.fetchall()]


async def next_seq(pool, user_id=USER, conversation_id=CONV):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT next_seq FROM conversations WHERE user_id = %s AND conversation_id = %s",
            (user_id, conversation_id),
        )
        return (await cur.fetchone())[0]


# ---------- Fixture de seguridad (no necesita base) ----------

@pytest.mark.parametrize(
    "url",
    ["postgresql://postgres:x@localhost:5432/memory", "postgresql://postgres:x@localhost:5432/test_memory"],
)
def test_se_niega_a_correr_contra_base_no_test(url):
    with pytest.raises(pytest.fail.Exception, match="no termina en '_test'"):
        verificar_base_de_test(url)


# ---------- Lectura y escritura ----------

def test_a_guardar_y_leer_en_orden(db_url):
    async def prueba(pool):
        assert await guardar_turno(pool, USER, CONV, uuid4(), turno("Hola", "¡Hola!")) == 2
        assert await guardar_turno(pool, USER, CONV, uuid4(), turno("¿Todo bien?", "Sí")) == 2
        return await leer_historial(pool, USER, CONV, 200)

    historial = correr(db_url, prueba)
    assert [(m["role"], m["content"]) for m in historial] == [
        ("user", "Hola"), ("assistant", "¡Hola!"), ("user", "¿Todo bien?"), ("assistant", "Sí"),
    ]
    assert all(set(m) == {"role", "content", "created_at"} for m in historial)
    assert all(type(m["created_at"]) is int for m in historial)


def test_b_limit_devuelve_los_ultimos_en_orden_ascendente(db_url):
    async def prueba(pool):
        for i in range(1, 4):
            await guardar_turno(pool, USER, CONV, uuid4(), turno(f"p{i}", f"r{i}"))
        return await leer_historial(pool, USER, CONV, 3)

    assert [m["content"] for m in correr(db_url, prueba)] == ["r2", "p3", "r3"]


def test_c_conversacion_inexistente_devuelve_lista_vacia(db_url):
    async def prueba(pool):
        return await leer_historial(pool, USER, "no-existe", 20)

    assert correr(db_url, prueba) == []


# ---------- Idempotencia por turn_id ----------

def test_d_mismo_turn_id_dos_veces_no_duplica(db_url):
    async def prueba(pool):
        t = uuid4()
        primero = await guardar_turno(pool, USER, CONV, t, turno("Hola", "¡Hola!"))
        # Contenido distinto con el mismo turn_id: se ignora (contrato, pendiente de la sección 6).
        segundo = await guardar_turno(pool, USER, CONV, t, turno("Otra cosa", "Otra respuesta"))
        return primero, segundo, await leer_historial(pool, USER, CONV, 200)

    primero, segundo, historial = correr(db_url, prueba)
    assert (primero, segundo) == (2, 0)
    assert [m["content"] for m in historial] == ["Hola", "¡Hola!"]


def test_d_mismo_turn_id_concurrente_se_guarda_una_sola_vez(db_url):
    async def prueba(pool):
        t = uuid4()
        resultados = await asyncio.gather(
            *(guardar_turno(pool, USER, CONV, t, turno("Hola", "¡Hola!")) for _ in range(5))
        )
        return resultados, await seqs(pool)

    resultados, s = correr(db_url, prueba)
    assert sorted(resultados) == [0, 0, 0, 0, 2]
    assert s == [1, 2]


# ---------- Aislamiento entre usuarios ----------

def test_e_mismo_conversation_id_con_usuarios_distintos_son_independientes(db_url):
    async def prueba(pool):
        await guardar_turno(pool, "ana", CONV, uuid4(), turno("Soy Ana", "Hola Ana"))
        await guardar_turno(pool, "beto", CONV, uuid4(), turno("Soy Beto", "Hola Beto"))
        return (
            await leer_historial(pool, "ana", CONV, 200),
            await leer_historial(pool, "beto", CONV, 200),
            await seqs(pool, "ana"),
            await seqs(pool, "beto"),
        )

    ana, beto, seqs_ana, seqs_beto = correr(db_url, prueba)
    assert [m["content"] for m in ana] == ["Soy Ana", "Hola Ana"]
    assert [m["content"] for m in beto] == ["Soy Beto", "Hola Beto"]
    # Cada una numera por su cuenta: no comparten next_seq.
    assert seqs_ana == seqs_beto == [1, 2]


# ---------- seq ----------

def test_f_seq_consecutivos_sin_huecos_incluso_tras_un_duplicado(db_url):
    async def prueba(pool):
        t1 = uuid4()
        await guardar_turno(pool, USER, CONV, t1, turno("p1", "r1"))
        await guardar_turno(pool, USER, CONV, uuid4(), turno("p2", "r2"))
        assert await guardar_turno(pool, USER, CONV, t1, turno("p1", "r1")) == 0
        await guardar_turno(pool, USER, CONV, uuid4(), turno("p3", "r3"))
        return await seqs(pool), await next_seq(pool)

    s, siguiente = correr(db_url, prueba)
    assert s == [1, 2, 3, 4, 5, 6]
    assert siguiente == 7


def test_g_guardados_concurrentes_en_la_misma_conversacion(db_url):
    async def prueba(pool):
        resultados = await asyncio.gather(
            *(guardar_turno(pool, USER, CONV, uuid4(), turno(f"p{i}", f"r{i}")) for i in range(10))
        )
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT seq, role, turn_id FROM messages WHERE user_id = %s AND conversation_id = %s "
                "ORDER BY seq",
                (USER, CONV),
            )
            filas = await cur.fetchall()
        return resultados, filas, await next_seq(pool)

    resultados, filas, siguiente = correr(db_url, prueba)
    assert resultados == [2] * 10
    assert [f[0] for f in filas] == list(range(1, 21))
    assert siguiente == 21
    # Los dos mensajes de cada turno quedan contiguos (user, assistant): los guardados no se intercalan.
    for user_row, assistant_row in zip(filas[0::2], filas[1::2]):
        assert (user_row[1], assistant_row[1]) == ("user", "assistant")
        assert user_row[2] == assistant_row[2]


# ---------- Restricciones del esquema ----------

def test_h_role_system_rechazado_y_el_turno_no_se_guarda_a_medias(db_url):
    async def prueba(pool):
        mensajes = [{"role": "user", "content": "Hola"}, {"role": "system", "content": "Sos..."}]
        with pytest.raises(errors.CheckViolation):
            await guardar_turno(pool, USER, CONV, uuid4(), mensajes)
        # Atomicidad: tampoco quedó el mensaje "user" ni la conversación.
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) FROM conversations")
            conversaciones = (await cur.fetchone())[0]
        return await leer_historial(pool, USER, CONV, 200), conversaciones

    historial, conversaciones = correr(db_url, prueba)
    assert historial == []
    assert conversaciones == 0


def test_roles_repetidos_en_un_turno_se_rechazan_sin_guardar_nada(db_url):
    async def prueba(pool):
        mensajes = [{"role": "user", "content": "Hola"}, {"role": "user", "content": "¿Estás?"}]
        with pytest.raises(ValueError, match="como máximo un mensaje por rol"):
            await guardar_turno(pool, USER, CONV, uuid4(), mensajes)
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT (SELECT count(*) FROM messages), (SELECT count(*) FROM conversations)")
            return await cur.fetchone()

    assert correr(db_url, prueba) == (0, 0)


@pytest.mark.parametrize(
    "user_id, conversation_id",
    [
        (USER, "con espacios"),
        (USER, ""),
        (USER, "a" * 65),
        (USER, "c/../otra"),
        ("usuario@dominio", CONV),
        ("", CONV),
    ],
    ids=["conv_espacios", "conv_vacio", "conv_65_chars", "conv_barra", "user_arroba", "user_vacio"],
)
def test_i_identificadores_con_formato_invalido_rechazados(db_url, user_id, conversation_id):
    async def prueba(pool):
        with pytest.raises(errors.CheckViolation):
            await guardar_turno(pool, user_id, conversation_id, uuid4(), turno("Hola", "¡Hola!"))

    correr(db_url, prueba)


def test_j_borrar_conversacion_borra_sus_mensajes(db_url):
    async def prueba(pool):
        await guardar_turno(pool, USER, CONV, uuid4(), turno("Hola", "¡Hola!"))
        await guardar_turno(pool, USER, "otra", uuid4(), turno("Sigo", "acá"))
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM conversations WHERE user_id = %s AND conversation_id = %s", (USER, CONV)
            )
            cur = await conn.execute(
                "SELECT count(*) FROM messages WHERE user_id = %s AND conversation_id = %s", (USER, CONV)
            )
            restantes = (await cur.fetchone())[0]
        return restantes, await leer_historial(pool, USER, "otra", 200)

    restantes, otra = correr(db_url, prueba)
    assert restantes == 0
    assert len(otra) == 2  # el CASCADE solo afecta a la conversación borrada
