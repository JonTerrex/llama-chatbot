"""
Capa de datos del servicio de memoria (contrato de memoria, secciones 3.1-3.3 y 3.6).

Todas las consultas filtran por (user_id, conversation_id): no existe acceso por
conversation_id solo. Las excepciones de psycopg se propagan a propósito: la capa
HTTP las traduce a 503. No se loguea contenido de mensajes.
"""
from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.memory.config import Config

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

def crear_pool(config: Config) -> AsyncConnectionPool:
    """
    Sin abrir (open=False): el lifespan decide cuándo abrirlo con `await pool.open()`.

    min_size=1 porque el default de psycopg_pool es 4, que fallaría con DB_POOL_MAX_SIZE < 4.
    """
    return AsyncConnectionPool(
        config.database_url, min_size=1, max_size=config.pool_max_size, open=False
    )


async def aplicar_esquema(pool: AsyncConnectionPool) -> None:
    """Idempotente (CREATE TABLE IF NOT EXISTS): se puede llamar en cada arranque."""
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    async with pool.connection() as conn:
        # Sin parámetros, psycopg admite varias sentencias en un solo execute.
        await conn.execute(sql)


async def guardar_turno(
    pool: AsyncConnectionPool,
    user_id: str,
    conversation_id: str,
    turn_id: str | UUID,
    mensajes: list[dict[str, str]],
) -> int:
    """
    Guarda los mensajes de un turno de forma atómica y devuelve cuántos guardó.

    Devuelve 0 si la conversación ya tenía mensajes con ese turn_id (idempotencia):
    en ese caso no consume seq. Todo ocurre en una sola transacción: el bloque
    pool.connection() hace commit al salir bien y rollback si hay excepción.
    """
    if not mensajes:
        # 0 significa "duplicado"; una lista vacía es un error de quien llama, no un duplicado.
        raise ValueError("mensajes no puede estar vacío.")
    # El esquema exige UNIQUE (..., turn_id, role): se rechaza antes de abrir la conexión,
    # así el error es de quien llama y no una UniqueViolation de la base.
    roles = [m["role"] for m in mensajes]
    if len(roles) != len(set(roles)):
        raise ValueError("un turno admite como máximo un mensaje por rol.")

    async with pool.connection() as conn:
        # 1) La conversación se crea implícitamente con el primer turno.
        await conn.execute(
            "INSERT INTO conversations (user_id, conversation_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (user_id, conversation_id),
        )
        # 2) El bloqueo de la fila serializa los guardados concurrentes de esta conversación.
        cur = await conn.execute(
            "SELECT next_seq FROM conversations "
            "WHERE user_id = %s AND conversation_id = %s FOR UPDATE",
            (user_id, conversation_id),
        )
        (next_seq,) = await cur.fetchone()

        # 3) Turno ya guardado: no insertar ni avanzar next_seq.
        cur = await conn.execute(
            "SELECT 1 FROM messages "
            "WHERE user_id = %s AND conversation_id = %s AND turn_id = %s LIMIT 1",
            (user_id, conversation_id, turn_id),
        )
        if await cur.fetchone() is not None:
            return 0

        # 4) seq consecutivos a partir de next_seq, y next_seq avanza en la misma transacción.
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO messages (user_id, conversation_id, seq, role, content, turn_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (user_id, conversation_id, next_seq + i, m["role"], m["content"], turn_id)
                    for i, m in enumerate(mensajes)
                ],
            )
        await conn.execute(
            "UPDATE conversations SET next_seq = %s WHERE user_id = %s AND conversation_id = %s",
            (next_seq + len(mensajes), user_id, conversation_id),
        )

    return len(mensajes)


async def leer_historial(
    pool: AsyncConnectionPool, user_id: str, conversation_id: str, limit: int
) -> list[dict[str, Any]]:
    """
    Los últimos `limit` mensajes, del más antiguo al más nuevo.

    Ordena por seq y no por created_at: los mensajes de un turno comparten timestamp.
    created_at se trunca a epoch entero en SQL (como int(time.time())).
    Conversación inexistente o ajena: lista vacía, indistinguibles a propósito.
    """
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT role, content, created_at FROM ("
                "  SELECT seq, role, content,"
                "         floor(extract(epoch FROM created_at))::bigint AS created_at"
                "  FROM messages"
                "  WHERE user_id = %s AND conversation_id = %s"
                "  ORDER BY seq DESC LIMIT %s"
                ") AS ultimos ORDER BY seq ASC",
                (user_id, conversation_id, limit),
            )
            return await cur.fetchall()
