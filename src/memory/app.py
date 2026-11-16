"""
Servicio de memoria: historial de conversaciones sobre PostgreSQL (contrato de memoria, sección 3).

Único dueño de la base: el orquestador habla con este servicio por HTTP y nunca
toca PostgreSQL. Nunca se loguea ni se devuelve contenido de mensajes, la URL de
la base, el host ni la contraseña: los logs llevan operación, conversation_id y
la clase de la excepción.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional
from uuid import UUID

import psycopg
from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.memory.config import cargar_config
from src.memory.db import aplicar_esquema, crear_pool, guardar_turno, leer_historial

logger = logging.getLogger("memory")

# ⚠️ Convención propia (contrato de memoria, 3.5): centralizadas aquí para cambiarlas fácil.
ERROR_TYPE_INVALID_REQUEST = "invalid_request_error"
ERROR_TYPE_MEMORY = "memory_error"

# Mismo patrón que los CHECK de schema.sql: user_id y conversation_id (contrato, 3.1).
ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
HISTORY_LIMIT_MAX = 200

# Espera máxima al cerrar el pool por workers que siguen intentando conectar (con la base
# caída pueden estar colgados en un connect). Igual al default de psycopg_pool; explícito
# para que los tests lo bajen.
POOL_CLOSE_TIMEOUT_S = 5.0

# Tope de /health: menor que el timeout del healthcheck de compose (3 s), para que llegue a leer "loading".
HEALTH_TIMEOUT_S = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Config inválida -> el servicio no arranca. Base caída -> arranca igual, con
    /health en "loading", y el esquema se reintenta después (ver asegurar_esquema).
    """
    config = cargar_config()
    app.state.pool = crear_pool(config)
    # wait=False: no bloquear el arranque si la base todavía no está lista.
    await app.state.pool.open(wait=False)
    app.state.esquema_aplicado = False
    app.state.esquema_lock = asyncio.Lock()
    try:
        await asegurar_esquema(app)
    except Exception as e:
        logger.warning("No se pudo aplicar el esquema al arrancar (%s); se reintenta en /health.", type(e).__name__)
    yield
    await app.state.pool.close(timeout=POOL_CLOSE_TIMEOUT_S)


app = FastAPI(
    title="Memory Service",
    description="Historial de conversaciones por (user_id, conversation_id) sobre PostgreSQL.",
    version="0.1.0",
    lifespan=lifespan,
)


async def asegurar_esquema(app: FastAPI) -> None:
    """
    Aplica el esquema una sola vez. El lock evita dos CREATE TABLE simultáneos
    (p. ej. /health y una petición a la vez); la segunda verificación, dentro del
    lock, evita repetirlo si otro lo terminó mientras se esperaba.
    """
    if app.state.esquema_aplicado:
        return
    async with app.state.esquema_lock:
        if not app.state.esquema_aplicado:
            await aplicar_esquema(app.state.pool)
            app.state.esquema_aplicado = True


# ---------- Modelos de petición ----------

class Message(BaseModel):
    # Sin "system": el orquestador lo inyecta en cada pedido y no se guarda (contrato, 3.2).
    role: Literal["user", "assistant"]
    content: str


class SaveTurnRequest(BaseModel):
    user_id: str = Field(pattern=ID_PATTERN)
    turn_id: UUID
    messages: list[Message] = Field(min_length=1, max_length=2)


# ---------- Errores (contrato de memoria, 3.5) ----------

def error_response(
    status_code: int, message: str, type_: str, code: Optional[str], param: Optional[str] = None
) -> JSONResponse:
    """Único punto que construye errores, con la misma forma que el orquestador."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": type_, "param": param, "code": code}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """FastAPI responde 422 por defecto; el contrato pide 400 + invalid_request."""
    errors = exc.errors()
    first = errors[0] if errors else {}
    # loc es ("body"|"path"|"query", "campo", ...) en errores de campo, pero ("body", 14) si el
    # JSON no se pudo parsear: solo un string es nombre de campo, una posición no sirve como param.
    loc = first.get("loc", ())
    param = loc[1] if len(loc) > 1 and isinstance(loc[1], str) else None
    return error_response(
        400, first.get("msg", "Petición mal formada."), ERROR_TYPE_INVALID_REQUEST, "invalid_request", param
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    """Rutas o métodos inexistentes (404/405) también salen con el formato del contrato, no con {"detail"}."""
    return error_response(exc.status_code, str(exc.detail), ERROR_TYPE_INVALID_REQUEST, None)


def error_de_almacenamiento(operacion: str, conversation_id: str, exc: Exception) -> JSONResponse:
    """
    OperationalError cubre "la base no está": conexión caída, timeouts y, en psycopg_pool,
    PoolTimeout/PoolClosed/TooManyRequests (todas heredan de ella). Todo lo demás es un bug -> 500.

    Solo se loguea la clase: str(exc) puede traer el host o, en un CheckViolation,
    la fila con el contenido del mensaje.
    """
    if isinstance(exc, psycopg.OperationalError):
        logger.warning(
            "Base no disponible (op=%s, conversation_id=%s): %s", operacion, conversation_id, type(exc).__name__
        )
        return error_response(
            503, "El almacenamiento de memoria no está disponible.", ERROR_TYPE_MEMORY, "storage_unavailable"
        )
    logger.error("Error interno (op=%s, conversation_id=%s): %s", operacion, conversation_id, type(exc).__name__)
    return error_response(500, "Error interno del servicio de memoria.", ERROR_TYPE_MEMORY, "internal_error")


# ---------- Endpoints ----------

@app.get("/health")
async def health(request: Request):
    """
    "ready" solo si el esquema está aplicado y la base responde. Siempre HTTP 200:
    el estado va en el cuerpo, como en el contrato de proveedores (sección 7).
    """
    try:
        # Sin tope, con la base caída esperaría POOL_TIMEOUT_S (5 s). Al vencer levanta
        # TimeoutError, que sigue el camino de abajo: warning con la clase y "loading".
        async with asyncio.timeout(HEALTH_TIMEOUT_S):
            await asegurar_esquema(request.app)
            async with request.app.state.pool.connection() as conn:
                await conn.execute("SELECT 1")
    except Exception as e:
        logger.warning("Health: base no lista (%s).", type(e).__name__)
        return {"status": "loading"}
    return {"status": "ready"}


@app.post("/v1/conversations/{conversation_id}/messages")
async def save_turn(
    req: SaveTurnRequest,
    request: Request,
    conversation_id: str = Path(pattern=ID_PATTERN),
):
    """201 si guardó el turno; 200 con stored 0 si ese turn_id ya estaba (idempotencia, contrato 3.2)."""
    roles = [m.role for m in req.messages]
    if len(roles) != len(set(roles)):
        # Antes de tocar la base: es un error de quien llama, no de almacenamiento.
        return error_response(
            400, "Un turno admite como máximo un mensaje por rol.",
            ERROR_TYPE_INVALID_REQUEST, "invalid_request", "messages",
        )

    try:
        await asegurar_esquema(request.app)
        stored = await guardar_turno(
            request.app.state.pool,
            req.user_id,
            conversation_id,
            req.turn_id,
            [m.model_dump() for m in req.messages],
        )
    except Exception as e:
        return error_de_almacenamiento("write", conversation_id, e)

    return JSONResponse(
        status_code=201 if stored > 0 else 200,
        content={"conversation_id": conversation_id, "stored": stored},
    )


@app.get("/v1/conversations/{conversation_id}/messages")
async def read_history(
    request: Request,
    conversation_id: str = Path(pattern=ID_PATTERN),
    user_id: str = Query(pattern=ID_PATTERN),
    limit: int = Query(ge=1, le=HISTORY_LIMIT_MAX),
):
    """Últimos `limit` mensajes, del más antiguo al más nuevo; inexistente o ajena -> lista vacía (contrato 3.3)."""
    try:
        await asegurar_esquema(request.app)
        messages: list[dict[str, Any]] = await leer_historial(
            request.app.state.pool, user_id, conversation_id, limit
        )
    except Exception as e:
        return error_de_almacenamiento("read", conversation_id, e)

    return {"conversation_id": conversation_id, "messages": messages}
