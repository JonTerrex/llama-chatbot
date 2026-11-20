"""
Proveedor Llama local: servicio que cumple el contrato de proveedores
(docs/contrato-proveedores.md) con Llama 3 8B en GPU.

Modelado sobre el mock (src/providers/mock/app.py): mismos modelos de petición, errores y
handlers, copiados para no depender del resto de src/. No decide el system prompt: genera
con los mensajes que recibe. Sin x_debug ni /debug/requests (son solo del mock).
"""
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.providers.llama_local.engine import ContextoExcedido, MotorLlama, VRAMInsuficiente

logger = logging.getLogger("llama_local")

# ⚠️ Convención propia, no verificada contra OpenAI: centralizadas aquí para cambiarlas fácil.
ERROR_TYPE_INVALID_REQUEST = "invalid_request_error"
ERROR_TYPE_PROVIDER = "provider_error"
STREAMING_NOT_SUPPORTED_STATUS = 501

MODEL_PATH_DEFAULT = "modelos/llama3"
TOP_P_DEFAULT = "0.9"

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
# Nombre del handler propio: permite reconocerlo y no duplicarlo si el lifespan corre varias veces.
NOMBRE_HANDLER = "llama_local_stderr"


def configurar_logging() -> None:
    """
    Handler propio a stderr para el logger "llama_local", con nivel de LOG_LEVEL (default INFO).
    Sin esto, en Docker no se ve nada: el root no está configurado y uvicorn solo configura
    sus propios loggers. propagate=False evita que la misma línea salga dos veces si alguien
    configura el root. Idempotente. LOG_LEVEL inválido -> ValueError: el servicio no arranca.

    ⚠️ LOG_LEVEL=DEBUG imprime tracebacks de las fallas de generación, que pueden incluir
    texto del usuario: no usarlo con datos reales.
    """
    raw = os.getenv("LOG_LEVEL") or "INFO"
    nivel = logging.getLevelNamesMapping().get(raw.upper())
    if nivel is None:
        raise ValueError(f"LOG_LEVEL={raw!r} no es un nivel válido (DEBUG, INFO, WARNING, ERROR, CRITICAL).")

    log = logging.getLogger("llama_local")
    log.setLevel(nivel)
    log.propagate = False
    if not any(h.get_name() == NOMBRE_HANDLER for h in log.handlers):
        handler = logging.StreamHandler()  # stderr
        handler.set_name(NOMBRE_HANDLER)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        log.addHandler(handler)


def crear_motor() -> MotorLlama:
    """
    Lee la config del entorno y crea el motor, sin cargarlo. Función aparte para que los
    tests la sustituyan por un motor falso (mismo patrón que crear_cliente en el orquestador).
    Config inválida -> ValueError en el lifespan: el servicio no arranca.
    """
    path_modelo = os.getenv("MODEL_PATH") or MODEL_PATH_DEFAULT
    raw = os.getenv("TOP_P") or TOP_P_DEFAULT
    try:
        top_p = float(raw)
    except ValueError as e:
        raise ValueError(f"TOP_P={raw!r} no es un número.") from e
    if not 0 < top_p <= 1:
        raise ValueError(f"TOP_P={raw!r} fuera de rango: debe ser mayor que 0 y como máximo 1.")

    # Override manual del tope de contexto. Ausente o vacío: lo decide la VRAM al cargar.
    raw_ctx = os.getenv("CONTEXTO_MAX_TOKENS") or None
    contexto_max_config = None
    if raw_ctx is not None:
        try:
            contexto_max_config = int(raw_ctx)
        except ValueError as e:
            raise ValueError(f"CONTEXTO_MAX_TOKENS={raw_ctx!r} no es un entero.") from e
        if contexto_max_config <= 0:
            raise ValueError(f"CONTEXTO_MAX_TOKENS={raw_ctx!r} debe ser un entero mayor que 0.")
    return MotorLlama(path_modelo, top_p, contexto_max_config)


def _cargar_en_segundo_plano(app: FastAPI) -> None:
    """Corre en un hilo: la carga tarda minutos y el servidor tiene que responder /health mientras tanto."""
    t_inicio = time.perf_counter()
    try:
        app.state.motor.cargar()
    except Exception:
        # Con traceback: acá no hay contenido de usuario, y sin él no se diagnostica una carga fallida.
        logger.exception("Falló la carga del modelo; el proveedor queda en estado 'failed'.")
        app.state.estado = "failed"
        return
    app.state.estado = "ready"
    logger.info("Modelo cargado en %.1f s.", time.perf_counter() - t_inicio)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lee la config al arrancar (no al importar) y lanza la carga sin esperarla."""
    # Primero: para que la carga del modelo y cualquier error de config ya se vean.
    configurar_logging()
    app.state.motor = crear_motor()
    app.state.estado = "loading"
    # Una generación a la vez: la GPU no tiene VRAM para dos, y en paralelo se pisarían.
    app.state.lock_generacion = threading.Lock()
    # daemon: si se apaga el servidor durante la carga, el hilo no impide salir.
    app.state.hilo_carga = threading.Thread(
        target=_cargar_en_segundo_plano, args=(app,), name="carga-modelo", daemon=True
    )
    app.state.hilo_carga.start()
    yield
    logger.info("Apagando el proveedor.")
    if app.state.estado == "ready":
        app.state.motor.liberar()


app = FastAPI(
    title="Llama Local Provider",
    description="Proveedor compatible con /v1/chat/completions sobre Llama 3 8B en GPU (4 bits).",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Modelos de petición ----------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # Campos no soportados (top_p, etc.) se ignoran en vez de rechazarse (contrato, sección 3).
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    # Rango de OpenAI: fuera de él la generación no tiene sentido, así que es un 400 y no un 500.
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1)


# ---------- Errores (contrato, sección 6). Copiados del mock; x_provider es opcional ----------

def error_response(
    status_code: int,
    message: str,
    type_: str,
    code: Optional[str],
    param: Optional[str] = None,
    x_provider: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    """Único punto que construye errores, para que todos tengan la forma error.{message,type,param,code}."""
    error: dict[str, Any] = {"message": message, "type": type_, "param": param, "code": code}
    if x_provider is not None:
        error["x_provider"] = x_provider
    return JSONResponse(status_code=status_code, content={"error": error})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """FastAPI responde 422 por defecto; el contrato pide 400 + invalid_request."""
    errors = exc.errors()
    first = errors[0] if errors else {}
    # loc viene como ("body", "messages", 0, ...): el segundo elemento es el campo raíz.
    loc = first.get("loc", ())
    param = loc[1] if len(loc) > 1 and isinstance(loc[1], str) else None
    return error_response(
        400, first.get("msg", "Petición mal formada."), ERROR_TYPE_INVALID_REQUEST, "invalid_request", param
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    """Rutas o métodos inexistentes (404/405) también salen con el formato del contrato, no con {"detail"}."""
    return error_response(exc.status_code, str(exc.detail), ERROR_TYPE_INVALID_REQUEST, None)


# ---------- Endpoints ----------

@app.get("/health")
def health(request: Request):
    """'ready' solo con el modelo cargado; 'loading' y 'failed' van con 503 (contrato, sección 7)."""
    estado = request.app.state.estado
    if estado == "ready":
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": estado})


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    Síncrono a propósito: FastAPI lo corre en un hilo, así generar (minutos de GPU) y la
    espera del lock no bloquean el event loop. Orden: validación -> stream -> estado -> generar.
    """
    if req.stream:
        # Temporal, igual que en el mock: hasta implementar SSE.
        return error_response(
            STREAMING_NOT_SUPPORTED_STATUS,
            "El streaming todavía no está implementado en el proveedor llama_local.",
            ERROR_TYPE_PROVIDER, "streaming_not_supported", "stream",
        )

    estado = request.app.state.estado
    if estado != "ready":
        return error_response(
            503, f"El proveedor '{req.model}' no está disponible (estado: {estado}).",
            ERROR_TYPE_PROVIDER, "provider_unavailable",
            x_provider={"name": req.model, "state": estado},
        )

    mensajes = [m.model_dump() for m in req.messages]
    t_llegada = time.perf_counter()
    with request.app.state.lock_generacion:
        # latency_s cuenta desde que se obtiene el lock: tokenización + generación, sin la cola.
        t_inicio = time.perf_counter()
        espera_s = t_inicio - t_llegada
        try:
            resultado = request.app.state.motor.generar(mensajes, req.temperature, req.max_tokens)
        except ContextoExcedido as e:
            logger.info(
                "Contexto excedido: prompt_tokens=%d max_new_tokens=%s limite=%d.",
                e.prompt_tokens, e.max_new_tokens, e.limite,
            )
            return error_response(
                400,
                f"La petición no entra en el contexto del modelo: {e.prompt_tokens} tokens de entrada "
                f"+ {e.max_new_tokens} de max_tokens superan el límite de {e.limite}.",
                ERROR_TYPE_INVALID_REQUEST, "context_length_exceeded", "messages",
            )
        except VRAMInsuficiente as e:
            # 503: no hay capacidad para este pedido en este momento o en esta GPU. Viene de la
            # verificación previa o de un OOM real al generar, y puede ser permanente para ese
            # tamaño. El 400 context_length_exceeded es un límite ESTIMADO y conservador, no una
            # garantía. Solo números, sin traceback ni texto de torch.
            logger.info(
                "VRAM insuficiente: necesaria_mib=%.1f disponible_mib=%.1f.", e.necesaria_mb, e.disponible_mb
            )
            return error_response(
                503,
                f"VRAM insuficiente: el pedido necesita {e.necesaria_mb:.0f} MiB "
                f"y hay {e.disponible_mb:.0f} MiB disponibles.",
                ERROR_TYPE_PROVIDER, "insufficient_vram",
            )
        except Exception as e:
            # Solo la clase: el texto de una excepción de generación puede traer contenido del prompt.
            # El traceback completo, únicamente en DEBUG.
            logger.error("Falló la generación (%s).", type(e).__name__)
            logger.debug("Traceback de la falla de generación.", exc_info=True)
            return error_response(
                500, "Error interno al generar la respuesta.", ERROR_TYPE_PROVIDER, "inference_error",
            )
        latency_s = round(time.perf_counter() - t_inicio, 3)

    logger.info(
        "Generación: prompt_tokens=%d completion_tokens=%d finish_reason=%s "
        "espera_s=%.3f latency_s=%.3f generation_s=%.3f",
        resultado.prompt_tokens, resultado.completion_tokens, resultado.finish_reason,
        espera_s, latency_s, resultado.generation_s,
    )

    x_metrics: dict[str, Any] = {"latency_s": latency_s}
    # Velocidad de generación pura (sin tokenización): se omite si no hay un tiempo medible.
    if round(resultado.generation_s, 3) > 0:
        x_metrics["tokens_per_s"] = round(resultado.completion_tokens / resultado.generation_s, 2)
    if resultado.vram_peak_mb is not None:
        x_metrics["vram_peak_mb"] = resultado.vram_peak_mb
    if resultado.vram_reserved_mb is not None:
        x_metrics["vram_reserved_mb"] = resultado.vram_reserved_mb

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": resultado.texto},
                "finish_reason": resultado.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": resultado.prompt_tokens,
            "completion_tokens": resultado.completion_tokens,
            "total_tokens": resultado.prompt_tokens + resultado.completion_tokens,
        },
        "x_metrics": x_metrics,
    }
