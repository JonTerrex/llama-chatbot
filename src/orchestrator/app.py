"""
Orquestador: punto de entrada único compatible con /v1/chat/completions.

Resuelve el proveedor a partir de `model`, antepone el system prompt y reenvía.
Los proveedores son "tontos" (contrato, sección 1): todo lo que es política
(qué modelo, qué system prompt) se decide aquí. Todavía sin streaming.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import httpx2
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.orchestrator.config import Config, cargar_config

logger = logging.getLogger("orchestrator")

# ⚠️ Convención propia, no verificada contra OpenAI: centralizadas aquí para cambiarlas fácil.
ERROR_TYPE_INVALID_REQUEST = "invalid_request_error"
ERROR_TYPE_PROVIDER = "provider_error"
STREAMING_NOT_SUPPORTED_STATUS = 501

CONNECT_TIMEOUT_S = 5.0


def crear_cliente(config: Config) -> httpx2.AsyncClient:
    """
    Función aparte para que los tests la sustituyan (monkeypatch) por un cliente con MockTransport.

    connect corto: si el proveedor no acepta la conexión en 5 s, está caído; esperar
    timeout_s (pensado para la generación, que es lenta) solo retrasaría el error.
    """
    return httpx2.AsyncClient(timeout=httpx2.Timeout(config.timeout_s, connect=CONNECT_TIMEOUT_S))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga la configuración al arrancar (si es inválida, el servicio no arranca) y abre un cliente HTTP compartido."""
    app.state.config = cargar_config()
    # Un solo cliente para todo el proceso: reutiliza conexiones con los proveedores.
    app.state.client = crear_cliente(app.state.config)
    app.state.started_at = int(time.time())
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="LLM Orchestrator",
    description="Orquestador compatible con /v1/chat/completions que enruta a proveedores LLM.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Modelos de petición ----------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # "allow" en vez de "ignore": los extras no se reenvían, pero así se pueden loguear (model_extra).
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ---------- Errores (contrato, sección 6) ----------

def error_response(
    status_code: int,
    message: str,
    type_: str,
    code: Optional[str],
    param: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    """Único punto que construye errores; `extra` se mezcla dentro de error (p. ej. x_provider)."""
    error: dict[str, Any] = {"message": message, "type": type_, "param": param, "code": code}
    if extra:
        error.update(extra)
    return JSONResponse(status_code=status_code, content={"error": error})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """FastAPI responde 422 por defecto; el contrato pide 400 + invalid_request."""
    errors = exc.errors()
    first = errors[0] if errors else {}
    # loc es ("body", "messages", ...) en errores de campo, pero ("body", 14) si el JSON
    # no se pudo parsear: solo un string es nombre de campo, una posición no sirve como param.
    loc = first.get("loc", ())
    param = loc[1] if len(loc) > 1 and isinstance(loc[1], str) else None
    return error_response(
        400, first.get("msg", "Petición mal formada."), ERROR_TYPE_INVALID_REQUEST, "invalid_request", param
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    """Rutas o métodos inexistentes (404/405) también salen con el formato del contrato, no con {"detail"}."""
    return error_response(exc.status_code, str(exc.detail), ERROR_TYPE_INVALID_REQUEST, None)


def error_proveedor(status_code: int, code: str, modelo: str, causa: str) -> JSONResponse:
    """Loguea y construye un fallo del proveedor; x_provider permite al cliente saber cuál falló."""
    logger.warning("Proveedor '%s' falló (%s): %s", modelo, code, causa)
    return error_response(
        status_code,
        f"El proveedor '{modelo}' falló: {causa}",
        ERROR_TYPE_PROVIDER,
        code,
        extra={"x_provider": {"name": modelo}},
    )


# ---------- Endpoints ----------

@app.get("/health")
def health():
    """Listo en cuanto arrancó: la config ya se validó en el lifespan. No sondea proveedores (eso es de otro tramo)."""
    return {"status": "ready"}


@app.get("/v1/models")
def list_models(request: Request):
    """
    Un objeto por proveedor del registro, en el orden del YAML.

    ⚠️ Formato del objeto model (id, object, created, owned_by) de memoria de la API de OpenAI,
    no verificado. Por ahora lista todos los registrados, no solo los activos (falta el sondeo).
    """
    config: Config = request.app.state.config
    created = request.app.state.started_at
    return {
        "object": "list",
        "data": [
            {"id": nombre, "object": "model", "created": created, "owned_by": "llama-chatbot"}
            for nombre in config.providers
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """Resuelve modelo, valida, antepone el system prompt y reenvía al proveedor (pass-through de la respuesta)."""
    config: Config = request.app.state.config

    # 1. Resolver el modelo.
    modelo = req.model or config.default_model
    if modelo is None:
        return error_response(
            400, "Falta 'model' y no hay DEFAULT_MODEL configurado.",
            ERROR_TYPE_INVALID_REQUEST, "invalid_request", "model",
        )
    if modelo not in config.providers:
        return error_response(
            404, f"El modelo '{modelo}' no existe.",
            ERROR_TYPE_INVALID_REQUEST, "model_not_found", "model",
        )

    # 2. Streaming: temporal, hasta implementar SSE. No se reenvía nada.
    if req.stream:
        return error_response(
            STREAMING_NOT_SUPPORTED_STATUS,
            "El streaming todavía no está implementado en el orquestador.",
            ERROR_TYPE_PROVIDER, "streaming_not_supported", "stream",
        )

    # 3. Los extras se descartan en silencio para el cliente, pero quedan en el log para diagnosticar.
    if req.model_extra:
        logger.warning("Campos no soportados descartados: %r", sorted(req.model_extra))
        
    # 4. Cuerpo para el proveedor: solo campos del contrato. Nuestro system va primero;
    #    los system del cliente se conservan detrás (política provisional, contrato sección 9 ❓).
    body: dict[str, Any] = {
        "model": modelo,
        "messages": [{"role": "system", "content": config.system_prompt}]
        + [m.model_dump() for m in req.messages],
        "stream": False,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens

    # 5. Reenvío. Orden de captura importante: ConnectTimeout hereda de TimeoutException,
    #    no de ConnectError, así que hay que nombrarlo antes que el timeout genérico.
    url = f"{config.providers[modelo]}/v1/chat/completions"
    try:
        resp = await request.app.state.client.post(url, json=body)
    except (httpx2.ConnectError, httpx2.ConnectTimeout) as e:
        # Un connect timeout significa que el proveedor no aceptó la conexión: está caído
        # o inalcanzable, igual que un rechazo. No es "tardó en generar" (eso es el 504).
        return error_proveedor(502, "provider_error", modelo, f"no se pudo conectar ({type(e).__name__})")
    except httpx2.TimeoutException as e:
        return error_proveedor(504, "provider_timeout", modelo, f"no respondió a tiempo ({type(e).__name__})")
    except httpx2.TransportError as e:
        return error_proveedor(502, "provider_error", modelo, f"error de transporte ({type(e).__name__})")

    # 6 y 7. Pass-through solo si es 2xx con JSON; cualquier otra cosa es un fallo del proveedor.
    if not resp.is_success:
        return error_proveedor(502, "provider_error", modelo, f"respondió HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        return error_proveedor(502, "provider_error", modelo, f"respondió HTTP {resp.status_code} con un cuerpo que no es JSON")

    return JSONResponse(status_code=resp.status_code, content=data)
