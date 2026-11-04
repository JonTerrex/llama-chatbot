"""
Proveedor mock: servicio independiente que cumple el contrato de proveedores
(docs/contrato-proveedores.md) sin LLM real.

Sirve como baseline de rendimiento y para probar el orquestador sin GPU.
No importa nada del resto de src/ para que su imagen Docker sea mínima.
Solo para desarrollo: x_debug y /debug/requests exponen la conversación completa.
"""
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

# ⚠️ Convención propia, no verificada contra OpenAI: centralizadas aquí para cambiarlas fácil.
ERROR_TYPE_INVALID_REQUEST = "invalid_request_error"
ERROR_TYPE_PROVIDER = "provider_error"
STREAMING_NOT_SUPPORTED_STATUS = 501

# El mock no depende de nada del resto de src/, para que su imagen Docker sea mínima.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lee la configuración del entorno al arrancar (no al importar), para que los tests usen monkeypatch.setenv."""
    app.state.latency = float(os.getenv("MOCK_LATENCY", "0"))
    app.state.history = deque(maxlen=int(os.getenv("MOCK_HISTORY_SIZE", "20")))
    yield
    app.state.history.clear()

# ---------- FastAPI app ----------
app = FastAPI(
    title="Mock LLM Provider",
    description="Proveedor simulado compatible con /v1/chat/completions. Solo desarrollo.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Modelos de petición ----------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # Campos no soportados (top_p, etc.) se ignoran en vez de rechazarse.
    model_config = ConfigDict(extra="ignore")

    model: str = "mock"
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ---------- Errores (contrato, sección 6) ----------

def error_response(
    status_code: int, message: str, type_: str, code: Optional[str], param: Optional[str] = None
) -> JSONResponse:
    """Único punto que construye errores, para que todos tengan la forma error.{message,type,param,code}."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": type_, "param": param, "code": code}},
    )

# ---------- Handlers de errores ----------
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


# ---------- Lógica del mock ----------

def generar_texto(messages: list[dict]) -> str:
    """Copia de src/mock_llm.generar_respuesta (sin importarla, para no depender del resto de src/)."""
    ultimo_usuario = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    turnos = sum(1 for m in messages if m["role"] == "user")
    return (
        "**[MOCK]** Respuesta simulada, sin LLM real.\n\n"
        f"Recibí tu mensaje: **{ultimo_usuario}**\n\n"
        f"Mensajes del usuario en el historial: **{turnos}**"
    )


def contar_palabras(texto: str) -> int:
    return len(texto.split())


def params_de(req: ChatCompletionRequest) -> dict[str, Any]:
    return {
        "model": req.model,
        "stream": req.stream,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }


def construir_x_debug(req: ChatCompletionRequest, messages: list[dict]) -> dict[str, Any]:
    """Todo x_debug sale de aquí: quitar la extensión es borrar esta función y su uso."""
    return {"messages": messages, "params": params_de(req)}


# ---------- Endpoints ----------

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    Respuesta sin streaming en formato OpenAI.

    Endpoint síncrono a propósito: FastAPI lo ejecuta en un hilo, así time.sleep
    (MOCK_LATENCY) no bloquea el event loop.
    """
    t_inicio = time.perf_counter()
    messages = [m.model_dump() for m in req.messages]

    # Se registra toda petición válida, incluidas las de streaming, para ver qué llegó.
    request.app.state.history.append({
        "received_at": datetime.now(timezone.utc).isoformat(),
        "request": {"model": req.model, "messages": messages, "params": params_de(req)},
    })

    if req.stream:
        # Temporal: hasta implementar SSE en un tramo posterior.
        return error_response(
            STREAMING_NOT_SUPPORTED_STATUS,
            "El streaming todavía no está implementado en el proveedor mock.",
            ERROR_TYPE_PROVIDER,
            "streaming_not_supported",
            "stream",
        )

    if request.app.state.latency > 0:
        time.sleep(request.app.state.latency)

    texto = generar_texto(messages)

    # Cifras SIMULADAS (palabras, no tokens reales): estables y deterministas,
    # solo sirven para probar que usage se propaga de punta a punta.
    prompt_tokens = sum(contar_palabras(m["content"]) for m in messages)
    completion_tokens = contar_palabras(texto)
    # Métricas de rendimiento (latencia, tokens/s) se calculan en el mock, pero no VRAM.
    latency_s = round(time.perf_counter() - t_inicio, 3)
    x_metrics = {"latency_s": latency_s}
    if latency_s > 0:
        x_metrics["tokens_per_s"] = round(completion_tokens / latency_s, 2)
    # Sin claves vram_*: el contrato dice que solo aparecen si aplican (proveedores con GPU).

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "x_metrics": x_metrics,
        "x_debug": construir_x_debug(req, messages),
    }


@app.get("/health")
def health():
    """El mock no carga nada: siempre está listo para atender."""
    return {"status": "ready"}


@app.get("/debug/requests")
def debug_requests(request: Request):
    """Últimas N peticiones en orden cronológico (la más nueva al final, como las deja el deque)."""
    history = request.app.state.history
    return {"count": len(history), "max": history.maxlen, "requests": list(history)}
