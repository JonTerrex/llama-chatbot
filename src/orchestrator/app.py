"""
Orquestador: punto de entrada único compatible con /v1/chat/completions.

Resuelve el proveedor a partir de `model`, antepone el system prompt y reenvía.
Los proveedores son "tontos" (contrato, sección 1): todo lo que es política
(qué modelo, qué system prompt) se decide aquí. Todavía sin streaming.
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import httpx2
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

# USER_ID_PATTERN es el patrón de ids de memoria (contrato de memoria, 3.1); vale igual para
# conversation_id. Se importa en vez de duplicarlo para que no puedan divergir.
from src.orchestrator.config import USER_ID_PATTERN as MEMORY_ID_PATTERN
from src.orchestrator.config import Config, cargar_config
from src.orchestrator.logging_config import configurar_logging
from src.orchestrator.memory_client import MemoryClient, MemoryUnavailable, crear_cliente_memoria

logger = logging.getLogger("orchestrator")

# ⚠️ Convención propia, no verificada contra OpenAI: centralizadas aquí para cambiarlas fácil.
ERROR_TYPE_INVALID_REQUEST = "invalid_request_error"
ERROR_TYPE_PROVIDER = "provider_error"
ERROR_TYPE_MEMORY = "memory_error"
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
    # Primero: antes de crear cualquier cliente HTTP, que httpx2 no pueda loguear URLs.
    configurar_logging()
    app.state.config = cargar_config()
    # Un solo cliente para todo el proceso: reutiliza conexiones con los proveedores.
    app.state.client = crear_cliente(app.state.config)
    # Memoria: solo si está habilitada. El cliente httpx2 se guarda aparte para cerrarlo al
    # apagar sin tocar los atributos internos de MemoryClient.
    app.state.memory_http = None
    app.state.memory_client = None
    if app.state.config.memory_url is not None:
        app.state.memory_http = crear_cliente_memoria(app.state.config)
        app.state.memory_client = MemoryClient(app.state.memory_http, app.state.config)
    app.state.started_at = int(time.time())
    yield
    await app.state.client.aclose()
    if app.state.memory_http is not None:
        await app.state.memory_http.aclose()


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
    # Campo propio (contrato de memoria, 4.1). Declarado para que no caiga en model_extra:
    # no se loguea como extra y nunca se reenvía al proveedor.
    x_conversation_id: Optional[str] = None


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


# Lista blanca de codes del proveedor que se reconocen (llama-local: src/providers/llama_local/app.py).
# Un code fuera de esta lista, o con otro status, cae en el 502 genérico. streaming_not_supported
# no está: el orquestador nunca le pide stream al proveedor.
CODES_DEL_PROVEEDOR = frozenset({"context_length_exceeded", "insufficient_vram", "provider_unavailable", "inference_error"})


def code_del_proveedor(resp: httpx2.Response) -> Optional[str]:
    """
    El `error.code` del cuerpo, solo si está en la lista blanca; si no, None. Es lo ÚNICO que se
    lee del cuerpo de error: el resto (message incluido) puede traer texto del proveedor y nunca
    llega al cliente ni al log.
    """
    try:
        data = resp.json()
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code in CODES_DEL_PROVEEDOR else None


def error_desde_el_proveedor(resp: httpx2.Response, modelo: str) -> JSONResponse:
    """
    Traduce un no-2xx del proveedor. Se propagan solo los pares (status, code) conocidos, con un
    mensaje armado acá; el resto es el 502 provider_error de siempre. Log: modelo, status y code.
    """
    status, code = resp.status_code, code_del_proveedor(resp)
    x_provider = {"x_provider": {"name": modelo}}
    if status == 400 and code == "context_length_exceeded":
        logger.warning("Proveedor '%s' rechazó el pedido: status=%d code=%s.", modelo, status, code)
        return error_response(
            400,
            f"La petición (system, historial y mensaje nuevo) no entra en el contexto del modelo '{modelo}'.",
            ERROR_TYPE_INVALID_REQUEST, code, "messages", extra=x_provider,
        )
    if status == 503 and code in ("insufficient_vram", "provider_unavailable"):
        logger.warning("Proveedor '%s' no disponible: status=%d code=%s.", modelo, status, code)
        mensaje = (
            f"El proveedor '{modelo}' no tiene VRAM suficiente para este pedido."
            if code == "insufficient_vram"
            else f"El proveedor '{modelo}' no está disponible."
        )
        return error_response(503, mensaje, ERROR_TYPE_PROVIDER, code, extra=x_provider)
    if status == 500 and code == "inference_error":
        return error_proveedor(502, "provider_error", modelo, f"respondió HTTP {status} ({code})")
    return error_proveedor(502, "provider_error", modelo, f"respondió HTTP {status}")


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

    # 2b. Memoria (contrato de memoria, 4.1 y 4.4). Orden: los 400 de formato van antes que
    #     el 503 por memoria deshabilitada, y todo esto después de modelo y stream.
    if req.x_conversation_id is not None:
        # fullmatch sin anclas: con "^...$" y match, "$" aceptaría un "\n" final.
        if not MEMORY_ID_PATTERN.fullmatch(req.x_conversation_id):
            return error_response(
                400, "x_conversation_id debe tener 1 a 64 caracteres entre letras, dígitos, '_' y '-'.",
                ERROR_TYPE_INVALID_REQUEST, "invalid_request", "x_conversation_id",
            )
        # Con memoria, el historial lo pone el servicio: el cliente manda solo el turno nuevo.
        # Estricto a propósito (contrato 4.1): no se descartan ni se reenvían mensajes extra.
        if len(req.messages) != 1 or req.messages[0].role != "user":
            return error_response(
                400, "Con x_conversation_id, 'messages' debe tener exactamente un mensaje, con role 'user'.",
                ERROR_TYPE_INVALID_REQUEST, "invalid_request", "messages",
            )
        if config.memory_url is None:
            # Condición de configuración del servidor, no error del cliente: 503, no 400.
            # Ignorar el campo daría una respuesta normal desconectada de la conversación.
            logger.warning(
                "x_conversation_id recibido con la memoria deshabilitada (conversation_id=%r).",
                req.x_conversation_id,
            )
            return error_response(
                503, "La memoria de conversaciones no está habilitada en este servidor.",
                ERROR_TYPE_MEMORY, "memory_unavailable",
            )

    # 2c. Leer el historial (contrato de memoria, 4.2 paso 2). Si falla, 503 sin llamar al
    #     proveedor: responder sin historial daría una respuesta desconectada de la conversación.
    #     La causa ya la loguea MemoryClient; al cliente no se le expone.
    memory_client: Optional[MemoryClient] = request.app.state.memory_client
    usa_memoria = req.x_conversation_id is not None
    historial: list[dict[str, str]] = []
    if usa_memoria:
        try:
            guardados = await memory_client.leer_historial(req.x_conversation_id)
        except MemoryUnavailable:
            return error_response(
                503, "La memoria de conversaciones no está disponible en este momento.",
                ERROR_TYPE_MEMORY, "memory_unavailable",
            )
        # Solo role y content: created_at es de la memoria, no del contrato con proveedores.
        historial = [{"role": m.role, "content": m.content} for m in guardados]

    # 3. Los extras se descartan en silencio para el cliente, pero quedan en el log para diagnosticar.
    if req.model_extra:
        logger.warning("Campos no soportados descartados: %r", sorted(req.model_extra))

    # 4. Cuerpo para el proveedor: solo campos del contrato. Nuestro system va primero;
    #    los system del cliente se conservan detrás (política provisional, contrato sección 9 ❓).
    #    Con memoria: [system] + historial + el único mensaje user (validado en 2b).
    #    x_conversation_id es un campo declarado, no un extra: nunca llega acá.
    body: dict[str, Any] = {
        "model": modelo,
        "messages": [{"role": "system", "content": config.system_prompt}]
        + historial
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

    # 6 y 7. Pass-through solo si es 2xx con JSON. Un no-2xx se traduce (400/503 conocidos se
    #    propagan, el resto es 502) y, con memoria, sale acá sin guardar nada (memoria 4.2 paso 4).
    if not resp.is_success:
        return error_desde_el_proveedor(resp, modelo)
    try:
        data = resp.json()
    except ValueError:
        return error_proveedor(502, "provider_error", modelo, f"respondió HTTP {resp.status_code} con un cuerpo que no es JSON")

    if not usa_memoria:
        return JSONResponse(status_code=resp.status_code, content=data)

    # 8. Memoria: guardar el turno y agregar x_memory (contrato de memoria, 4.2 pasos 5-6 y 4.3).
    #    Llegar acá implica que el proveedor respondió bien: si falló, ya se devolvió su error
    #    sin guardar nada (4.2 paso 4).
    conversation_id = req.x_conversation_id
    if not isinstance(data, dict):
        # Sin un objeto no hay dónde poner x_memory, y tampoco se puede extraer la respuesta
        # del asistente: se devuelve tal cual, sin guardar, como en el pass-through de hoy.
        logger.error(
            "Respuesta del proveedor no es un objeto JSON: no se guarda el turno ni se agrega x_memory "
            "(conversation_id=%r).", conversation_id,
        )
        return JSONResponse(status_code=resp.status_code, content=data)

    data["x_memory"] = {"status": await guardar_turno_en_memoria(memory_client, conversation_id, req, data)}
    return JSONResponse(status_code=resp.status_code, content=data)


def contenido_del_asistente(data: dict[str, Any]) -> Optional[str]:
    """
    choices[0].message.content, o None si falta o no es str. Un "" se acepta: la memoria
    también lo acepta (Message.content: str sin largo mínimo, columna NOT NULL).
    """
    try:
        contenido = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return contenido if isinstance(contenido, str) else None


async def guardar_turno_en_memoria(
    memory_client: MemoryClient, conversation_id: str, req: ChatCompletionRequest, data: dict[str, Any]
) -> str:
    """
    Devuelve el x_memory.status: "saved" o "write_failed". Nunca falla el pedido: la respuesta
    del proveedor ya existe y se devuelve igual (contrato de memoria, 4.4). Sin reintentos.
    """
    asistente = contenido_del_asistente(data)
    if asistente is None:
        # El proveedor rompió el contrato (no un fallo de la memoria): ERROR, sin contenido.
        logger.error(
            "Respuesta del proveedor sin choices[0].message.content de texto: no se guarda el turno "
            "(conversation_id=%r).", conversation_id,
        )
        return "write_failed"

    # Uno por pedido: identifica el turno y hace idempotente un futuro reintento (contrato 3.2).
    turn_id = str(uuid.uuid4())
    try:
        await memory_client.guardar_turno(conversation_id, turn_id, req.messages[0].content, asistente)
    except MemoryUnavailable:
        # La causa ya la logueó MemoryClient; acá se agrega el turn_id para poder rastrearlo.
        logger.warning(
            "No se guardó el turno; se responde con write_failed (conversation_id=%r, turn_id=%s).",
            conversation_id, turn_id,
        )
        return "write_failed"
    return "saved"
