"""
Cliente HTTP del orquestador hacia el servicio de memoria (contrato de memoria, 3.2, 3.3 y 4.4).

Cualquier fallo se traduce a MemoryUnavailable con una `causa` corta: quien llama decide
qué hacer (503 al leer, "write_failed" al guardar). Nunca se loguea ni se pone en la
excepción la URL, el cuerpo de la respuesta ni el contenido de los mensajes.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, NoReturn
from urllib.parse import quote
from uuid import UUID

import httpx2

from src.orchestrator.config import Config

logger = logging.getLogger("orchestrator.memory")

CAUSAS = frozenset({"conexion", "timeout", "estado_http", "respuesta_invalida"})
ROLES_GUARDADOS = frozenset({"user", "assistant"})


class MemoryUnavailable(Exception):
    """La memoria no se pudo usar. El mensaje solo lleva la causa: nada de URL ni cuerpo."""

    def __init__(self, causa: str):
        if causa not in CAUSAS:
            raise ValueError(f"causa desconocida: {causa!r}")
        super().__init__(f"Memoria no disponible ({causa}).")
        self.causa = causa


@dataclass(frozen=True)
class MensajeGuardado:
    role: str
    content: str
    created_at: int


def crear_cliente_memoria(config: Config) -> httpx2.AsyncClient:
    """
    Cliente propio, separado del de proveedores: su timeout es corto (memory_timeout_s)
    para que una memoria colgada no congele el chat. Función aparte para que los tests
    la sustituyan por un cliente con MockTransport.
    """
    return httpx2.AsyncClient(timeout=httpx2.Timeout(config.memory_timeout_s))


def _es_int(valor: Any) -> bool:
    # bool es subclase de int: True no es un created_at ni un stored válido.
    return type(valor) is int


class MemoryClient:
    def __init__(self, client: httpx2.AsyncClient, config: Config):
        if config.memory_url is None:
            raise ValueError("MemoryClient requiere config.memory_url (memoria habilitada).")
        self._client = client
        self._base = config.memory_url
        # user_id sale siempre de la config: ningún método público lo recibe, así nadie
        # puede leer o escribir conversaciones de otro usuario a través de este cliente.
        self._user_id = config.dev_user_id
        self._limit = config.memory_history_limit
        self._timeout_s = config.memory_timeout_s

    def _url(self, conversation_id: str) -> str:
        # safe="": un "/" o "?" en el id no puede cambiar la ruta ni agregar parámetros.
        return f"{self._base}/v1/conversations/{quote(conversation_id, safe='')}/messages"

    def _fallar(self, operacion: str, conversation_id: str, causa: str, origen: str) -> NoReturn:
        # respuesta_invalida es un bug de contrato entre servicios, no una caída: error.
        nivel = logging.ERROR if causa == "respuesta_invalida" else logging.WARNING
        # %r en conversation_id: viene del cliente y acá todavía no está validado; así un
        # salto de línea no puede partir ni inventar líneas de log.
        logger.log(
            nivel,
            "Memoria no disponible (op=%s, conversation_id=%r, causa=%s, origen=%s).",
            operacion, conversation_id, causa, origen,
        )
        # from None: la excepción original de httpx2 trae la URL en su texto; no se encadena
        # para que un logger.exception aguas arriba tampoco la imprima.
        raise MemoryUnavailable(causa) from None

    async def _pedir(self, operacion: str, conversation_id: str, metodo: str, **kwargs) -> Any:
        """Hace el pedido con tope total, exige 2xx y devuelve el JSON. Los fallos van a _fallar."""
        try:
            # Tope total además del de httpx2 (que es por fase: conectar, leer...): un servidor
            # que gotea bytes no puede estirar el pedido más allá de memory_timeout_s.
            async with asyncio.timeout(self._timeout_s):
                respuesta = await self._client.request(metodo, self._url(conversation_id), **kwargs)
        # El orden importa: ConnectTimeout hereda de TimeoutException, pero no conectar es
        # "conexion" (memoria caída o inalcanzable), no "tardó en responder".
        except (httpx2.ConnectError, httpx2.ConnectTimeout) as e:
            self._fallar(operacion, conversation_id, "conexion", type(e).__name__)
        except (httpx2.TimeoutException, TimeoutError) as e:
            self._fallar(operacion, conversation_id, "timeout", type(e).__name__)
        except httpx2.TransportError as e:
            self._fallar(operacion, conversation_id, "conexion", type(e).__name__)
        # DecodingError y TooManyRedirects heredan de RequestError, no de TransportError.
        # Un cuerpo que no se puede decodificar (p. ej. gzip roto) es una respuesta inválida.
        except httpx2.DecodingError as e:
            self._fallar(operacion, conversation_id, "respuesta_invalida", type(e).__name__)
        # Catch-all: cualquier otro error del pedido es "no se pudo usar la memoria".
        except httpx2.RequestError as e:
            self._fallar(operacion, conversation_id, "conexion", type(e).__name__)

        if not respuesta.is_success:
            self._fallar(operacion, conversation_id, "estado_http", f"HTTP {respuesta.status_code}")
        try:
            return respuesta.json()
        except ValueError:  # JSONDecodeError y cuerpo no decodificable
            self._fallar(operacion, conversation_id, "respuesta_invalida", "json")

    async def leer_historial(self, conversation_id: str) -> list[MensajeGuardado]:
        """Últimos memory_history_limit mensajes, del más antiguo al más nuevo (contrato 3.3)."""
        datos = await self._pedir(
            "read", conversation_id, "GET", params={"user_id": self._user_id, "limit": self._limit}
        )
        # Validación estricta: lo que salga de acá va directo al prompt del proveedor.
        items = datos.get("messages") if isinstance(datos, dict) else None
        if not isinstance(items, list):
            self._fallar("read", conversation_id, "respuesta_invalida", "forma")
        mensajes: list[MensajeGuardado] = []
        for item in items:
            if not (
                isinstance(item, dict)
                and item.get("role") in ROLES_GUARDADOS
                and isinstance(item.get("content"), str)
                and _es_int(item.get("created_at"))
            ):
                self._fallar("read", conversation_id, "respuesta_invalida", "forma")
            mensajes.append(MensajeGuardado(item["role"], item["content"], item["created_at"]))
        return mensajes

    async def guardar_turno(
        self, conversation_id: str, turn_id: str | UUID, user_content: str, assistant_content: str
    ) -> None:
        """
        Guarda [user, assistant] en una sola llamada (contrato 3.2). 201 y 200 (stored 0, ese
        turn_id ya estaba) son éxito. Sin reintentos: hoy un fallo se informa como write_failed.
        """
        body = {
            "user_id": self._user_id,
            "turn_id": str(turn_id),
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        datos = await self._pedir("write", conversation_id, "POST", json=body)
        if not (isinstance(datos, dict) and _es_int(datos.get("stored"))):
            self._fallar("write", conversation_id, "respuesta_invalida", "forma")
