"""
Presupuesto de tokens del historial de memoria: estimación aproximada, sin tokenizer.

El orquestador no carga transformers ni torch, así que cuenta caracteres. Funciones puras:
no leen config ni loguean, y nunca modifican ni reordenan los mensajes que reciben.
"""
import math
from typing import Optional

# ❓ No medido: conservador para español con el tokenizer de Llama 3 (menos caracteres por
# token = más tokens estimados = se recorta antes).
CHARS_POR_TOKEN = 3.0
# Cabeceras del chat template por mensaje (rol y separadores).
OVERHEAD_POR_MENSAJE = 8
# Lo que el template agrega una vez al prompt completo (inicio y cabecera de la respuesta).
OVERHEAD_BASE = 16
# ❓ Acoplamiento conocido: copia de MAX_TOKENS_DEFAULT de src/providers/llama_local/engine.py
# (lo que genera llama-local si el pedido no trae max_tokens). Si cambia allá, cambiar acá.
MAX_TOKENS_DEFAULT_PROVEEDOR = 256


def tokens_de_mensaje(mensaje: dict[str, str]) -> int:
    return math.ceil(len(mensaje["content"]) / CHARS_POR_TOKEN) + OVERHEAD_POR_MENSAJE


def tokens_de_mensajes(mensajes: list[dict[str, str]]) -> int:
    return sum(tokens_de_mensaje(m) for m in mensajes)


def disponible_para_mensajes(context_tokens: int, max_tokens: Optional[int]) -> int:
    """Tokens estimados que pueden ocupar los mensajes. Puede ser negativo (nada entra)."""
    max_tokens_efectivo = MAX_TOKENS_DEFAULT_PROVEEDOR if max_tokens is None else max_tokens
    return context_tokens - max_tokens_efectivo - OVERHEAD_BASE


def recortar_historial(
    fijos: list[dict[str, str]], historial: list[dict[str, str]], disponible: int
) -> Optional[list[dict[str, str]]]:
    """
    El sufijo más largo del historial que entra junto con `fijos` (system + mensaje nuevo, que
    nunca se descartan). Se descarta desde el más viejo; si tras descartar algo el primero que
    queda es del asistente (par partido), se descarta también. None si `fijos` solos no entran.
    """
    fijos_tokens = tokens_de_mensajes(fijos)
    if fijos_tokens > disponible:
        return None
    inicio = 0
    total = fijos_tokens + tokens_de_mensajes(historial)
    while total > disponible:
        total -= tokens_de_mensaje(historial[inicio])
        inicio += 1
    if inicio > 0 and inicio < len(historial) and historial[inicio]["role"] == "assistant":
        inicio += 1
    return historial[inicio:]
