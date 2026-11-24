"""
Logging del orquestador: handler propio para el logger "orchestrator" y silencio de librerías HTTP.

Mismo patrón que llama-local (src/providers/llama_local/app.py), sin importarlo: handler a
stderr con nombre fijo, nivel desde LOG_LEVEL y propagate=False. No toca el root.
"""
import logging
import os

# httpcore2 loguea con loggers hijos ("httpcore2.connection", "httpcore2.http11", ...):
# fijar el padre alcanza, porque los hijos sin nivel propio heredan el suyo.
LOGGERS_SILENCIADOS = ("httpx2", "httpcore2")

# Padre de todos los loggers del servicio ("orchestrator", "orchestrator.memory", ...).
LOGGER_ORQUESTADOR = "orchestrator"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Nombre del handler propio: permite reconocerlo y no duplicarlo si el lifespan corre varias veces.
NOMBRE_HANDLER = "orchestrator_stderr"
# Diccionario propio: logging.getLevelNamesMapping solo existe desde Python 3.11.
NIVELES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configurar_logging() -> None:
    """
    1. Fija httpx2 y httpcore2 en WARNING, sea cual sea LOG_LEVEL: httpx2 escribe a INFO la
       URL completa de cada pedido (memoria y proveedores), que es interna.
    2. Handler a stderr para "orchestrator" con nivel de LOG_LEVEL (default INFO; vacío
       cuenta como ausente). Sin esto, en Docker los INFO se descartan: el root no está
       configurado y uvicorn solo configura sus propios loggers. propagate=False evita que
       la misma línea salga dos veces si alguien configura el root.

    Idempotente. LOG_LEVEL inválido -> ValueError: el servicio no arranca. Se valida antes
    de tocar nada, para no dejar el logging a medio configurar.
    """
    raw = os.getenv("LOG_LEVEL") or "INFO"
    nivel = NIVELES.get(raw.upper())
    if nivel is None:
        raise ValueError(f"LOG_LEVEL no es un nivel válido ({', '.join(NIVELES)}).")

    for nombre in LOGGERS_SILENCIADOS:
        logging.getLogger(nombre).setLevel(logging.WARNING)

    log = logging.getLogger(LOGGER_ORQUESTADOR)
    log.setLevel(nivel)
    log.propagate = False
    if not any(h.get_name() == NOMBRE_HANDLER for h in log.handlers):
        handler = logging.StreamHandler()  # stderr
        handler.set_name(NOMBRE_HANDLER)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        log.addHandler(handler)
