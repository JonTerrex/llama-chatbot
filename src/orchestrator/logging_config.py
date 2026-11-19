"""
Ajustes de logging del orquestador que protegen datos internos.

Por ahora solo silencia librerías HTTP: no configura el root, ni handlers, ni formato
(eso es un pendiente aparte).
"""
import logging

# httpcore2 loguea con loggers hijos ("httpcore2.connection", "httpcore2.http11", ...):
# fijar el padre alcanza, porque los hijos sin nivel propio heredan el suyo.
LOGGERS_SILENCIADOS = ("httpx2", "httpcore2")


def configurar_logging() -> None:
    """
    Fija httpx2 y httpcore2 en WARNING. httpx2 escribe a INFO la URL completa de cada
    pedido (memoria y proveedores): si alguien sube el root a INFO, aparecería en los logs.

    Consecuencia: para depurar httpx2 hay que cambiar esto a propósito. Idempotente.
    """
    for nombre in LOGGERS_SILENCIADOS:
        logging.getLogger(nombre).setLevel(logging.WARNING)
