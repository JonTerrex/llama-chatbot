"""Verifica configurar_logging() del orquestador: nivel, handler propio, formato y silencio de httpx2."""
import io
import logging
import re
import sys

import pytest
from fastapi.testclient import TestClient

from src.orchestrator.app import app
from src.orchestrator.logging_config import (
    LOG_FORMAT,
    LOGGER_ORQUESTADOR,
    LOGGERS_SILENCIADOS,
    NOMBRE_HANDLER,
    configurar_logging,
)

TODOS = (LOGGER_ORQUESTADOR, *LOGGERS_SILENCIADOS)


@pytest.fixture(autouse=True)
def estado_de_loggers(monkeypatch):
    """
    Sin LOG_LEVEL del entorno, y al final se restauran handlers, nivel y propagate de cada
    logger tocado: son globales al proceso y no deben contaminar a otros tests ni a caplog.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    guardado = {}
    for nombre in TODOS:
        log = logging.getLogger(nombre)
        guardado[nombre] = (list(log.handlers), log.level, log.propagate)
    yield
    for nombre, (handlers, nivel, propaga) in guardado.items():
        log = logging.getLogger(nombre)
        for h in list(log.handlers):
            if h not in handlers:
                log.removeHandler(h)
                h.close()
        log.setLevel(nivel)
        log.propagate = propaga


def orquestador():
    return logging.getLogger(LOGGER_ORQUESTADOR)


def handlers_propios():
    return [h for h in orquestador().handlers if h.get_name() == NOMBRE_HANDLER]


# ---------- Nivel ----------

@pytest.mark.parametrize("valor", [None, ""], ids=["ausente", "vacio"])
def test_default_es_info(monkeypatch, valor):
    # Vacío cuenta como ausente, igual que en llama-local ("LOG_LEVEL=" en compose).
    if valor is not None:
        monkeypatch.setenv("LOG_LEVEL", valor)
    configurar_logging()
    assert orquestador().level == logging.INFO


@pytest.mark.parametrize(
    "valor, nivel",
    [("DEBUG", logging.DEBUG), ("INFO", logging.INFO), ("WARNING", logging.WARNING),
     ("ERROR", logging.ERROR), ("CRITICAL", logging.CRITICAL),
     ("debug", logging.DEBUG), ("warning", logging.WARNING), ("Error", logging.ERROR)],
)
def test_niveles_validos(monkeypatch, valor, nivel):
    monkeypatch.setenv("LOG_LEVEL", valor)
    configurar_logging()
    assert orquestador().level == nivel


@pytest.mark.parametrize("valor", ["MUCHO", "5x", "20", " info", "WARN", "NOTSET"])
def test_nivel_invalido_da_value_error_sin_repetir_el_valor(monkeypatch, valor):
    monkeypatch.setenv("LOG_LEVEL", valor)
    antes = (list(orquestador().handlers), orquestador().level, orquestador().propagate)
    with pytest.raises(ValueError) as exc:
        configurar_logging()
    # Mensaje fijo: no repite el valor recibido.
    assert str(exc.value) == "LOG_LEVEL no es un nivel válido (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
    # Se valida antes de tocar nada: el logging no queda a medio configurar.
    assert (list(orquestador().handlers), orquestador().level, orquestador().propagate) == antes


def test_nivel_invalido_impide_arrancar_el_servicio(monkeypatch):
    # configurar_logging() es lo primero del lifespan: falla antes de leer providers.yaml.
    monkeypatch.setenv("LOG_LEVEL", "MUCHO")
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        with TestClient(app):
            pass


# ---------- Handler ----------

def test_idempotente_un_solo_handler():
    configurar_logging()
    configurar_logging()
    assert len(handlers_propios()) == 1


def test_propagate_false_y_handler_a_stderr():
    configurar_logging()
    assert orquestador().propagate is False
    (handler,) = handlers_propios()
    assert type(handler) is logging.StreamHandler
    assert handler.stream is sys.stderr
    assert handler.formatter._fmt == LOG_FORMAT == "%(asctime)s %(levelname)s %(name)s: %(message)s"


def test_info_sale_con_timestamp_nivel_y_logger():
    configurar_logging()
    (handler,) = handlers_propios()
    salida = io.StringIO()
    handler.setStream(salida)

    logging.getLogger("orchestrator").info("evento %d", 7)
    logging.getLogger("orchestrator.memory").warning("otro evento")
    logging.getLogger("orchestrator").debug("no sale con INFO")

    lineas = salida.getvalue().splitlines()
    ts = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"
    assert len(lineas) == 2
    assert re.fullmatch(ts + r" INFO orchestrator: evento 7", lineas[0])
    assert re.fullmatch(ts + r" WARNING orchestrator\.memory: otro evento", lineas[1])


def test_no_sale_duplicado_si_alguien_configura_el_root():
    configurar_logging()
    (handler,) = handlers_propios()
    handler.setStream(io.StringIO())
    del_root = io.StringIO()
    root_handler = logging.StreamHandler(del_root)
    root = logging.getLogger()
    nivel_root = root.level
    root.addHandler(root_handler)
    root.setLevel(logging.DEBUG)
    try:
        logging.getLogger("orchestrator").info("una sola vez")
    finally:
        root.removeHandler(root_handler)
        root.setLevel(nivel_root)
    assert del_root.getvalue() == ""
    assert handler.stream.getvalue().count("una sola vez") == 1


# ---------- Seguridad: httpx2 loguea URLs internas en INFO ----------

@pytest.mark.parametrize("valor", ["DEBUG", "INFO", "debug"])
def test_httpx2_y_httpcore2_siguen_en_warning_con_log_level_bajo(monkeypatch, valor):
    monkeypatch.setenv("LOG_LEVEL", valor)
    configurar_logging()
    for nombre in LOGGERS_SILENCIADOS:
        assert logging.getLogger(nombre).level == logging.WARNING
        assert not logging.getLogger(nombre).isEnabledFor(logging.INFO)
    # Los hijos de httpcore2 heredan el nivel del padre.
    assert not logging.getLogger("httpcore2.connection").isEnabledFor(logging.INFO)
    assert orquestador().level <= logging.INFO
