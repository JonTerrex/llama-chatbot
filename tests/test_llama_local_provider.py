"""
Verifica el proveedor Llama local contra el contrato (docs/contrato-proveedores.md) con un
motor falso: sin torch, sin GPU y sin MagicMock en sys.modules.
"""
import logging
import threading
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

import src.providers.llama_local.app as prov
from src.providers.llama_local.app import app
from src.providers.llama_local.engine import ContextoExcedido, MotorLlama, Resultado, VRAMInsuficiente

URL = "/v1/chat/completions"
MODELO = "llama3-8b-local"
MENSAJES = [
    {"role": "system", "content": "Sos el system del orquestador."},
    {"role": "user", "content": "Hola"},
]
SECRETO = "TEXTO-SECRETO-DEL-PROMPT-42"


def resultado(**cambios) -> Resultado:
    base = dict(
        texto="¡Hola! ¿En qué te ayudo?", prompt_tokens=30, completion_tokens=8,
        finish_reason="stop", generation_s=0.5, vram_peak_mb=5900.5, vram_reserved_mb=6050.0,
    )
    base.update(cambios)
    return Resultado(**base)


class FakeMotor:
    """
    Motor falso configurable. `puede_cargar` bloquea cargar() hasta que el test lo libere;
    `error_carga` / `error_generar` hacen fallar; registra las llamadas y la concurrencia.
    """

    def __init__(self):
        self.puede_cargar = threading.Event()
        self.puede_cargar.set()
        self.error_carga: Exception | None = None
        self.error_generar: Exception | None = None
        self.resultado = resultado()
        self.demora_generar = 0.0
        self.llamadas: list[tuple] = []
        self.liberado = False
        self._guardia = threading.Lock()
        self.en_curso = 0
        self.max_en_curso = 0

    def cargar(self):
        assert self.puede_cargar.wait(timeout=5), "el test nunca liberó la carga"
        if self.error_carga is not None:
            raise self.error_carga

    def generar(self, mensajes, temperature, max_tokens):
        with self._guardia:
            self.en_curso += 1
            self.max_en_curso = max(self.max_en_curso, self.en_curso)
            self.llamadas.append((mensajes, temperature, max_tokens))
        try:
            if self.demora_generar:
                time.sleep(self.demora_generar)
            if self.error_generar is not None:
                raise self.error_generar
            return self.resultado
        finally:
            with self._guardia:
                self.en_curso -= 1

    def liberar(self):
        self.liberado = True


@pytest.fixture(autouse=True)
def logger_del_proveedor(caplog, monkeypatch):
    """
    El lifespan llama a configurar_logging(), que deja "llama_local" con propagate=False:
    sus registros ya no llegan al root, donde caplog pone su handler. Por eso se agrega el
    handler de caplog directamente a este logger (lo que pytest documenta para loggers que
    no propagan). Al final se restauran handlers, nivel y propagate, porque el logger es
    global al proceso.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    log = logging.getLogger("llama_local")
    handlers, nivel, propaga = list(log.handlers), log.level, log.propagate
    log.addHandler(caplog.handler)
    yield log
    for h in list(log.handlers):
        if h not in handlers:
            log.removeHandler(h)
            if h is not caplog.handler:
                h.close()
    log.setLevel(nivel)
    log.propagate = propaga


@pytest.fixture
def motor(monkeypatch):
    falso = FakeMotor()
    monkeypatch.setattr(prov, "crear_motor", lambda: falso)
    yield falso
    # Liberar y esperar el hilo de carga: si terminara tarde, escribiría el estado de la app
    # (global) en medio del test siguiente.
    falso.puede_cargar.set()
    hilo = getattr(app.state, "hilo_carga", None)
    if hilo is not None:
        hilo.join(timeout=5)


@contextmanager
def cliente(esperar_carga=True):
    """TestClient con lifespan; por defecto espera (join) a que el hilo de carga termine."""
    with TestClient(app) as c:
        if esperar_carga:
            c.app.state.hilo_carga.join(timeout=5)
            assert not c.app.state.hilo_carga.is_alive()
        yield c


def pedir(c, **campos):
    body = {"model": MODELO, "messages": MENSAJES}
    body.update(campos)
    return c.post(URL, json=body)


def assert_error(r, status, code, param, type_):
    assert r.status_code == status
    error = r.json()["error"]
    assert {"message", "type", "param", "code"} <= set(error)
    assert (error["code"], error["param"], error["type"]) == (code, param, type_)
    return error


# ---------- Salud ----------

def test_health_pasa_de_loading_a_ready(motor):
    motor.puede_cargar.clear()
    with cliente(esperar_carga=False) as c:
        r = c.get("/health")
        assert (r.status_code, r.json()) == (503, {"status": "loading"})

        motor.puede_cargar.set()
        c.app.state.hilo_carga.join(timeout=5)

        r = c.get("/health")
        assert (r.status_code, r.json()) == (200, {"status": "ready"})
    assert motor.liberado  # al apagar con el modelo listo, se libera


def test_health_failed_si_la_carga_lanza(motor, caplog):
    motor.error_carga = RuntimeError("sin VRAM")
    with caplog.at_level(logging.ERROR, logger="llama_local"):
        with cliente() as c:
            r = c.get("/health")
    assert (r.status_code, r.json()) == (503, {"status": "failed"})
    assert any(rec.exc_info for rec in caplog.records if rec.name == "llama_local")  # con traceback
    assert not motor.liberado


@pytest.mark.parametrize("estado", ["loading", "failed"])
def test_post_sin_modelo_listo_da_503_sin_generar(motor, estado):
    if estado == "loading":
        motor.puede_cargar.clear()
    else:
        motor.error_carga = RuntimeError("falló")
    with cliente(esperar_carga=(estado == "failed")) as c:
        r = pedir(c)
    error = assert_error(r, 503, "provider_unavailable", None, "provider_error")
    assert error["x_provider"] == {"name": MODELO, "state": estado}
    assert motor.llamadas == []


# ---------- Camino feliz ----------

def test_forma_completa_de_la_respuesta(motor):
    with cliente() as c:
        r = pedir(c)
    assert r.status_code == 200
    body = r.json()
    assert body["id"].startswith("chatcmpl-") and len(body["id"]) == len("chatcmpl-") + 32
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == MODELO
    assert body["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "¡Hola! ¿En qué te ayudo?"}, "finish_reason": "stop"}
    ]
    assert body["usage"] == {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38}
    metrics = body["x_metrics"]
    assert set(metrics) == {"latency_s", "tokens_per_s", "vram_peak_mb", "vram_reserved_mb"}
    assert metrics["tokens_per_s"] == 16.0  # 8 tokens / 0.5 s de generación
    assert (metrics["vram_peak_mb"], metrics["vram_reserved_mb"]) == (5900.5, 6050.0)
    assert isinstance(metrics["latency_s"], (int, float))
    assert "x_debug" not in body


def test_finish_reason_length_se_propaga(motor):
    motor.resultado = resultado(finish_reason="length")
    with cliente() as c:
        assert pedir(c).json()["choices"][0]["finish_reason"] == "length"


def test_sin_vram_no_hay_claves_vram(motor):
    motor.resultado = resultado(vram_peak_mb=None, vram_reserved_mb=None)
    with cliente() as c:
        metrics = pedir(c).json()["x_metrics"]
    assert not any(k.startswith("vram_") for k in metrics)


@pytest.mark.parametrize("generation_s", [0.0, 0.0004], ids=["cero", "redondea_a_cero"])
def test_tokens_per_s_se_omite_sin_tiempo_de_generacion(motor, generation_s):
    motor.resultado = resultado(generation_s=generation_s)
    with cliente() as c:
        metrics = pedir(c).json()["x_metrics"]
    assert "tokens_per_s" not in metrics
    assert "latency_s" in metrics


def test_latency_s_mide_el_tiempo_dentro_del_lock(motor):
    motor.demora_generar = 0.1
    with cliente() as c:
        latency = pedir(c).json()["x_metrics"]["latency_s"]
    assert 0.1 <= latency < 1.0


def test_temperature_y_max_tokens_llegan_al_motor(motor):
    with cliente() as c:
        pedir(c, temperature=0, max_tokens=12)
        pedir(c)
    (_, t1, m1), (_, t2, m2) = motor.llamadas
    assert (t1, m1) == (0, 12)
    assert (t2, m2) == (None, None)


def test_el_motor_recibe_los_mensajes_tal_cual(motor):
    solo_user = [{"role": "user", "content": "Hola"}]
    with cliente() as c:
        pedir(c)
        pedir(c, messages=solo_user)
    assert motor.llamadas[0][0] == MENSAJES
    assert motor.llamadas[1][0] == solo_user  # sin system agregado por el proveedor


def test_campos_extra_se_ignoran(motor):
    with cliente() as c:
        r = pedir(c, top_p=0.1, n=3)
    assert r.status_code == 200
    assert len(motor.llamadas) == 1


# ---------- Validación y orden ----------

@pytest.mark.parametrize(
    "campos, param",
    [
        ({"temperature": -1}, "temperature"),
        ({"temperature": 2.5}, "temperature"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "messages"),
    ],
    ids=["temperature_negativa", "temperature_2_5", "max_tokens_0", "messages_vacio", "role_invalido"],
)
def test_peticion_invalida_da_400(motor, campos, param):
    with cliente() as c:
        r = pedir(c, **campos)
    assert_error(r, 400, "invalid_request", param, "invalid_request_error")
    assert motor.llamadas == []


def test_falta_model_da_400(motor):
    with cliente() as c:
        r = c.post(URL, json={"messages": MENSAJES})
    assert_error(r, 400, "invalid_request", "model", "invalid_request_error")


def test_stream_da_501(motor):
    with cliente() as c:
        r = pedir(c, stream=True)
    assert_error(r, 501, "streaming_not_supported", "stream", "provider_error")
    assert motor.llamadas == []


def test_stream_va_antes_que_el_estado(motor):
    # Con el modelo cargando, stream:true sigue siendo 501 y no 503.
    motor.puede_cargar.clear()
    with cliente(esperar_carga=False) as c:
        r = pedir(c, stream=True)
    assert_error(r, 501, "streaming_not_supported", "stream", "provider_error")


# ---------- Errores de generación ----------

def test_contexto_excedido_da_400_sin_contenido(motor):
    motor.error_generar = ContextoExcedido(8100, 8192, 256)
    with cliente() as c:
        r = pedir(c, messages=[{"role": "user", "content": SECRETO}])
    error = assert_error(r, 400, "context_length_exceeded", "messages", "invalid_request_error")
    for numero in ("8100", "256", "8192"):
        assert numero in error["message"]
    assert SECRETO not in r.text


def test_vram_insuficiente_da_503_con_los_numeros_y_sin_contenido(motor, caplog):
    motor.error_generar = VRAMInsuficiente(12562, 3140.5, 1200.0)
    with caplog.at_level(logging.INFO):
        with cliente() as c:
            r = pedir(c, messages=[{"role": "user", "content": SECRETO}])
    error = assert_error(r, 503, "insufficient_vram", None, "provider_error")
    # Solo los números: nada que prometa que es transitorio (bajo WSL2 puede no serlo).
    assert error["message"] == (
        "VRAM insuficiente: el pedido de 12562 tokens (estimado 3140 MiB) no entró en la GPU; "
        "presupuesto de arranque 1200 MiB."
    )
    assert "transitori" not in error["message"].lower() and "momento" not in error["message"].lower()
    assert "x_provider" not in error
    assert SECRETO not in r.text
    assert SECRETO not in caplog.text
    propios = [rec for rec in caplog.records if rec.name == "llama_local" and "VRAM insuficiente" in rec.getMessage()]
    assert len(propios) == 1
    assert propios[0].levelno == logging.INFO and not propios[0].exc_info
    # El proveedor sigue atendiendo: es transitorio.
    assert motor.llamadas and len(motor.llamadas) == 1


def test_oom_convertido_da_503_sin_texto_de_torch(motor, caplog):
    # Lo que lanza el motor real ante un OOM: VRAMInsuficiente encadenada a la excepción de torch.
    torch_texto = f"CUDA out of memory. Tried to allocate 64.00 MiB {SECRETO}"
    try:
        raise RuntimeError(torch_texto)
    except RuntimeError as original:
        convertida = VRAMInsuficiente(2256, 564.0, 380.5)
        convertida.__cause__ = original
    motor.error_generar = convertida
    with caplog.at_level(logging.INFO):
        with cliente() as c:
            r = pedir(c, messages=[{"role": "user", "content": SECRETO}])
    error = assert_error(r, 503, "insufficient_vram", None, "provider_error")
    assert error["message"] == (
        "VRAM insuficiente: el pedido de 2256 tokens (estimado 564 MiB) no entró en la GPU; "
        "presupuesto de arranque 380 MiB."
    )
    for texto in (SECRETO, "Tried to allocate", "CUDA out of memory"):
        assert texto not in r.text
        assert texto not in caplog.text


def test_excepcion_generica_da_500_sin_su_texto(motor, caplog):
    motor.error_generar = RuntimeError(f"CUDA explotó con el prompt {SECRETO}")
    with caplog.at_level(logging.INFO):
        with cliente() as c:
            r = pedir(c, messages=[{"role": "user", "content": SECRETO}])
    error = assert_error(r, 500, "inference_error", None, "provider_error")
    assert SECRETO not in r.text
    assert "CUDA" not in error["message"]
    assert SECRETO not in caplog.text
    assert any("RuntimeError" in rec.getMessage() for rec in caplog.records if rec.name == "llama_local")


def test_el_traceback_de_generacion_solo_va_a_debug(motor, caplog, monkeypatch):
    # El lifespan fija el nivel desde LOG_LEVEL (pisa el at_level): DEBUG se habilita así.
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    motor.error_generar = RuntimeError(SECRETO)
    with caplog.at_level(logging.DEBUG, logger="llama_local"):
        with cliente() as c:
            pedir(c)
    con_traceback = [rec for rec in caplog.records if rec.name == "llama_local" and rec.exc_info]
    assert con_traceback and all(rec.levelno == logging.DEBUG for rec in con_traceback)


# ---------- Serialización ----------

def test_dos_pedidos_simultaneos_no_generan_a_la_vez(motor):
    # Dos hilos contra el mismo TestClient: el endpoint es síncrono y FastAPI lo corre en su
    # pool de hilos, así que sin el lock las dos generaciones se solaparían.
    motor.demora_generar = 0.3
    with cliente() as c:
        respuestas = []
        hilos = [threading.Thread(target=lambda: respuestas.append(pedir(c))) for _ in range(2)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(timeout=10)
    assert [r.status_code for r in respuestas] == [200, 200]
    assert len(motor.llamadas) == 2
    assert motor.max_en_curso == 1


# ---------- Configuración ----------

def test_crear_motor_lee_el_entorno(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/modelos/otro")
    monkeypatch.setenv("TOP_P", "0.5")
    monkeypatch.setenv("CONTEXTO_MAX_TOKENS", "2048")
    m = prov.crear_motor()
    assert isinstance(m, MotorLlama)
    assert (m.path_modelo, m.top_p, m.contexto_max_config) == ("/modelos/otro", 0.5, 2048)


def test_crear_motor_defaults(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("TOP_P", raising=False)
    monkeypatch.delenv("CONTEXTO_MAX_TOKENS", raising=False)
    m = prov.crear_motor()
    assert (m.path_modelo, m.top_p, m.contexto_max_config) == ("modelos/llama3", 0.9, None)


@pytest.mark.parametrize("valor, esperado", [(None, None), ("", None), ("2048", 2048)], ids=["ausente", "vacio", "2048"])
def test_contexto_max_tokens_valido(monkeypatch, valor, esperado):
    if valor is None:
        monkeypatch.delenv("CONTEXTO_MAX_TOKENS", raising=False)
    else:
        monkeypatch.setenv("CONTEXTO_MAX_TOKENS", valor)
    assert prov.crear_motor().contexto_max_config == esperado


@pytest.mark.parametrize("valor", ["0", "-5", "abc", "1.5"])
def test_contexto_max_tokens_invalido_impide_arrancar(monkeypatch, valor):
    monkeypatch.setenv("CONTEXTO_MAX_TOKENS", valor)
    with pytest.raises(ValueError, match=f"CONTEXTO_MAX_TOKENS='{valor}'"):
        prov.crear_motor()
    with pytest.raises(ValueError, match="CONTEXTO_MAX_TOKENS"):
        with TestClient(app):
            pass


def test_al_apagar_se_loguea_antes_de_liberar(motor, caplog):
    with caplog.at_level(logging.INFO, logger="llama_local"):
        with cliente():
            pass
    mensajes = [rec.getMessage() for rec in caplog.records if rec.name == "llama_local"]
    assert "Apagando el proveedor." in mensajes
    assert motor.liberado


@pytest.mark.parametrize("valor", ["0", "1.5", "-0.1", "mucho"])
def test_top_p_invalido_impide_arrancar(monkeypatch, valor):
    monkeypatch.setenv("TOP_P", valor)
    with pytest.raises(ValueError, match="TOP_P"):
        with TestClient(app):
            pass


# ---------- Logging del proveedor ----------

def handlers_propios(log):
    return [h for h in log.handlers if h.get_name() == prov.NOMBRE_HANDLER]


def test_el_handler_se_instala_una_sola_vez(motor, logger_del_proveedor):
    with cliente():
        pass
    with cliente():
        pass
    (handler,) = handlers_propios(logger_del_proveedor)
    assert isinstance(handler, logging.StreamHandler)
    assert handler.formatter._fmt == "%(asctime)s %(levelname)s %(name)s %(message)s"
    assert logger_del_proveedor.propagate is False


@pytest.mark.parametrize("valor, nivel", [(None, logging.INFO), ("warning", logging.WARNING), ("DEBUG", logging.DEBUG)])
def test_log_level_se_aplica(motor, logger_del_proveedor, monkeypatch, valor, nivel):
    if valor is not None:
        monkeypatch.setenv("LOG_LEVEL", valor)
    with cliente():
        assert logger_del_proveedor.level == nivel


@pytest.mark.parametrize("valor", ["MUCHO", "5x"])
def test_log_level_invalido_impide_arrancar(motor, logger_del_proveedor, monkeypatch, valor):
    monkeypatch.setenv("LOG_LEVEL", valor)
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        with TestClient(app):
            pass
