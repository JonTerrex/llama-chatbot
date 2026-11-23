"""Reglas puras del motor del proveedor Llama local: se prueban sin torch ni GPU."""
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest

import src.providers.llama_local.engine as engine
from src.providers.llama_local.engine import (
    MARGEN_VRAM_MB,
    RESERVA_PEDIDO_MB,
    ContextoExcedido,
    MotorLlama,
    SinGPU,
    VRAMInsuficiente,
    bytes_por_token,
    contexto_efectivo,
    contexto_por_vram,
    decidir_finish_reason,
    fraccion_vram,
    presupuesto_vram_b,
    resolver_parametros,
    validar_contexto,
)

GIB = 1024 ** 3
MIB = 1024 ** 2
# Llama 3 8B: 32 capas, 8 cabezas de KV, head_dim 128, bf16.
KV_LLAMA3_8B_SIN_FACTOR = 131072
COSTO_LLAMA3_8B = 2 * KV_LLAMA3_8B_SIN_FACTOR  # con FACTOR_OVERHEAD_KV = 2: 0,25 MiB por token

EOS = {128001, 128009}


# ---------- resolver_parametros ----------

def test_sin_parametros_usa_los_defaults():
    assert resolver_parametros(None, None) == (True, 0.6, 256)


def test_temperature_cero_es_greedy():
    do_sample, temperature, max_new = resolver_parametros(0, 64)
    assert do_sample is False
    assert temperature == 0.0
    assert max_new == 64


def test_temperature_y_max_tokens_de_la_peticion():
    assert resolver_parametros(1.2, 10) == (True, 1.2, 10)


def test_temperature_none_con_max_tokens():
    assert resolver_parametros(None, 1) == (True, 0.6, 1)


def test_max_tokens_none_con_temperature():
    assert resolver_parametros(0.3, None) == (True, 0.3, 256)


# ---------- validar_contexto ----------

def test_exactamente_en_el_limite_pasa():
    validar_contexto(prompt_tokens=8000, max_new_tokens=192, contexto_max=8192)


def test_uno_por_encima_del_limite_falla():
    with pytest.raises(ContextoExcedido) as exc:
        validar_contexto(prompt_tokens=8001, max_new_tokens=192, contexto_max=8192)
    assert (exc.value.prompt_tokens, exc.value.max_new_tokens, exc.value.limite) == (8001, 192, 8192)


def test_contexto_excedido_solo_lleva_numeros():
    assert str(ContextoExcedido(10, 8, 2)) == "prompt_tokens=10, max_new_tokens=2, limite=8"


# ---------- decidir_finish_reason ----------

def test_tope_alcanzado_sin_token_de_fin_es_length():
    assert decidir_finish_reason(42, EOS, completion_tokens=256, max_new_tokens=256) == "length"


def test_token_de_fin_justo_en_el_tope_es_stop():
    assert decidir_finish_reason(128009, EOS, completion_tokens=256, max_new_tokens=256) == "stop"


def test_termina_antes_del_tope_es_stop():
    assert decidir_finish_reason(128009, EOS, completion_tokens=10, max_new_tokens=256) == "stop"


def test_bajo_el_tope_sin_token_de_fin_es_stop():
    # Puede pasar si la secuencia se cortó por otro motivo: no es "length" porque no llegó al tope.
    assert decidir_finish_reason(42, EOS, completion_tokens=10, max_new_tokens=256) == "stop"


def test_sin_tokens_generados_es_stop():
    assert decidir_finish_reason(None, EOS, completion_tokens=0, max_new_tokens=256) == "stop"


def test_eos_como_lista():
    assert decidir_finish_reason(128001, [128001], completion_tokens=5, max_new_tokens=5) == "stop"


# ---------- cargar(): chequeos previos, con un torch falso ----------

def torch_falso(hay_cuda: bool, libre_gb: float = 0.0, total_gb: float = 0.0):
    """Solo lo que cargar() toca antes de importar transformers. Sin MagicMock: un atributo de más falla."""
    cuda = types.SimpleNamespace(is_available=lambda: hay_cuda)
    if hay_cuda:
        cuda.mem_get_info = lambda device=0: (int(libre_gb * GIB), int(total_gb * GIB))
    return types.SimpleNamespace(cuda=cuda)


def test_sin_cuda_lanza_sin_gpu_antes_de_importar_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", torch_falso(hay_cuda=False))
    # None en sys.modules: cualquier "import transformers" lanza ImportError.
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(SinGPU):
        MotorLlama("modelos/llama3", 0.9).cargar()


def test_vram_insuficiente_lanza_con_los_numeros(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", torch_falso(hay_cuda=True, libre_gb=4.0, total_gb=8.0))
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError) as exc:
        MotorLlama("modelos/llama3", 0.9).cargar()
    assert not isinstance(exc.value, SinGPU)
    mensaje = str(exc.value)
    for numero in ("4.00", "8.00", "5.0"):
        assert numero in mensaje


# ---------- Funciones puras de VRAM y contexto ----------

def test_bytes_por_token_llama3_8b():
    assert bytes_por_token(32, 8, 128, factor=1) == KV_LLAMA3_8B_SIN_FACTOR
    assert bytes_por_token(32, 8, 128) == COSTO_LLAMA3_8B  # factor por defecto 2
    assert bytes_por_token(32, 8, 128, bytes_dtype=4, factor=1) == 2 * KV_LLAMA3_8B_SIN_FACTOR


def test_contexto_por_vram_presupuesto_justo():
    assert contexto_por_vram(1000 * COSTO_LLAMA3_8B, COSTO_LLAMA3_8B, 8192) == 1000
    assert contexto_por_vram(1000 * COSTO_LLAMA3_8B - 1, COSTO_LLAMA3_8B, 8192) == 999


def test_contexto_por_vram_sobrado_topea_en_el_modelo():
    assert contexto_por_vram(100 * GIB, COSTO_LLAMA3_8B, 8192) == 8192


def test_contexto_por_vram_menos_de_un_token_falla():
    with pytest.raises(ValueError, match="no alcanza"):
        contexto_por_vram(COSTO_LLAMA3_8B - 1, COSTO_LLAMA3_8B, 8192)
    with pytest.raises(ValueError):
        contexto_por_vram(-5 * MIB, COSTO_LLAMA3_8B, 8192)


@pytest.mark.parametrize(
    "modelo, por_vram, configurado, esperado",
    [
        (8192, 4797, None, 4797),
        (8192, 4797, 1024, 1024),
        (8192, 4797, 99999, 4797),  # un configurado mayor no amplía
        (8192, None, None, 8192),
        (8192, None, 2048, 2048),
        (None, 4797, None, 4797),
        (None, None, 300, 300),
        (2048, 4797, 3000, 2048),
    ],
)
def test_contexto_efectivo_es_el_menor_conocido(modelo, por_vram, configurado, esperado):
    assert contexto_efectivo(modelo, por_vram, configurado) == esperado


def test_contexto_efectivo_sin_ningun_tope_falla():
    with pytest.raises(ValueError):
        contexto_efectivo(None, None, None)


def test_fraccion_vram_normal():
    assert fraccion_vram(6 * GIB, 8 * GIB, 256 * MIB) == pytest.approx((6 * GIB - 256 * MIB) / (8 * GIB))


@pytest.mark.parametrize("libre_mib", [256, 100], ids=["margen_igual_a_lo_libre", "margen_mayor"])
def test_fraccion_vram_sin_lugar_falla(libre_mib):
    with pytest.raises(ValueError, match="Sin VRAM utilizable"):
        fraccion_vram(libre_mib * MIB, 8 * GIB, 256 * MIB)


def test_presupuesto_vram_sin_cache():
    assert presupuesto_vram_b(1000 * MIB, 0, 256 * MIB, 0) == 744 * MIB


def test_presupuesto_vram_suma_la_cache():
    assert presupuesto_vram_b(500 * MIB, 300 * MIB, 256 * MIB, 0) == 544 * MIB


def test_presupuesto_vram_descuenta_la_reserva():
    assert presupuesto_vram_b(1000 * MIB, 0, 256 * MIB, 150 * MIB) == 594 * MIB
    # Reserva 0 equivale al resultado anterior (solo margen).
    assert presupuesto_vram_b(1000 * MIB, 0, 256 * MIB, 0) == 1000 * MIB - 256 * MIB


def test_presupuesto_vram_reserva_mayor_que_lo_disponible_da_negativo():
    assert presupuesto_vram_b(300 * MIB, 50 * MIB, 256 * MIB, 150 * MIB) == -56 * MIB


def test_presupuesto_vram_negativo_y_contexto_por_vram_lo_rechaza():
    presupuesto = presupuesto_vram_b(100 * MIB, 50 * MIB, 256 * MIB, 0)
    assert presupuesto == -106 * MIB
    with pytest.raises(ValueError, match="no alcanza"):
        contexto_por_vram(presupuesto, COSTO_LLAMA3_8B, 8192)


# ---------- cargar() completo, con torch y transformers falsos ----------

def config_llama3(**cambios):
    base = dict(num_hidden_layers=32, num_key_value_heads=8, head_dim=128, hidden_size=4096,
                num_attention_heads=32, max_position_embeddings=8192)
    base.update(cambios)
    return types.SimpleNamespace(**base)


# Texto que "genera" el modelo falso: no debe aparecer en ningún log.
SALIDA_CALENTAMIENTO = "SALIDA-GENERADA-XYZ"


class Secuencia(list):
    """Lo mínimo de un tensor 1D que usa _generar: shape, slicing y elementos enteros."""

    @property
    def shape(self):
        return (len(self),)

    def __getitem__(self, k):
        r = list.__getitem__(self, k)
        return Secuencia(r) if isinstance(k, slice) else r


class TensorFalso:
    def __init__(self, n, orden):
        self.shape, self._orden = (1, n), orden

    def to(self, device):
        self._orden.append(("to_gpu",))
        return self


class GPUFalsa:
    """
    torch y transformers falsos que registran en `orden` cada llamada relevante, en secuencia.
    `despues` ({libre_b, asignada_b, reservada_b}) se aplica cuando corre model.generate: simula la
    VRAM que ocupa el calentamiento. `durante` se aplica al terminar from_pretrained.
    """

    def __init__(self, libre_b, total_b, asignada_b, reservada_b=None, config=None,
                 despues=None, durante=None, error_generate=None):
        self.orden: list[tuple] = []
        self.libre_b, self.total_b = libre_b, total_b
        self.asignada_b = asignada_b
        self.reservada_b = asignada_b if reservada_b is None else reservada_b
        self.config = config or config_llama3()
        self.despues, self.durante, self.error_generate = despues, durante, error_generate
        self.motor = None  # si se asigna, generate registra su contexto_max en ese momento
        self.kwargs_generate: list[dict] = []
        # sys.exc_info() en el momento de cada empty_cache: con una excepción viva no libera nada.
        self.exc_info_en_empty_cache: list[tuple] = []

    def _empty_cache(self):
        self.orden.append(("empty_cache",))
        self.exc_info_en_empty_cache.append(sys.exc_info())

    def _aplicar(self, cambios):
        for k, v in (cambios or {}).items():
            setattr(self, k, v)

    def mem_get_info(self, device=0):
        self.orden.append(("mem_get_info", self.libre_b))
        return self.libre_b, self.total_b

    def torch(self):
        o = self.orden
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=self.mem_get_info,
            get_device_name=lambda device=0: "GPU falsa",
            set_per_process_memory_fraction=lambda f, device=0: o.append(("set_fraction", f, device)),
            memory_allocated=lambda device=None: self.asignada_b,
            memory_reserved=lambda device=None: self.reservada_b,
            max_memory_allocated=lambda device=None: self.asignada_b,
            empty_cache=self._empty_cache,
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
        )
        return types.SimpleNamespace(cuda=cuda, bfloat16="bf16", __version__="falso", version=types.SimpleNamespace(cuda="falso"))

    def generate(self, input_ids, max_new_tokens, **kwargs):
        self.orden.append(("generate", None if self.motor is None else self.motor.contexto_max))
        self.kwargs_generate.append(dict(kwargs, max_new_tokens=max_new_tokens))
        self._aplicar(self.despues)
        if self.error_generate is not None:
            raise self.error_generate
        n = input_ids.shape[-1]
        return [Secuencia([7] * n + [42] * max_new_tokens)]

    def transformers(self):
        gpu, o, config = self, self.orden, self.config

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path):
                o.append(("tokenizer",))
                return types.SimpleNamespace(
                    eos_token_id=128009, pad_token_id=None, decode=lambda ids, **k: SALIDA_CALENTAMIENTO,
                    apply_chat_template=lambda *a, **k: {"input_ids": TensorFalso(5, o)},
                )

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(path, **kwargs):
                o.append(("modelo", kwargs["device_map"]))
                gpu._aplicar(gpu.durante)
                return types.SimpleNamespace(
                    config=config, generation_config=types.SimpleNamespace(eos_token_id=[128001, 128009]),
                    device=types.SimpleNamespace(type="cuda"), generate=gpu.generate,
                )

        return types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModelForCausalLM, BitsAndBytesConfig=lambda **kw: kw
        )

    def instalar(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self.torch())
        monkeypatch.setitem(sys.modules, "transformers", self.transformers())


# Antes del calentamiento: mucha VRAM libre (el tope saldría 8192, el del modelo). Después: el
# calentamiento ocupó VRAM fuera del allocator, así que el tope correcto es mucho menor.
LIBRE_B, TOTAL_B, ASIGNADA_B = int(6.89 * GIB), 8 * GIB, 5600 * MIB
DESPUES = {"libre_b": 900 * MIB, "asignada_b": 5700 * MIB, "reservada_b": 5800 * MIB}
PRESUPUESTO_DESPUES_B = (900 + (5800 - 5700) - MARGEN_VRAM_MB - RESERVA_PEDIDO_MB) * MIB  # 594 MiB
CONTEXTO_DESPUES = int(PRESUPUESTO_DESPUES_B // COSTO_LLAMA3_8B)                         # 2376 tokens
FRACCION_FINAL = (5800 + 900 - MARGEN_VRAM_MB) * MIB / TOTAL_B


def gpu_con_calentamiento(**cambios):
    return GPUFalsa(LIBRE_B, TOTAL_B, ASIGNADA_B, despues=DESPUES, **cambios)


def test_cargar_calcula_el_tope_con_la_vram_posterior_al_calentamiento(monkeypatch):
    gpu = gpu_con_calentamiento()
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    motor.cargar()
    assert motor.contexto_max == CONTEXTO_DESPUES == 2376  # con reserva; sin ella serían 2976
    assert motor._presupuesto_arranque_b == PRESUPUESTO_DESPUES_B  # lo informa VRAMInsuficiente
    assert motor._costo_por_token_b == COSTO_LLAMA3_8B
    assert motor._eos_ids == {128001, 128009}


def test_cargar_calienta_antes_de_fijar_el_tope(monkeypatch):
    gpu = gpu_con_calentamiento()
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    gpu.motor = motor
    motor.cargar()
    nombres = [p[0] for p in gpu.orden]
    (generate,) = [p for p in gpu.orden if p[0] == "generate"]
    assert generate == ("generate", None)  # durante el calentamiento el tope todavía no existe
    # Después del generate se vuelve a medir la VRAM, ya con los valores posteriores.
    i_gen = nombres.index("generate")
    assert ("mem_get_info", DESPUES["libre_b"]) in gpu.orden[i_gen:]
    # Calentamiento fijo: greedy y 8 tokens.
    (kw,) = gpu.kwargs_generate
    assert kw["max_new_tokens"] == 8 and kw["do_sample"] is False


def test_cargar_fija_el_limite_dos_veces(monkeypatch):
    gpu = gpu_con_calentamiento()
    gpu.instalar(monkeypatch)
    MotorLlama("modelos/llama3", 0.9).cargar()
    nombres = [p[0] for p in gpu.orden]
    fracciones = [p for p in gpu.orden if p[0] == "set_fraction"]
    assert len(fracciones) == 2, gpu.orden
    i_primera, i_segunda = [i for i, n in enumerate(nombres) if n == "set_fraction"]
    assert i_primera < nombres.index("modelo")          # protege la carga
    assert i_segunda > nombres.index("generate")         # valor preciso, tras el calentamiento
    assert fracciones[0][1] == pytest.approx((LIBRE_B - MARGEN_VRAM_MB * MIB) / TOTAL_B)
    assert fracciones[1][1] == pytest.approx(FRACCION_FINAL)
    assert all(device == 0 for _, _, device in fracciones)


def test_si_el_calentamiento_falla_cargar_lanza_y_no_fija_el_tope(monkeypatch):
    gpu = gpu_con_calentamiento(error_generate=RuntimeError("falla de kernel"))
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    with pytest.raises(RuntimeError, match="falla de kernel"):
        motor.cargar()
    assert motor.contexto_max is None and motor._model is None
    assert len([p for p in gpu.orden if p[0] == "set_fraction"]) == 1  # no llegó a la segunda


def test_el_calentamiento_sin_vram_libre_llega_a_generar_y_no_loguea_contenido(monkeypatch, caplog):
    # Tras cargar el modelo casi no queda VRAM libre: el calentamiento igual llega a generate (no
    # hay verificación previa). Después del generate, la VRAM vuelve a alcanzar para el tope.
    gpu = gpu_con_calentamiento(durante={"libre_b": 0, "reservada_b": ASIGNADA_B})
    gpu.instalar(monkeypatch)
    log = logging.getLogger("llama_local")
    propaga, log.propagate = log.propagate, False
    log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="llama_local"):
            motor = MotorLlama("modelos/llama3", 0.9)
            motor.cargar()
    finally:
        log.removeHandler(caplog.handler)
        log.propagate = propaga
    assert [p[0] for p in gpu.orden].count("generate") == 1
    assert motor.contexto_max == CONTEXTO_DESPUES
    mensajes = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Calentamiento: ") for m in mensajes)
    assert not any("Hola" in m or SALIDA_CALENTAMIENTO in m for m in mensajes)  # ni el mensaje ni lo generado


def test_cargar_sin_head_dim_lo_deriva_de_hidden_size(monkeypatch):
    config = config_llama3()
    del config.head_dim
    gpu = gpu_con_calentamiento(config=config)
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    motor.cargar()
    assert motor._costo_por_token_b == COSTO_LLAMA3_8B  # 4096 // 32 = 128


@pytest.mark.parametrize("configurado, gana_configurado", [(1024, True), (99999, False)])
def test_cargar_con_override_manda_el_menor(monkeypatch, configurado, gana_configurado):
    gpu = gpu_con_calentamiento()
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9, contexto_max_config=configurado)
    motor.cargar()
    assert motor.contexto_max == (configurado if gana_configurado else CONTEXTO_DESPUES)


def test_cargar_falla_si_no_queda_vram_para_contexto(monkeypatch):
    # Tras el calentamiento quedan 300 MiB libres: menos que margen + reserva (406 MiB).
    gpu = GPUFalsa(LIBRE_B, TOTAL_B, ASIGNADA_B, despues={"libre_b": 300 * MIB, "reservada_b": ASIGNADA_B})
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    with pytest.raises(RuntimeError) as exc:
        motor.cargar()
    assert not isinstance(exc.value, ValueError)
    mensaje = str(exc.value)
    for numero in ("-106.0", "300.0", "256.0", "150.0"):  # presupuesto, libre, margen, reserva
        assert numero in mensaje, mensaje
    assert "reserva" in mensaje and "margen" in mensaje
    assert motor.contexto_max is None and motor._model is None


# ---------- generar(): sin verificación de VRAM por pedido ----------

class LlegoAGenerar(Exception):
    pass


# Presupuesto de arranque del motor armado a mano (el medido en vivo en el paso 3f).
PRESUPUESTO_ARRANQUE_MB = 598.0


def motor_cargado(gpu: GPUFalsa, prompt_tokens: int, error_generate=None, genera=False, contexto_max=8192):
    """
    MotorLlama con modelo y tokenizador falsos, sin pasar por cargar(). generate lanza
    `error_generate`; si no hay error, con genera=True devuelve max_new_tokens tokens sin fin
    (finish_reason "length") y si no lanza LlegoAGenerar.
    """
    o = gpu.orden

    class TensorFalso:
        shape = (1, prompt_tokens)

        def to(self, device):
            o.append(("to_gpu",))
            return self

    def generate(max_new_tokens, **kwargs):
        o.append(("generate",))
        if error_generate is not None:
            raise error_generate
        if not genera:
            raise LlegoAGenerar()
        return [Secuencia([7] * prompt_tokens + [42] * max_new_tokens)]

    motor = MotorLlama("modelos/llama3", 0.9)
    motor._tokenizer = types.SimpleNamespace(
        apply_chat_template=lambda *a, **k: {"input_ids": TensorFalso()}, pad_token_id=None, eos_token_id=128009,
        decode=lambda ids, **k: SALIDA_CALENTAMIENTO,
    )
    motor._model = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"), generate=generate)
    motor._eos_ids = {128009}
    motor._costo_por_token_b = COSTO_LLAMA3_8B
    motor._presupuesto_arranque_b = PRESUPUESTO_ARRANQUE_MB * MIB
    motor.contexto_max = contexto_max
    return motor


class GPURuidosa(GPUFalsa):
    """mem_get_info devuelve valores MUY distintos en cada llamada, como el ruido medido en el paso 3f."""

    def __init__(self, libres_mib, **kw):
        super().__init__(libre_b=libres_mib[0] * MIB, total_b=TOTAL_B, asignada_b=ASIGNADA_B, **kw)
        self._libres = [x * MIB for x in libres_mib]
        self._i = 0

    def mem_get_info(self, device=0):
        self.libre_b = self._libres[self._i % len(self._libres)]
        self._i += 1
        return super().mem_get_info(device)


def test_pedido_justo_en_el_tope_se_acepta_siempre_aunque_la_vram_libre_oscile(monkeypatch):
    # El bug del paso 3f: 2390 tokens contra un tope de 2391 dieron 503 en 5 de 9 pedidos idénticos
    # porque la verificación por pedido re-medía mem_get_info (375-598 MiB). Ahora la capacidad la
    # decide el tope de arranque: con la VRAM libre oscilando entre 100, 900 y 400 MiB, el pedido
    # exactamente en el tope llega a generar en TODAS las repeticiones.
    gpu = GPURuidosa([100, 900, 400])
    gpu.instalar(monkeypatch)
    tope = 2391
    motor = motor_cargado(gpu, prompt_tokens=tope - 256, genera=True, contexto_max=tope)
    resultados = [motor.generar([{"role": "user", "content": "x"}], 0, 256) for _ in range(9)]
    assert [r.completion_tokens for r in resultados] == [256] * 9
    assert all(r.prompt_tokens + 256 == tope and r.finish_reason == "length" for r in resultados)


def test_durante_un_pedido_no_se_llama_a_mem_get_info(monkeypatch):
    # cargar() mide la VRAM (antes de cargar y tras el calentamiento); los pedidos, nunca.
    gpu = gpu_con_calentamiento()
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    motor.cargar()
    llamadas_carga = [p for p in gpu.orden if p[0] == "mem_get_info"]
    assert len(llamadas_carga) == 2, gpu.orden
    for _ in range(3):
        motor.generar([{"role": "user", "content": "x"}], 0, 16)
    assert [p for p in gpu.orden if p[0] == "mem_get_info"] == llamadas_carga
    assert [p[0] for p in gpu.orden].count("generate") == 4  # calentamiento + 3 pedidos


def test_generar_sin_vram_libre_igual_llega_a_generate(monkeypatch):
    # Antes, 300 MiB libres rechazaban este pedido de 314 MiB antes de tocar la GPU; ahora decide el tope.
    gpu = GPUFalsa(libre_b=300 * MIB, total_b=TOTAL_B, asignada_b=ASIGNADA_B)
    gpu.instalar(monkeypatch)
    motor = motor_cargado(gpu, prompt_tokens=1000)
    with pytest.raises(LlegoAGenerar):
        motor.generar([{"role": "user", "content": "x"}], 0, 256)
    assert [p[0] for p in gpu.orden] == ["to_gpu", "generate"]


def capturar_logs(caplog):
    """Handler de caplog directo en el logger del proveedor, sin propagar (sin duplicados)."""
    log = logging.getLogger("llama_local")
    estado = (log.propagate, log.level)
    log.propagate = False
    log.addHandler(caplog.handler)
    return log, estado


def soltar_logs(caplog, log, estado):
    log.removeHandler(caplog.handler)
    log.propagate, nivel = estado
    log.setLevel(nivel)


def test_oom_al_generar_se_convierte_en_vram_insuficiente(monkeypatch, caplog):
    gpu = GPUFalsa(libre_b=2000 * MIB, total_b=TOTAL_B, asignada_b=ASIGNADA_B, reservada_b=ASIGNADA_B + 100 * MIB)
    torch_falso = gpu.torch()
    monkeypatch.setitem(sys.modules, "torch", torch_falso)
    original = torch_falso.cuda.OutOfMemoryError(f"CUDA out of memory. Tried to allocate 64.00 MiB. {SALIDA_CALENTAMIENTO}")
    motor = motor_cargado(gpu, prompt_tokens=1000, error_generate=original)
    log, estado = capturar_logs(caplog)
    try:
        with caplog.at_level(logging.INFO, logger="llama_local"):
            with pytest.raises(VRAMInsuficiente) as exc:
                motor.generar([{"role": "user", "content": "x"}], 0, 256)
    finally:
        soltar_logs(caplog, log, estado)
    assert ("empty_cache",) in gpu.orden
    assert ("mem_get_info", gpu.libre_b) not in gpu.orden  # no re-mide: informa el presupuesto de arranque
    assert exc.value.tokens_pedido == 1256
    assert exc.value.necesaria_mb == pytest.approx(1256 * COSTO_LLAMA3_8B / MIB)
    assert exc.value.presupuesto_arranque_mb == pytest.approx(PRESUPUESTO_ARRANQUE_MB)
    assert "Tried to allocate" not in str(exc.value) and SALIDA_CALENTAMIENTO not in str(exc.value)
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1
    mensaje = avisos[0].getMessage()
    assert "estimación" in mensaje and "1256" in mensaje and "314.0" in mensaje and "598.0" in mensaje
    assert "Tried to allocate" not in caplog.text and SALIDA_CALENTAMIENTO not in caplog.text
    assert not avisos[0].exc_info


SENTINELA_TORCH = "SENTINELA-TORCH-3H"


def provocar_oom(monkeypatch, caplog=None):
    """
    Un OOM del generate falso con un texto sentinela, y gc.collect falso que se registra en el
    mismo `orden` que empty_cache. Devuelve (gpu, excepción lanzada, mensajes de log).
    """
    gpu = GPUFalsa(libre_b=2000 * MIB, total_b=TOTAL_B, asignada_b=ASIGNADA_B)
    torch_falso = gpu.torch()
    monkeypatch.setitem(sys.modules, "torch", torch_falso)
    monkeypatch.setattr(engine, "gc", types.SimpleNamespace(collect=lambda: gpu.orden.append(("gc_collect",))))
    original = torch_falso.cuda.OutOfMemoryError(f"CUDA out of memory. {SENTINELA_TORCH}")
    motor = motor_cargado(gpu, prompt_tokens=1000, error_generate=original)
    mensajes = []
    if caplog is not None:
        log, estado = capturar_logs(caplog)
    try:
        with pytest.raises(VRAMInsuficiente) as exc:
            if caplog is not None:
                with caplog.at_level(logging.INFO, logger="llama_local"):
                    motor.generar([{"role": "user", "content": "x"}], 0, 256)
            else:
                motor.generar([{"role": "user", "content": "x"}], 0, 256)
    finally:
        if caplog is not None:
            mensajes = [r.getMessage() for r in caplog.records]
            soltar_logs(caplog, log, estado)
    return gpu, exc.value, mensajes


def test_tras_un_oom_empty_cache_corre_sin_ninguna_excepcion_activa(monkeypatch):
    # Dentro del except la excepción viva retiene el traceback y los tensores del generate
    # fallido: empty_cache ahí no libera esa VRAM (paso 3g: ~420 MiB quedaban reservados).
    gpu, _, _ = provocar_oom(monkeypatch)
    assert gpu.exc_info_en_empty_cache == [(None, None, None)]


def test_tras_un_oom_gc_collect_corre_antes_de_empty_cache(monkeypatch):
    gpu, _, _ = provocar_oom(monkeypatch)
    nombres = [p[0] for p in gpu.orden]
    assert nombres.count("gc_collect") == 1 and nombres.count("empty_cache") == 1
    assert nombres.index("generate") < nombres.index("gc_collect") < nombres.index("empty_cache")


def test_vram_insuficiente_no_encadena_la_excepcion_de_torch(monkeypatch, caplog):
    _, exc, mensajes = provocar_oom(monkeypatch, caplog)
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert SENTINELA_TORCH not in str(exc)
    (aviso,) = [m for m in mensajes if m.startswith("OOM al generar")]
    assert "1256" in aviso and "314.0" in aviso and "598.0" in aviso
    assert all(SENTINELA_TORCH not in m for m in mensajes)
    assert SENTINELA_TORCH not in caplog.text


def test_otras_excepciones_al_generar_no_se_convierten(monkeypatch):
    gpu = GPUFalsa(libre_b=2000 * MIB, total_b=TOTAL_B, asignada_b=ASIGNADA_B)
    gpu.instalar(monkeypatch)
    motor = motor_cargado(gpu, prompt_tokens=10, error_generate=RuntimeError("otra falla"))
    with pytest.raises(RuntimeError, match="otra falla"):
        motor.generar([{"role": "user", "content": "x"}], 0, 8)
    assert ("empty_cache",) not in gpu.orden


def test_vram_insuficiente_solo_lleva_numeros():
    assert str(VRAMInsuficiente(1256, 314.0, 598.0)) == (
        "tokens_pedido=1256, necesaria_mb=314.0, presupuesto_arranque_mb=598.0"
    )
    # OOM en el calentamiento: todavía no hay presupuesto de arranque.
    assert str(VRAMInsuficiente(13, 3.25, None)).endswith("presupuesto_arranque_mb=None")


# ---------- liberar(): logs de diagnóstico del apagado ----------

def test_liberar_loguea_cada_paso_en_orden(monkeypatch, caplog):
    gpu = GPUFalsa(LIBRE_B, TOTAL_B, ASIGNADA_B)
    gpu.instalar(monkeypatch)
    log = logging.getLogger("llama_local")
    # El handler de caplog va directo al logger y sin propagar, así cada registro llega una sola
    # vez esté como esté propagate (la app lo pone en False). Se restaura al final.
    propaga = log.propagate
    log.propagate = False
    log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="llama_local"):
            motor = MotorLlama("modelos/llama3", 0.9)
            motor._model = object()
            motor.liberar()
    finally:
        log.removeHandler(caplog.handler)
        log.propagate = propaga
    mensajes = [r.getMessage() for r in caplog.records if r.name == "llama_local"]
    assert mensajes == [
        "Liberando: gc.collect() empieza.",
        "Liberando: gc.collect() terminó.",
        "Liberando: torch.cuda.empty_cache() empieza.",
        "Liberando: torch.cuda.empty_cache() terminó.",
        "Modelo liberado.",
    ]
    assert ("empty_cache",) in gpu.orden
    assert motor._model is None


# ---------- Imports diferidos ----------

def test_importar_el_proveedor_no_carga_torch():
    # En un proceso aparte: en este, otro test podría haber importado torch antes. Con None en
    # sys.modules, cualquier import de nivel de módulo falla, esté o no instalado torch.
    codigo = (
        "import sys\n"
        "for m in ('torch', 'transformers', 'bitsandbytes'):\n"
        "    sys.modules[m] = None\n"
        "import src.providers.llama_local.app, src.providers.llama_local.engine\n"
        "pesados = [m for m in ('torch', 'transformers', 'bitsandbytes') if sys.modules.get(m) is not None]\n"
        "assert not pesados, pesados\n"
    )
    raiz = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=raiz)
    assert r.returncode == 0, r.stderr
