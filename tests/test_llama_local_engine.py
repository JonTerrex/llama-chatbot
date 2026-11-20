"""Reglas puras del motor del proveedor Llama local: se prueban sin torch ni GPU."""
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.providers.llama_local.engine import (
    MARGEN_VRAM_MB,
    ContextoExcedido,
    MotorLlama,
    SinGPU,
    VRAMInsuficiente,
    bytes_por_token,
    contexto_efectivo,
    contexto_por_vram,
    decidir_finish_reason,
    fraccion_vram,
    resolver_parametros,
    validar_contexto,
    vram_alcanza,
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


def test_vram_alcanza_justo_en_el_borde():
    # necesaria = 1000 tokens * costo; disponible = libre + cache - margen, exactamente igual.
    necesaria = 1000 * COSTO_LLAMA3_8B
    assert vram_alcanza(1000, COSTO_LLAMA3_8B, necesaria + 256 * MIB, 0, 256 * MIB)


def test_vram_alcanza_un_byte_por_encima_no():
    necesaria = 1000 * COSTO_LLAMA3_8B
    assert not vram_alcanza(1000, COSTO_LLAMA3_8B, necesaria + 256 * MIB - 1, 0, 256 * MIB)


def test_vram_alcanza_gracias_a_la_cache_reutilizable():
    necesaria = 1000 * COSTO_LLAMA3_8B
    assert not vram_alcanza(1000, COSTO_LLAMA3_8B, 256 * MIB, 0, 256 * MIB)
    assert vram_alcanza(1000, COSTO_LLAMA3_8B, 256 * MIB, necesaria, 256 * MIB)


# ---------- cargar() completo, con torch y transformers falsos ----------

def config_llama3(**cambios):
    base = dict(num_hidden_layers=32, num_key_value_heads=8, head_dim=128, hidden_size=4096,
                num_attention_heads=32, max_position_embeddings=8192)
    base.update(cambios)
    return types.SimpleNamespace(**base)


class GPUFalsa:
    """torch y transformers falsos que registran en `orden` cada llamada relevante, en secuencia."""

    def __init__(self, libre_b, total_b, asignada_b, reservada_b=None, config=None):
        self.orden: list[tuple] = []
        self.libre_b, self.total_b = libre_b, total_b
        self.asignada_b = asignada_b
        self.reservada_b = asignada_b if reservada_b is None else reservada_b
        self.config = config or config_llama3()

    def torch(self):
        o = self.orden
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda device=0: (self.libre_b, self.total_b),
            get_device_name=lambda device=0: "GPU falsa",
            set_per_process_memory_fraction=lambda f, device=0: o.append(("set_fraction", f, device)),
            memory_allocated=lambda device=None: self.asignada_b,
            memory_reserved=lambda device=None: self.reservada_b,
            empty_cache=lambda: o.append(("empty_cache",)),
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
        )
        return types.SimpleNamespace(cuda=cuda, bfloat16="bf16", __version__="falso", version=types.SimpleNamespace(cuda="falso"))

    def transformers(self):
        o, config = self.orden, self.config

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path):
                o.append(("tokenizer",))
                return types.SimpleNamespace(eos_token_id=128009)

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(path, **kwargs):
                o.append(("modelo", kwargs["device_map"]))
                return types.SimpleNamespace(config=config, generation_config=types.SimpleNamespace(eos_token_id=[128001, 128009]))

        return types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModelForCausalLM, BitsAndBytesConfig=lambda **kw: kw
        )

    def instalar(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self.torch())
        monkeypatch.setitem(sys.modules, "transformers", self.transformers())


LIBRE_B, TOTAL_B, ASIGNADA_B = int(6.89 * GIB), 8 * GIB, 5600 * MIB


def contexto_esperado_por_vram():
    limite_b = (LIBRE_B - MARGEN_VRAM_MB * MIB) / TOTAL_B * TOTAL_B
    return int((limite_b - ASIGNADA_B) // COSTO_LLAMA3_8B)


def test_cargar_sin_override_usa_el_tope_por_vram(monkeypatch):
    gpu = GPUFalsa(LIBRE_B, TOTAL_B, ASIGNADA_B)
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    motor.cargar()
    assert motor.contexto_max == contexto_esperado_por_vram()
    assert 0 < motor.contexto_max < 8192  # acá manda la VRAM, no el modelo
    assert motor._costo_por_token_b == COSTO_LLAMA3_8B
    assert motor._eos_ids == {128001, 128009}


def test_cargar_sin_head_dim_lo_deriva_de_hidden_size(monkeypatch):
    config = config_llama3()
    del config.head_dim
    gpu = GPUFalsa(LIBRE_B, TOTAL_B, ASIGNADA_B, config=config)
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9)
    motor.cargar()
    assert motor._costo_por_token_b == COSTO_LLAMA3_8B  # 4096 // 32 = 128


@pytest.mark.parametrize("configurado, gana_configurado", [(1024, True), (99999, False)])
def test_cargar_con_override_manda_el_menor(monkeypatch, configurado, gana_configurado):
    gpu = GPUFalsa(LIBRE_B, TOTAL_B, ASIGNADA_B)
    gpu.instalar(monkeypatch)
    motor = MotorLlama("modelos/llama3", 0.9, contexto_max_config=configurado)
    motor.cargar()
    assert motor.contexto_max == (configurado if gana_configurado else contexto_esperado_por_vram())


def test_cargar_limita_la_vram_antes_de_cargar_el_modelo(monkeypatch):
    gpu = GPUFalsa(LIBRE_B, TOTAL_B, ASIGNADA_B)
    gpu.instalar(monkeypatch)
    MotorLlama("modelos/llama3", 0.9).cargar()
    nombres = [paso[0] for paso in gpu.orden]
    assert "set_fraction" in nombres, gpu.orden
    assert nombres.index("set_fraction") < nombres.index("modelo")
    (_, fraccion, device), = [p for p in gpu.orden if p[0] == "set_fraction"]
    assert fraccion == pytest.approx((LIBRE_B - MARGEN_VRAM_MB * MIB) / TOTAL_B)
    assert device == 0


def test_cargar_falla_si_no_queda_vram_para_contexto(monkeypatch):
    # El modelo ocupó todo el límite: ni un token de contexto entra.
    gpu = GPUFalsa(LIBRE_B, TOTAL_B, LIBRE_B)
    gpu.instalar(monkeypatch)
    with pytest.raises(ValueError, match="no alcanza"):
        MotorLlama("modelos/llama3", 0.9).cargar()


# ---------- generar(): verificación de VRAM por pedido ----------

class LlegoAGenerar(Exception):
    pass


def motor_cargado(gpu: GPUFalsa, prompt_tokens: int):
    """MotorLlama con modelo y tokenizador falsos, sin pasar por cargar()."""
    o = gpu.orden

    class TensorFalso:
        shape = (1, prompt_tokens)

        def to(self, device):
            o.append(("to_gpu",))
            return self

    def generate(**kwargs):
        o.append(("generate",))
        raise LlegoAGenerar()

    motor = MotorLlama("modelos/llama3", 0.9)
    motor._tokenizer = types.SimpleNamespace(
        apply_chat_template=lambda *a, **k: {"input_ids": TensorFalso()}, pad_token_id=None, eos_token_id=128009
    )
    motor._model = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"), generate=generate)
    motor._eos_ids = {128009}
    motor._costo_por_token_b = COSTO_LLAMA3_8B
    motor.contexto_max = 8192
    return motor


def test_generar_sin_vram_suficiente_falla_antes_de_tocar_la_gpu(monkeypatch):
    # 1000 + 256 tokens * 0,25 MiB = 314 MiB necesarios; disponibles 300 + 0 - 256 = 44 MiB.
    gpu = GPUFalsa(libre_b=300 * MIB, total_b=TOTAL_B, asignada_b=ASIGNADA_B)
    gpu.instalar(monkeypatch)
    motor = motor_cargado(gpu, prompt_tokens=1000)
    with pytest.raises(VRAMInsuficiente) as exc:
        motor.generar([{"role": "user", "content": "x"}], 0, 256)
    assert exc.value.necesaria_mb == pytest.approx(1256 * COSTO_LLAMA3_8B / MIB)
    assert exc.value.disponible_mb == pytest.approx(300 - MARGEN_VRAM_MB)
    assert ("generate",) not in gpu.orden and ("to_gpu",) not in gpu.orden


def test_generar_con_vram_suficiente_sigue_hasta_generate(monkeypatch):
    # La cache reservada y libre del proceso cuenta: 300 MiB libres + 400 MiB de cache.
    gpu = GPUFalsa(libre_b=300 * MIB, total_b=TOTAL_B, asignada_b=ASIGNADA_B, reservada_b=ASIGNADA_B + 400 * MIB)
    gpu.instalar(monkeypatch)
    motor = motor_cargado(gpu, prompt_tokens=1000)
    with pytest.raises(LlegoAGenerar):
        motor.generar([{"role": "user", "content": "x"}], 0, 256)
    assert [p[0] for p in gpu.orden] == ["to_gpu", "generate"]


def test_vram_insuficiente_solo_lleva_numeros():
    assert str(VRAMInsuficiente(314.0, 44.0)) == "necesaria_mb=314.0, disponible_mb=44.0"


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
