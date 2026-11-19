"""Reglas puras del motor del proveedor Llama local: se prueban sin torch ni GPU."""
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.providers.llama_local.engine import (
    ContextoExcedido,
    MotorLlama,
    SinGPU,
    decidir_finish_reason,
    resolver_parametros,
    validar_contexto,
)

GIB = 1024 ** 3

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
