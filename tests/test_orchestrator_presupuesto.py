"""Verifica el estimador de tokens y el recorte del historial del orquestador (funciones puras)."""
import pytest

from src.orchestrator.presupuesto import (
    CHARS_POR_TOKEN,
    MAX_TOKENS_DEFAULT_PROVEEDOR,
    OVERHEAD_BASE,
    OVERHEAD_POR_MENSAJE,
    disponible_para_mensajes,
    recortar_historial,
    tokens_de_mensaje,
)


def msg(role, n, letra="a"):
    """Mensaje con content de exactamente n caracteres."""
    return {"role": role, "content": letra * n}


# ---------- Estimador ----------

def test_constantes_acordadas():
    assert CHARS_POR_TOKEN == 3.0
    assert OVERHEAD_POR_MENSAJE == 8
    assert OVERHEAD_BASE == 16
    assert MAX_TOKENS_DEFAULT_PROVEEDOR == 256


@pytest.mark.parametrize(
    "n, esperado",
    [(0, 8), (1, 9), (3, 9), (30, 18), (4, 10), (31, 19), (32, 19)],
    ids=["vacio", "un_caracter", "multiplo_3", "multiplo_30", "no_multiplo_4", "no_multiplo_31", "no_multiplo_32"],
)
def test_tokens_de_mensaje_es_ceil_mas_overhead(n, esperado):
    assert tokens_de_mensaje(msg("user", n)) == esperado


def test_el_rol_no_cambia_la_estimacion():
    assert tokens_de_mensaje(msg("system", 30)) == tokens_de_mensaje(msg("assistant", 30)) == 18


# ---------- Presupuesto ----------

def test_disponible_con_max_tokens_del_pedido():
    assert disponible_para_mensajes(1800, 100) == 1800 - 100 - 16


def test_disponible_sin_max_tokens_usa_el_default_del_proveedor():
    assert disponible_para_mensajes(1800, None) == 1800 - 256 - 16


# ---------- Recorte ----------
# Todos de 30 caracteres = 18 tokens estimados. fijos (system + nuevo) = 36.

FIJOS = [msg("system", 30, "s"), msg("user", 30, "n")]
U1, A1, U2, A2, U3, A3 = (msg(r, 30, str(i)) for i, r in enumerate(["user", "assistant"] * 3))
HISTORIAL = [U1, A1, U2, A2, U3, A3]  # 108 tokens; total con fijos: 144


def test_historial_que_entra_completo_no_se_toca():
    assert recortar_historial(FIJOS, HISTORIAL, 144) == HISTORIAL  # justo en el límite


def test_se_descartan_los_mas_viejos_hasta_entrar():
    # 109 disponibles: hay que sacar 2 mensajes (144 - 36 = 108 <= 109).
    assert recortar_historial(FIJOS, HISTORIAL, 109) == [U2, A2, U3, A3]
    assert recortar_historial(FIJOS, HISTORIAL, 108) == [U2, A2, U3, A3]
    assert recortar_historial(FIJOS, HISTORIAL, 107) == [U3, A3]


def test_par_partido_tambien_se_descarta():
    # 143 disponibles: alcanzaría con sacar U1, pero A1 quedaría primero sin su pregunta.
    assert recortar_historial(FIJOS, HISTORIAL, 143) == [U2, A2, U3, A3]


def test_par_partido_con_tamanos_distintos():
    largo_a = msg("assistant", 300)  # 108 tokens
    historial = [msg("user", 30), largo_a, U2, A2]
    # total = 36 + 18 + 108 + 36 = 198; con 197, sacar el user alcanza, pero queda el assistant primero.
    assert recortar_historial(FIJOS, historial, 197) == [U2, A2]


def test_si_no_se_descarto_nada_un_assistant_inicial_se_conserva():
    # Sin recorte no hay par partido por culpa del orquestador: se reenvía lo que dio la memoria.
    historial = [A1, U2, A2]
    assert recortar_historial(FIJOS, historial, 1000) == historial


def test_puede_quedar_vacio():
    assert recortar_historial(FIJOS, HISTORIAL, 36) == []
    assert recortar_historial(FIJOS, [], 36) == []


def test_fijos_que_no_entran_devuelven_none():
    assert recortar_historial(FIJOS, HISTORIAL, 35) is None
    assert recortar_historial(FIJOS, [], 35) is None
    assert recortar_historial(FIJOS, HISTORIAL, -10) is None


def test_no_modifica_ni_reordena_las_entradas():
    historial = [dict(m) for m in HISTORIAL]
    fijos = [dict(m) for m in FIJOS]
    resultado = recortar_historial(fijos, historial, 109)
    assert historial == HISTORIAL and fijos == FIJOS
    assert resultado == historial[2:]
