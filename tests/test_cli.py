"""Verifica la consola para desarrolladores (src/cli.py) sin red real: sesión HTTP inyectada."""
import io
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from src import cli

ORQ = "http://127.0.0.1:8002"
MEM = "http://127.0.0.1:8003"
PROV = "http://127.0.0.1:8004"


# ---------- Dobles ----------

class RespuestaFalsa:
    def __init__(self, status_code=200, cuerpo=None, json_invalido=False):
        self.status_code = status_code
        self._cuerpo = cuerpo
        self._json_invalido = json_invalido

    def json(self):
        if self._json_invalido:
            raise ValueError("no es JSON")
        return self._cuerpo


class SesionFalsa:
    """Responde según (método, URL) con respuestas o excepciones encoladas, y registra los pedidos."""

    def __init__(self, rutas=None):
        self.rutas = {k: list(v) if isinstance(v, list) else [v] for k, v in (rutas or {}).items()}
        self.pedidos = []

    def _responder(self, metodo, url, **kwargs):
        self.pedidos.append({"metodo": metodo, "url": url, **kwargs})
        cola = self.rutas.get((metodo, url))
        if not cola:
            raise AssertionError(f"pedido inesperado: {metodo} {url}")
        r = cola.pop(0) if len(cola) > 1 else cola[0]
        if isinstance(r, BaseException):
            raise r
        return r

    def get(self, url, **kwargs):
        return self._responder("get", url, **kwargs)

    def post(self, url, **kwargs):
        return self._responder("post", url, **kwargs)


def respuesta_chat(texto="hola", x_memory=None, x_metrics=None, usage=True):
    cuerpo = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": "stop"}],
    }
    if usage:
        cuerpo["usage"] = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    if x_memory is not None:
        cuerpo["x_memory"] = x_memory
    if x_metrics is not None:
        cuerpo["x_metrics"] = x_metrics
    return RespuestaFalsa(200, cuerpo)


def error(status, code, message="mensaje del servicio"):
    return RespuestaFalsa(status, {"error": {"message": message, "type": "t", "param": None, "code": code}})


def correr(argv, sesion=None, entradas=(), entorno=None):
    """Corre main y devuelve (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    cola = list(entradas)

    def entrada(_prompt):
        if not cola:
            raise EOFError
        valor = cola.pop(0)
        if isinstance(valor, BaseException):
            raise valor
        return valor

    codigo = cli.main(argv, session=sesion or SesionFalsa(), entrada=entrada,
                      entorno={} if entorno is None else entorno, out=out, err=err)
    return codigo, out.getvalue(), err.getvalue()


def salida_de_uso(argv, entorno=None):
    with pytest.raises(SystemExit) as exc:
        cli.parsear(argv, {} if entorno is None else entorno)
    return exc.value.code


# ---------- Parseo y configuración ----------

def test_defaults_sin_variables_de_entorno():
    args = cli.parsear(["status"], {})
    assert args.orchestrator_url == ORQ
    assert args.memory_url == MEM
    assert args.provider_url == PROV
    assert args.user_id == "dev-user"
    assert args.debug is False


def test_variables_de_entorno_y_flags():
    entorno = {"ORCHESTRATOR_URL": "http://localhost:9002/", "MEMORY_URL": "http://localhost:9003",
               "PROVIDER_URL": "http://localhost:9004", "DEV_USER_ID": "otro_user"}
    args = cli.parsear(["status"], entorno)
    # La barra final se saca para no armar "//health".
    assert args.orchestrator_url == "http://localhost:9002"
    assert args.memory_url == "http://localhost:9003"
    assert args.provider_url == "http://localhost:9004"
    assert args.user_id == "otro_user"
    # El flag gana sobre la variable.
    args = cli.parsear(["--orchestrator-url", "http://127.0.0.1:1", "--debug", "models"], entorno)
    assert args.orchestrator_url == "http://127.0.0.1:1"
    assert args.debug is True


def test_variable_vacia_usa_el_default():
    args = cli.parsear(["status"], {"ORCHESTRATOR_URL": ""})
    assert args.orchestrator_url == ORQ


def test_defaults_de_chat_ask_y_history():
    args = cli.parsear(["chat"], {})
    assert (args.conversation, args.model, args.max_tokens, args.temperature) == (None, None, None, None)
    assert args.timeout == cli.PEDIDO_TIMEOUT_S
    args = cli.parsear(["ask", "hola"], {})
    assert args.target == "orquestador"
    assert args.texto == "hola"
    args = cli.parsear(["history", "conv-1"], {})
    assert args.limit == 20
    args = cli.parsear(["chat", "-c", "abc_1", "--model", "m", "--max-tokens", "50", "--temperature", "0.2"], {})
    assert (args.conversation, args.model, args.max_tokens, args.temperature) == ("abc_1", "m", 50, 0.2)


@pytest.mark.parametrize("argv", [
    [],
    ["inexistente"],
    ["chat", "--conversation", "con espacio"],
    ["chat", "-c", "x" * 65],
    ["chat", "-c", "abc\n"],
    ["history", "no/valido"],
    ["history", "conv-1", "--limit", "0"],
    ["history", "conv-1", "--limit", "201"],
    ["history", "conv-1", "--limit", "diez"],
    ["ask", "hola", "--target", "memoria"],
    ["ask", "hola", "--target", "proveedor"],
    ["chat", "--max-tokens", "mucho"],
])
def test_uso_incorrecto_sale_con_2(argv):
    assert salida_de_uso(argv) == 2


def test_limit_maximo_aceptado():
    assert cli.parsear(["history", "conv-1", "--limit", "200"], {}).limit == 200


def test_dev_user_id_invalido_sale_con_2_en_history(capsys):
    assert salida_de_uso(["history", "conv-1"], {"DEV_USER_ID": "con espacio"}) == 2
    assert "DEV_USER_ID" in capsys.readouterr().err
    assert salida_de_uso(["--user-id", "x" * 65, "history", "conv-1"]) == 2


def test_dev_user_id_solo_se_valida_en_history():
    # status no lo usa: no tiene por qué fallar.
    assert cli.parsear(["status"], {"DEV_USER_ID": "con espacio"}).comando == "status"


def test_id_valido_cumple_el_patron_en_ambos_extremos():
    assert cli.parsear(["chat", "-c", "a"], {}).conversation == "a"
    assert cli.parsear(["chat", "-c", "A" * 64], {}).conversation == "A" * 64


# ---------- status ----------

def sesion_de_status(orq="ready", mem="ready", prov="ready"):
    def health(estado):
        if isinstance(estado, BaseException):
            return estado
        # El proveedor contesta 503 en loading/failed; la memoria, 200 siempre.
        return RespuestaFalsa(200 if estado == "ready" else 503, {"status": estado})
    return SesionFalsa({
        ("get", ORQ + "/health"): health(orq),
        ("get", MEM + "/health"): RespuestaFalsa(200, {"status": mem}) if isinstance(mem, str) else mem,
        ("get", PROV + "/health"): health(prov),
        ("get", ORQ + "/v1/models"): RespuestaFalsa(200, {"object": "list", "data": [
            {"id": "mock"}, {"id": "llama3-8b-local"}]}),
    })


def test_status_tres_ready_sale_con_0():
    sesion = sesion_de_status()
    codigo, out, err = correr(["status"], sesion)
    assert codigo == 0
    lineas = out.splitlines()
    assert lineas[0].split()[:2] == ["orquestador", "ready"]
    assert lineas[1].split()[:2] == ["memoria", "ready"]
    assert lineas[2].split()[:2] == ["proveedor", "ready"]
    assert all(" ms " in l for l in lineas[:3])
    assert lineas[3] == "modelos      mock, llama3-8b-local"
    assert err == ""
    # Timeout corto en todos los pedidos.
    assert all(p["timeout"] == 5.0 for p in sesion.pedidos)


def test_status_proveedor_loading_sale_con_1():
    codigo, out, _ = correr(["status"], sesion_de_status(prov="loading"))
    assert codigo == 1
    assert out.splitlines()[2].split()[:2] == ["proveedor", "loading"]


def test_status_memoria_loading_con_200_sale_con_1():
    codigo, out, _ = correr(["status"], sesion_de_status(mem="loading"))
    assert codigo == 1
    assert out.splitlines()[1].split()[:2] == ["memoria", "loading"]


def test_status_proveedor_failed():
    codigo, out, _ = correr(["status"], sesion_de_status(prov="failed"))
    assert codigo == 1
    assert out.splitlines()[2].split()[:2] == ["proveedor", "failed"]


def test_status_servicio_caido_dice_no_disponible():
    codigo, out, err = correr(["status"], sesion_de_status(mem=requests.ConnectionError("rechazada")))
    assert codigo == 1
    assert "memoria      no disponible" in out.splitlines()[1]
    assert "Traceback" not in err


def test_status_cuerpo_no_json_dice_no_disponible_con_el_http():
    sesion = sesion_de_status()
    sesion.rutas[("get", PROV + "/health")] = [RespuestaFalsa(502, json_invalido=True)]
    codigo, out, _ = correr(["status"], sesion)
    assert codigo == 1
    assert "no disponible (HTTP 502)" in out.splitlines()[2]


def test_status_modelos_caidos_no_cambia_el_exit_code():
    sesion = sesion_de_status()
    sesion.rutas[("get", ORQ + "/v1/models")] = [requests.ConnectionError()]
    codigo, out, _ = correr(["status"], sesion)
    assert codigo == 0
    assert out.splitlines()[3].startswith("modelos      no disponibles")


# ---------- models ----------

def test_models_lista_los_ids():
    sesion = SesionFalsa({("get", ORQ + "/v1/models"): RespuestaFalsa(200, {"data": [{"id": "mock"}, {"id": "otro"}]})})
    codigo, out, _ = correr(["models"], sesion)
    assert codigo == 0
    assert out.splitlines() == ["mock", "otro"]


def test_models_error_del_servicio_sale_con_1():
    sesion = SesionFalsa({("get", ORQ + "/v1/models"): error(500, "internal_error", "falló")})
    codigo, out, err = correr(["models"], sesion)
    assert codigo == 1
    assert out == ""
    assert "HTTP 500 internal_error: falló" in err


# ---------- chat ----------

def sesion_de_chat(*respuestas):
    return SesionFalsa({("post", ORQ + "/v1/chat/completions"): list(respuestas)})


def test_chat_un_turno_manda_un_solo_mensaje_y_el_id():
    sesion = sesion_de_chat(respuesta_chat("respuesta uno", x_memory={"status": "saved"}))
    codigo, out, _ = correr(["chat", "-c", "conv-1"], sesion, entradas=["hola"])
    assert codigo == 0
    assert len(sesion.pedidos) == 1
    cuerpo = sesion.pedidos[0]["json"]
    assert cuerpo == {"messages": [{"role": "user", "content": "hola"}], "x_conversation_id": "conv-1"}
    assert "stream" not in cuerpo
    assert "respuesta uno" in out
    assert "memoria saved" in out


def test_chat_pasa_model_max_tokens_y_temperature():
    sesion = sesion_de_chat(respuesta_chat())
    correr(["chat", "-c", "c", "--model", "mock", "--max-tokens", "40", "--temperature", "0"], sesion, entradas=["hola"])
    cuerpo = sesion.pedidos[0]["json"]
    assert cuerpo["model"] == "mock"
    assert cuerpo["max_tokens"] == 40
    assert cuerpo["temperature"] == 0.0
    assert sesion.pedidos[0]["timeout"] == cli.PEDIDO_TIMEOUT_S


def test_chat_genera_un_id_valido_y_lo_imprime():
    sesion = sesion_de_chat(respuesta_chat())
    _, out, _ = correr(["chat"], sesion, entradas=["hola"])
    generado = sesion.pedidos[0]["json"]["x_conversation_id"]
    assert cli.ID_PATTERN.fullmatch(generado)
    assert generado.startswith("cli-") and len(generado) == 12
    assert f"--conversation {generado}" in out.splitlines()[0]


def test_chat_ids_generados_distintos_por_sesion():
    ids = set()
    for _ in range(5):
        sesion = sesion_de_chat(respuesta_chat())
        correr(["chat"], sesion, entradas=["hola"])
        ids.add(sesion.pedidos[0]["json"]["x_conversation_id"])
    assert len(ids) == 5


def test_chat_dos_turnos_mismo_id_sin_acumular_historial():
    sesion = sesion_de_chat(respuesta_chat("r1"), respuesta_chat("r2"))
    correr(["chat"], sesion, entradas=["primero", "segundo"])
    assert len(sesion.pedidos) == 2
    c1, c2 = (p["json"] for p in sesion.pedidos)
    assert c1["x_conversation_id"] == c2["x_conversation_id"]
    assert c1["messages"] == [{"role": "user", "content": "primero"}]
    assert c2["messages"] == [{"role": "user", "content": "segundo"}]


@pytest.mark.parametrize("status,code", [
    (400, "context_length_exceeded"),
    (503, "provider_unavailable"),
    (502, "provider_error"),
])
def test_chat_error_en_un_turno_se_informa_y_la_sesion_sigue(status, code):
    sesion = sesion_de_chat(error(status, code, "texto del servicio"), respuesta_chat("despues del error"))
    codigo, out, err = correr(["chat", "-c", "c"], sesion, entradas=["uno", "dos"])
    assert codigo == 0
    assert len(sesion.pedidos) == 2
    assert f"HTTP {status} {code}: texto del servicio" in err
    assert "despues del error" in out
    assert "Traceback" not in err


def test_chat_error_de_conexion_en_un_turno_no_termina_la_sesion():
    sesion = sesion_de_chat(requests.ConnectionError(), respuesta_chat("ok"))
    codigo, out, err = correr(["chat", "-c", "c"], sesion, entradas=["uno", "dos"])
    assert codigo == 0
    assert "No se pudo conectar con el orquestador en http://127.0.0.1:8002" in err
    assert "ok" in out


@pytest.mark.parametrize("salida", ["salir", "exit", "chau", "CHAU", "  salir  "])
def test_chat_palabras_de_salida_no_mandan_nada(salida):
    sesion = sesion_de_chat(respuesta_chat())
    codigo, _, _ = correr(["chat"], sesion, entradas=[salida, "no se manda"])
    assert codigo == 0
    assert sesion.pedidos == []


@pytest.mark.parametrize("corte", [EOFError(), KeyboardInterrupt()])
def test_chat_eof_y_ctrl_c_terminan_con_0(corte):
    sesion = sesion_de_chat(respuesta_chat())
    codigo, _, _ = correr(["chat"], sesion, entradas=["hola", corte])
    assert codigo == 0
    assert len(sesion.pedidos) == 1


def test_chat_linea_vacia_no_manda_nada():
    sesion = sesion_de_chat(respuesta_chat())
    correr(["chat"], sesion, entradas=["", "   ", "hola"])
    assert len(sesion.pedidos) == 1


def test_chat_respuesta_sin_choices_es_error_del_turno():
    sesion = sesion_de_chat(RespuestaFalsa(200, {"object": "chat.completion"}), respuesta_chat("bien"))
    codigo, out, err = correr(["chat", "-c", "c"], sesion, entradas=["uno", "dos"])
    assert codigo == 0
    assert "choices[0].message.content" in err
    assert "bien" in out


# ---------- Línea de datos ----------

def test_linea_de_datos_muestra_solo_lo_que_existe():
    data = respuesta_chat(x_memory={"status": "write_failed"},
                          x_metrics={"latency_s": 1.5, "tokens_per_s": 30.1, "vram_peak_mb": 6000}).json()
    linea = cli.linea_de_datos(2.0, data)
    assert linea == ("[total 2.00 s | prompt_tokens 12, completion_tokens 3 | memoria write_failed"
                     " | latency_s=1.5 tokens_per_s=30.1 vram_peak_mb=6000]")


def test_linea_de_datos_sin_usage_ni_extras():
    data = respuesta_chat(usage=False).json()
    assert cli.linea_de_datos(0.5, data) == "[total 0.50 s]"


def test_linea_de_datos_ignora_metricas_desconocidas():
    data = respuesta_chat(usage=False, x_metrics={"latency_s": 0.1, "otra": 1}).json()
    assert cli.linea_de_datos(0.5, data) == "[total 0.50 s | latency_s=0.1]"


# ---------- ask ----------

def test_ask_orquestador_no_manda_x_conversation_id():
    sesion = sesion_de_chat(respuesta_chat("suelta", x_metrics={"latency_s": 0.2}))
    codigo, out, _ = correr(["ask", "pregunta"], sesion)
    assert codigo == 0
    cuerpo = sesion.pedidos[0]["json"]
    assert cuerpo == {"messages": [{"role": "user", "content": "pregunta"}]}
    assert out.splitlines()[0] == "suelta"
    assert "latency_s=0.2" in out


def test_ask_proveedor_pega_a_provider_url_y_manda_model():
    sesion = SesionFalsa({("post", PROV + "/v1/chat/completions"): respuesta_chat("directo")})
    codigo, out, _ = correr(["ask", "pregunta", "--target", "proveedor", "--model", "llama3-8b-local"], sesion)
    assert codigo == 0
    assert sesion.pedidos[0]["url"] == PROV + "/v1/chat/completions"
    cuerpo = sesion.pedidos[0]["json"]
    assert cuerpo["model"] == "llama3-8b-local"
    assert "x_conversation_id" not in cuerpo
    assert "directo" in out


def test_ask_error_sale_con_1():
    sesion = sesion_de_chat(error(404, "model_not_found", "El modelo 'x' no existe."))
    codigo, out, err = correr(["ask", "hola", "--model", "x"], sesion)
    assert codigo == 1
    assert out == ""
    assert "HTTP 404 model_not_found: El modelo 'x' no existe." in err


def test_error_sin_cuerpo_reconocible_no_vuelca_el_cuerpo():
    sesion = sesion_de_chat(RespuestaFalsa(500, {"detalle": "CUERPO-CRUDO"}))
    codigo, _, err = correr(["ask", "hola"], sesion)
    assert codigo == 1
    assert "HTTP 500 sin cuerpo de error reconocible" in err
    assert "CUERPO-CRUDO" not in err


def test_respuesta_2xx_no_json_es_error():
    sesion = sesion_de_chat(RespuestaFalsa(200, json_invalido=True))
    codigo, _, err = correr(["ask", "hola"], sesion)
    assert codigo == 1
    assert "no es JSON" in err


# ---------- history ----------

def test_history_arma_la_query_e_imprime_en_orden():
    url = MEM + "/v1/conversations/conv-1/messages"
    sesion = SesionFalsa({("get", url): RespuestaFalsa(200, {"conversation_id": "conv-1", "messages": [
        {"role": "user", "content": "primero", "created_at": 1},
        {"role": "assistant", "content": "segundo", "created_at": 1},
        {"role": "user", "content": "tercero", "created_at": 2},
    ]})})
    codigo, out, _ = correr(["history", "conv-1", "--limit", "5"], sesion, entorno={"DEV_USER_ID": "u_1"})
    assert codigo == 0
    assert sesion.pedidos[0]["params"] == {"user_id": "u_1", "limit": 5}
    assert out.splitlines() == ["[user] primero", "[assistant] segundo", "[user] tercero"]


def test_history_default_limit_y_usuario():
    url = MEM + "/v1/conversations/c/messages"
    sesion = SesionFalsa({("get", url): RespuestaFalsa(200, {"conversation_id": "c", "messages": []})})
    codigo, out, _ = correr(["history", "c"], sesion)
    assert codigo == 0
    assert sesion.pedidos[0]["params"] == {"user_id": "dev-user", "limit": 20}
    assert out.strip() == "(sin mensajes)"


def test_history_error_de_la_memoria_sale_con_1():
    url = MEM + "/v1/conversations/c/messages"
    sesion = SesionFalsa({("get", url): error(503, "storage_unavailable", "La base no responde.")})
    codigo, _, err = correr(["history", "c"], sesion)
    assert codigo == 1
    assert "la memoria: HTTP 503 storage_unavailable: La base no responde." in err


# ---------- Errores de conexión y --debug ----------

def test_error_de_conexion_mensaje_claro_sin_traceback():
    sesion = sesion_de_chat(requests.ConnectionError("detalle interno"))
    codigo, out, err = correr(["ask", "hola"], sesion)
    assert codigo == 1
    assert "No se pudo conectar con el orquestador en http://127.0.0.1:8002" in err
    assert "python -m src.cli status" in err
    assert "Traceback" not in err


def test_error_de_conexion_con_debug_agrega_traceback():
    sesion = sesion_de_chat(requests.ConnectionError("detalle interno"))
    codigo, _, err = correr(["--debug", "ask", "hola"], sesion)
    assert codigo == 1
    assert "No se pudo conectar" in err
    assert "Traceback" in err


def test_timeout_mensaje_claro():
    sesion = sesion_de_chat(requests.ConnectTimeout())
    codigo, _, err = correr(["ask", "hola", "--timeout", "3"], sesion)
    assert codigo == 1
    assert "el orquestador (http://127.0.0.1:8002) no respondió en 3 s" in err
    assert sesion.pedidos[0]["timeout"] == 3.0


def test_url_con_credenciales_no_las_muestra():
    sesion = SesionFalsa({("post", "http://user:clave@127.0.0.1:8002/v1/chat/completions"): requests.ConnectionError()})
    _, _, err = correr(["--orchestrator-url", "http://user:clave@127.0.0.1:8002", "ask", "hola"], sesion)
    assert "http://127.0.0.1:8002" in err
    assert "clave" not in err and "user:" not in err


# ---------- Aviso de URL no loopback ----------

def test_aviso_si_la_url_no_es_loopback():
    sesion = SesionFalsa({("get", "http://10.0.0.5:8002/v1/models"): RespuestaFalsa(200, {"data": []})})
    codigo, _, err = correr(["--orchestrator-url", "http://10.0.0.5:8002", "models"], sesion)
    # Avisa pero no bloquea.
    assert codigo == 0
    assert "Aviso: ORCHESTRATOR_URL apunta a http://10.0.0.5:8002" in err
    assert len(sesion.pedidos) == 1


@pytest.mark.parametrize("url", ["http://127.0.0.1:8002", "http://localhost:8002", "http://[::1]:8002"])
def test_sin_aviso_en_loopback(url):
    sesion = SesionFalsa({("get", url + "/v1/models"): RespuestaFalsa(200, {"data": []})})
    _, _, err = correr(["--orchestrator-url", url, "models"], sesion)
    assert "Aviso" not in err


def test_status_avisa_por_cada_url_no_loopback():
    sesion = SesionFalsa({
        ("get", "http://10.0.0.1/health"): RespuestaFalsa(200, {"status": "ready"}),
        ("get", MEM + "/health"): RespuestaFalsa(200, {"status": "ready"}),
        ("get", "http://10.0.0.2/health"): RespuestaFalsa(200, {"status": "ready"}),
        ("get", "http://10.0.0.1/v1/models"): RespuestaFalsa(200, {"data": []}),
    })
    _, _, err = correr(["status"], sesion, entorno={"ORCHESTRATOR_URL": "http://10.0.0.1",
                                                   "PROVIDER_URL": "http://10.0.0.2"})
    assert "ORCHESTRATOR_URL" in err and "PROVIDER_URL" in err
    assert "MEMORY_URL" not in err


# ---------- Imports ----------

def test_importar_la_cli_no_carga_torch_ni_transformers():
    # En un proceso aparte, como en test_llama_local_engine: con None en sys.modules,
    # cualquier import de nivel de módulo falla, esté o no instalado.
    codigo = (
        "import sys\n"
        "for m in ('torch', 'transformers', 'bitsandbytes'):\n"
        "    sys.modules[m] = None\n"
        "import src.cli\n"
        "pesados = [m for m in ('torch', 'transformers', 'bitsandbytes') if sys.modules.get(m) is not None]\n"
        "assert not pesados, pesados\n"
    )
    raiz = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=raiz)
    assert r.returncode == 0, r.stderr


def test_importar_la_cli_no_importa_otros_servicios():
    codigo = (
        "import sys\n"
        "import src.cli\n"
        "otros = [m for m in sys.modules if m.startswith(('src.orchestrator', 'src.memory', 'src.providers', 'yaml', 'httpx2'))]\n"
        "assert not otros, otros\n"
    )
    raiz = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=raiz)
    assert r.returncode == 0, r.stderr
