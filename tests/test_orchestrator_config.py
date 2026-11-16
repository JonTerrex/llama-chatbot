"""Verifica la carga y validación de la configuración del orquestador."""
import pytest

from src.orchestrator.config import cargar_config


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Registro y prompt válidos en tmp_path; cada test sobrescribe lo que necesita romper."""
    providers = tmp_path / "providers.yaml"
    providers.write_text(
        "providers:\n"
        "  mock:\n"
        "    url: http://mock:8000/\n"
        "  local:\n"
        "    url: https://local:9000\n",
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Sos un asistente de prueba. ñ\n\n", encoding="utf-8")

    monkeypatch.setenv("PROVIDERS_FILE", str(providers))
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(prompt))
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("PROVIDER_TIMEOUT_S", raising=False)
    for variable in ("MEMORY_URL", "DEV_USER_ID", "MEMORY_HISTORY_LIMIT", "MEMORY_TIMEOUT_S"):
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def escribir_registro(tmp_path, contenido):
    (tmp_path / "providers.yaml").write_text(contenido, encoding="utf-8")


# ---------- Carga correcta ----------

def test_carga_correcta(entorno, monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "local")
    monkeypatch.setenv("PROVIDER_TIMEOUT_S", "30")
    config = cargar_config()
    assert list(config.providers) == ["mock", "local"]  # orden del YAML
    assert config.default_model == "local"
    assert config.timeout_s == 30.0
    assert config.system_prompt == "Sos un asistente de prueba. ñ"  # UTF-8 y sin espacios finales


def test_quita_barra_final(entorno):
    assert cargar_config().providers["mock"] == "http://mock:8000"


def test_defaults(entorno):
    config = cargar_config()
    assert config.default_model is None
    assert config.timeout_s == 120.0


# ---------- Errores de configuración ----------

def test_falta_clave_providers(entorno):
    escribir_registro(entorno, "otra_cosa: 1\n")
    with pytest.raises(ValueError, match="no tiene la clave 'providers'"):
        cargar_config()


def test_providers_vacio(entorno):
    escribir_registro(entorno, "providers: {}\n")
    with pytest.raises(ValueError, match="vacío"):
        cargar_config()


def test_entrada_sin_url(entorno):
    escribir_registro(entorno, "providers:\n  mock:\n    puerto: 8000\n")
    with pytest.raises(ValueError, match="'mock' necesita 'url'"):
        cargar_config()


def test_url_sin_esquema_http(entorno):
    escribir_registro(entorno, "providers:\n  mock:\n    url: mock:8000\n")
    with pytest.raises(ValueError, match="'mock' necesita 'url'"):
        cargar_config()


def test_default_model_inexistente(entorno, monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-99")
    with pytest.raises(ValueError, match="DEFAULT_MODEL='gpt-99'"):
        cargar_config()


def test_falta_archivo_de_providers(entorno, monkeypatch):
    monkeypatch.setenv("PROVIDERS_FILE", str(entorno / "no-existe.yaml"))
    with pytest.raises(ValueError, match="registro de proveedores"):
        cargar_config()


def test_falta_archivo_de_system_prompt(entorno, monkeypatch):
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(entorno / "no-existe.txt"))
    with pytest.raises(ValueError, match="system prompt"):
        cargar_config()


def test_timeout_no_numerico(entorno, monkeypatch):
    monkeypatch.setenv("PROVIDER_TIMEOUT_S", "mucho")
    with pytest.raises(ValueError, match="PROVIDER_TIMEOUT_S='mucho'"):
        cargar_config()


def test_yaml_con_sintaxis_invalida(entorno):
    escribir_registro(entorno, "providers:\n  mock: [url: http://mock:8000\n")
    with pytest.raises(ValueError, match="no es un YAML válido"):
        cargar_config()


# ---------- Casos borde de lectura ----------

def test_default_model_vacio_equivale_a_ausente(entorno, monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "")
    assert cargar_config().default_model is None


def test_system_prompt_con_bom_se_carga_sin_bom(entorno):
    # Editores de Windows (p. ej. Notepad) pueden guardar UTF-8 con BOM.
    (entorno / "prompt.txt").write_text("Prompt con BOM", encoding="utf-8-sig")
    prompt = cargar_config().system_prompt
    assert "\ufeff" not in prompt
    assert prompt == "Prompt con BOM"


# ---------- Memoria (contrato de memoria, 4.2) ----------

def test_memoria_deshabilitada_por_defecto(entorno):
    config = cargar_config()
    assert config.memory_url is None
    assert config.dev_user_id is None
    assert config.memory_history_limit == 20
    assert config.memory_timeout_s == 5.0


def test_memory_url_vacia_equivale_a_ausente(entorno, monkeypatch):
    monkeypatch.setenv("MEMORY_URL", "")
    monkeypatch.setenv("DEV_USER_ID", "no valido")  # se ignora sin memoria
    config = cargar_config()
    assert config.memory_url is None
    assert config.dev_user_id is None


def test_memoria_configurada_quita_barra_final(entorno, monkeypatch):
    monkeypatch.setenv("MEMORY_URL", "http://memory:8000/")
    monkeypatch.setenv("DEV_USER_ID", "dev-user_1")
    monkeypatch.setenv("MEMORY_HISTORY_LIMIT", "200")
    monkeypatch.setenv("MEMORY_TIMEOUT_S", "4.5")
    config = cargar_config()
    assert config.memory_url == "http://memory:8000"
    assert config.dev_user_id == "dev-user_1"
    assert config.memory_history_limit == 200
    assert config.memory_timeout_s == 4.5


def test_memory_url_sin_esquema(entorno, monkeypatch):
    monkeypatch.setenv("MEMORY_URL", "memory-interno:8000")
    monkeypatch.setenv("DEV_USER_ID", "dev-user")
    with pytest.raises(ValueError, match="MEMORY_URL debe empezar") as exc:
        cargar_config()
    assert "memory-interno" not in str(exc.value)  # el mensaje no repite la URL


@pytest.mark.parametrize("valor", [None, ""], ids=["ausente", "vacio"])
def test_memoria_sin_dev_user_id(entorno, monkeypatch, valor):
    monkeypatch.setenv("MEMORY_URL", "http://memory:8000")
    if valor is not None:
        monkeypatch.setenv("DEV_USER_ID", valor)
    with pytest.raises(ValueError, match="Falta DEV_USER_ID"):
        cargar_config()


@pytest.mark.parametrize(
    "valor",
    ["con espacio", "a/b", "x" * 65, "dev-user\n"],
    ids=["espacio", "barra", "65_chars", "salto_de_linea"],
)
def test_memoria_con_dev_user_id_invalido(entorno, monkeypatch, valor):
    monkeypatch.setenv("MEMORY_URL", "http://memory:8000")
    monkeypatch.setenv("DEV_USER_ID", valor)
    with pytest.raises(ValueError, match="DEV_USER_ID=.* no es válido"):
        cargar_config()


@pytest.mark.parametrize(
    "valor, mensaje",
    [("0", "fuera de rango"), ("201", "fuera de rango"), ("-5", "fuera de rango"),
     ("muchos", "no es un entero"), ("20.5", "no es un entero")],
    ids=["cero", "201", "negativo", "texto", "decimal"],
)
def test_history_limit_invalido(entorno, monkeypatch, valor, mensaje):
    monkeypatch.setenv("MEMORY_HISTORY_LIMIT", valor)
    with pytest.raises(ValueError, match=f"MEMORY_HISTORY_LIMIT.*{mensaje}"):
        cargar_config()


@pytest.mark.parametrize(
    "valor, mensaje",
    [("mucho", "no es un número"), ("0", "mayor que 0"), ("-1", "mayor que 0"),
     ("nan", "mayor que 0"), ("inf", "mayor que 0")],
    ids=["texto", "cero", "negativo", "nan", "inf"],
)
def test_memory_timeout_invalido(entorno, monkeypatch, valor, mensaje):
    monkeypatch.setenv("MEMORY_TIMEOUT_S", valor)
    with pytest.raises(ValueError, match=f"MEMORY_TIMEOUT_S='{valor}' .*{mensaje}"):
        cargar_config()
