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
