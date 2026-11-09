"""
Configuración del orquestador, leída una sola vez al arrancar (desde el lifespan).

Cualquier problema levanta ValueError: es preferible que el servicio no arranque
a que arranque con un registro de proveedores inválido y falle en cada petición.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Config:
    # nombre de modelo -> URL base del proveedor, en el orden del YAML (lo usa /v1/models).
    providers: dict[str, str]
    default_model: Optional[str]
    timeout_s: float
    system_prompt: str


def _leer_archivo(ruta: Path, descripcion: str) -> str:
    if not ruta.is_file():
        raise ValueError(f"No se encontró el archivo de {descripcion}: '{ruta}'.")
    return ruta.read_text(encoding="utf-8-sig")


def _cargar_providers(ruta: Path) -> dict[str, str]:
    """Valida la forma del registro y normaliza las URLs (sin barra final, para concatenar rutas sin '//')."""
    try:
        datos = yaml.safe_load(_leer_archivo(ruta, "registro de proveedores"))
    except yaml.YAMLError as e:
        raise ValueError(f"'{ruta}' no es un YAML válido: {e}") from e

    if not isinstance(datos, dict) or "providers" not in datos:
        raise ValueError(f"'{ruta}' no tiene la clave 'providers'.")

    entradas = datos["providers"]
    if not isinstance(entradas, dict) or not entradas:
        raise ValueError(f"'{ruta}': 'providers' está vacío o no es un mapa nombre -> {{url}}.")

    providers: dict[str, str] = {}
    for nombre, entrada in entradas.items():
        url = entrada.get("url") if isinstance(entrada, dict) else None
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(
                f"'{ruta}': el proveedor '{nombre}' necesita 'url' que empiece con http:// o https://."
            )
        providers[str(nombre)] = url.rstrip("/")
    return providers


def cargar_config() -> Config:
    """Lee el entorno en el momento de la llamada (no al importar), para que los tests usen monkeypatch.setenv."""
    providers_file = Path(os.getenv("PROVIDERS_FILE", "providers.yaml"))
    prompt_file = Path(os.getenv("SYSTEM_PROMPT_FILE", "prompts/system_prompt.txt"))

    providers = _cargar_providers(providers_file)

    # Vacío cuenta como ausente: en compose es común dejar "DEFAULT_MODEL=" sin valor.
    default_model = os.getenv("DEFAULT_MODEL") or None
    if default_model is not None and default_model not in providers:
        raise ValueError(
            f"DEFAULT_MODEL='{default_model}' no existe en '{providers_file}'. "
            f"Disponibles: {', '.join(providers)}."
        )

    timeout_raw = os.getenv("PROVIDER_TIMEOUT_S", "120")
    try:
        timeout_s = float(timeout_raw)
    except ValueError as e:
        raise ValueError(f"PROVIDER_TIMEOUT_S='{timeout_raw}' no es un número.") from e

    system_prompt = _leer_archivo(prompt_file, "system prompt").rstrip()

    return Config(
        providers=providers,
        default_model=default_model,
        timeout_s=timeout_s,
        system_prompt=system_prompt,
    )
