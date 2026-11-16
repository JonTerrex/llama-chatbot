"""
Configuración del orquestador, leída una sola vez al arrancar (desde el lifespan).

Cualquier problema levanta ValueError: es preferible que el servicio no arranque
a que arranque con un registro de proveedores inválido y falle en cada petición.
"""
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Mismo patrón que el servicio de memoria (contrato de memoria, 3.1): validar acá evita
# descubrirlo recién en el primer pedido, como un 400 de la memoria.
# Sin anclas y con fullmatch: con "^...$" y match, "$" acepta un "\n" final.
USER_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
# 200 es el máximo de `limit` que acepta el servicio de memoria.
MEMORY_HISTORY_LIMIT_MAX = 200


@dataclass(frozen=True)
class Config:
    # nombre de modelo -> URL base del proveedor, en el orden del YAML (lo usa /v1/models).
    providers: dict[str, str]
    default_model: Optional[str]
    timeout_s: float
    system_prompt: str
    # Memoria (contrato de memoria, 4.2). Con defaults para no romper construcciones existentes.
    # memory_url None = memoria deshabilitada; en ese caso dev_user_id también es None.
    memory_url: Optional[str] = None
    dev_user_id: Optional[str] = None
    memory_history_limit: int = 20
    # Debe ser MAYOR que POOL_TIMEOUT_S de src/memory/db.py (hoy 4 s): así, con la base
    # caída, la memoria alcanza a contestar su 503. No se puede validar entre servicios.
    memory_timeout_s: float = 5.0


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

    memory_url, dev_user_id = _cargar_memoria()
    memory_history_limit = _cargar_history_limit()
    memory_timeout_s = _cargar_memory_timeout()

    return Config(
        providers=providers,
        default_model=default_model,
        timeout_s=timeout_s,
        system_prompt=system_prompt,
        memory_url=memory_url,
        dev_user_id=dev_user_id,
        memory_history_limit=memory_history_limit,
        memory_timeout_s=memory_timeout_s,
    )


def _cargar_memoria() -> tuple[Optional[str], Optional[str]]:
    """
    MEMORY_URL ausente o vacía deshabilita la memoria, y entonces DEV_USER_ID se ignora.
    El mensaje de error no repite la URL: es interna y puede terminar en un log.
    """
    memory_url = os.getenv("MEMORY_URL") or None
    if memory_url is None:
        return None, None
    if not memory_url.startswith(("http://", "https://")):
        raise ValueError("MEMORY_URL debe empezar con http:// o https://.")

    dev_user_id = os.getenv("DEV_USER_ID") or None
    if dev_user_id is None:
        raise ValueError("Falta DEV_USER_ID: es obligatoria cuando MEMORY_URL está definida.")
    if not USER_ID_PATTERN.fullmatch(dev_user_id):
        raise ValueError(
            f"DEV_USER_ID={dev_user_id!r} no es válido: 1 a 64 caracteres entre letras, dígitos, '_' y '-'."
        )
    # Sin barra final, como las URLs de providers: se concatenan rutas sin '//'.
    return memory_url.rstrip("/"), dev_user_id


def _cargar_history_limit() -> int:
    # Se valida aunque la memoria esté deshabilitada: un valor roto es un error de config igual.
    raw = os.getenv("MEMORY_HISTORY_LIMIT") or "20"
    try:
        limite = int(raw)
    except ValueError as e:
        raise ValueError(f"MEMORY_HISTORY_LIMIT={raw!r} no es un entero.") from e
    if not 1 <= limite <= MEMORY_HISTORY_LIMIT_MAX:
        raise ValueError(f"MEMORY_HISTORY_LIMIT={limite} fuera de rango (1 a {MEMORY_HISTORY_LIMIT_MAX}).")
    return limite


def _cargar_memory_timeout() -> float:
    raw = os.getenv("MEMORY_TIMEOUT_S") or "5"
    try:
        timeout = float(raw)
    except ValueError as e:
        raise ValueError(f"MEMORY_TIMEOUT_S={raw!r} no es un número.") from e
    # isfinite descarta "inf" (sin tope, justo lo que este timeout quiere evitar); "nan" ya falla en > 0.
    if not (timeout > 0 and math.isfinite(timeout)):
        raise ValueError(f"MEMORY_TIMEOUT_S={raw!r} debe ser un número mayor que 0.")
    return timeout
