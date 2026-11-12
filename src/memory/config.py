"""
Configuración del servicio de memoria, leída una sola vez al arrancar (desde el lifespan).

Cualquier problema levanta ValueError: es preferible que el servicio no arranque
a que arranque sin base de datos y falle en cada petición.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Contiene la contraseña: nunca incluirla en mensajes de error ni en logs.
    database_url: str
    pool_max_size: int


def cargar_config() -> Config:
    """Lee el entorno en el momento de la llamada (no al importar), para que los tests usen monkeypatch.setenv."""
    # Vacío cuenta como ausente: en compose es común dejar "DATABASE_URL=" sin valor.
    database_url = os.getenv("DATABASE_URL") or None
    if database_url is None:
        raise ValueError("Falta la variable de entorno DATABASE_URL (obligatoria).")

    pool_raw = os.getenv("DB_POOL_MAX_SIZE", "5")
    try:
        pool_max_size = int(pool_raw)
    except ValueError as e:
        raise ValueError(f"DB_POOL_MAX_SIZE='{pool_raw}' no es un entero.") from e
    if pool_max_size < 1:
        raise ValueError(f"DB_POOL_MAX_SIZE='{pool_raw}' tiene que ser al menos 1.")

    return Config(database_url=database_url, pool_max_size=pool_max_size)
