"""
Backend simulado (mock) del LLM.

Respeta las mismas firmas que src/chatbot.py para que la API pueda
intercambiar backends sin cambios:
    - inicializar_modelo_y_tokenizador(path_modelo) -> (model, tokenizer)
    - generar_respuesta(historial, model, tokenizer) -> str

No importa torch ni transformers: permite levantar todo el stack
(API + UI + Docker) en máquinas sin GPU y en CI.
"""
import os
import time

MOCK_MODEL = "MOCK_MODEL"
MOCK_TOKENIZER = "MOCK_TOKENIZER"


def inicializar_modelo_y_tokenizador(path_modelo: str = "modelos/llama3"):
    """Simula el Cold Start sin cargar pesos en memoria."""
    print(f"[INFO] Modo MOCK activo: se omite la carga de '{path_modelo}'.")
    return MOCK_MODEL, MOCK_TOKENIZER


def generar_respuesta(historial: list, model=None, tokenizer=None, metricas: dict | None = None) -> str:
    """
    Devuelve una respuesta en Markdown (con negritas) a partir del último
    mensaje del usuario. Opcionalmente simula latencia con MOCK_LATENCY (segundos).
    """
    latencia = float(os.getenv("MOCK_LATENCY", "0"))
    if latencia > 0:
        time.sleep(latencia)

    ultimo_usuario = next(
        (m.get("content", "") for m in reversed(historial) if m.get("role") == "user"),
        "",
    )
    turnos = sum(1 for m in historial if m.get("role") == "user")

    return (
        "**[MOCK]** Respuesta simulada, sin LLM real.\n\n"
        f"Recibí tu mensaje: **{ultimo_usuario}**\n\n"
        f"Mensajes del usuario en el historial: **{turnos}**"
    )
