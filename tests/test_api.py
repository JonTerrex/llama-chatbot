import sys
from unittest.mock import MagicMock

# 1. MOCK DE MÓDULOS DE HARDWARE EN SYS.MODULES
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["bitsandbytes"] = MagicMock()

import pytest
from fastapi.testclient import TestClient

# 2. DEFINICIÓN DE MOCKS RESPETANDO LAS FIRMAS DE PRODUCCIÓN
def mock_inicializar_modelo_y_tokenizador(path_modelo="modelos/llama3", *args, **kwargs):
    """Sustituye el Cold Start real de Llama 3."""
    return "MOCK_MODEL", "MOCK_TOKENIZER"

# CORRECCIÓN: Agregamos model y tokenizer a la firma del Mock
def mock_generar_respuesta(historial, model=None, tokenizer=None, **kwargs):
    """
    Sustituye la generación de tokens en GPU.
    Respeta la firma (historial, model, tokenizer) esperada por api.py.
    """
    return "Respuesta generada por Pytest Mock (Sin GPU)."

# 3. TEST DE INTEGRACIÓN PARA LA API REST
def test_chat_endpoint_con_mock(monkeypatch):
    # Parcheamos las funciones de src.chatbot ANTES de iniciar FastAPI
    monkeypatch.setattr("src.chatbot.inicializar_modelo_y_tokenizador", mock_inicializar_modelo_y_tokenizador)
    monkeypatch.setattr("src.chatbot.generar_respuesta", mock_generar_respuesta)
    
    # Importamos la app de FastAPI
    from src.api import app

    # Ejecutamos con TestClient en un contexto 'with' para disparar el evento lifespan
    with TestClient(app) as client:
        payload = {
            "historial": [
                {"role": "user", "content": "Hola, ¿estás funcionando?"}
            ]
        }
        
        # Petición POST al endpoint /v1/chat
        response = client.post("/v1/chat", json=payload)
        
        # Aserciones de validación
        assert response.status_code == 200
        json_data = response.json()
        assert "respuesta" in json_data
        assert json_data["respuesta"] == "Respuesta generada por Pytest Mock (Sin GPU)."