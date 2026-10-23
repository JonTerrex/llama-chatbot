import pytest
import requests

# 1. Definimos una clase Mock para simular el objeto de respuesta de la librería 'requests'
class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

# 2. Test unitario para verificar la lógica de red del cliente (UI / CLI)
def test_cliente_peticion_http(monkeypatch):
    
    # 3. Función que reemplazará a requests.post durante la prueba
    def mock_post(url, json, timeout):
        # Validamos que el cliente esté llamando a la URL y con los parámetros correctos
        assert url == "http://127.0.0.1:8000/v1/chat"
        assert "historial" in json
        assert timeout == 60
        
        # Devolvemos un HTTP 200 OK simulado con el formato JSON esperado
        return MockResponse({"respuesta": "Respuesta interceptada y simulada para el cliente."}, 200)
    
    # 4. Parcheamos requests.post globalmente
    monkeypatch.setattr(requests, "post", mock_post)
    
    # 5. Simulamos la ejecución exacta que ocurre en app_ui.py (Línea 43-44)
    payload = {"historial": [{"role": "user", "content": "Mensaje de prueba"}]}
    API_URL = "http://127.0.0.1:8000/v1/chat"
    
    response = requests.post(API_URL, json=payload, timeout=60)
    
    # 6. Aserciones para asegurar que el cliente procesa la respuesta adecuadamente
    assert response.status_code == 200
    respuesta_texto = response.json().get("respuesta", "")
    assert respuesta_texto == "Respuesta interceptada y simulada para el cliente."