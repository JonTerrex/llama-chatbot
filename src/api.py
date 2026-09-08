from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Importamos directamente desde el módulo chatbot para mantener la consistencia y evitar dependencias circulares
from src.chatbot import cargar_system_prompt, inicializar_modelo_y_tokenizador, generar_respuesta

# Estructura del objeto global de estado para el modelo
ml_models: Dict[str, Any] = {}


@asynccontextmanager

async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.
    Carga el modelo y tokenizador en VRAM al arrancar (Cold Start)
    y libera recursos al apagar el servidor.
    """
    # 1. Mensaje informativo al iniciar el servidor
    print("\n[INFO] Iniciando servidor API... Cargando modelo Llama 3 en GPU...")
    path_modelo = "modelos/llama3"
    
    # 2. Carga única en VRAM
    model, tokenizer = inicializar_modelo_y_tokenizador(path_modelo)
    system_prompt = cargar_system_prompt("system_prompt.txt")
    
    ml_models["model"] = model
    ml_models["tokenizer"] = tokenizer
    ml_models["system_prompt"] = system_prompt
    
    print("[INFO] Modelo cargado exitosamente. Servidor listo para recibir peticiones.\n")
    yield
    
    # Limpieza al apagar el servidor
    ml_models.clear()
    print("[INFO] Servidor detenido y recursos liberados.")

# Inicializamos la aplicación FastAPI con metadatos y el contexto de vida definido
app = FastAPI(
    title="Llama 3 Chatbot API",
    description="API REST para inferencia interactiva con Llama 3 cuantizado a 4 bits.",
    version="1.0.0",
    lifespan=lifespan
)


# Modelos Pydantic para validación de datos
class ChatRequest(BaseModel):
    historial: List[Dict[str, str]] = Field(
        ..., 
        description="Historial de conversación con roles ('user', 'assistant', 'system')."
    )


class ChatResponse(BaseModel):
    respuesta: str = Field(..., description="Respuesta generada por el modelo Llama 3.")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Endpoint de monitoreo para verificar que la API está activa."""
    return {"status": "ok", "model_loaded": "model" in ml_models}


@app.post("/v1/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal para generar respuestas de chat utilizando el modelo cargado.
    """
    if "model" not in ml_models or "tokenizer" not in ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado en la memoria de la GPU."
        )
    
    try:
        # Si el historial enviado no incluye system prompt, le inyectamos el prompt por defecto
        historial_completo = request.historial
        if not any(msg.get("role") == "system" for msg in historial_completo):
            historial_completo.insert(0, {"role": "system", "content": ml_models["system_prompt"]})

        respuesta_texto = generar_respuesta(
            historial=historial_completo,
            model=ml_models["model"],
            tokenizer=ml_models["tokenizer"]
        )
        
        return ChatResponse(respuesta=respuesta_texto)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia del modelo: {str(e)}"
        )