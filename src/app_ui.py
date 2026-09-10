import streamlit as st
import requests

# 1. Configuración principal de la página
st.set_page_config(
    page_title="Llama 3 Chatbot UI",
    page_icon="🤖",
    layout="centered"
)
 # 2. Título y descripción de la aplicación
st.title("🤖 Llama 3 Interactive Assistant")
st.caption("Interfaz conectada a la API REST de FastAPI con cuantización a 4 bits.")

# URL de la API de FastAPI y configuración de tiempo de espera para las solicitudes HTTP
API_URL = "http://127.0.0.1:8000/v1/chat"
DEFAULT_TIMEOUT = 120  # Tiempo de espera por defecto para la respuesta de la API en segundos

# --- PANEL LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Configuración del Bot")
    st.info("Modelo: **Llama 3 (4-bit quantization)**")
    
    # Botón para reiniciar la conversación
    if st.button("🗑️ Limpiar Historial", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 2. Inicializamos el historial en la memoria de la sesión gráfica de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Renderizamos los mensajes guardados en st.session_state en cada re-ejecución
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 3. Capturamos la entrada del usuario mediante el widget de chat
if prompt := st.chat_input("Escribí tu mensaje aquí..."):
    
    # Mostramos el mensaje del usuario inmediatamente en la UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Agregamos el mensaje del usuario al historial local de la sesión
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Enviamos el historial completo a la API REST de FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Pensando respuesta en GPU..."):
            try:
                payload = {"historial": st.session_state.messages}
                response = requests.post(API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    respuesta_texto = response.json().get("respuesta", "")
                    st.markdown(respuesta_texto)
                    # Guardamos la respuesta del asistente en el historial
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
                else:
                    error_msg = f"Error HTTP {response.status_code}: {response.text}"
                    st.error(error_msg)

            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API de FastAPI. ¿Está el servidor corriendo en http://127.0.0.1:8000?")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {str(e)}")