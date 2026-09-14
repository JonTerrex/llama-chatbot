import argparse
import requests
import traceback


def parsear_argumentos():
    """
    Configura y procesa los argumentos pasados por la línea de comandos (CLI).
    """
    #1. Creamos un parser de argumentos con una descripción del programa
    parser = argparse.ArgumentParser(
        description="Interfaz CLI que consume la API REST de Llama 3."
    )
   
    # 2. Reemplazamos --path-modelo por --api-url
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/v1/chat",
        help="URL del endpoint de chat de la API REST (default: 'http://127.0.0.1:8000/v1/chat')"
    )
    
    # 3. Retornamos los argumentos parseados para su uso en la función main()
    return parser.parse_args()

def enviar_peticion_chat(api_url: str, historial: list) -> str:
    """
    Encapsula la comunicación HTTP POST con la API REST de FastAPI.
    """
    # 1. Preparamos el payload con el historial de conversación
    payload = {"historial": historial}
    # 2. Enviamos la solicitud POST a la API REST con un tiempo de espera de 120 segundos
    response = requests.post(api_url, json=payload, timeout=120)
    
    # Lanza una excepción si el código de estado es HTTP 4xx o 5xx
    response.raise_for_status()
    # 3. Extraemos la respuesta JSON y retornamos el campo 'respuesta'
    data = response.json()
    # Retornamos la respuesta generada por el modelo Llama 3
    return data.get("respuesta", "")

def main():
    """
    Bucle principal de interacción por terminal consumiendo el backend web.
    """
    # 1. Parseamos los argumentos de la línea de comandos
    args = parsear_argumentos()

    # 2. Mostramos un mensaje informativo al iniciar la interfaz CLI
    print(f"\n[INFO] Conectando cliente CLI a la API en: {args.api_url}")
    print("[INFO] ¡Chatbot Activo (vía API REST)! Escribí 'salir' para finalizar.\n")
    # 3. Inicializamos el historial de conversación vacío
    historial_conversacion = []

    # 4. Iniciamos un bucle interactivo para recibir entradas del usuario y generar respuestas del modelo
    while True:
        # 5. Capturamos la entrada del usuario mediante input()
        try:
            user_input = input("Tú > ")
            
            if user_input.lower() in ['salir', 'exit', 'chau']:
                print("[INFO] ¡Hasta luego!")
                break
                
            if not user_input.strip():
                continue
            
            historial_conversacion.append({"role": "user", "content": user_input})

            # 6. Intentamos enviar la solicitud a la API REST y manejar posibles errores de conexión o HTTP
            try:
                # Llamada modular a la función que consume la API REST
                respuesta = enviar_peticion_chat(args.api_url, historial_conversacion)
                print(f"\nLlama 3 > {respuesta}\n")
                # Guardamos la respuesta del asistente en el historial
                historial_conversacion.append({"role": "assistant", "content": respuesta})
            
            except requests.exceptions.ConnectionError:
                print("\n[ERROR] No se pudo conectar a la API. ¿Asegurate de que Uvicorn esté corriendo en el puerto 8000?\n")
                # Removemos el último mensaje del usuario para no corromper el historial
                historial_conversacion.pop()
            except requests.exceptions.HTTPError as http_err:
                print(f"\n[ERROR HTTP] El servidor devolvió un error: {http_err}\n")
                historial_conversacion.pop()
            except Exception as e:
                print(f"\n[ERROR] Error inesperado en la llamada: {e}\n")
                historial_conversacion.pop()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupción detectada. Saliendo...")
            break
        except Exception as e:
            print(f"\n[ERROR CRÍTICO] Fallo en el bucle principal: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()