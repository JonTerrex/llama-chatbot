import argparse
import traceback
from pathlib import Path
from src.chatbot import cargar_system_prompt, inicializar_modelo_y_tokenizador, generar_respuesta

def parsear_argumentos():
    """
    Configura y procesa los argumentos pasados por la línea de comandos (CLI).
    """
    #1. Creamos un parser de argumentos con una descripción del programa
    parser = argparse.ArgumentParser(
        description="Interfaz de línea de comandos para interactuar con Llama 3."
    )
    # 2. Definimos los argumentos que el usuario puede pasar al script
    parser.add_argument(
        "--path-modelo",
        type=str,
        default="modelos/llama3",
        help="Ruta local o identificador del modelo LLM (default: 'modelos/llama3')"
    )
    # 3. Definimos un argumento opcional para especificar el archivo del System Prompt
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="system_prompt.txt",
        help="Nombre del archivo .txt con el System Prompt en la carpeta /prompts"
    )
    # 4. Retornamos los argumentos parseados para su uso en la función main()
    return parser.parse_args()


def main():
    """
    Función principal que inicia el chatbot, carga el modelo y el tokenizador, y gestiona la interacción con el usuario.
    """
    # 1. Parseamos los argumentos de la línea de comandos
    args = parsear_argumentos()

    # 2. Intentamos cargar el modelo y el System Prompt, manejando cualquier excepción que pueda ocurrir
    try:
        model, tokenizer = inicializar_modelo_y_tokenizador(args.path_modelo)
        system_prompt = cargar_system_prompt(args.prompt_file)
        print("\n[INFO] System prompt cargado exitosamente.\n")
        
        # 3. Inicializamos el historial de conversación con el System Prompt
        historial_conversacion = [
                        {"role": "system", "content": system_prompt}
                ]
        print("[INFO] ¡Chatbot Activo! Escribí 'salir' para finalizar.\n")

        # 4. Iniciamos un bucle interactivo para recibir entradas del usuario y generar respuestas del modelo
        while True:
            user_input = input("Tú > ")
            # 5. Verificamos si el usuario quiere salir del chat
            if user_input.lower() in ['salir', 'exit', 'chau']:
                print("[INFO] ¡Hasta luego!")
                break

            # 6. Ignoramos entradas vacías para evitar errores en la generación de respuestas    
            if not user_input.strip():
                continue

            # 7. Guardamos la entrada del usuario con rol 'user' para la próxima generación de respuesta
            historial_conversacion.append({"role": "user", "content": user_input})

            # 8. Intentamos generar la respuesta del modelo y manejar cualquier excepción que pueda ocurrir
            try:
                respuesta = generar_respuesta(historial_conversacion, model, tokenizer)
                print(f"\nLlama 3 > {respuesta}\n")
                historial_conversacion.append({"role": "assistant", "content": respuesta})
            
            except Exception as e:
                print(f"\n[ERROR] Ocurrió un fallo al generar la respuesta: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un fallo crítico en la ejecución: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()