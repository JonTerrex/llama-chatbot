import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Importamos BitsAndBytesConfig para la configuración de carga en 4 bits


def cargar_system_prompt(nombre_archivo: str = "system_prompt.txt") -> str:
    """
    Carga el System Prompt desde un archivo externo .txt ubicado en la carpeta /prompts.
    Utiliza rutas absolutas dinámicas basadas en la ubicación de chatbot.py.
    """
    # 1. Determinamos la raíz del proyecto subiendo dos niveles desde src/chatbot.py
    RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
    
    # 2. Construimos la ruta absoluta hacia el archivo del prompt
    ruta_prompt = RAIZ_PROYECTO / "prompts" / nombre_archivo
    
    # 3. Intentamos leer el archivo con codificación UTF-8
    try:
        with open(ruta_prompt, "r", encoding="utf-8") as f:
            contenido = f.read().strip()
            
        if not contenido:
            print(f"[ADVERTENCIA] El archivo '{nombre_archivo}' está vacío. Usando prompt por defecto.")
            return "Sos un asistente de IA muy conciso, profesional y servicial."
            
        print(f"[INFO] System Prompt cargado exitosamente desde: {ruta_prompt.name}")
        return contenido

    except FileNotFoundError:
        print(f"[ADVERTENCIA] No se encontró el archivo en {ruta_prompt}. Usando prompt por defecto.")
        # Fallback de seguridad en caso de que el archivo no exista
        return "Sos un asistente de IA muy conciso, profesional y servicial."
    

def inicializar_modelo_y_tokenizador(path_modelo: str):
    """
    Inicializa el hardware de Nvidia CUDA y carga el modelo Llama 3 cuantizado a 4 bits.
    Se ejecuta una sola vez al arrancar la aplicación (Cold Start).
    """
    print("[INFO] Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(path_modelo)

    print("[INFO] Configurando cuantización de 4 bits...")
    configuracion_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    print("[INFO] Cargando modelo en GPU con CUDA (4 bits)...")
    model = AutoModelForCausalLM.from_pretrained(
        path_modelo,
        quantization_config=configuracion_4bit,
        device_map="cuda"
    )
    
    # Devolvemos los dos objetos cargados en la memoria de la GPU
    return model, tokenizer


def generar_respuesta(historial: list, model, tokenizer) -> str:
    """
    Genera una respuesta del modelo Llama 3 Instruct basado en el historial de conversación.
    
    """
    # 1. Aplicamos la plantilla de chat del modelo y obtenemos los input_ids directamente
    inputs = tokenizer.apply_chat_template(
        historial,
        add_generation_prompt=True, # Agrega un prompt de generación para que el modelo sepa que debe responder
        return_tensors="pt", # Convertimos la entrada a tensores de PyTorch
        return_dict=True  # Nos asegura el formato correcto estructurado
    )
    #2. Enviamos el diccionario de tensores a la GPU para acelerar la inferencia
    inputs = {k: v.to("cuda") for k, v in inputs.items()} 

    print("[INFO] Generando respuesta...")
    # 3. Configuramos de forma segura el token_pad_id para evitar advertencias de padding en la generación
    #tokenizer.pad_token_id = tokenizer.eos_token_id    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # 4. Pasamos todos los tensores al modelo para generar la respuesta
    outputs = model.generate(
        **inputs, # desempaquetamos los inputs y attention mask juntos
        max_new_tokens=256, # Limita la cantidad de tokens generados para evitar respuestas demasiado largas
        do_sample=True, # Permite la generación de texto con cierta aleatoriedad
        temperature=0.6, # Controla la aleatoriedad de la generación
        top_p=0.9, # Filtra los tokens más probables para mantener coherencia
        eos_token_id=tokenizer.eos_token_id, # Indica al modelo cuándo detener la generación
        pad_token_id=pad_token_id # silencia la advertencia de padding y evita errores de longitud
    )
    
    # 5. Decodificar solo los tokens nuevos generados por el modelo
    input_length = inputs["input_ids"].shape[-1]
    respuesta_tokens = outputs[0][input_length:]
    respuesta_texto = tokenizer.decode(respuesta_tokens, skip_special_tokens=True)
    
    return respuesta_texto

def main():
    """
    Función principal que inicia el chatbot, carga el modelo y el tokenizador, y gestiona la interacción con el usuario.
    """
    
    PATH_MODELO = "modelos/llama3"# Ruta local donde descargamos el modelo en el paso anterior
    try:
        """
        # Generamos la respuesta del modelo usando el prompt proporcionado
        # y el historial de conversación acumulado.
        """
        # 1. Carga del modelo (Cold Start)
        model, tokenizer = inicializar_modelo_y_tokenizador(PATH_MODELO)
        # Inicializamos el historial con las directivas del sistema (System Prompt)
        
        # 2. Cargamos el System Prompt dinámicamente desde el archivo externo .txt
        system_prompt = cargar_system_prompt("system_prompt.txt")
        print("\n🤖 ¡Nueva personalidad cargada! \n")

        # 3. inicializamos el historial de conversación con el System Prompt
        historial_conversacion = [
                {"role": "system", "content": system_prompt}
        ]
              
        print("\n🤖 ¡Chatbot Activo! Escribí 'salir' para finalizar.\n")
        
        # 4. Iniciamos el bucle interactivo continuo
        while True:
            
            # 5. Capturamos la entrada del usuario
            user_input = input("Tú > ")
            
            # 6. Verificamos si el usuario quiere salir del chat
            if user_input.lower() in ['salir', 'exit', 'chau']:
                print("🤖 ¡Hasta luego!")
                break
                
            # 7. Ignoramos entradas vacías para evitar errores en la generación de respuestas
            if not user_input.strip():
                continue
            
            # 8. Guardamos la entrada del usuario con rol 'user' para la proxima generación de respuesta
            historial_conversacion.append({"role": "user", "content": user_input})
            
            # 9. Intentamos generar la respuesta del modelo y manejar cualquier excepción que pueda ocurrir
            try:
                # 10. Generamos la respuesta del modelo usando el historial de conversación acumulado
                respuesta = generar_respuesta(historial_conversacion, model, tokenizer)
                
                # 11. Mostramos la respuesta generada por el modelo en la consola
                print(f"\nLlama 3 > {respuesta}\n")
                
                # 12. CRUCIAL: Guardamos la respuesta del bot con rol 'assistant' para el próximo turno
                historial_conversacion.append({"role": "assistant", "content": respuesta})
            
            # 13. Capturamos cualquier excepción que ocurra durante la generación de la respuesta
            except Exception as e:
                import traceback
                print("\n[ERROR] Ocurrió un fallo al generar la respuesta:")
                traceback.print_exc()
    # 14.Capturamos cualquier excepción que ocurra durante la ejecución del programa
    except Exception as e:
        import traceback
        print("\n[ERROR] Ocurrió un fallo en la ejecución:")
        traceback.print_exc() # Imprimimos el traceback completo del error para facilitar la depuración
    
if __name__ == "__main__":
    main()