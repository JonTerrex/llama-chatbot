import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Importamos BitsAndBytesConfig para la configuración de carga en 4 bits

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

    # Ruta local donde descargamos el modelo en el paso anterior
    PATH_MODELO = "modelos/llama3"
    model, tokenizer = inicializar_modelo_y_tokenizador(PATH_MODELO)
    try:
        """
        # Generamos la respuesta del modelo usando el prompt proporcionado
        resultado = generar_respuesta(args.prompt, model, tokenizer)
        print("\n=== RESPUESTA DEL CHATBOT ===")
        print(resultado)
        print("=============================\n")"""
        
        # Inicializamos el historial con las directivas del sistema (System Prompt)
        historial_conversacion = [
            {"role": "system", "content": "Sos un asistente de IA muy conciso, profesional y servicial."}
        ]
        
        print("\n🤖 ¡Chatbot Activo! Escribí 'salir' para finalizar.\n")
        
        # Iniciamos el bucle interactivo continuo
        while True:
            # Capturamos lo que el usuario escribe en tiempo real
            user_input = input("Tú > ")
            
            # Condición de quiebre: si escribe 'salir', romper el bucle
            if user_input.lower() in ['salir', 'exit', 'chau']:
                print("🤖 ¡Hasta luego!")
                break
                
            # Si el usuario le da al Enter sin escribir nada, salteamos la vuelta
            if not user_input.strip():
                continue
            # Agregamos el nuevo mensaje del usuario al historial acumulativo
            historial_conversacion.append({"role": "user", "content": user_input})
            
            # Modificamos temporalmente el eco para verificar que la memoria acumula bien
            print(f"Historial actual (Tokens guardados): {len(historial_conversacion)} mensajes en memoria.\n")    
            # Imprimimos un eco temporal para verificar que el bucle está vivo
            print(f"Llama 3 (Prueba) > Recibí tu mensaje: '{user_input}'\n")

            # Llamamos a la función para generar la respuesta del modelo
            try:
                # 2. Llamamos a la función enviándole el historial y la ruta del modelo
                respuesta = generar_respuesta(historial_conversacion, model, tokenizer)
                
                # 3. Imprimimos la respuesta oficial en la pantalla
                print(f"\nLlama 3 > {respuesta}\n")
                
                # 4. CRUCIAL: Guardamos la respuesta del bot con rol 'assistant' para el próximo turno
                historial_conversacion.append({"role": "assistant", "content": respuesta})
            # Capturamos cualquier excepción que ocurra durante la generación de la respuesta
            except Exception as e:
                import traceback
                print("\n[ERROR] Ocurrió un fallo al generar la respuesta:")
                traceback.print_exc()
    # Capturamos cualquier excepción que ocurra durante la ejecución del programa
    except Exception as e:
        import traceback
        print("\n[ERROR] Ocurrió un fallo en la ejecución:")
        # Imprimimos el mensaje de error y el traceback completo para facilitar la depuración
        traceback.print_exc() # Imprimimos el traceback completo del error para facilitar la depuración
    
if __name__ == "__main__":
    main()