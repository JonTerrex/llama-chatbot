import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Importamos BitsAndBytesConfig para la configuración de carga en 4 bits

def generar_respuesta(prompt: str, path_modelo: str) -> str:
    """
    Carga el modelo en GPU usando CUDA, procesa el prompt y retorna la respuesta.
    prompt: Mensaje de entrada del usuario.
    path_modelo: Ruta local donde se encuentra el modelo Llama 3.
    """
    print("[INFO] Cargando tokenizador...")
    # Cargamos el tokenizador desde la ruta local del modelo
    tokenizer = AutoTokenizer.from_pretrained(path_modelo) 
    
    print("[INFO] Configurando cuantización de 4 bits...")
    # Creamos el objeto de configuración moderno que exige transformers
    configuracion_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4" # NormalFloat4, optimizado para modelos como Llama
    )
    
    print("[INFO] Cargando modelo en GPU con CUDA (bfloat16)...")
    # Cargamos con torch_dtype=torch.bfloat16 y device_map="cuda" para forzar el uso de la GPU
    model = AutoModelForCausalLM.from_pretrained(
        path_modelo, # Cargamos el modelo desde la ruta local
        dtype=torch.bfloat16, # Usamos bfloat16 para optimizar el uso de memoria
        device_map="cuda", # Asignamos automáticamente el modelo a la GPU compatible con CUDAdisponible
        quantization_config=configuracion_4bit # Aplicamos la configuración de cuantización de 4 bits para reducir el uso de memoria y mejorar la velocidad de inferencia
    )
    
    # Estructura del prompt usando la plantilla oficial de Llama 3 Instruct
    mensajes = [
        {"role": "system", "content": "Sos un asistente de IA muy conciso y profesional."},
        {"role": "user", "content": prompt}
    ]
    # Aplicar la plantilla de chat del modelo y obtener los input_ids directamente
    inputs = tokenizer.apply_chat_template(
        mensajes,
        add_generation_prompt=True, # Agrega un prompt de generación para que el modelo sepa que debe responder
        return_tensors="pt", # Convertimos la entrada a tensores de PyTorch
        return_dict=True  # Nos asegura el formato correcto estructurado
    )
    
    # Extraemos explícitamente los ID de los tokens y los enviamos a la GPU
    input_ids = inputs["input_ids"].to("cuda") # Enviamos los tensores de entrada a la GPU
    
    print("[INFO] Generando respuesta...")
    # Configuración de generación optimizada para producción
    outputs = model.generate(
        input_ids,
        max_new_tokens=256, # Limita la cantidad de tokens generados para evitar respuestas demasiado largas
        do_sample=True, # Permite la generación de texto con cierta aleatoriedad
        temperature=0.6, # Controla la aleatoriedad de la generación
        top_p=0.9, # Filtra los tokens más probables para mantener coherencia
        eos_token_id=tokenizer.eos_token_id # Indica al modelo cuándo detener la generación
    )
    
    # Decodificar solo los tokens nuevos generados por el modelo
    respuesta_tokens = outputs[0][input_ids.shape[-1]:]
    respuesta_texto = tokenizer.decode(respuesta_tokens, skip_special_tokens=True)
    
    return respuesta_texto

def main():
    # Configuración del parser de argumentos para recibir el prompt desde la línea de comandos
    parser = argparse.ArgumentParser(
        description="Chatbot profesional utilizando Meta-Llama-3-8B-Instruct local con soporte CUDA."
    )
    # Agregamos un argumento obligatorio para el prompt
    parser.add_argument(
        "prompt", 
        type=str, 
        help="El mensaje o pregunta que le vas a enviar al modelo Llama 3."
    )
    args = parser.parse_args()
    
    # Ruta local donde descargamos el modelo en el paso anterior
    PATH_MODELO = "modelos/llama3"
    
    try:
        # Generamos la respuesta del modelo usando el prompt proporcionado
        resultado = generar_respuesta(args.prompt, PATH_MODELO) 
        print("\n=== RESPUESTA DEL CHATBOT ===")
        print(resultado)
        print("=============================\n")
    except Exception as e:
        import traceback
        print("\n[ERROR] Ocurrió un fallo en la ejecución:")
        # Imprimimos el mensaje de error y el traceback completo para facilitar la depuración
        traceback.print_exc() # Imprimimos el traceback completo del error para facilitar la depuración
    
if __name__ == "__main__":
    main()