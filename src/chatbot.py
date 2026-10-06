import torch
import psutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Importamos BitsAndBytesConfig para la configuración de carga en 4 bits
from typing import TypedDict, Optional


class SystemCapabilities(TypedDict):
    has_gpu: bool
    gpu_name: Optional[str]
    vram_gb: float
    ram_available_gb: float


def get_system_capabilities() -> SystemCapabilities:
    """
    Inspecciona el hardware del sistema (GPU/VRAM y RAM) para determinar
    el presupuesto de memoria disponible antes de cargar un modelo.
    """
    ram_bytes = psutil.virtual_memory().available
    ram_gb = round(ram_bytes / (1024 ** 3), 2)
    
    has_gpu = torch.cuda.is_available()
    gpu_name: Optional[str] = None
    vram_gb: float = 0.0

    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        free_vram_bytes, _ = torch.cuda.mem_get_info(0)
        vram_gb = round(free_vram_bytes / (1024 ** 3), 2)

    return {
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "ram_available_gb": ram_gb
    }

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
        # 4. Retornamos el contenido del System Prompt
        return contenido

    except FileNotFoundError:
        print(f"[ADVERTENCIA] No se encontró el archivo en {ruta_prompt}. Usando prompt por defecto.")
        # Fallback de seguridad en caso de que el archivo no exista
        return "Sos un asistente de IA muy conciso, profesional y servicial."
    

def inicializar_modelo_y_tokenizador(path_modelo: str):
    """
    Inicializa el modelo evaluando dinámicamente las capacidades del sistema.
    Aplica cuantización en GPU si hay VRAM suficiente, o fallback a CPU.
    Se ejecuta una sola vez al arrancar la aplicación (Cold Start).
    """
    #1. verificamos las capacidades del sistema antes de cargar el modelo
    caps = get_system_capabilities()
    print(f"\n[INFO] Evaluando capacidades del sistema para el Cold Start:")
    print(f"       - RAM disponible: {caps['ram_available_gb']} GB")
    print(f"       - GPU detectada: {caps['has_gpu']} ({caps['gpu_name'] or 'N/A'})")
    # 2. verificamos si hay GPU y mostramos la VRAM disponible
    if caps['has_gpu']:
        print(f"       - VRAM libre: {caps['vram_gb']} GB")
    # 3. guardrail de seguridad para prevenir que el modelo se cargue en sistemas con recursos insuficientes
    MIN_RAM_REQUIRED_GB = 8.0
    if not caps['has_gpu'] and caps['ram_available_gb'] < MIN_RAM_REQUIRED_GB:
        raise RuntimeError(
            f"[ERROR CRÍTICO] Recursos insuficientes. Se requieren al menos {MIN_RAM_REQUIRED_GB} GB "
            f"de RAM libre para ejecutar en CPU, pero solo se detectaron {caps['ram_available_gb']} GB."
        )
    
    # 4. cargamos el tokenizador desde la ruta del modelo, independientemente de la estrategia de carga
    print(f"[INFO] Cargando tokenizador desde: {path_modelo}")
    tokenizer = AutoTokenizer.from_pretrained(path_modelo)

    # 5. Decisión de estrategia de carga, dependiendo de la disponibilidad de GPU y VRAM
    if caps['has_gpu'] and caps['vram_gb'] >= 5.0:
        print(f"[INFO] Arquitectura de la GPU: {torch.cuda.get_device_properties(0).name}, Memoria total: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        # Estrategia de carga en GPU con cuantización a 4 bits (NF4)
        print("[INFO] Estrategia seleccionada: CUDA GPU con cuantización a 4 bits (NF4).")
        # Mostramos la versión de PyTorch y CUDA para depuración
        print(f"[INFO] Versión de PyTorch: {torch.__version__}, Versión de CUDA: {torch.version.cuda}")
        configuracion_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        # 6. Cargamos el modelo en GPU con la configuración de cuantización a 4 bits
        model = AutoModelForCausalLM.from_pretrained(
            path_modelo,
            quantization_config=configuracion_4bit,
            device_map="cuda"
        )
    else:
        # 7. Estrategia de fallback a CPU con precisión Float32
        print("[ADVERTENCIA] GPU no disponible o VRAM insuficiente. Ejecutando fallback en CPU (Float32)...")
        model = AutoModelForCausalLM.from_pretrained(
            path_modelo,
            dtype=torch.float32,
            device_map="cpu"
        )

    # 8. Devolvemos los dos objetos cargados en la memoria
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
    # 2. Enviamos el diccionario de tensores al mismo dispositivo donde reside el modelo (cuda o cpu)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

