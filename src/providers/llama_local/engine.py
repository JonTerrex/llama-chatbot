"""
Motor de inferencia del proveedor Llama local: carga en GPU (4 bits NF4) y generación.

Sin FastAPI, para poder probar sus reglas sin servidor. torch, transformers y
bitsandbytes se importan solo dentro de MotorLlama (imports diferidos): importar este
módulo no requiere torch, y las funciones puras se prueban sin GPU.

Autocontenido como el mock: la lógica de carga y generación es una copia adaptada de
src/chatbot.py, sin importarla, para que la imagen del proveedor no dependa del resto de src/.
"""
import gc
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("llama_local")

TEMPERATURE_DEFAULT = 0.6
MAX_TOKENS_DEFAULT = 256
# Mismo umbral que chatbot.py: por debajo, Llama 3 8B en 4 bits no entra con margen.
VRAM_MIN_GB = 5.0

MIB = 1024 ** 2
# VRAM que se deja sin usar, para Windows y para la fragmentación del allocator de PyTorch.
MARGEN_VRAM_MB = 256
# El KV cache de Llama 3 8B en bf16 cuesta ~0,125 MiB por token (fórmula de bytes_por_token sin
# factor); el pico medido en vivo fue ~0,2275 MiB por token de entrada (activaciones y buffers
# temporales además del KV). El factor 2 lleva la estimación a ~0,25 MiB: margen sobre lo medido.
FACTOR_OVERHEAD_KV = 2.0


@dataclass
class Resultado:
    texto: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length"
    generation_s: float
    vram_peak_mb: Optional[float] = None
    vram_reserved_mb: Optional[float] = None


class ContextoExcedido(Exception):
    """El prompt más max_tokens no entra en el contexto del modelo. Solo lleva números, nunca contenido."""

    def __init__(self, prompt_tokens: int, limite: int, max_new_tokens: Optional[int] = None):
        self.prompt_tokens = prompt_tokens
        self.limite = limite
        self.max_new_tokens = max_new_tokens
        super().__init__(f"prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens}, limite={limite}")


class SinGPU(Exception):
    """No hay CUDA. A diferencia de chatbot.py no hay fallback a CPU: 8B en float32 no es usable."""


class VRAMInsuficiente(Exception):
    """La VRAM disponible ahora no alcanza para este pedido (p. ej. otro proceso la ocupó). Solo números."""

    def __init__(self, necesaria_mb: float, disponible_mb: float):
        self.necesaria_mb = necesaria_mb
        self.disponible_mb = disponible_mb
        super().__init__(f"necesaria_mb={necesaria_mb:.1f}, disponible_mb={disponible_mb:.1f}")


# ---------- Reglas puras (testeables sin torch) ----------

def resolver_parametros(temperature: Optional[float], max_tokens: Optional[int]) -> tuple[bool, float, int]:
    """
    (do_sample, temperature, max_new_tokens). Sin valores en la petición se usan los de
    chatbot.py (0.6 y 256). temperature 0 significa greedy: transformers no admite
    muestrear con temperatura 0, y "0" en la API de OpenAI es "lo más determinista posible".
    """
    t = TEMPERATURE_DEFAULT if temperature is None else float(temperature)
    max_new = MAX_TOKENS_DEFAULT if max_tokens is None else int(max_tokens)
    if t == 0:
        return False, 0.0, max_new
    return True, t, max_new


def validar_contexto(prompt_tokens: int, max_new_tokens: int, contexto_max: int) -> None:
    """Rechaza antes de generar: pasarse del contexto degrada o rompe la salida sin avisar."""
    if prompt_tokens + max_new_tokens > contexto_max:
        raise ContextoExcedido(prompt_tokens, contexto_max, max_new_tokens)


def decidir_finish_reason(
    ultimo_token_id: Optional[int], eos_ids: Iterable[int], completion_tokens: int, max_new_tokens: int
) -> str:
    """
    "length" solo si se agotó max_new_tokens sin terminar en un token de fin. Si el último
    token generado es de fin, el modelo terminó solo aunque coincida con el tope: "stop".
    """
    if completion_tokens >= max_new_tokens and ultimo_token_id not in set(eos_ids):
        return "length"
    return "stop"


def bytes_por_token(
    num_capas: int, num_kv_heads: int, head_dim: int, bytes_dtype: int = 2, factor: float = FACTOR_OVERHEAD_KV
) -> float:
    """
    Costo estimado en VRAM de un token de contexto: K y V (el 2 inicial) por capa, por cabeza
    de KV y por dimensión, en bf16 (2 bytes, el compute dtype), multiplicado por `factor`.
    """
    return 2 * num_capas * num_kv_heads * head_dim * bytes_dtype * factor


def contexto_por_vram(presupuesto_b: float, costo_por_token_b: float, contexto_modelo: int) -> int:
    """Cuántos tokens de contexto entran en el presupuesto, sin pasar del contexto del modelo."""
    tokens = min(contexto_modelo, int(presupuesto_b // costo_por_token_b))
    if tokens < 1:
        raise ValueError(
            f"La VRAM que queda tras cargar el modelo ({presupuesto_b / MIB:.1f} MiB) no alcanza "
            f"ni para un token de contexto ({costo_por_token_b / MIB:.4f} MiB por token)."
        )
    return tokens


def contexto_efectivo(contexto_modelo: Optional[int], por_vram: Optional[int], configurado: Optional[int]) -> int:
    """El menor de los topes conocidos: un CONTEXTO_MAX_TOKENS mayor que el automático no lo amplía."""
    candidatos = [c for c in (contexto_modelo, por_vram, configurado) if c is not None]
    if not candidatos:
        raise ValueError("No hay ningún tope de contexto conocido.")
    return min(candidatos)


def fraccion_vram(libre_b: float, total_b: float, margen_b: float) -> float:
    """Fracción del total de la GPU que puede usar este proceso: lo libre al arrancar menos el margen."""
    fraccion = (libre_b - margen_b) / total_b
    if not fraccion > 0:
        raise ValueError(
            f"Sin VRAM utilizable: {libre_b / MIB:.1f} MiB libres menos {margen_b / MIB:.1f} MiB de margen."
        )
    return fraccion


def vram_alcanza(
    tokens_pedido: int, costo_por_token_b: float, libre_b: float, cache_reutilizable_b: float, margen_b: float
) -> bool:
    """
    Si la VRAM de este momento alcanza para el pedido. Cuenta lo libre en el dispositivo más lo
    que PyTorch ya tiene reservado y no usa (lo reutiliza sin pedirle al driver), menos el margen.
    """
    necesaria = tokens_pedido * costo_por_token_b
    disponible = libre_b + cache_reutilizable_b - margen_b
    return necesaria <= disponible


# ---------- Motor real (requiere GPU) ----------

class MotorLlama:
    def __init__(self, path_modelo: str, top_p: float, contexto_max_config: Optional[int] = None):
        self.path_modelo = path_modelo
        self.top_p = top_p
        # Override manual (CONTEXTO_MAX_TOKENS): solo puede achicar el tope automático.
        self.contexto_max_config = contexto_max_config
        self._model: Any = None
        self._tokenizer: Any = None
        self._eos_ids: set[int] = set()
        self._costo_por_token_b: Optional[float] = None
        self.contexto_max: Optional[int] = None

    def cargar(self) -> None:
        """Carga tokenizador y modelo en GPU con cuantización NF4 (mismos parámetros que chatbot.py)."""
        import torch

        # Chequeos antes de importar transformers: fallan rápido y se prueban con un torch falso.
        if not torch.cuda.is_available():
            raise SinGPU("CUDA no está disponible: este proveedor solo corre en GPU.")
        libre_b, total_b = torch.cuda.mem_get_info(0)
        libre_gb, total_gb = libre_b / (1024 ** 3), total_b / (1024 ** 3)
        if libre_gb < VRAM_MIN_GB:
            raise RuntimeError(
                f"VRAM libre insuficiente: {libre_gb:.2f} GB libres de {total_gb:.2f} GB; "
                f"se requieren al menos {VRAM_MIN_GB} GB."
            )
        logger.info(
            "Cargando modelo en %s (VRAM libre %.2f GB de %.2f GB, torch %s, CUDA %s).",
            torch.cuda.get_device_name(0), libre_gb, total_gb, torch.__version__, torch.version.cuda,
        )

        # Límite de VRAM del proceso, ANTES de cualquier from_pretrained. Bajo el driver de
        # Windows la VRAM puede derramarse a la RAM del sistema en vez de dar OOM, con latencias
        # de minutos (medido: 423 s con ~8000 tokens). Con el límite, PyTorch lanza
        # OutOfMemoryError, que generar() ya maneja (empty_cache + 500 inference_error).
        fraccion = fraccion_vram(libre_b, total_b, MARGEN_VRAM_MB * MIB)
        torch.cuda.set_per_process_memory_fraction(fraccion, 0)
        limite_b = fraccion * total_b
        logger.info("Límite de VRAM del proceso: %.1f MiB (fracción %.4f del total).", limite_b / MIB, fraccion)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(self.path_modelo)
        configuracion_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.path_modelo, quantization_config=configuracion_4bit, device_map="cuda"
        )

        # Llama 3 Instruct termina el turno con <|eot_id|>, que está en generation_config
        # pero puede no ser el eos del tokenizador: se juntan ambos para cortar y para
        # decidir finish_reason con el mismo criterio.
        eos: set[int] = set()
        for fuente in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
            if isinstance(fuente, int):
                eos.add(fuente)
            elif fuente is not None:
                eos.update(int(x) for x in fuente)

        # Tope de contexto según la VRAM que quedó tras cargar el modelo.
        config = model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        costo_b = bytes_por_token(config.num_hidden_layers, config.num_key_value_heads, head_dim)
        asignada_b = torch.cuda.memory_allocated()
        contexto_modelo = int(config.max_position_embeddings)
        por_vram = contexto_por_vram(limite_b - asignada_b, costo_b, contexto_modelo)
        efectivo = contexto_efectivo(contexto_modelo, por_vram, self.contexto_max_config)
        logger.info(
            "Tope de contexto: libre_mib=%.1f limite_mib=%.1f asignada_mib=%.1f costo_mib_por_token=%.4f "
            "contexto_modelo=%d por_vram=%d configurado=%s efectivo=%d",
            libre_b / MIB, limite_b / MIB, asignada_b / MIB, costo_b / MIB,
            contexto_modelo, por_vram, self.contexto_max_config, efectivo,
        )

        self.contexto_max = efectivo
        self._costo_por_token_b = costo_b
        self._eos_ids = eos
        self._tokenizer = tokenizer
        self._model = model

    def generar(self, mensajes: list[dict], temperature: Optional[float], max_tokens: Optional[int]) -> Resultado:
        """
        Aplica la plantilla de chat del tokenizador a los mensajes TAL CUAL llegan: el
        proveedor no agrega system prompt (lo decide el orquestador). No muta `mensajes`.
        """
        import torch

        if self._model is None:
            raise RuntimeError("El modelo no está cargado.")
        model, tokenizer = self._model, self._tokenizer
        do_sample, temp, max_new = resolver_parametros(temperature, max_tokens)

        # Copia superficial: apply_chat_template no debería mutar, pero no depende de eso.
        inputs = tokenizer.apply_chat_template(
            [dict(m) for m in mensajes], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        validar_contexto(prompt_tokens, max_new, self.contexto_max)

        # VRAM de ESTE momento, antes de mover nada a la GPU: otro proceso (un juego) pudo
        # ocuparla después del arranque, y entonces el tope calculado al cargar ya no alcanza.
        libre_b = torch.cuda.mem_get_info(0)[0]
        cache_reutilizable_b = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        tokens_pedido = prompt_tokens + max_new
        margen_b = MARGEN_VRAM_MB * MIB
        if not vram_alcanza(tokens_pedido, self._costo_por_token_b, libre_b, cache_reutilizable_b, margen_b):
            raise VRAMInsuficiente(
                tokens_pedido * self._costo_por_token_b / MIB,
                (libre_b + cache_reutilizable_b - margen_b) / MIB,
            )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "eos_token_id": sorted(self._eos_ids),
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            # En greedy no se pasan: transformers avisa si llegan temperature/top_p sin muestreo.
            kwargs["temperature"] = temp
            kwargs["top_p"] = self.top_p

        hay_cuda = model.device.type == "cuda"
        if hay_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_inicio = time.perf_counter()
        try:
            outputs = model.generate(**inputs, **kwargs)
            if hay_cuda:
                # generate es asíncrono en GPU: esperar antes de parar el cronómetro.
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            # Devolver al pool lo que quedó reservado, para que el próximo pedido tenga chance.
            torch.cuda.empty_cache()
            raise
        generation_s = time.perf_counter() - t_inicio

        nuevos = outputs[0][prompt_tokens:]
        completion_tokens = int(nuevos.shape[-1])
        ultimo = int(nuevos[-1]) if completion_tokens else None
        texto = tokenizer.decode(nuevos, skip_special_tokens=True)

        return Resultado(
            texto=texto,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=decidir_finish_reason(ultimo, self._eos_ids, completion_tokens, max_new),
            generation_s=generation_s,
            vram_peak_mb=round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if hay_cuda else None,
            vram_reserved_mb=round(torch.cuda.memory_reserved() / (1024 ** 2), 1) if hay_cuda else None,
        )

    def liberar(self) -> None:
        """Suelta modelo y tokenizador y devuelve la VRAM. Sin efecto si no se cargó."""
        if self._model is None:
            return
        import torch

        # Logs de cada paso: el apagado da exit 137 y hay que ver en cuál se corta.
        self._model = None
        self._tokenizer = None
        logger.info("Liberando: gc.collect() empieza.")
        gc.collect()
        logger.info("Liberando: gc.collect() terminó.")
        if torch.cuda.is_available():
            logger.info("Liberando: torch.cuda.empty_cache() empieza.")
            torch.cuda.empty_cache()
            logger.info("Liberando: torch.cuda.empty_cache() terminó.")
        logger.info("Modelo liberado.")
