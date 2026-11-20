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
# Mensaje fijo del calentamiento al cargar: nunca se loguea ni él ni lo generado.
MENSAJES_CALENTAMIENTO = ({"role": "user", "content": "Hola"},)
# VRAM que se deja sin usar, para Windows y para la fragmentación del allocator de PyTorch.
MARGEN_VRAM_MB = 256
# El KV cache de Llama 3 8B en bf16 cuesta ~0,125 MiB por token (fórmula de bytes_por_token sin
# factor); el pico medido en vivo fue ~0,2275 MiB por token de entrada (activaciones y buffers
# temporales además del KV). El factor 2 lleva la estimación a ~0,25 MiB: margen sobre lo medido.
FACTOR_OVERHEAD_KV = 2.0
# Reserva por pedido que se descuenta del presupuesto de contexto (no del límite de proceso).
# Origen (paso 3e): ~119 MiB de costo fijo por pedido (ordenada del pico sobre la memoria en
# reposo; hipótesis sin medir: descuantización temporal de bitsandbytes) más fragmentación
# (17-58 MiB medidos) más holgura por la histéresis vista con expandable_segments tras varios
# OOM seguidos. Medido para Llama 3 8B NF4 en una RTX 4070 Laptop bajo WSL2: NO extrapolar a
# otras GPUs, modelos o drivers sin volver a medir.
RESERVA_PEDIDO_MB = 150


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
    """
    No hay capacidad de VRAM para este pedido: lo rechaza la verificación previa o lo provocó un
    OOM real al generar. Puede ser permanente para ese tamaño en esta GPU. Solo números.
    """

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


def fraccion_de_limite(limite_b: float, total_b: float) -> float:
    """Fracción del total para set_per_process_memory_fraction a partir de un límite en bytes (como máximo 1)."""
    fraccion = limite_b / total_b
    if not fraccion > 0:
        raise ValueError(f"Límite de VRAM no positivo: {limite_b / MIB:.1f} MiB.")
    return min(fraccion, 1.0)


def presupuesto_vram_b(libre_b: float, cache_reutilizable_b: float, margen_b: float, reserva_b: float) -> float:
    """
    VRAM utilizable para contexto: lo libre según el driver más lo que PyTorch ya tiene reservado
    y no usa (lo reutiliza sin pedirle nada al driver), menos el margen y la reserva por pedido.
    ÚNICA fórmula: la usan el tope de arranque y la verificación por pedido, para que no puedan
    contradecirse. Puede dar negativo; cargar() lo rechaza con un mensaje claro.
    """
    return libre_b + cache_reutilizable_b - margen_b - reserva_b


def vram_alcanza(
    tokens_pedido: int,
    costo_por_token_b: float,
    libre_b: float,
    cache_reutilizable_b: float,
    margen_b: float,
    reserva_b: float,
) -> bool:
    """Si el presupuesto de este momento alcanza para el pedido (prompt + max_tokens)."""
    return tokens_pedido * costo_por_token_b <= presupuesto_vram_b(libre_b, cache_reutilizable_b, margen_b, reserva_b)


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
        # OutOfMemoryError, que _generar() convierte en VRAMInsuficiente (empty_cache + 503).
        # Primera pasada: protege la carga. Se reajusta con el valor preciso tras el calentamiento.
        fraccion = fraccion_vram(libre_b, total_b, MARGEN_VRAM_MB * MIB)
        torch.cuda.set_per_process_memory_fraction(fraccion, 0)
        logger.info(
            "Límite de VRAM del proceso (carga): %.1f MiB (fracción %.4f del total).", fraccion * total_b / MIB, fraccion
        )

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

        config = model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        costo_b = bytes_por_token(config.num_hidden_layers, config.num_key_value_heads, head_dim)
        contexto_modelo = int(config.max_position_embeddings)

        # Calentamiento: el primer generate carga kernels y workspaces de CUDA (~3,4 tokens/s
        # medido contra ~29 después) y ocupa VRAM fuera del allocator que memory_allocated() no
        # ve. Se hace por el mismo camino que generar(), ANTES de medir la VRAM para el tope,
        # para que ese consumo ya esté descontado y el primer pedido real no pague el arranque.
        self._model, self._tokenizer, self._eos_ids = model, tokenizer, eos
        # El costo ya se conoce; _generar lo necesita si un OOM del calentamiento se convierte en VRAMInsuficiente.
        self._costo_por_token_b = costo_b
        t_calentamiento = time.perf_counter()
        try:
            self._generar(MENSAJES_CALENTAMIENTO, 0, 8, contexto_max=contexto_modelo, verificar_vram=False)
        except Exception:
            # Sin modelo a medias: el estado queda "failed" y el tope sin fijar.
            self._model, self._tokenizer, self._eos_ids = None, None, set()
            self._costo_por_token_b = None
            raise
        logger.info("Calentamiento: %.2f s.", time.perf_counter() - t_calentamiento)

        # Tope de contexto con la MISMA fórmula que la verificación por pedido, en reposo.
        libre_b = torch.cuda.mem_get_info(0)[0]
        reservada_b = torch.cuda.memory_reserved()
        cache_reutilizable_b = reservada_b - torch.cuda.memory_allocated()
        margen_b = MARGEN_VRAM_MB * MIB
        reserva_b = RESERVA_PEDIDO_MB * MIB
        presupuesto_b = presupuesto_vram_b(libre_b, cache_reutilizable_b, margen_b, reserva_b)
        if presupuesto_b < costo_b:
            # Mensaje con todos los números: es el motivo del estado "failed" en el log.
            self._model, self._tokenizer, self._eos_ids, self._costo_por_token_b = None, None, set(), None
            raise RuntimeError(
                f"No queda VRAM para contexto tras cargar el modelo: presupuesto {presupuesto_b / MIB:.1f} MiB "
                f"(libre {libre_b / MIB:.1f} + cache reutilizable {cache_reutilizable_b / MIB:.1f} "
                f"- margen {margen_b / MIB:.1f} - reserva {reserva_b / MIB:.1f}), menos que un token "
                f"({costo_b / MIB:.4f} MiB)."
            )
        # El tope es una ESTIMACIÓN conservadora (paso 3e), no una garantía: un pedido que entra
        # puede igual dar OOM, que llega al cliente como 503 insufficient_vram.
        por_vram = contexto_por_vram(presupuesto_b, costo_b, contexto_modelo)
        efectivo = contexto_efectivo(contexto_modelo, por_vram, self.contexto_max_config)
        logger.info(
            "Tope de contexto: libre_mib=%.1f cache_reutilizable_mib=%.1f margen_mib=%.1f reserva_mib=%.1f "
            "presupuesto_mib=%.1f costo_mib_por_token=%.4f contexto_modelo=%d por_vram=%d configurado=%s efectivo=%d",
            libre_b / MIB, cache_reutilizable_b / MIB, margen_b / MIB, reserva_b / MIB, presupuesto_b / MIB,
            costo_b / MIB, contexto_modelo, por_vram, self.contexto_max_config, efectivo,
        )

        # Segunda pasada del límite: lo máximo que PyTorch puede llegar a reservar sin pasar la
        # VRAM física menos el margen (lo que ya reservó más lo libre, menos el margen). Sin la
        # reserva por pedido: es holgura del presupuesto, sumarla acá la penalizaría dos veces.
        limite_b = reservada_b + libre_b - margen_b
        fraccion = fraccion_de_limite(limite_b, total_b)
        torch.cuda.set_per_process_memory_fraction(fraccion, 0)
        logger.info("Límite de VRAM del proceso (final): %.1f MiB (fracción %.4f del total).", limite_b / MIB, fraccion)

        self.contexto_max = efectivo

    def generar(self, mensajes: list[dict], temperature: Optional[float], max_tokens: Optional[int]) -> Resultado:
        """
        Aplica la plantilla de chat del tokenizador a los mensajes TAL CUAL llegan: el
        proveedor no agrega system prompt (lo decide el orquestador). No muta `mensajes`.
        """
        return self._generar(mensajes, temperature, max_tokens, contexto_max=self.contexto_max, verificar_vram=True)

    def _generar(
        self,
        mensajes: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
        contexto_max: Optional[int],
        verificar_vram: bool,
    ) -> Resultado:
        """Camino común. El calentamiento lo usa sin verificación de VRAM: no debe poder rechazarse."""
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
        validar_contexto(prompt_tokens, max_new, contexto_max)

        # VRAM de ESTE momento, antes de mover nada a la GPU. Solo cubre lo que ve mem_get_info:
        # en Linux nativo incluye a los demás procesos, pero bajo WSL2/WDDM no los ve (medido en
        # el paso 3c): el chequeo cubre únicamente el estado del propio allocator. El 400
        # context_length_exceeded es un límite ESTIMADO y conservador, no una garantía.
        libre_b = torch.cuda.mem_get_info(0)[0]
        cache_reutilizable_b = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        tokens_pedido = prompt_tokens + max_new
        margen_b = MARGEN_VRAM_MB * MIB
        reserva_b = RESERVA_PEDIDO_MB * MIB
        if verificar_vram and not vram_alcanza(
            tokens_pedido, self._costo_por_token_b, libre_b, cache_reutilizable_b, margen_b, reserva_b
        ):
            raise VRAMInsuficiente(
                tokens_pedido * self._costo_por_token_b / MIB,
                presupuesto_vram_b(libre_b, cache_reutilizable_b, margen_b, reserva_b) / MIB,
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
        except torch.cuda.OutOfMemoryError as e:
            # Devolver al pool lo que quedó reservado, para que el próximo pedido tenga chance.
            torch.cuda.empty_cache()
            # La estimación no alcanzó: OOM real del límite de proceso. Se informa como falta de
            # capacidad (503), con los números; nunca el texto de torch (de ahí el `from` sin str).
            necesaria_mb = tokens_pedido * self._costo_por_token_b / MIB
            disponible_mb = presupuesto_vram_b(
                torch.cuda.mem_get_info(0)[0],
                torch.cuda.memory_reserved() - torch.cuda.memory_allocated(),
                margen_b,
                reserva_b,
            ) / MIB
            logger.warning(
                "OOM al generar: la estimación de VRAM no alcanzó (tokens_pedido=%d necesaria_mib=%.1f "
                "disponible_mib=%.1f).", tokens_pedido, necesaria_mb, disponible_mb,
            )
            raise VRAMInsuficiente(necesaria_mb, disponible_mb) from e
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
