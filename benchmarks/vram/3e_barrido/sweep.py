"""
Diagnóstico de la curva VRAM contra largo del prompt para llama-local (paso 3e).
Corre contra el commit 541bca3 (con el código actual falla; ver benchmarks/README.md). Usa la API real de MotorLlama: cargar() hace el
calentamiento y fija tope y límite de proceso como el servicio; _generar(verificar_vram=False)
con contexto_max alto se salta la verificación por pedido y el tope, para poder llegar a OOM.
"""
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)

import torch  # noqa: E402

from src.providers.llama_local.engine import MotorLlama  # noqa: E402

MIB = 1024 ** 2
SIN_TOPE = 10 ** 9
MAX_NEW = int(os.getenv("MAX_NEW", "8"))
TOPE_BARRIDO = int(os.getenv("TOPE_BARRIDO", "2300"))
UNIDAD = "La reunión del lunes se pasó al martes por la mañana. "


def emitir(**d):
    print("DIAG " + json.dumps(d, ensure_ascii=True), flush=True)


def mensajes(reps, extra):
    return [{"role": "user", "content": UNIDAD * reps + " x" * extra + " Resumí lo anterior en una línea."}]


def contar(tok, msgs):
    return int(tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True)["input_ids"].shape[-1])


def armar(tok, objetivo):
    """Mensajes con exactamente `objetivo` tokens de prompt (plantilla de chat incluida)."""
    p0 = contar(tok, mensajes(0, 0))
    por_rep = (contar(tok, mensajes(10, 0)) - p0) / 10
    reps = max(0, int((objetivo - p0) // por_rep) - 1)
    extra = 0
    for _ in range(200):
        p = contar(tok, mensajes(reps, extra))
        if p == objetivo:
            return mensajes(reps, extra), p
        if p > objetivo:
            reps, extra = max(0, reps - 1), 0
        else:
            extra += max(1, objetivo - p)
    raise RuntimeError(f"no se pudo armar {objetivo}")


def punto(motor, objetivo, fase):
    msgs, p = armar(motor._tokenizer, objetivo)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        r = motor._generar(msgs, 0, MAX_NEW, contexto_max=SIN_TOPE, verificar_vram=False)
        d = dict(fase=fase, objetivo=objetivo, prompt_tokens=r.prompt_tokens, resultado="OK", completion_tokens=r.completion_tokens,
                 pico_mib=round(torch.cuda.max_memory_allocated() / MIB, 1),
                 reservada_mib=round(torch.cuda.memory_reserved() / MIB, 1),
                 generation_s=round(r.generation_s, 3))
    except torch.cuda.OutOfMemoryError as e:
        # _generar ya hizo empty_cache antes de relanzar; se repite por las dudas.
        torch.cuda.empty_cache()
        d = dict(fase=fase, objetivo=objetivo, prompt_tokens=p, resultado="OOM",
                 pico_mib=round(torch.cuda.max_memory_allocated() / MIB, 1),
                 reservada_mib=round(torch.cuda.memory_reserved() / MIB, 1),
                 generation_s=round(time.perf_counter() - t0, 3), oom=str(e))
    d["libre_mib"] = round(torch.cuda.mem_get_info(0)[0] / MIB, 1)
    d["asignada_mib"] = round(torch.cuda.memory_allocated() / MIB, 1)
    emitir(**d)
    return d["resultado"] == "OK"


def main():
    variante = os.getenv("VARIANTE", "?")
    motor = MotorLlama(os.getenv("MODEL_PATH", "/app/modelos/llama3"), 0.9)
    t0 = time.perf_counter()
    motor.cargar()
    emitir(variante=variante, torch=torch.__version__, cuda=torch.version.cuda,
           attn=getattr(motor._model.config, "_attn_implementation", None),
           alloc_conf=os.getenv("PYTORCH_CUDA_ALLOC_CONF"), max_new=MAX_NEW, contexto_efectivo=motor.contexto_max,
           carga_s=round(time.perf_counter() - t0, 1),
           reservada_mib=round(torch.cuda.memory_reserved() / MIB, 1),
           asignada_mib=round(torch.cuda.memory_allocated() / MIB, 1),
           libre_mib=round(torch.cuda.mem_get_info(0)[0] / MIB, 1))

    # 3. Ascendente.
    objetivos = [250, 500, 750, 1000, 1250, 1500] + list(range(1550, TOPE_BARRIDO + 1, 50))
    ultimo_ok, primer_oom = None, None
    for o in objetivos:
        if punto(motor, o, "asc"):
            if primer_oom is None:
                ultimo_ok = o
        elif primer_oom is None:
            primer_oom = o

    # 4. Bisección entre el último OK (antes del primer OOM) y el primer OOM, hasta ±8.
    if ultimo_ok is not None and primer_oom is not None:
        lo, hi = ultimo_ok, primer_oom
        while hi - lo > 8:
            medio = (lo + hi) // 2
            if punto(motor, medio, "bis"):
                lo = medio
            else:
                hi = medio
        emitir(variante=variante, umbral_ultimo_ok=lo, umbral_primer_oom=hi)

    # 5. Descendente, mismo proceso.
    for o in range(TOPE_BARRIDO, 1499, -100):
        punto(motor, o, "desc")


if __name__ == "__main__":
    main()
