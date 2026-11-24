"""
X4 del paso 3g: OOM real en vivo, con el código del commit a1f6444 (ver benchmarks/README.md). Carga el motor como el servicio
(cargar() con calentamiento, tope y límite de proceso) y pide un prompt muy por encima del umbral
real saltándose validar_contexto (contexto_max alto). No imprime contenido ni texto de torch.
"""
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)

import torch  # noqa: E402

from src.providers.llama_local.engine import MotorLlama, VRAMInsuficiente  # noqa: E402

MIB = 1024 ** 2
SIN_TOPE = 10 ** 9
OBJETIVO = int(os.getenv("OBJETIVO", "3200"))
UNIDAD = "La reunión del lunes se pasó al martes por la mañana. "


def emitir(**d):
    print("DIAG " + json.dumps(d, ensure_ascii=True), flush=True)


def mensajes(reps, extra):
    return [{"role": "user", "content": UNIDAD * reps + " x" * extra + " Resumí lo anterior en una línea."}]


def contar(tok, msgs):
    return int(tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True)["input_ids"].shape[-1])


def armar(tok, objetivo):
    p0 = contar(tok, mensajes(0, 0))
    por_rep = (contar(tok, mensajes(10, 0)) - p0) / 10
    reps, extra = max(0, int((objetivo - p0) // por_rep) - 1), 0
    for _ in range(200):
        p = contar(tok, mensajes(reps, extra))
        if p == objetivo:
            return mensajes(reps, extra)
        if p > objetivo:
            reps, extra = max(0, reps - 1), 0
        else:
            extra += max(1, objetivo - p)
    raise RuntimeError("no se pudo armar")


def vram():
    return dict(reservada_mib=round(torch.cuda.memory_reserved() / MIB, 1),
                asignada_mib=round(torch.cuda.memory_allocated() / MIB, 1))


def main():
    motor = MotorLlama(os.getenv("MODEL_PATH", "/app/modelos/llama3"), 0.9)
    motor.cargar()
    emitir(fase="cargado", contexto_efectivo=motor.contexto_max,
           alloc_conf=os.getenv("PYTORCH_CUDA_ALLOC_CONF"), **vram())

    msgs = armar(motor._tokenizer, OBJETIVO)
    # Antes: un pedido normal, para que la reservada refleje un pedido real.
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        motor._generar(msgs, 0, 256, contexto_max=SIN_TOPE)
        emitir(fase="grande", resultado="OK_SIN_OOM", s=round(time.perf_counter() - t0, 3), **vram())
    except VRAMInsuficiente as e:
        dt = time.perf_counter() - t0
        emitir(fase="grande", resultado="VRAMInsuficiente", s=round(dt, 3),
               tokens_pedido=e.tokens_pedido, necesaria_mb=round(e.necesaria_mb, 1),
               presupuesto_arranque_mb=round(e.presupuesto_arranque_mb, 1),
               causa=type(e.__cause__).__name__, str_sin_torch=("memory" not in str(e).lower() and "allocate" not in str(e)),
               str_exc=str(e), pico_mib=round(torch.cuda.max_memory_allocated() / MIB, 1), **vram())
    except Exception as e:  # OutOfMemoryError crudo u otra cosa: solo la clase
        emitir(fase="grande", resultado="OTRA", clase=type(e).__name__, s=round(time.perf_counter() - t0, 3), **vram())

    corto = [{"role": "user", "content": "Contá del 1 al 40 en palabras."}]
    torch.cuda.reset_peak_memory_stats()
    r = motor._generar(corto, 0, 128, contexto_max=motor.contexto_max)
    emitir(fase="corto", prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
           finish=r.finish_reason, generation_s=round(r.generation_s, 3),
           tokens_por_s=round(r.completion_tokens / r.generation_s, 2), pico_mib=r.vram_peak_mb, **vram())
    motor.liberar()


if __name__ == "__main__":
    main()
