"""
Y1-Y4 del paso 3h: VRAM tras un OOM real, con el código del commit ae1894f (ver benchmarks/README.md). Carga el motor como el
servicio y alterna OOM (prompt ~3400, max_new 256, sin validar_contexto) y pedidos cortos.
No imprime contenido ni texto de torch.
"""
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)

import torch  # noqa: E402

from src.providers.llama_local.engine import MotorLlama, VRAMInsuficiente  # noqa: E402

MIB = 1024 ** 2
SIN_TOPE = 10 ** 9
OBJETIVO = int(os.getenv("OBJETIVO", "3400"))
UNIDAD = "La reunión del lunes se pasó al martes por la mañana. "


def emitir(**d):
    print("DIAG " + json.dumps(d, ensure_ascii=True), flush=True)


def smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception as e:  # sin nvidia-smi en el contenedor
        return f"n/d ({type(e).__name__})"


def vram():
    return dict(asignada_mib=round(torch.cuda.memory_allocated() / MIB, 1),
                reservada_mib=round(torch.cuda.memory_reserved() / MIB, 1), smi_mib=smi())


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


CORTO = [{"role": "user", "content": "Contá del 1 al 40 en palabras."}]


def oom(motor, msgs, etiqueta):
    t0 = time.perf_counter()
    try:
        motor._generar(msgs, 0, 256, contexto_max=SIN_TOPE)
        emitir(fase=etiqueta, resultado="OK_SIN_OOM", s=round(time.perf_counter() - t0, 3), **vram())
    except VRAMInsuficiente as e:
        dt = time.perf_counter() - t0
        # Medición INMEDIATA, antes de soltar `e` (ya no retiene nada: no encadena).
        d = vram()
        emitir(fase=etiqueta, resultado="VRAMInsuficiente", s=round(dt, 3), tokens_pedido=e.tokens_pedido,
               causa=type(e.__cause__).__name__, contexto=type(e.__context__).__name__, **d)
    except Exception as e:
        emitir(fase=etiqueta, resultado="OTRA", clase=type(e).__name__, **vram())


def corto(motor, etiqueta):
    r = motor._generar(CORTO, 0, 128, contexto_max=motor.contexto_max)
    emitir(fase=etiqueta, completion_tokens=r.completion_tokens, generation_s=round(r.generation_s, 3),
           tokens_por_s=round(r.completion_tokens / r.generation_s, 2), **vram())


def main():
    motor = MotorLlama(os.getenv("MODEL_PATH", "/app/modelos/llama3"), 0.9)
    motor.cargar()
    msgs = armar(motor._tokenizer, OBJETIVO)
    emitir(fase="Y1_reposo", contexto_efectivo=motor.contexto_max, **vram())
    oom(motor, msgs, "Y2_oom")
    corto(motor, "Y3_corto")
    for i in range(1, 4):
        oom(motor, msgs, f"Y4.{i}_oom")
        corto(motor, f"Y4.{i}_corto")
    motor.liberar()


if __name__ == "__main__":
    main()
