"""X1-X3 del paso 3g contra llama-local en 127.0.0.1:8004. No imprime contenido de prompts ni respuestas."""
import json
import re
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8004/v1/chat/completions"
TOPE = int(sys.argv[1])
UNIDAD = "La reunión del lunes se pasó al martes por la mañana. "
CIERRE_CORTO = " Resumí lo anterior en una línea."
CIERRE_LARGO = " Ahora escribí una lista numerada larguísima, del 1 al 500, con una frase distinta por número, sin cortar."


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status, data = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        status, data = e.code, json.load(e)
    return status, data, time.perf_counter() - t0


def msgs(reps, extra, cierre):
    return [{"role": "user", "content": UNIDAD * reps + " x" * extra + cierre}]


def contar(m):
    s, d, _ = post({"model": "llama3-8b-local", "messages": m, "max_tokens": 100000, "temperature": 0})
    assert s == 400, s
    return int(re.search(r"(\d+) tokens de entrada", d["error"]["message"]).group(1))


def armar(objetivo, cierre):
    p0 = contar(msgs(0, 0, cierre))
    por_rep = (contar(msgs(10, 0, cierre)) - p0) / 10
    reps, extra = max(0, int((objetivo - p0) // por_rep) - 1), 0
    for _ in range(200):
        p = contar(msgs(reps, extra, cierre))
        if p == objetivo:
            return msgs(reps, extra, cierre)
        if p > objetivo:
            reps, extra = reps - 1, 0
        else:
            extra += max(1, objetivo - p)
    raise RuntimeError("no se pudo armar")


def resumen(s, d, dt):
    if s == 200:
        u, x, ch = d["usage"], d.get("x_metrics", {}), d["choices"][0]
        return dict(status=s, prompt=u["prompt_tokens"], completion=u["completion_tokens"], finish=ch["finish_reason"],
                    cliente_s=round(dt, 3), **{k: v for k, v in x.items()})
    return dict(status=s, code=d["error"].get("code"), message=d["error"]["message"], cliente_s=round(dt, 3))


def pedir(m, max_tokens, temperature=0):
    return resumen(*post({"model": "llama3-8b-local", "messages": m, "max_tokens": max_tokens, "temperature": temperature}))


prompt_x1 = TOPE - 1 - 256
m1 = armar(prompt_x1, CIERRE_CORTO)
print(f"X1: prompt {prompt_x1} + 256 = {prompt_x1 + 256} (tope {TOPE})")
for i in range(10):
    print(" X1", i + 1, json.dumps(pedir(m1, 256)))

m2 = armar(prompt_x1, CIERRE_LARGO)
print(f"X2: prompt {prompt_x1} + 256 = {prompt_x1 + 256}, salida larga")
print(" X2", json.dumps(pedir(m2, 256)))
print(" X2 corto", json.dumps(pedir([{"role": "user", "content": "Contá del 1 al 40 en palabras."}], 128)))

m3 = armar(TOPE - 256 + 1, CIERRE_CORTO)
print(f"X3: prompt {TOPE - 255} + 256 = {TOPE + 1}")
print(" X3", json.dumps(pedir(m3, 256)))
