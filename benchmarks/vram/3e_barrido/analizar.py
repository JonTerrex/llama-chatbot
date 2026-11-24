import json, sys
def cargar(ruta):
    cab, pts, umbral = None, [], None
    for l in open(ruta, encoding="utf-8", errors="replace"):
        if not l.startswith("DIAG "): continue
        d = json.loads(l[5:])
        if "fase" in d: pts.append(d)
        elif "umbral_ultimo_ok" in d: umbral = d
        else: cab = d
    return cab, pts, umbral
def ajuste(pts, campo="pico_mib"):
    ok = [p for p in pts if p["resultado"] == "OK"]
    xs = [p["prompt_tokens"] for p in ok]; ys = [p[campo] for p in ok]
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    desv = max(ok, key=lambda p: abs(p[campo] - (a + b * p["prompt_tokens"])))
    return a, b, desv["prompt_tokens"], desv[campo] - (a + b * desv["prompt_tokens"]), n
for ruta in sys.argv[1:]:
    cab, pts, umbral = cargar(ruta)
    print("==", ruta.rsplit("/", 1)[-1], "| max_new", cab.get("max_new", 8), "| efectivo", cab["contexto_efectivo"], "| alloc", cab["alloc_conf"])
    a, b, x, d, n = ajuste(pts)
    print(f"   pico = {a:.1f} + {b:.4f}*tokens  (n={n} OK; mayor desvío {d:+.1f} MiB en {x})")
    a2, b2, x2, d2, _ = ajuste(pts, "reservada_mib")
    print(f"   reservada = {a2:.1f} + {b2:.4f}*tokens  (mayor desvío {d2:+.1f} MiB en {x2})")
    oom = [p for p in pts if p["resultado"] == "OOM"]
    print("   OOM:", len(oom), "| umbral:", umbral)
