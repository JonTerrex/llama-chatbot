"""
Prueba en vivo de R1 + R2 + R3 (código del commit 84ea34f; ver benchmarks/README.md). Solo imprime números, status y
codes: nunca contenido de mensajes. Uso: python vivo_r123.py paso2|esperar|pruebaA|pruebaB
Se corre con cwd = repo y PYTHONPATH=. para reusar (solo lectura) src/orchestrator/presupuesto.py.
"""
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from src.orchestrator.presupuesto import OVERHEAD_BASE, recortar_historial, disponible_para_mensajes, tokens_de_mensajes

ORQ = "http://127.0.0.1:8002"
PROV = "http://127.0.0.1:8004"
MEM = "http://127.0.0.1:8003"
MODELO = "llama3-8b-local"
C_ORQ = "llama-chatbot-orchestrator-1"
C_PROV = "llama-chatbot-llama-local-1"
TOPE = 1800
MAX_TOKENS = 150
HISTORY_LIMIT = 8  # MEMORY_HISTORY_LIMIT de docker-compose.yml


def http(metodo, url, cuerpo=None, timeout=300):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    req = urllib.request.Request(url, data=datos, method=metodo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except ValueError:
            return e.code, None
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return None, type(e).__name__


def logs(contenedor):
    r = subprocess.run(["docker", "logs", contenedor], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr).splitlines()


def code_de(cuerpo):
    return cuerpo.get("error", {}).get("code") if isinstance(cuerpo, dict) else None


def estado_proveedor():
    s, b = http("GET", f"{PROV}/health", timeout=5)
    return s, (b.get("status") if isinstance(b, dict) else b)


# ---------- Paso 2: pedido mientras el proveedor carga ----------

def paso2():
    for _ in range(120):
        s, _b = http("GET", f"{ORQ}/health", timeout=5)
        if s == 200:
            break
        time.sleep(1)
    print(f"orquestador /health: {s}")
    print(f"proveedor /health antes: {estado_proveedor()}")
    s, b = http("POST", f"{ORQ}/v1/chat/completions",
                {"model": MODELO, "messages": [{"role": "user", "content": "Hola"}], "max_tokens": 8}, timeout=60)
    xp = b.get("error", {}).get("x_provider") if isinstance(b, dict) else None
    print(f"pedido al orquestador: status={s} code={code_de(b)} type={b.get('error', {}).get('type') if isinstance(b, dict) else None} x_provider={xp}")
    print(f"proveedor /health después: {estado_proveedor()}")


# ---------- Paso 3: esperar ready ----------

def esperar():
    t0 = time.time()
    while time.time() - t0 < 900:
        s, st = estado_proveedor()
        if s == 200:
            break
        if st == "failed":
            break
        time.sleep(5)
    print(f"proveedor /health: status={s} estado={st} tras {time.time() - t0:.0f} s de espera")
    for l in logs(C_PROV):
        if any(k in l for k in ("Tope de contexto", "Cargando modelo", "Modelo cargado", "Límite de VRAM", "Calentamiento")):
            # Líneas del proveedor: solo números de VRAM y tiempos, sin contenido.
            print("LOG", l[l.find(" ") + 1:] if l[:4].isdigit() else l)


# ---------- Prueba A ----------

BASE = ("Estoy organizando un viaje en tren por el norte del país y quiero que me ayudes a planificar "
        "cada etapa con detalle, incluyendo horarios, comidas típicas y lugares para visitar. ")
CIERRE = " Respondé con una descripción larga y detallada."


def mensaje_usuario(i):
    largo = 400 + (i * 73) % 201  # 400..600, varía por turno
    cuerpo = f"Turno {i}. " + BASE * 10
    cuerpo = cuerpo[: largo - len(CIERRE)] + CIERRE
    assert len(cuerpo) == largo
    return cuerpo


def lineas_recorte():
    return [l for l in logs(C_ORQ) if "Historial recortado" in l]


PATRON = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (INFO) orchestrator: Historial recortado: modelo=(\S+) "
    r"descartados=(\d+) tokens_estimados_antes=(\d+) tokens_estimados_despues=(\d+) disponible=(-?\d+) tope=(\d+)\.$"
)


def pruebaA(conv):
    system = open("prompts/system_prompt.txt", encoding="utf-8-sig").read().rstrip()
    guardados = []  # [{"role","content"}] del lado del cliente, solo para reconstruir la estimación
    previas = len(lineas_recorte())
    filas = []
    for i in range(1, 15):
        nuevo = {"role": "user", "content": mensaje_usuario(i)}
        # Lo que el orquestador debería mandar: últimos 8 guardados, recortados con el mismo estimador.
        historial = guardados[-HISTORY_LIMIT:]
        fijos = [{"role": "system", "content": system}, nuevo]
        disp = disponible_para_mensajes(TOPE, MAX_TOKENS)
        recortado = recortar_historial(fijos, historial, disp)
        est_local = None if recortado is None else tokens_de_mensajes(fijos + recortado)

        t0 = time.perf_counter()
        s, b = http("POST", f"{ORQ}/v1/chat/completions",
                    {"model": MODELO, "messages": [nuevo], "max_tokens": MAX_TOKENS, "x_conversation_id": conv})
        dt = time.perf_counter() - t0
        nuevas = lineas_recorte()[previas:]
        previas += len(nuevas)
        fila = {"turno": i, "len_user": len(nuevo["content"]), "status": s, "code": code_de(b), "s": round(dt, 1),
                "hist_leido": len(historial), "hist_enviado_local": None if recortado is None else len(recortado),
                "est_local": est_local}
        if isinstance(b, dict) and s == 200:
            u = b.get("usage", {})
            fila.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                        finish=b["choices"][0].get("finish_reason"), x_memory=(b.get("x_memory") or {}).get("status"),
                        tps=(b.get("x_metrics") or {}).get("tokens_per_s"))
            if fila["x_memory"] == "saved":
                guardados += [nuevo, {"role": "assistant", "content": b["choices"][0]["message"]["content"]}]
        if nuevas:
            m = PATRON.match(nuevas[-1])
            fila["log_recorte"] = None if m is None else {
                "ts": m.group(1), "nivel": m.group(2), "modelo": m.group(3), "descartados": int(m.group(4)),
                "antes": int(m.group(5)), "despues": int(m.group(6)), "disponible": int(m.group(7)), "tope": int(m.group(8))}
            fila["log_recorte_n"] = len(nuevas)
            fila["log_formato_ok"] = m is not None
        print("FILA " + json.dumps(fila, ensure_ascii=True), flush=True)
        filas.append(fila)
    # Memoria: cuántos mensajes quedaron guardados (solo el número).
    s, b = http("GET", f"{MEM}/v1/conversations/{conv}/messages?user_id=dev-user&limit=200", timeout=10)
    print(f"MEMORIA status={s} mensajes={len(b.get('messages', [])) if isinstance(b, dict) else None}")


# ---------- Prueba B ----------

def pruebaB(conv):
    gen_antes = sum("Generación:" in l for l in logs(C_PROV))
    post_antes = sum("POST /v1/chat/completions" in l for l in logs(C_PROV))
    largo = (BASE * 60)[:8000]
    s, b = http("POST", f"{ORQ}/v1/chat/completions",
                {"model": MODELO, "messages": [{"role": "user", "content": largo}], "max_tokens": MAX_TOKENS,
                 "x_conversation_id": conv}, timeout=60)
    err = b.get("error", {}) if isinstance(b, dict) else {}
    print(f"status={s} code={err.get('code')} type={err.get('type')} param={err.get('param')} "
          f"mensaje_ok={'aunque se descarte todo el historial' in (err.get('message') or '')} "
          f"claves={sorted(b) if isinstance(b, dict) else None} len_user={len(largo)}")
    time.sleep(1)
    print(f"proveedor: lineas 'Generación:' antes={gen_antes} después={sum('Generación:' in l for l in logs(C_PROV))}; "
          f"POST /v1/chat/completions antes={post_antes} después={sum('POST /v1/chat/completions' in l for l in logs(C_PROV))}")
    sm, bm = http("GET", f"{MEM}/v1/conversations/{conv}/messages?user_id=dev-user&limit=200", timeout=10)
    print(f"memoria: status={sm} mensajes={len(bm.get('messages', [])) if isinstance(bm, dict) else None}")
    ultima = None
    for l in logs(C_ORQ):
        if "Pedido que no entra sin historial" in l:
            ultima = l  # solo números: modelo, tokens_estimados, disponible, tope
    print("LOG_ORQ_400 " + (re.sub(r"[^\x20-\x7e]", "?", ultima) if ultima else "ninguna"))


if __name__ == "__main__":
    accion = sys.argv[1]
    if accion == "paso2":
        paso2()
    elif accion == "esperar":
        esperar()
    elif accion == "pruebaA":
        pruebaA(sys.argv[2])
    elif accion == "pruebaB":
        pruebaB(sys.argv[2])
    else:
        sys.exit(f"acción desconocida: {accion}")
