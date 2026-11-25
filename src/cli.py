"""
Consola para desarrolladores: prueba el stack levantado en contenedores hablando SOLO por HTTP.

Nunca carga el modelo ni importa torch/transformers: la GPU es del contenedor llama-local.
Uso: python -m src.cli [--debug] [--orchestrator-url URL] ... <subcomando> [opciones]

Subcomandos: status, models, chat, ask, history (ver --help de cada uno).
"""
import argparse
import os
import re
import secrets
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, TextIO
from urllib.parse import urlsplit

import requests

ORCHESTRATOR_URL_DEFAULT = "http://127.0.0.1:8002"
MEMORY_URL_DEFAULT = "http://127.0.0.1:8003"
PROVIDER_URL_DEFAULT = "http://127.0.0.1:8004"
DEV_USER_ID_DEFAULT = "dev-user"

# Mismo patrón que src/orchestrator/config.py y src/memory/app.py (copiado: importar config
# arrastra pyyaml). Sin anclas y con fullmatch: con "^...$" y match, "$" acepta un "\n" final.
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

HOSTS_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
STATUS_TIMEOUT_S = 5.0
# Mayor que PROVIDER_TIMEOUT_S del orquestador (120 s): así llega su 504 limpio antes del corte.
PEDIDO_TIMEOUT_S = 130.0
HISTORY_LIMIT_DEFAULT = 20
HISTORY_LIMIT_MAX = 200  # HISTORY_LIMIT_MAX de src/memory/app.py
PALABRAS_DE_SALIDA = {"salir", "exit", "chau"}
# Métricas de x_metrics que informan los proveedores (llama_local/app.py y mock/app.py), en este orden.
METRICAS = ("latency_s", "tokens_per_s", "vram_peak_mb", "vram_reserved_mb")


class FalloDeServicio(Exception):
    """Error de servicio o de conexión, con un mensaje listo para mostrar (sin contenido de usuario)."""


@dataclass
class Contexto:
    session: Any
    out: TextIO
    err: TextIO
    debug: bool


# ---------- Utilidades ----------

def url_visible(url: str) -> str:
    """scheme://host[:puerto] de la URL base, sin credenciales ni ruta."""
    partes = urlsplit(url)
    host = partes.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    puerto = f":{partes.port}" if partes.port else ""
    return f"{partes.scheme}://{host}{puerto}"


def es_loopback(url: str) -> bool:
    return (urlsplit(url).hostname or "") in HOSTS_LOOPBACK


def avisar_si_no_es_loopback(ctx: Contexto, nombre_var: str, url: str) -> None:
    if not es_loopback(url):
        print(
            f"Aviso: {nombre_var} apunta a {url_visible(url)}, que no es loopback; "
            "los servicios no autentican.",
            file=ctx.err,
        )


def describir_error(servicio: str, resp: Any) -> str:
    """HTTP status + error.code + error.message. Nunca el cuerpo crudo."""
    try:
        data = resp.json()
    except ValueError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return f"{servicio}: HTTP {resp.status_code} sin cuerpo de error reconocible."
    code = error.get("code") or "sin code"
    message = error.get("message") or "sin mensaje"
    return f"{servicio}: HTTP {resp.status_code} {code}: {message}"


def pedir(
    ctx: Contexto, servicio: str, metodo: str, url_base: str, ruta: str, *, timeout: float, **kwargs: Any
) -> dict[str, Any]:
    """Hace el pedido y devuelve el JSON (un objeto). Cualquier falla sale como FalloDeServicio."""
    visible = url_visible(url_base)
    sugerencia = "Probá `python -m src.cli status`."
    try:
        resp = getattr(ctx.session, metodo)(url_base + ruta, timeout=timeout, **kwargs)
    # Timeout antes que ConnectionError: ConnectTimeout es las dos cosas.
    except requests.Timeout as e:
        raise FalloDeServicio(f"{servicio} ({visible}) no respondió en {timeout:g} s. {sugerencia}") from e
    except requests.ConnectionError as e:
        raise FalloDeServicio(f"No se pudo conectar con {servicio} en {visible}. {sugerencia}") from e
    except requests.RequestException as e:
        raise FalloDeServicio(f"Falló el pedido a {servicio} en {visible} ({type(e).__name__}).") from e
    if not 200 <= resp.status_code < 300:
        raise FalloDeServicio(describir_error(servicio, resp))
    try:
        data = resp.json()
    except ValueError as e:
        raise FalloDeServicio(f"{servicio}: HTTP {resp.status_code} con un cuerpo que no es JSON.") from e
    if not isinstance(data, dict):
        raise FalloDeServicio(f"{servicio}: HTTP {resp.status_code} con un JSON que no es un objeto.")
    return data


def informar_fallo(ctx: Contexto, e: FalloDeServicio) -> None:
    print(f"Error: {e}", file=ctx.err)
    if ctx.debug:
        traceback.print_exception(e, file=ctx.err)


def cuerpo_del_pedido(texto: str, args: argparse.Namespace) -> dict[str, Any]:
    """Un solo mensaje user y los parámetros que se hayan dado. Nunca "stream"."""
    cuerpo: dict[str, Any] = {"messages": [{"role": "user", "content": texto}]}
    if args.model is not None:
        cuerpo["model"] = args.model
    if args.max_tokens is not None:
        cuerpo["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        cuerpo["temperature"] = args.temperature
    return cuerpo


def contenido_de(data: dict[str, Any]) -> str:
    try:
        contenido = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise FalloDeServicio("La respuesta no trae choices[0].message.content.") from e
    if not isinstance(contenido, str):
        raise FalloDeServicio("choices[0].message.content no es texto.")
    return contenido


def linea_de_datos(latencia_s: float, data: dict[str, Any]) -> str:
    """Latencia total, usage, x_memory.status y las x_metrics presentes; solo lo que exista."""
    partes = [f"total {latencia_s:.2f} s"]
    usage = data.get("usage")
    if isinstance(usage, dict):
        tokens = [f"{k} {usage[k]}" for k in ("prompt_tokens", "completion_tokens") if k in usage]
        if tokens:
            partes.append(", ".join(tokens))
    memoria = data.get("x_memory")
    if isinstance(memoria, dict) and "status" in memoria:
        partes.append(f"memoria {memoria['status']}")
    metricas = data.get("x_metrics")
    if isinstance(metricas, dict):
        presentes = [f"{k}={metricas[k]}" for k in METRICAS if k in metricas]
        if presentes:
            partes.append(" ".join(presentes))
    return "[" + " | ".join(partes) + "]"


def turno(ctx: Contexto, servicio: str, url_base: str, cuerpo: dict[str, Any], timeout: float) -> None:
    """Un pedido de chat: imprime la respuesta y la línea de datos. Falla con FalloDeServicio."""
    inicio = time.perf_counter()
    data = pedir(ctx, servicio, "post", url_base, "/v1/chat/completions", json=cuerpo, timeout=timeout)
    latencia_s = time.perf_counter() - inicio
    print(contenido_de(data), file=ctx.out)
    print(linea_de_datos(latencia_s, data), file=ctx.out)


# ---------- Subcomandos ----------

def cmd_status(ctx: Contexto, args: argparse.Namespace) -> int:
    servicios = (
        ("orquestador", "ORCHESTRATOR_URL", args.orchestrator_url),
        ("memoria", "MEMORY_URL", args.memory_url),
        ("proveedor", "PROVIDER_URL", args.provider_url),
    )
    listos = 0
    for nombre, var, url in servicios:
        avisar_si_no_es_loopback(ctx, var, url)
        inicio = time.perf_counter()
        try:
            resp = ctx.session.get(url + "/health", timeout=STATUS_TIMEOUT_S)
        except requests.RequestException as e:
            estado = "no disponible"
            if ctx.debug:
                traceback.print_exception(e, file=ctx.err)
        else:
            # El estado va en el cuerpo: la memoria responde 200 también en "loading", y el
            # proveedor 503 en loading/failed.
            try:
                cuerpo = resp.json()
            except ValueError:
                cuerpo = None
            valor = cuerpo.get("status") if isinstance(cuerpo, dict) else None
            estado = valor if isinstance(valor, str) else f"no disponible (HTTP {resp.status_code})"
        ms = (time.perf_counter() - inicio) * 1000
        if estado == "ready":
            listos += 1
        print(f"{nombre:<12} {estado:<14} {ms:>7.0f} ms  {url_visible(url)}", file=ctx.out)

    try:
        data = pedir(ctx, "el orquestador", "get", args.orchestrator_url, "/v1/models", timeout=STATUS_TIMEOUT_S)
        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        print(f"modelos      {', '.join(str(i) for i in ids) or '(ninguno)'}", file=ctx.out)
    except FalloDeServicio as e:
        print(f"modelos      no disponibles ({e})", file=ctx.out)
        if ctx.debug:
            traceback.print_exception(e, file=ctx.err)
    return 0 if listos == len(servicios) else 1


def cmd_models(ctx: Contexto, args: argparse.Namespace) -> int:
    avisar_si_no_es_loopback(ctx, "ORCHESTRATOR_URL", args.orchestrator_url)
    try:
        data = pedir(ctx, "el orquestador", "get", args.orchestrator_url, "/v1/models", timeout=STATUS_TIMEOUT_S)
    except FalloDeServicio as e:
        informar_fallo(ctx, e)
        return 1
    modelos = data.get("data")
    if not isinstance(modelos, list):
        informar_fallo(ctx, FalloDeServicio("el orquestador: /v1/models no trae una lista en 'data'."))
        return 1
    for m in modelos:
        if isinstance(m, dict) and "id" in m:
            print(m["id"], file=ctx.out)
    return 0


def cmd_chat(ctx: Contexto, args: argparse.Namespace, entrada: Callable[[str], str]) -> int:
    avisar_si_no_es_loopback(ctx, "ORCHESTRATOR_URL", args.orchestrator_url)
    conversacion = args.conversation or f"cli-{secrets.token_hex(4)}"
    print(f"Conversación: {conversacion} (retomala con --conversation {conversacion})", file=ctx.out)
    print("Salir con salir, exit, chau, Ctrl+C o EOF.", file=ctx.out)
    while True:
        try:
            texto = entrada("vos> ").strip()
            if not texto:
                continue
            if texto.lower() in PALABRAS_DE_SALIDA:
                break
            # Solo el mensaje nuevo: el historial lo pone el orquestador desde la memoria
            # (contrato de memoria, 4.1). El cliente no guarda nada.
            cuerpo = cuerpo_del_pedido(texto, args)
            cuerpo["x_conversation_id"] = conversacion
            try:
                turno(ctx, "el orquestador", args.orchestrator_url, cuerpo, args.timeout)
            except FalloDeServicio as e:
                informar_fallo(ctx, e)  # un turno fallido no termina la sesión
        except (EOFError, KeyboardInterrupt):
            print(file=ctx.out)
            break
    return 0


def cmd_ask(ctx: Contexto, args: argparse.Namespace) -> int:
    if args.target == "proveedor":
        servicio, var, url = "el proveedor", "PROVIDER_URL", args.provider_url
    else:
        servicio, var, url = "el orquestador", "ORCHESTRATOR_URL", args.orchestrator_url
    avisar_si_no_es_loopback(ctx, var, url)
    try:
        turno(ctx, servicio, url, cuerpo_del_pedido(args.texto, args), args.timeout)
    except FalloDeServicio as e:
        informar_fallo(ctx, e)
        return 1
    return 0


def cmd_history(ctx: Contexto, args: argparse.Namespace) -> int:
    avisar_si_no_es_loopback(ctx, "MEMORY_URL", args.memory_url)
    try:
        data = pedir(
            ctx, "la memoria", "get", args.memory_url, f"/v1/conversations/{args.conversation_id}/messages",
            params={"user_id": args.user_id, "limit": args.limit}, timeout=STATUS_TIMEOUT_S,
        )
    except FalloDeServicio as e:
        informar_fallo(ctx, e)
        return 1
    mensajes = data.get("messages")
    if not isinstance(mensajes, list):
        informar_fallo(ctx, FalloDeServicio("la memoria: la respuesta no trae una lista en 'messages'."))
        return 1
    if not mensajes:
        print("(sin mensajes)", file=ctx.out)
    for m in mensajes:  # la memoria ya los devuelve del más antiguo al más nuevo
        if isinstance(m, dict):
            print(f"[{m.get('role')}] {m.get('content')}", file=ctx.out)
    return 0


# ---------- Parseo ----------

def tipo_id(valor: str) -> str:
    if not ID_PATTERN.fullmatch(valor):
        raise argparse.ArgumentTypeError(
            f"{valor!r} no es válido: 1 a 64 caracteres entre letras, dígitos, '_' y '-'."
        )
    return valor


def tipo_limit(valor: str) -> int:
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{valor!r} no es un entero.")
    if not 1 <= n <= HISTORY_LIMIT_MAX:
        raise argparse.ArgumentTypeError(f"debe estar entre 1 y {HISTORY_LIMIT_MAX}.")
    return n


def construir_parser(entorno: dict[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli", description="Consola para probar el stack por HTTP (orquestador, memoria, proveedor)."
    )
    parser.add_argument("--debug", action="store_true", help="agrega el traceback a los errores")
    parser.add_argument("--orchestrator-url", default=entorno.get("ORCHESTRATOR_URL") or ORCHESTRATOR_URL_DEFAULT,
                        help="default: $ORCHESTRATOR_URL o %(default)s")
    parser.add_argument("--memory-url", default=entorno.get("MEMORY_URL") or MEMORY_URL_DEFAULT,
                        help="default: $MEMORY_URL o %(default)s")
    parser.add_argument("--provider-url", default=entorno.get("PROVIDER_URL") or PROVIDER_URL_DEFAULT,
                        help="default: $PROVIDER_URL o %(default)s")
    # Se valida solo en history, el único que lo usa.
    parser.add_argument("--user-id", default=entorno.get("DEV_USER_ID") or DEV_USER_ID_DEFAULT,
                        help="usuario para history; default: $DEV_USER_ID o %(default)s")
    sub = parser.add_subparsers(dest="comando", required=True, metavar="subcomando")

    sub.add_parser("status", help="estado de los tres servicios y modelos del orquestador")
    sub.add_parser("models", help="modelos que informa el orquestador")

    def params_de_generacion(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", help="modelo; en el orquestador, default DEFAULT_MODEL")
        p.add_argument("--max-tokens", type=int)
        p.add_argument("--temperature", type=float)
        p.add_argument("--timeout", type=float, default=PEDIDO_TIMEOUT_S, help="segundos; default %(default)g")

    p_chat = sub.add_parser("chat", help="conversación interactiva por el orquestador, con memoria")
    p_chat.add_argument("--conversation", "-c", type=tipo_id, help="id a retomar; sin él se genera uno nuevo")
    params_de_generacion(p_chat)

    p_ask = sub.add_parser("ask", help="un pedido suelto, sin memoria")
    p_ask.add_argument("texto")
    p_ask.add_argument("--target", choices=("orquestador", "proveedor"), default="orquestador")
    params_de_generacion(p_ask)

    p_hist = sub.add_parser("history", help="historial leído directo de la memoria")
    p_hist.add_argument("conversation_id", type=tipo_id)
    p_hist.add_argument("--limit", type=tipo_limit, default=HISTORY_LIMIT_DEFAULT,
                        help=f"1 a {HISTORY_LIMIT_MAX}; default %(default)s")
    return parser


def parsear(argv: Optional[list[str]], entorno: dict[str, str]) -> argparse.Namespace:
    """Parsea y valida. Uso incorrecto sale con SystemExit(2), como argparse."""
    parser = construir_parser(entorno)
    args = parser.parse_args(argv)
    for campo in ("orchestrator_url", "memory_url", "provider_url"):
        setattr(args, campo, getattr(args, campo).rstrip("/"))
    if args.comando == "ask" and args.target == "proveedor" and not args.model:
        parser.error("--target proveedor exige --model: el proveedor no tiene modelo por defecto.")
    if args.comando == "history" and not ID_PATTERN.fullmatch(args.user_id):
        parser.error(
            f"DEV_USER_ID/--user-id={args.user_id!r} no es válido: 1 a 64 caracteres entre letras, dígitos, '_' y '-'."
        )
    return args


def main(
    argv: Optional[list[str]] = None,
    *,
    session: Any = None,
    entrada: Callable[[str], str] = input,
    entorno: Optional[dict[str, str]] = None,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
) -> int:
    """Punto de entrada. session, entrada y entorno se inyectan en los tests (sin red real)."""
    args = parsear(argv, dict(os.environ) if entorno is None else entorno)
    ctx = Contexto(
        session=session if session is not None else requests.Session(),
        out=sys.stdout if out is None else out,
        err=sys.stderr if err is None else err,
        debug=args.debug,
    )
    if args.comando == "status":
        return cmd_status(ctx, args)
    if args.comando == "models":
        return cmd_models(ctx, args)
    if args.comando == "chat":
        return cmd_chat(ctx, args, entrada)
    if args.comando == "ask":
        return cmd_ask(ctx, args)
    return cmd_history(ctx, args)


if __name__ == "__main__":
    sys.exit(main())
