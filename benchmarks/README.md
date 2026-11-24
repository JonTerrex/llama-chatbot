# Benchmarks: registro de mediciones

> **Qué es:** la evidencia medida detrás de las decisiones del proveedor `llama-local` (tope de contexto, reserva por pedido, manejo del OOM) y del orquestador (propagación de errores, recorte del historial por tokens, logging).
> **Marcas:** ✅ medido, con salida cruda en esta carpeta · ⚠️ decisión propia, inferencia o riesgo · ❓ sin salida cruda o no verificado.
> **Importante:** son **mediciones puntuales de desarrollo**, hechas mientras se construía cada paso. Las definitivas se van a repetir cuando el proyecto esté terminado.

## Contenido

```
benchmarks/
├── README.md
├── vram/
│   ├── 3e_barrido/         sweep.py, analizar.py y 4 salidas (salida_a, salida_b, salida_a256, salida_b256)
│   ├── 3g_tope_y_oom/      oom.py + salida_oom.txt (X4), x123.py (X1 a X3, sin salida cruda)
│   └── 3h_oom_repetido/    oom3h.py + salida_oom3h.txt y salida_oom3h_b.txt (dos corridas)
└── orquestador/            vivo_r123.py + salida_pruebaA.txt (prueba en vivo de R1 a R3)
```

Hay una carpeta por medición porque cada una se corrió contra un commit distinto: el script, sus salidas y el commit quedan juntos.

Las salidas son la salida estándar cruda de cada corrida. Contienen:
- **Líneas `DIAG {json}` o `FILA {json}`:** solo números, estados y códigos.
- **Logs de `llama_local`:** también solo números.
- **Barras de progreso y avisos de `transformers`:** traen URLs públicas de huggingface.co.
- **Texto de la excepción OOM de torch:** solo números y una URL pública de docs.pytorch.org.

Ninguna salida trae contenido de mensajes ni de prompts. Los prompts son texto sintético repetido, armado por cada script.

## Hardware y entorno

- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB (según el log del proveedor: "de 8.00 GB").
- **Sistema:** Windows 11 con Docker Desktop sobre WSL2. Los diagnósticos de VRAM corren dentro de la imagen de `llama-local` (`python:3.12-slim`).
- **Modelo:** Llama 3 8B Instruct en 4 bits, NF4 con `bitsandbytes`, compute dtype bf16. Atención `sdpa` (campo `attn` de las salidas del barrido).
- **Librerías** (de `uv.lock`):
  - `torch` 2.15.0; dentro del contenedor, `2.15.0+cu132` con CUDA 13.2, según las salidas.
  - `transformers` 5.22.1.
  - `bitsandbytes` 0.50.2.
  - `accelerate` 1.15.0.
- **Orquestador:** `fastapi` 0.162.0 y `httpx2` 2.16.0.

## Cómo se corren

**Diagnósticos de VRAM.** Se corren en Git Bash, desde la raíz del repo, dentro de la imagen de `llama-local`, con el modelo y la carpeta del script montados en solo lectura. Cada script importa `src.providers.llama_local.engine`, así que hay que montar el `src/` **del commit indicado en cada sección** (por ejemplo, con `git worktree add <dir> <commit>`):

```bash
MW=$(cygpath -m "$(pwd)/modelos")
DW=$(cygpath -m "$(pwd)/benchmarks/vram/<carpeta>")
SRC=$(cygpath -m "<worktree del commit>/src")
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
  -v "$MW:/app/modelos:ro" -v "$DW:/diag:ro" -v "$SRC:/app/src:ro" \
  -e PYTHONPATH=/app [-e VARIABLE=valor ...] \
  --entrypoint python llama-chatbot-llama-local /diag/<script>.py > salida.txt
```

- ⚠️ **El comando está reconstruido:** es el patrón con que se corrieron los diagnósticos, pero no se volvió a ejecutar al armar esta carpeta.
- ⚠️ **El allocator de la imagen:** desde el commit 2c201ff, `Dockerfile.llama` fija `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Para repetir una variante con el allocator por defecto hay que usar una imagen construida antes de ese commit. ❓ No se probó si pasar `-e PYTORCH_CUDA_ALLOC_CONF=` vacío equivale al allocator por defecto.
- **Antes de medir:** cerrar las aplicaciones que usan GPU. Cambian la VRAM libre al arrancar, y con ella el tope de contexto (ver "Limitaciones").

**Prueba del orquestador:** ver su sección.

---

## 1. Curva de VRAM contra largo del prompt (paso 3e)

**Pregunta:** ¿cuánto crece la VRAM con el largo del prompt, dónde está el umbral real de OOM y qué cambia con `expandable_segments`? Con esto se fijaron el costo por token, la reserva por pedido y el tope de contexto.

**Script:** `vram/3e_barrido/sweep.py`, más `analizar.py` (regresión lineal sobre las salidas).

**Qué hace:** carga el motor como el servicio (calentamiento, tope y límite de proceso). Después pide prompts de largo exacto saltándose el tope: primero ascendentes (250 a 1500 y luego de a 50), después una bisección hasta ±8 tokens alrededor del primer OOM, y por último descendentes de a 100.

**Commit:** 541bca3. ⚠️ Deducido: las salidas son del 2026-11-20, de 11:18 a 11:32, posteriores al commit (11:14). Su log del tope no tiene `reserva_mib`, que entra en 2c201ff. El script usa `_generar(..., verificar_vram=False)`, un parámetro que ya no existe: **con el código actual falla**.

**Variables por salida** (de las cabeceras de cada salida; el script lee `VARIANTE`, `MAX_NEW`, `TOPE_BARRIDO`, `PYTORCH_CUDA_ALLOC_CONF` y `MODEL_PATH`):

| Salida | `MAX_NEW` | `PYTORCH_CUDA_ALLOC_CONF` | Barrido |
|---|---|---|---|
| `salida_a.txt` | 8 | (sin definir) | 250 a 2300 |
| `salida_b.txt` | 8 | `expandable_segments:True` | 250 a 2300 |
| `salida_a256.txt` | 256 | (sin definir) | 250 a 3200 (`TOPE_BARRIDO=3200`) |
| `salida_b256.txt` | 256 | `expandable_segments:True` | 250 a 3200 (`TOPE_BARRIDO=3200`) |

**Análisis:** `python benchmarks/vram/3e_barrido/analizar.py benchmarks/vram/3e_barrido/salida_*.txt`. No usa la GPU.

**Resultados** ✅ (las rectas son de `analizar.py`; los demás números, de las salidas):

| Variante | Pico de VRAM (MiB) | OK / OOM | Umbral de OOM (último OK / primer OOM) | Tope calculado al arrancar | Carga |
|---|---|---|---|---|---|
| a (defecto, `max_new` 8) | 5900,0 + 0,2134 × tokens (n=31; desvío máx. +6,3) | 31 / 0 | sin OOM hasta 2300 | 3001 | 125,8 s |
| b (expandable, `max_new` 8) | 5867,0 + 0,2124 × tokens (n=31; desvío máx. −3,4) | 31 / 0 | sin OOM hasta 2300 | 3234 | 122,0 s |
| a256 (defecto, `max_new` 256) | 5900,7 + 0,2130 × tokens (n=32; desvío máx. +6,5) | 32 / 29 | **2312 / 2318** | 2943 | 125,9 s |
| b256 (expandable, `max_new` 256) | 5867,4 + 0,2121 × tokens (n=41; desvío máx. −3,7) | 41 / 20 | **2700 / 2706** | 3127 | 169,0 s |

- **Costo fijo por pedido:** la ordenada del pico menos la memoria asignada en reposo de la misma salida da 5900,0 − 5781,1 = 118,9 MiB (a) y 5867,0 − 5748,6 = 118,4 MiB (b). Es el "~119 MiB" de la reserva por pedido. ⚠️ Su causa no está medida; la hipótesis es una descuantización temporal de `bitsandbytes`.
- **Pendiente:** 0,212 a 0,213 MiB por token de prompt. El motor usa 0,25 MiB por token (factor 2 sobre el KV cache teórico), que queda por encima de lo medido.
- **Reservada:** con el allocator por defecto, la recta de la reservada (a256) es 6047,2 + 0,2229 × tokens; con expandable (b256), 6079,6 + 0,1759 × tokens. En las cuatro salidas, el mayor desvío de la reservada contra su recta está en 1500 tokens: entre +116,5 y +182,5 MiB.
- ⚠️ **Histéresis con expandable (b256):** la bisección dio 2700 como último OK. En la fase descendente, después de 13 OOM (10 ascendentes y 3 de bisección), también dieron OOM 2700 y 2600, y recién 2500 entró. Durante esos OOM, la reservada quedó entre 6384 y 6424 MiB. Es una observación sin diagnóstico.
- **Conclusión usada en el código:** con `max_new` 256, `expandable_segments` sube el umbral real de 2312 a 2700 tokens de prompt. Se adoptó en 2c201ff.
- ⚠️ **Los topes de esta tabla son de 541bca3**, que todavía no descontaba la reserva por pedido de 150 MiB. No son comparables con los topes de las secciones 2 a 4.

## 2. Pedidos en el tope de contexto (paso 3g, X1 a X3)

**Pregunta:** sin la verificación de VRAM por pedido, ¿los pedidos justo en el tope entran siempre?

**Script:** `vram/3g_tope_y_oom/x123.py <tope>`. Corre en el host contra `llama-local` levantado en `127.0.0.1:8004`, y el tope se pasa como argumento (se lee del log "Tope de contexto" del proveedor). Arma los prompts contando tokens con el `400 context_length_exceeded` del propio proveedor:
- **X1:** 10 pedidos con prompt + 256 = tope − 1.
- **X2:** lo mismo, con una salida larga, y después un pedido corto.
- **X3:** tope + 1, que tiene que dar 400.

**Commit:** a1f6444 (paso 3g). ⚠️ Deducido: el script es del 2026-11-23 10:00, con los cambios del paso todavía sin commitear; el commit que los incluye es de las 10:46.

**Resultado:** ❓ **reportado en sesión, sin salida cruda:**
- **X1:** 10 de 10 pedidos con `200`.
- **X2:** el primer intento cortó solo, en 180 tokens; con una instrucción de repetición dio 256 tokens y `finish_reason: "length"`.
- **X3:** no quedó registrado.

La medición anterior (paso 3f), con la verificación por pedido, dio 5 rechazos falsos en 9 pedidos idénticos de 2390 tokens contra 598 MiB de presupuesto, con ±200 MiB de ruido de `mem_get_info` entre pedidos. Ese dato está en un comentario de `src/providers/llama_local/engine.py`. ❓ Tampoco tiene salida cruda en esta carpeta.

## 3. VRAM tras un OOM real, antes del arreglo (paso 3g, X4)

**Pregunta:** después de un OOM real al generar, ¿el proceso devuelve la VRAM?

**Script:** `vram/3g_tope_y_oom/oom.py`. `OBJETIVO` es el largo del prompt en tokens (por defecto 3200), con `max_new` 256 y saltándose el tope. Después corre un pedido corto. Esta salida fue con `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (campo `alloc_conf`).

**Commit:** a1f6444 (mismo criterio que la sección 2; salida del 2026-11-23 10:05).

**Resultados** (`salida_oom.txt`) ✅:

| Momento | Reservada (MiB) | Asignada (MiB) | Otros |
|---|---|---|---|
| En reposo, tras la carga | 5906,0 | 5748,6 | tope calculado: 2215 |
| Tras el OOM | **6304,0** | **6187,3** | `tokens_pedido` 3456; necesaria estimada 864,0 MiB contra un presupuesto de 554,0; pico 6357,9; `causa` = `OutOfMemoryError` (excepción encadenada) |
| Tras un pedido corto | 6324,0 | 5748,6 | 128 tokens, 23,38 tokens/s |

- ⚠️ **Hallazgo:** después del OOM quedaron ~439 MiB asignados de más (6187,3 contra 5748,6) mientras la excepción seguía viva. La causa es que la excepción, encadenada, retenía los tensores del intento. Esto motivó el paso 3h.

## 4. VRAM tras OOMs repetidos, con el arreglo (paso 3h, Y1 a Y4)

**Pregunta:** si la VRAM se libera fuera del `except` y sin encadenar la excepción, ¿vuelve la memoria al reposo? ¿Se mantiene estable con OOMs seguidos?

**Script:** `vram/3h_oom_repetido/oom3h.py`. `OBJETIVO` vale 3400 por defecto (más `max_new` 256, `tokens_pedido` 3656). Hace:
- **Y1:** mide en reposo.
- **Y2:** provoca un OOM.
- **Y3:** corre un pedido corto.
- **Y4.1 a Y4.3:** tres ciclos más de OOM y pedido corto.

También lee `nvidia-smi`. El allocator lo fija la imagen (`expandable_segments`, ver "Cómo se corren").

**Commit:** ae1894f. La corrida del 2026-11-23 11:33 fue con los cambios del paso todavía sin commitear; la del 2026-11-24 10:53, sobre el commit.

**Resultados** ✅:

| | `salida_oom3h.txt` (2026-11-23) | `salida_oom3h_b.txt` (2026-11-24) |
|---|---|---|
| Tope calculado al arrancar | 1759 | 1938 |
| En reposo: reservada / asignada (MiB) | 5906,0 / 5748,6 | 5906,0 / 5748,6 |
| Tras cada uno de los 4 OOM: reservada / asignada | **5802,0 / 5748,6** (las 4 veces) | **5802,0 / 5748,6** (las 4 veces) |
| Tras cada pedido corto: reservada / asignada | 5844,0 / 5748,6 | 5844,0 / 5748,6 |
| `__cause__` / `__context__` de `VRAMInsuficiente` | `None` / `None` | `None` / `None` |
| Tokens/s de los pedidos cortos (128 tokens) | 25,25 a 27,98 | 25,57 a 27,93 |
| Duración del OOM | 0,573 a 1,295 s | 0,729 a 1,019 s |
| `nvidia-smi` en reposo / tras el OOM (MiB) | 7557 / 7101 a 7105 | 6873 / 6769 |

- **Conclusión:** tras el OOM, la asignada vuelve exactamente al reposo (5748,6). La reservada queda en 5802,0, por debajo del reposo (5906,0) y de los 6304,0 del paso 3g. Con 4 OOM seguidos no crece nada.
- ⚠️ **`nvidia-smi` es absoluto:** incluye las demás aplicaciones de Windows. La primera corrida tenía más VRAM ocupada por otros procesos (7557 contra 6873 MiB en reposo), y por eso su tope salió en 1759.
- ⚠️ **Rango del tope:** 1759 queda **por debajo** del rango "1938 a 3234" que citan `docs/contrato-proveedores.md` y el comentario de `providers.yaml`. Con la fórmula actual, los topes con salida cruda en esta carpeta son 1759, 1938 y 2215; el 2527 de la prueba en vivo está solo en el reporte de la sesión (sección 5).

## 5. Orquestador contra los servicios reales (R1 a R3)

**Pregunta:**
- **R1:** ¿el orquestador propaga el 503 del proveedor cargando?
- **R2:** ¿el recorte del historial por tokens mantiene una conversación larga dentro del tope, y el estimador es conservador?
- **R3:** ¿los logs salen con timestamp y nivel, y sin URLs internas?

**Script:** `orquestador/vivo_r123.py`. Usa `src/orchestrator/presupuesto.py` del repo, en modo solo lectura, para reconstruir la estimación de cada turno. Se corre desde la raíz del repo, con el stack levantado:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile dev --profile llama build orchestrator llama-local
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile dev --profile llama up -d orchestrator llama-local
PYTHONPATH=. uv run python benchmarks/orquestador/vivo_r123.py paso2          # apenas arranca, con el proveedor cargando
PYTHONPATH=. uv run python benchmarks/orquestador/vivo_r123.py esperar        # hasta /health 200
PYTHONPATH=. uv run python benchmarks/orquestador/vivo_r123.py pruebaA <conversation_id nuevo> > salida_pruebaA.txt
PYTHONPATH=. uv run python benchmarks/orquestador/vivo_r123.py pruebaB <conversation_id nuevo>
```

El script usa los puertos publicados en loopback (8002 el orquestador, 8003 la memoria, 8004 el proveedor) y `docker logs` de `llama-chatbot-orchestrator-1` y `llama-chatbot-llama-local-1`. Imprime solo números, estados y códigos.

**Commit:** 84ea34f (2026-11-24).

**Prueba A: 14 turnos con memoria** (`salida_pruebaA.txt`) ✅. Parámetros: `max_tokens=150`, mensajes de usuario de 417 a 600 caracteres, `context_tokens` = 1800. Resultados:
- **Respuestas:** 14 de 14 con `200` y `x_memory.status = "saved"`. Al final, la memoria tenía 28 mensajes.
- **Recorte:** desde el turno 5, 10 turnos con recorte, de 2 mensajes cada uno, con `disponible` 1634 y tope 1800.
- **Estimación local:** en los 10 turnos con recorte coincide exactamente con `tokens_estimados_despues` del log.
- **Prompt real:** de 257 a 1191 tokens. El máximo de prompt real + 150 es 1341, por debajo de 1800.
- **Cociente estimado/real:** (estimado + 16) / `usage.prompt_tokens`. Va de **1,188 a 1,304**: 1,304 en el turno 1, 1,188 a 1,204 en los turnos con recorte. Siempre ≥ 1, así que el estimador fue conservador.
- **Velocidad y tiempos:** de 24,14 a 26,7 tokens/s, con 5,7 a 6,3 s por turno.
- **Logs (R3):** 10 líneas "Historial recortado", todas con timestamp y nivel (`log_formato_ok`).

| Turno | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Prompt real | 257 | 564 | 838 | 1131 | 1154 | 1124 | 1147 | 1171 | 1143 | 1166 | 1191 | 1162 | 1183 | 1155 |
| Cociente | 1,304 | 1,229 | 1,214 | 1,201 | 1,198 | 1,195 | 1,192 | 1,194 | 1,188 | 1,197 | 1,203 | 1,204 | 1,202 | 1,194 |

**Pasos `paso2`, `esperar` y `pruebaB`:** ❓ reportado en sesión, sin salida cruda (se imprimieron en la consola y no se guardaron):
- **R1:** con el proveedor en `/health` 503 `loading`, el orquestador respondió `503 provider_unavailable` con `x_provider.name`.
- **Arranque del proveedor:** cargó en 120,4 s; el log dio el tope `efectivo=2527` (`libre_mib` 880,4, `presupuesto_mib` 631,8); `nvidia-smi` ya listo marcaba 7216 MiB usados.
- **Prueba B (caso 6 de R2):** un mensaje de 8000 caracteres dio `400 context_length_exceeded` del orquestador (`tokens_estimados` 2828 contra `disponible` 1634). Los pedidos al proveedor no aumentaron (15 antes y 15 después) y la conversación quedó con 0 mensajes.
- **R3:** 0 líneas de `httpx2`/`httpcore2` en el log del orquestador; la única URL era la del arranque de uvicorn.

## Valores reportados en sesión, sin salida cruda

Estos valores se citan en la documentación o en el código, pero no tienen salida cruda en esta carpeta. ❓ No se recalcularon:

| Valor | Origen | Con salida cruda |
|---|---|---|
| Carga en frío de 120 a 375 s | Sesión, y el comentario de `docker-compose.gpu.yml` ("de 194 a 375 s") | Solo 122,0 a 169,0 s (sección 1); 120,4 s solo en el reporte de la sección 5 |
| Generación de 22 a 28 tokens/s | Sesión | De 23,38 a 27,98 tokens/s en las secciones 3, 4 y 5 |
| Tope de contexto de 1938 a 3234 tokens | `docs/contrato-proveedores.md`, `providers.yaml` | Con la fórmula actual: 1759, 1938 y 2215 (secciones 3 y 4). El 3234 es de la sección 1, con la fórmula anterior. Ver la advertencia de la sección 4 |
| 10 de 10 en el tope (X1) y X2 | Sesión, paso 3g | No (sección 2) |
| 5 rechazos falsos en 9 pedidos (paso 3f) | Comentario de `engine.py` | No |

## Limitaciones

- **Una sola máquina** (la de la sección "Hardware y entorno") y **una sola corrida por prueba**, salvo el paso 3h, que tiene dos. No hay dispersión medida.
- **Las aplicaciones abiertas en Windows** cambian la VRAM libre al arrancar y, con ella, el tope de contexto: de 1759 a 2215 con la misma fórmula en estas salidas. También cambian el valor absoluto de `nvidia-smi`.
- **Bajo WSL2/WDDM, `mem_get_info` no ve la VRAM que usan otros procesos**, y la VRAM puede derramarse a la RAM del sistema sin que se detecte. Ningún benchmark mide VRAM competida por otra aplicación.
- **Textos de prueba sintéticos y repetitivos en español.** El cociente del estimador (sección 5) puede cambiar con código, números u otros idiomas.
- **Linux nativo sin probar.**
- **Los scripts de VRAM dependen del commit indicado** (usan métodos internos del motor) y no corren contra el código actual sin adaptarlos.
