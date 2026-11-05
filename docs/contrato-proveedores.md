# Contrato entre orquestador y proveedores LLM

> **Estado:** borrador revisado (Fase 1). El proveedor **mock** ya implementa este contrato **sin streaming** (`src/providers/mock/`); el orquestador, el streaming y el proveedor GPU todavía no.
> **Marcas de verificación:** ✅ verificado con fuente · ⚠️ convención propia o evidencia indirecta · ❓ pendiente de decidir · sin marca: decisión tomada.

## 1. Principios

1. **Compatibilidad con OpenAI** como idioma común: todos los servicios hablan el mismo formato (`/v1/chat/completions`).
2. **Extensiones opcionales y aisladas.** Todo lo que no es estándar viaja en campos con prefijo `x_` (`x_metrics`, `x_debug`, `x_provider`). Son opcionales, solo aparecen cuando aplican, y quitarlos no debe cambiar nada de lo estándar. Si dieran problemas con algún cliente, se eliminan.
3. **Proveedores "tontos":** reciben `messages`, generan, devuelven. No tocan base de datos ni deciden el system prompt.
4. **Streaming desde el diseño:** cada proveedor soporta `stream: true` y `stream: false`.

## 2. Servicios y endpoints

| Servicio | Endpoints |
|---|---|
| **Orquestador** | `POST /v1/chat/completions`, `GET /v1/models`, `GET /health` |
| **Proveedor** (local, mock, online) | `POST /v1/chat/completions`, `GET /health` |
| **Mock** (solo desarrollo) | los del proveedor + `GET /debug/requests` |

## 3. Petición

`POST /v1/chat/completions`

```json
{
  "model": "llama3-8b-local",
  "messages": [{"role": "user", "content": "Hola"}],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 256
}
```

- Conjunto de parámetros: `model`, `messages`, `stream`, `temperature`, `max_tokens`.
- Los parámetros no soportados (`top_p`, `n`, etc.) **se ignoran**, no se rechazan con `400`: los clientes compatibles con OpenAI mandan muchos y rechazarlos los rompería. El orquestador registra un warning cuando descarta alguno; el mock los ignora en silencio.
- `messages[].role` ∈ `system`, `user`, `assistant`, y `content` es texto. ⚠️ Un `content` en forma de lista de partes, o los roles `developer`/`tool`, devuelven `400` por ahora (ver sección 11).

## 4. Respuesta sin streaming

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1762170000,
  "model": "llama3-8b-local",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 42, "completion_tokens": 128, "total_tokens": 170},
  "x_metrics": {
    "tokens_per_s": 27.4,
    "latency_s": 4.7,
    "vram_peak_mb": 5900,
    "vram_reserved_mb": 6050
  }
}
```

- `usage` es **estándar y solo tokens**. ✅ Los nombres `prompt_tokens`, `completion_tokens` y `total_tokens` coinciden con la salida de ejemplo del Cookbook de OpenAI (sección 12). El mock los **simula** contando palabras: cifras estables, no tokens reales.
- `x_metrics` es opcional. Cada clave aparece **solo si aplica**: `vram_*` solo en proveedores con GPU; el mock y los online no las envían. `tokens_per_s` se omite cuando `latency_s`, redondeada a milisegundos, vale 0: no hay una velocidad significativa que informar.
- Claves de `x_metrics`: `latency_s`, `tokens_per_s`, `vram_peak_mb` y `vram_reserved_mb` (inglés y snake_case, coherentes con `usage`). Quien las consuma debe tolerar que falten.
- Cambio respecto a hoy: `tokens_prompt` y `tokens_generados` pasan a `usage.prompt_tokens` y `usage.completion_tokens`.

## 5. Streaming (SSE)

`stream: true` → `Content-Type: text/event-stream`. Secuencia (la petición incluye `stream_options: {"include_usage": true}` para recibir el chunk final):

1. Primer chunk: `delta` con `{"role": "assistant"}`.
2. Chunks de texto: `choices[0].delta.content`.
3. Último chunk de texto: `delta` vacío y `finish_reason` (`stop` o `length`).
4. **Chunk final de métricas** (solo si el cliente envió `stream_options: {"include_usage": true}`): `choices: []`, más `usage` y `x_metrics`. ✅ Es el comportamiento de OpenAI: el chunk de `usage` se emite únicamente cuando se pide, y su `choices` es siempre un arreglo vacío (sección 12).
5. `data: [DONE]`.

```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Ho"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":42,"completion_tokens":128,"total_tokens":170},"x_metrics":{"tokens_per_s":27.4}}

data: [DONE]
```

- **Proveedores:** siguen el estándar al pie de la letra y emiten el chunk final solo si reciben `stream_options.include_usage`. Sin él, el stream termina con el chunk de `finish_reason` y `[DONE]`.
- **Orquestador → proveedores:** el orquestador **siempre** pide `include_usage` a los proveedores, para tener las métricas. ❓ Qué reenvía al cliente cuando este no lo pidió (propuesta: reenviar el chunk solo si lo pidió, y que la UI propia lo pida siempre).
- **Por qué no se emite siempre:** OpenAI advierte que el código que lee `choices[0]` debe comprobar antes que no se trata del chunk de `usage`, que trae `choices` vacío (sección 12). Emitirlo siempre, como proponía la versión anterior de este documento, rompería a los clientes que no lo pidieron.
- ⚠️ Con `include_usage` activo, OpenAI incluye `"usage": null` en el resto de los chunks; el ejemplo de arriba no lo muestra. Detalle a decidir al implementar el streaming.
- Aparcado para la Fase 2: tiempo hasta el primer token, y cómo se mide tokens/s con `TextIteratorStreamer` en `transformers`.

## 6. Errores

Formato de OpenAI más una extensión opcional. ✅ Los cuatro campos `message`, `type`, `param` y `code` son obligatorios en el esquema de OpenAI; `param` y `code` pueden ser `null` (sección 12).

```json
{
  "error": {
    "message": "El proveedor 'llama3-8b-local' no está disponible.",
    "type": "provider_error",
    "param": null,
    "code": "provider_unavailable",
    "x_provider": {"name": "llama3-8b-local", "state": "down"}
  }
}
```

| Situación | HTTP | `code` |
|---|---|---|
| Petición mal formada | 400 | `invalid_request` |
| `model` desconocido | 404 | `model_not_found` |
| Proveedor marcado como caído o cargando | 503 | `provider_unavailable` |
| Falla la conexión o respuesta inválida del proveedor | 502 | `provider_error` |
| El proveedor no responde a tiempo | 504 | `provider_timeout` |

- ⚠️ La tabla de códigos es **propuesta propia**. 502 y 503 coinciden con la práctica de OpenRouter, pero no es un estándar de OpenAI.
- `type`: `invalid_request_error` para peticiones mal formadas ✅ (valor que OpenAI devuelve en sus errores reales) y `provider_error` para fallos del proveedor ⚠️ (convención propia).
- `param` nombra el campo que falló (por ejemplo `messages`) o es `null`; nunca una posición dentro del cuerpo. Con JSON no parseable, FastAPI informa la posición del fallo y el mock la descarta (cubierto por test).
- Rutas o métodos inexistentes (404/405) usan el mismo objeto `error`, con `code: null` ✅ (OpenAI también devuelve `code: null` cuando no hay código).
- **Temporal, solo en el mock:** `stream: true` devuelve `501` con `code: "streaming_not_supported"` y `type: "provider_error"` hasta que se implemente el streaming. ⚠️
- `x_provider.state` ∈ `down`, `loading`, `timeout`. ❓ lista final de estados.
- **Errores en mitad de un stream** (el HTTP ya es 200): se envía un chunk `data:` con el mismo objeto `error` y se cierra con `data: [DONE]`. ⚠️ **Convención propia inspirada en OpenRouter**, que usa un campo `error` de primer nivel; no encontré un formato oficial de OpenAI. ❓ Si además se añade una opción con `finish_reason: "error"` como hace OpenRouter.

## 7. Salud y disponibilidad

- `GET /health` significa **"listo para atender"**, no "vivo":
  - `200` con `{"status": "ready"}`.
  - `503` con `{"status": "loading"}` mientras el modelo carga (el local tarda unos 2:10-2:40; medida del autor).
- El orquestador **sondea periódicamente** (intervalo configurable por variable de entorno) y guarda el estado en caché. No consulta a los proveedores en cada petición de `/v1/models`.
- **Marca pasiva:** si una petición real falla por conexión, el orquestador marca ese proveedor como caído al instante; vuelve a listarlo cuando un sondeo lo vea sano. ❓ Qué errores cuentan como "caído" y cuáles son errores normales de la petición.
- Un pequeño desfase es aceptable: si se muestra un proveedor como disponible y no lo está, se devuelve el error correspondiente (sección 6).

## 8. Selección de proveedor

- El campo estándar `model` decide a qué proveedor va la petición, mediante un **registro en configuración** (nombre de modelo → URL del proveedor) escrito en un archivo **YAML** (`providers.yaml`) montado como volumen. Si falta `model`, se usa el modelo indicado en la variable de entorno `DEFAULT_MODEL`; si tampoco está definida, la respuesta es `400`. ❓ Estructura exacta del archivo (se define al implementar el orquestador).
- `GET /v1/models` devuelve **solo los proveedores activos** en formato lista de OpenAI (`object: "list"`, `data: [...]`). La UI lo usa para el selector y para ofrecer un cambio de modelo cuando uno falla.
- Restricción de hardware: con ~5.9 GB de VRAM pico para Llama 3 8B en 4 bits sobre 8 GB, dos proveedores de GPU de ese tamaño probablemente no caben a la vez (inferencia a partir de medidas del autor, sin probar). El registro puede contener proveedores configurados pero no arrancados.

## 9. System prompt

- Lo inyecta **el orquestador** como primer mensaje `{"role": "system"}` antes de reenviar. Los proveedores no lo conocen.
- Inicialmente es **estático**, leído de `prompts/system_prompt.txt`. Escalable después (por nivel de acceso, etc.).
- Si el cliente ya envía un mensaje `system`, el orquestador **antepone el suyo** y conserva el del cliente (dos mensajes `system`, el nuestro primero). ⚠️ Habrá que revisarlo cuando exista acceso por niveles de información: un `system` del cliente no debería poder anular las reglas del orquestador.
- El formato de plantilla de chat (tokens especiales de Llama 3) es responsabilidad de cada proveedor, no del orquestador.

## 10. Mock (proveedor solo de desarrollo)

- Servicio propio (`src/providers/mock/`), con su imagen (`Dockerfile.mock`, unos 200 MB en disco, **sin torch**). Cumple el mismo contrato que cualquier proveedor: `usage`, errores y `/health`. **Streaming: pendiente**; mientras tanto `stream: true` devuelve el `501` temporal de la sección 6. No importa nada del resto de `src/`.
- Respuesta de contenido **fijo y predecible**, para que sirva de baseline de rendimiento y para pruebas rápidas.
- Devuelve `x_debug` con las `messages` y los `params` que recibió (`model`, `stream`, `temperature`, `max_tokens`), de modo que se puede ver el contexto real (system prompt incluido) antes de que lo toque un modelo. En streaming viajará en el chunk final.
- Guarda las últimas N peticiones, incluidas las de `stream: true`, y las expone en `GET /debug/requests` como `{"count", "max", "requests": [...]}`, en orden cronológico (la más nueva al final). Cada entrada: `{"received_at", "request": {"model", "messages", "params"}}`.
- Configuración por variables de entorno, leídas al arrancar: `MOCK_HISTORY_SIZE` (N, por defecto 20) y `MOCK_LATENCY` (segundos de espera simulada por petición, por defecto 0).
- **Nunca para producción:** `x_debug` y el endpoint exponen la conversación completa.
- Quitar `x_debug` y `/debug/requests` no debe cambiar el comportamiento estándar.

## 11. Pendientes

- ❓ Estados posibles en `x_provider.state` (hoy: `down`, `loading`, `timeout`).
- ❓ Si los errores en mitad de un stream añaden también `finish_reason: "error"`, como hace OpenRouter.
- ❓ Qué errores cuentan como "proveedor caído" para la marca pasiva (sección 7) y cuáles son errores normales de la petición.
- ❓ Estructura del archivo `providers.yaml` (sección 8).
- ❓ Qué reenvía el orquestador al cliente cuando este no pidió `include_usage` (sección 5).
- ⚠️ Compatibilidad con clientes externos: hoy `content` debe ser texto y los roles se limitan a `system`, `user` y `assistant`; un `content` en forma de lista de partes o los roles `developer`/`tool` devuelven `400`.
- ❓ Traducción de este documento al inglés (el repo es de portfolio): se hará cuando la especificación se estabilice; por ahora, español.

## 12. Evidencia consultada

| Punto | Estado | Fuente |
|---|---|---|
| Objeto `error`: `message`, `type`, `param` y `code` obligatorios; `param` y `code` anulables | ✅ indirecto | Esquemas generados a partir del OpenAPI de OpenAI: [ex_openai](https://hexdocs.pm/ex_openai/2.0.0-beta2/ExOpenAI.Components.Error.html), [openai-struct](https://docs.rs/openai-struct/latest/src/openai_struct/models/error.rs.html) |
| `type: "invalid_request_error"` y `code: null` en errores reales | ✅ indirecto | [Mensaje de error citado en OpenAI Developer Community](https://community.openai.com/t/improve-the-error-messages-for-json-schema-structured-output/1126645) |
| Chunk de `usage` solo con `include_usage`; `choices` siempre vacío | ✅ indirecto | [OpenAI Cookbook, "How to stream completions"](https://nbviewer.org/format/script/github/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb); [anuncio en OpenAI Developer Community](https://community.openai.com/t/usage-stats-now-available-when-using-streaming-with-the-chat-completions-api-or-completions-api/738156); [openai-struct, `ChatCompletionStreamOptions`](https://docs.rs/crate/openai-struct/latest/source/src/models/chat_completion_stream_options.rs) |
| Advertencia: el código que lee `choices[0]` debe comprobar el chunk de `usage`; `"usage": null` en el resto de chunks | ✅ indirecto | Anuncio en OpenAI Developer Community (enlace anterior) |
| Nombres `prompt_tokens`, `completion_tokens`, `total_tokens` | ✅ indirecto | Salida de ejemplo del OpenAI Cookbook (enlace anterior) |
| Clientes que fallan con `choices` vacío | ✅ indirecto | Documentación de streaming de OpenRouter, y la advertencia de OpenAI de arriba |
| Campos extra tolerados por el SDK `openai` | ⚠️ circunstancial | OpenRouter añade campos propios y recomienda ese SDK; no hay prueba directa |
| Formato de error en mitad del stream | ⚠️ convención | OpenRouter; no hay formato oficial confirmado |
| Página oficial de OpenAI | ❌ no verificada | `platform.openai.com` no estaba en la lista de sitios permitidos del entorno de trabajo (2026-11-05). Las fuentes de arriba son de OpenAI pero secundarias; contrastar al implementar el streaming |
