# Contrato entre orquestador y proveedores LLM

> **Estado:** borrador para revisión (Fase 1). Nada de esto toca código todavía.
> **Marcas de verificación:** ✅ verificado con fuente · ⚠️ convención propia o evidencia indirecta · ❓ pendiente de decidir.

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

- Conjunto inicial de parámetros: `model`, `messages`, `stream`, `temperature`, `max_tokens`. ❓ (pendiente de confirmar)
- ❓ Qué se hace con parámetros no soportados: ignorar en silencio, o rechazar con `400`.

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

- `usage` es **estándar y solo tokens**. ✅ (estructura de OpenAI; los nombres de tokens vienen de memoria, contrastar al implementar)
- `x_metrics` es opcional. Cada clave aparece **solo si aplica**: `vram_*` solo en proveedores con GPU; el mock y los online no las envían.
- ❓ Nombres exactos de las claves (propuesta: inglés y snake_case, coherente con `usage`).
- Cambio respecto a hoy: `tokens_prompt` y `tokens_generados` pasan a `usage.prompt_tokens` y `usage.completion_tokens`.

## 5. Streaming (SSE)

`stream: true` → `Content-Type: text/event-stream`. Secuencia:

1. Primer chunk: `delta` con `{"role": "assistant"}`.
2. Chunks de texto: `choices[0].delta.content`.
3. Último chunk de texto: `delta` vacío y `finish_reason` (`stop` o `length`).
4. **Chunk final de métricas:** `choices: []`, más `usage` y `x_metrics`. ✅ La especificación de OpenAI emite el chunk de `usage` con `choices` vacío (fuente: documentación de OpenRouter, que lo describe).
5. `data: [DONE]`.

```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Ho"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":42,"completion_tokens":128,"total_tokens":170},"x_metrics":{"tokens_per_s":27.4}}

data: [DONE]
```

- Nuestros proveedores emiten el chunk final **siempre**, aunque el cliente no pida `stream_options.include_usage`, porque las métricas son parte del valor del proyecto.
- ⚠️ **Riesgo conocido:** algunos clientes fallan al leer `choices[0]` cuando `choices` está vacío (OpenRouter se desvía por eso). Decisión tomada: seguir el estándar estricto. Si un cliente real se rompe, se cambia de forma puntual y documentada.
- Aparcado para la Fase 2: tiempo hasta el primer token, y cómo se mide tokens/s con `TextIteratorStreamer` en `transformers`.

## 6. Errores

Formato de OpenAI más una extensión opcional. ✅ (estructura `error.{message,type,param,code}` verificada en librerías cliente)

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

- El campo estándar `model` decide a qué proveedor va la petición, mediante un **registro en configuración** (nombre de modelo → URL del proveedor). ❓ Formato del registro (variables de entorno o YAML) y qué pasa si falta `model`.
- `GET /v1/models` devuelve **solo los proveedores activos** en formato lista de OpenAI (`object: "list"`, `data: [...]`). La UI lo usa para el selector y para ofrecer un cambio de modelo cuando uno falla.
- Restricción de hardware: con ~5.9 GB de VRAM pico para Llama 3 8B en 4 bits sobre 8 GB, dos proveedores de GPU de ese tamaño probablemente no caben a la vez (inferencia a partir de medidas del autor, sin probar). El registro puede contener proveedores configurados pero no arrancados.

## 9. System prompt

- Lo inyecta **el orquestador** como primer mensaje `{"role": "system"}` antes de reenviar. Los proveedores no lo conocen.
- Inicialmente es **estático**, leído de `prompts/system_prompt.txt`. Escalable después (por nivel de acceso, etc.).
- ❓ Si el cliente ya envía un mensaje `system`: reemplazar, anteponer el nuestro o rechazar.
- El formato de plantilla de chat (tokens especiales de Llama 3) es responsabilidad de cada proveedor, no del orquestador.

## 10. Mock (proveedor solo de desarrollo)

- Servicio propio, con su imagen y **sin torch**. Cumple el mismo contrato que cualquier proveedor, incluido streaming, `usage`, errores y `/health`.
- Respuesta de contenido **fijo y predecible**, para que sirva de baseline de rendimiento y para pruebas rápidas.
- Devuelve `x_debug` con las `messages` y los parámetros que recibió, de modo que se puede ver el contexto real (system prompt incluido) antes de que lo toque un modelo. En streaming viaja en el chunk final.
- Guarda las últimas N peticiones y las expone en `GET /debug/requests`. ❓ N por defecto y si se configura por variable de entorno.
- **Nunca para producción:** `x_debug` y el endpoint exponen la conversación completa.
- Quitar `x_debug` y `/debug/requests` no debe cambiar el comportamiento estándar.

## 11. Pendientes menores (para decidir al revisar el borrador)

- ❓ Conjunto final de parámetros y qué hacer con los no soportados.
- ❓ Nombres definitivos de las claves de `x_metrics`.
- ❓ Estados posibles en `x_provider.state`.
- ❓ Política cuando el cliente manda su propio mensaje `system`.
- ❓ Formato del registro de proveedores y comportamiento sin `model`.
- ❓ Valor de N del mock y su variable de entorno.
- ❓ Idioma de esta documentación en el repo (español o inglés, dado que es un proyecto de portfolio).

## 12. Evidencia consultada

| Punto | Estado | Fuente |
|---|---|---|
| Estructura del objeto `error` | ✅ indirecto | Librerías cliente de OpenAI (p. ej. `async-openai`) |
| Chunk de `usage` con `choices` vacío | ✅ indirecto | Documentación de streaming de OpenRouter |
| Clientes que fallan con `choices` vacío | ✅ indirecto | Documentación de streaming de OpenRouter |
| Campos extra tolerados por el SDK `openai` | ⚠️ circunstancial | OpenRouter añade campos propios y recomienda ese SDK; no hay prueba directa |
| Formato de error en mitad del stream | ⚠️ convención | OpenRouter; no hay formato oficial confirmado |
| Página oficial de OpenAI | ❌ no verificada | No se pudo abrir desde el entorno de trabajo; contrastar al implementar |
