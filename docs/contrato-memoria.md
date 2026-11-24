# Contrato del servicio de memoria y su integración con el orquestador

> **Estado:** borrador (Fase 2). Nada de esto está implementado todavía.
> **Marcas de verificación:** ✅ verificado con fuente · ⚠️ convención o propuesta propia · ❓ pendiente de decidir · sin marca: decisión tomada.
> Este documento **extiende** `contrato-proveedores.md` para el orquestador. **No cambia** el contrato con los proveedores: ellos siguen recibiendo `messages` completos y no conocen la memoria.

## 1. Principios

1. **Alcance: solo historial de conversaciones.** Los recuerdos duraderos y los que cruzan chats distintos quedan fuera: requieren separar lo importante de lo que no lo es, y eso es otro servicio (sección 6).
2. **El orquestador nunca toca la base de datos.** Habla por HTTP con el servicio de memoria, que es el único dueño de PostgreSQL.
3. **La fuente de verdad del historial es el servicio de memoria.** La UI no envía historial: envía el mensaje nuevo y un `x_conversation_id`.
4. **El `user_id` lo pone el orquestador, nunca el cliente.** Hoy sale de configuración; con login saldrá de un token. Un cliente no puede declararse otro usuario.
5. **Ante un fallo de memoria no hay crash ni respuestas inconexas.** Se informa con un error claro y se registra para poder trabajarlo (sección 4.4).
6. **Sin `x_conversation_id`, todo funciona como antes:** sin estado, con el historial completo en `messages`, compatible con clientes OpenAI estándar.
7. **El servicio de memoria no autentica:** confía en el `user_id` que recibe. Por eso solo puede ser alcanzable desde el orquestador: red interna de Docker y, en desarrollo, publicado únicamente en `127.0.0.1`. Nunca en una interfaz pública. ⚠️

## 2. Servicios y endpoints

| Servicio | Endpoints |
|---|---|
| **Orquestador** | los de siempre; `POST /v1/chat/completions` gana el campo opcional `x_conversation_id` |
| **Memoria** | `POST /v1/conversations/{conversation_id}/messages`, `GET /v1/conversations/{conversation_id}/messages`, `GET /health` |

## 3. Servicio de memoria

### 3.1 Identificadores

- Una conversación se identifica por el **par `(user_id, conversation_id)`** ⚠️. Dos usuarios pueden usar el mismo `conversation_id` sin chocar, y una conversación ajena es indistinguible de una inexistente: no hay forma de averiguar si existe. Esto resuelve la separación de acceso en la capa de datos desde el principio.
- `user_id` y `conversation_id`: texto opaco, `[A-Za-z0-9_-]`, entre 1 y 64 caracteres ⚠️. Se recomienda UUID para `conversation_id`.
- Todo acceso exige `user_id`. No existe ninguna consulta por `conversation_id` solo.

### 3.2 Agregar un turno

`POST /v1/conversations/{conversation_id}/messages`

```json
{
  "user_id": "dev-user",
  "turn_id": "7b1f6c0e-5d0a-4c6e-9f3a-2f8e1d4b9a10",
  "messages": [
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "¡Hola! ¿En qué te ayudo?"}
  ]
}
```

Respuesta `201`:

```json
{"conversation_id": "c-123", "stored": 2}
```

- La conversación se crea implícitamente con el primer turno ⚠️.
- `messages[].role` ∈ `user`, `assistant`. El `system` **no se guarda**: el orquestador lo inyecta en cada pedido, así que guardarlo duplicaría y congelaría un prompt que puede cambiar.
- **Atómico:** se guardan todos los mensajes del pedido o ninguno.
- `turn_id` (UUID, obligatorio): lo genera el orquestador **una vez por pedido** e identifica el turno. Hace el guardado **idempotente**: si la conversación ya tiene mensajes con ese `turn_id`, no se guarda nada y la respuesta es `200` con `{"conversation_id": "c-123", "stored": 0}` en vez de `201`. ⚠️
- Hoy el orquestador no reintenta guardados, así que esta garantía todavía no se usa (sección 6). **No protege** contra un cliente que envía el mismo mensaje dos veces: cada pedido tiene su propio `turn_id`.

### 3.3 Leer el historial

`GET /v1/conversations/{conversation_id}/messages?user_id=dev-user&limit=20`

Respuesta `200`:

```json
{
  "conversation_id": "c-123",
  "messages": [
    {"role": "user", "content": "Hola", "created_at": 1794305797},
    {"role": "assistant", "content": "¡Hola! ¿En qué te ayudo?", "created_at": 1794305798}
  ]
}
```

- Devuelve **los últimos `limit` mensajes**, ordenados del más antiguo al más nuevo.
- Una conversación inexistente (o ajena, ver 3.1) devuelve `200` con `"messages": []`. Así el primer turno de una conversación no es un caso especial.
- `limit` es obligatorio, entero, entre 1 y 200 ⚠️. Fuera de rango → `400`.
- El **valor** de `limit` lo decide el orquestador (sección 4.2): `MEMORY_HISTORY_LIMIT`, con un default de 20 en la configuración, que `docker-compose.yml` fija en 8 por el contexto útil de llama-local. Además, si el proveedor declara `context_tokens` en `providers.yaml`, el orquestador recorta ese historial por tokens estimados antes de reenviarlo (4.2, paso 3). La memoria siempre devuelve los últimos `limit` mensajes; el recorte no es responsabilidad suya.

### 3.4 Salud

`GET /health` → `{"status": "ready"}` solo si la base responde; `{"status": "loading"}` mientras no ⚠️.

### 3.5 Errores

Mismo formato que el contrato de proveedores (objeto `error` con `message`, `type`, `param`, `code`).

| Situación | HTTP | `code` | `type` |
|---|---|---|---|
| Petición mal formada (JSON inválido, `conversation_id`, `user_id` o `turn_id` con formato inválido, `limit` fuera de rango, role no permitido o repetido dentro del turno) | 400 | `invalid_request` | `invalid_request_error` |
| Ruta o método inexistente | 404 / 405 | `null` | `invalid_request_error` |
| Base de datos no disponible | 503 | `storage_unavailable` ⚠️ | `memory_error` ⚠️ |
| Error interno inesperado | 500 | `internal_error` ⚠️ | `memory_error` ⚠️ |

- `param` nombra el campo que falló o es `null`, igual que en proveedores.
- Los mensajes de error no incluyen contenido de mensajes ni detalles de la base de datos.
- Los logs propios del servicio llevan la operación (`read`/`write`), el `conversation_id` y la clase de la excepción. El logger de `psycopg_pool` incluye el **host** de la base cuando no logra conectar (observado en Linux, dentro del contenedor: `failed to resolve host 'postgres'`); en ese fallo de DNS no aparecieron ni el puerto ni la contraseña. ⚠️ Con la base viva pero rechazando conexiones el mensaje de libpq podría traer IP y puerto: no se probó. Los logs son internos y el cuerpo del 503 nunca los incluye.
- **Timeouts** ⚠️: con la base caída, cada petición espera `POOL_TIMEOUT_S` antes de dar el 503 (`PoolTimeout`); se midió 5,00 s con 5.0 y 4,02 s con el valor actual de 4.0 (Docker, 2026-11-16; las primeras peticiones tras parar la base fallan al instante porque el pool aún tiene conexiones viejas). `/health` responde `loading` en `HEALTH_TIMEOUT_S` = 2 s como máximo (medido: 2,00 y 2,01 s). Invariante: `MEMORY_TIMEOUT_S` del orquestador (5 s) debe ser **mayor** que `POOL_TIMEOUT_S` (4 s); hoy hay 1 s de margen.

### 3.6 Esquema de PostgreSQL

⚠️ Diseño propio.

```sql
CREATE TABLE IF NOT EXISTS conversations (
    user_id          text        NOT NULL,
    conversation_id  text        NOT NULL,
    next_seq         integer     NOT NULL DEFAULT 1,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, conversation_id),
    CHECK (user_id         ~ '^[A-Za-z0-9_-]{1,64}$'),
    CHECK (conversation_id ~ '^[A-Za-z0-9_-]{1,64}$')
);

CREATE TABLE IF NOT EXISTS messages (
    user_id          text        NOT NULL,
    conversation_id  text        NOT NULL,
    seq              integer     NOT NULL,
    role             text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text        NOT NULL,
    turn_id          uuid        NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, conversation_id, seq),
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations ON DELETE CASCADE,
    UNIQUE (user_id, conversation_id, turn_id, role)
);
```

- **Claves naturales compuestas:** `user_id` es el prefijo de toda clave e índice, así que el aislamiento entre usuarios es estructural y no depende solo del código. No existe forma de leer mensajes sin pasar por el usuario.
- **`seq` ordena, no el timestamp:** los dos mensajes de un turno se insertan en una misma transacción y comparten `created_at`. La lectura de "los últimos N" es `ORDER BY seq DESC LIMIT N` sobre la clave primaria, y se devuelve invertida.
- **`next_seq`** sirve para serializar los guardados concurrentes de una misma conversación.
- **Los `CHECK`** repiten lo que ya valida el servicio (defensa en profundidad). El `role` impide guardar un `system`.
- **`ON DELETE CASCADE`:** borrar una conversación borra sus mensajes. Todavía no hay endpoint de borrado.

**Algoritmo del guardado** (una sola transacción):

1. Crear la conversación si no existe (`INSERT … ON CONFLICT DO NOTHING`).
2. Bloquear su fila y leer `next_seq` (`SELECT … FOR UPDATE`).
3. Si ya existen mensajes con ese `turn_id`: no insertar nada y responder `200`, `stored: 0`.
4. Insertar los mensajes con `seq = next_seq, next_seq + 1, …` y actualizar `next_seq`.

Los duplicados se detectan en el paso 3, bajo el bloqueo de la fila. El `UNIQUE (…, turn_id, role)` mantiene la integridad en la base: una violación es un error real y **nunca** se interpreta como duplicado (`stored: 0`). Consecuencia: un turno admite como máximo un mensaje por rol, y el servicio rechaza con `400` uno que repita rol. Alcanza para `user` + `assistant`; habrá que revisarlo si algún día se guardan otros roles.

**Cómo se crea el esquema:** el servicio aplica este SQL al arrancar, y mientras no termine `/health` responde `loading`. No sabe modificar tablas existentes: con el primer cambio de esquema se migra a Alembic. ⚠️ Por esto el rol de base de datos del servicio necesita permiso de DDL.

## 4. Integración en el orquestador

### 4.1 Petición del cliente

`POST /v1/chat/completions` acepta un campo opcional `x_conversation_id` (texto, mismo formato que 3.1).

- **Con `x_conversation_id`:** `messages` debe contener **exactamente un mensaje, con rol `user`**: el turno nuevo. Cualquier otra cosa → `400 invalid_request` con `param: "messages"`. ⚠️ Es estricto a propósito para el MVP (❓ sección 6).
- **Sin `x_conversation_id`:** comportamiento actual, sin memoria.
- `"x_conversation_id": null` explícito equivale a no mandarlo: sin memoria y sin `x_memory`. Una cadena vacía, en cambio, es un formato inválido → `400`.
- Un cliente que mande `x_user_id` (o cualquier otro campo que no sea de la lista) lo tiene descartado como cualquier extra, con warning en el log. El `user_id` nunca se toma del cuerpo.
- El `x_conversation_id` **no se reenvía** al proveedor.

### 4.2 Flujo con memoria

1. Validar la petición y resolver el modelo (como hoy). Si hay un error, no se toca la memoria.
2. Leer el historial: `GET .../messages?user_id=<el del orquestador>&limit=<MEMORY_HISTORY_LIMIT>`. Si falla → `503 memory_unavailable` (4.4).
3. Armar `messages` para el proveedor: `[system del orquestador] + historial + mensaje nuevo`. Si el proveedor declara `context_tokens` (contrato de proveedores, sección 8), antes se **recorta el historial** por tokens estimados ⚠️:
   - **Cuándo no se recorta:** sin `context_tokens` no se recorta nada. Sin memoria tampoco: el `messages` que arma el cliente se reenvía entero.
   - **Estimación** ⚠️ (sin tokenizer: el orquestador no carga `transformers`):
     - Por mensaje: `ceil(len(content) / 3.0) + 8`.
     - El total tiene que entrar en `disponible = context_tokens − max_tokens − 16`, con `max_tokens` = el del pedido, o 256 (el default de llama-local) si no viene.
   - **Qué se descarta:** el system y el mensaje nuevo **nunca** se descartan. Se descartan mensajes del historial desde el **más viejo** hasta que la estimación entre. Si hubo recorte y el primer mensaje que queda es del asistente (par partido), también se descarta. Nunca se reordena ni se edita contenido.
   - **Si no entran ni el system y el mensaje nuevo solos:** `400 context_length_exceeded` (`type: "invalid_request_error"`, `param: "messages"`), con el mensaje "aunque se descarte todo el historial". No se llama al proveedor ni se guarda nada.
   - ⚠️ El 256 es una copia del default de `src/providers/llama_local/engine.py`: si cambia allá, hay que cambiarlo en el orquestador.
4. Reenviar al proveedor (sin cambios respecto de proveedores). Si el proveedor falla, **no se guarda nada**: ni el mensaje del usuario. Se evita dejar un mensaje colgado sin respuesta, y un reintento no lo duplica. ⚠️
5. Si el proveedor respondió, guardar el turno: `[mensaje del usuario, respuesta del asistente]` en una sola llamada, con un `turn_id` (UUID v4) generado una vez por pedido. Si falla → se registra y la respuesta se devuelve igual con `x_memory.status = "write_failed"` (4.4).
6. Devolver la respuesta del proveedor más `x_memory` (4.3).

Configuración nueva del orquestador (nombres ⚠️): `MEMORY_URL`, `DEV_USER_ID`, `MEMORY_HISTORY_LIMIT` (default 20 en la configuración, entre 1 y 200; `docker-compose.yml` lo fija en 8 ⚠️), `MEMORY_TIMEOUT_S` (propuesto: 5 ⚠️, **independiente** de `PROVIDER_TIMEOUT_S` para que una memoria colgada no congele el chat). Debe ser mayor que `POOL_TIMEOUT_S` de la memoria (4 s, sección 3.5), para que con la base caída la memoria alcance a responder su 503.

`MEMORY_URL` es **opcional**: ausente o vacía significa **memoria deshabilitada**, y sin `x_conversation_id` el orquestador se comporta como hoy. `DEV_USER_ID` solo se exige si `MEMORY_URL` está definida.

### 4.3 Respuesta

Se agrega a la respuesta del proveedor un objeto opcional `x_memory`, **solo si el cliente mandó `x_conversation_id`** (con una única excepción: si el JSON del proveedor no es un objeto, ver abajo):

```json
{"x_memory": {"status": "saved"}}
```

Valores de `status` en una respuesta exitosa: `saved` o `write_failed`. Quitar el campo no cambia nada de lo estándar (principio 2 del contrato de proveedores).

- `x_memory` contiene solo `status`: el `turn_id` y la causa interna de un fallo no se exponen; van al log.
- **El proveedor responde 2xx sin `choices[0].message.content` de texto** (falta, o no es una cadena): la respuesta se devuelve tal cual con `x_memory.status = "write_failed"`, **sin intentar guardar**, y se registra a nivel ERROR. El proveedor rompió su contrato; no es un fallo de la memoria. ⚠️
- **`content` igual a `""`** sí se guarda: el servicio de memoria lo acepta (`Message.content` no tiene largo mínimo).
- **El proveedor responde 2xx pero el JSON no es un objeto** (p. ej. una lista): se devuelve tal cual, **sin `x_memory`** (no hay dónde agregarlo), sin guardar y con log ERROR. Es la excepción a la regla de que `x_memory` va siempre que el cliente mandó `x_conversation_id`. ⚠️

### 4.4 Fallos de memoria

| Cuándo falla | Qué ve el cliente | Qué hace el orquestador |
|---|---|---|
| **Al leer** (antes de generar) | `503`, `code: "memory_unavailable"`, `type: "memory_error"` ⚠️, con un mensaje claro | No llama al proveedor. No se genera ni se guarda nada. |
| **Al guardar** (después de generar) | La respuesta normal, con `x_memory.status = "write_failed"` | Registra el error y no falla el pedido |
| **Memoria deshabilitada** (`MEMORY_URL` vacía) y el cliente manda `x_conversation_id` | `503`, `code: "memory_unavailable"`, `type: "memory_error"` ⚠️, con un mensaje claro | No llama a la memoria ni al proveedor. Es una condición de configuración del servidor, no un error del cliente. |

- Se considera fallo cualquier cosa que impida usar la memoria: no conecta, timeout, respuesta no JSON, estado no 2xx. Un timeout se informa como `memory_unavailable`, igual que una conexión caída ⚠️.
- **Por qué `503` al leer y no degradar:** responder sin historial produce una respuesta que parece normal y está desconectada de la conversación; además ese turno quedaría guardado en un hilo con un hueco.
- **Memoria deshabilitada:** se evalúa en el paso 1, **después** de validar la petición: un `400` por formato (`x_conversation_id` inválido, o `messages` que no es exactamente un mensaje `user`) tiene prioridad sobre este `503`. ⚠️ No se ignora el campo en silencio por la misma razón que al fallar la lectura: la respuesta parecería normal y estaría desconectada de la conversación. Tampoco es un `400`: el cliente no hizo nada mal.
- **Log:** `MemoryClient` registra `op` (`read`/`write`), `conversation_id`, `causa` (cerrada: `conexion`, `timeout`, `estado_http`, `respuesta_invalida`) y `origen`: la clase de la excepción cuando el pedido no llegó a completarse (incluye el tope total de tiempo y los errores de decodificación), o `HTTP <código>`, `json` o `forma` cuando hubo respuesta pero no sirvió. Nivel WARNING, salvo `respuesta_invalida`, que va en ERROR. Además, el orquestador registra: WARNING con `conversation_id` y `turn_id` si falla el guardado; WARNING con `conversation_id` si llega `x_conversation_id` con la memoria deshabilitada; ERROR con `conversation_id` si el proveedor responde sin `content` de texto o con un JSON que no es objeto. La `causa` solo está en los logs: al leer, el cliente recibe siempre `memory_unavailable`; al guardar, `write_failed`, sea cual sea la causa. **Nunca** el contenido de los mensajes ni la URL interna. El orquestador fija los loggers `httpx2` y `httpcore2` en WARNING, **sea cual sea `LOG_LEVEL`**, porque `httpx2` escribe a INFO la URL completa de cada pedido. Sus propios loggers, `orchestrator` y `orchestrator.memory`, salen a stderr con timestamp, nivel y nombre (`propagate=False`, para no duplicar líneas si alguien configura el root). Su nivel es el de `LOG_LEVEL`: por defecto `INFO`; un valor inválido impide arrancar. El recorte del historial (4.2, paso 3) registra un INFO solo con números: modelo, mensajes descartados, tokens estimados antes y después, disponible y tope. El `400` por no entrar ni sin historial registra un WARNING, también solo con números.
- El mensaje de error al cliente no incluye el host de la memoria (misma regla que con los proveedores).

## 5. Fuera de alcance de esta fase

Recuerdos duraderos y entre chats · listar y borrar conversaciones · resúmenes · login y tokens · retención de datos.

## 6. Pendientes

- ❓ Reintento automático del guardado con el mismo `turn_id`: el esquema ya lo soporta, el orquestador todavía no lo usa.
- ❓ Mismo `turn_id` con contenido distinto: hoy se ignora en silencio; podría ser un `409`.
- ❓ Tope de tamaño de `content`, a nivel del servicio.
- ❓ Concurrencia en lectura: dos pedidos simultáneos a la misma conversación no ven el turno del otro: cada uno lee el historial antes de que el otro guarde. El guardado sí está serializado (3.6, `next_seq`), así que el orden por `seq` se mantiene, pero la segunda respuesta no ve el turno de la primera. No se probó ni se mitiga.
- ❓ Reintento del cliente tras perder la respuesta: guarda el turno dos veces (ver 3.2, último punto). Una salida posible, no decidida: que el cliente provea una clave de idempotencia.
- Presupuesto de tokens: hecho, con un conteo aproximado (4.2, paso 3). ❓ Falta medir `CHARS_POR_TOKEN` (3,0) y los overheads (8 por mensaje, 16 de base) contra el tokenizer real de Llama 3 con otros tipos de texto: código, números, otros idiomas.
- ❓ La memoria crece sin tope: cada turno guarda dos mensajes, aunque el orquestador solo lea los últimos 8 y envíe 6 a 8.
- ❓ Streaming: con `stream: true` el turno se guardaría al terminar el flujo. Qué pasa si el cliente se desconecta a mitad.
- ❓ `/health` del orquestador: ¿depende de la memoria? Hoy no depende de los proveedores.
- ❓ Flexibilizar la regla de "exactamente un mensaje `user`" cuando haya clientes que la necesiten.
- ❓ Retención y borrado de conversaciones.
- ⚠️ Recuperación del pool tras una caída de la base (medido en Docker, con `reconnect_timeout` por defecto de `psycopg_pool`, 300 s): una caída de 20 s se recupera en el primer intento; una de ~4,5 min tardó ~70 s en volver tras reiniciar Postgres, sin reiniciar `memory`. Explicación probable, ❓ sin confirmar: backoff creciente entre reintentos. No se probó una caída más larga ni se ajustó el pool.
- ❓ Autenticación entre servicios (orquestador → memoria), cuando haya login real y niveles de acceso.

## 7. Evidencia consultada

Ninguna fuente externa. Todo este documento es **diseño propio** (⚠️). Lo único heredado es el formato de errores y las extensiones `x_`, que vienen del contrato de proveedores y tienen allí su evidencia. El comportamiento de PostgreSQL que se da por sentado en 3.6 (`ON CONFLICT`, bloqueo de fila con `FOR UPDATE`, claves foráneas compuestas) es conocimiento general y ⚠️ **no está verificado** contra la documentación: se contrasta al implementar el servicio. Contrastado hasta ahora de forma empírica, no contra la documentación: los 47 tests de memoria pasan contra PostgreSQL 18 real (incluidos los de concurrencia) y el servicio se verificó en Docker (idempotencia, persistencia del volumen, 503 sin detalles, recuperación).

Flujo del orquestador (4.1 a 4.4): probado con `TestClient` y `httpx2.MockTransport` (sin red ni servicios reales) y con cinco mutaciones del flujo aplicadas en una copia temporal fuera del repo (guardar antes de llamar al proveedor, seguir sin historial si la lectura falla, devolver `saved` ante un guardado fallido, no proyectar el historial, reenviar `x_conversation_id`): los tests fallaron en cada una. Esa copia usó Python 3.10 con un reemplazo de `asyncio.timeout`, así que el caso `timeout` de la lectura no se pudo evaluar allí; la suite completa pasa (190 passed, 22 skipped por falta de `TEST_DATABASE_URL`) en Python 3.14. Verificado además de punta a punta en Docker, con el orquestador, la memoria y PostgreSQL reales y el proveedor mock: (1) dos turnos en la misma conversación, donde el segundo pedido recibe el historial del primero y el proveedor no recibe `x_conversation_id` ni `created_at`; (2) memoria detenida, y memoria viva con PostgreSQL detenido: `503 memory_unavailable` sin llamar al proveedor, con `causa=conexion` y `causa=estado_http` en el log respectivamente; (3) recuperación sin reiniciar el orquestador al volver PostgreSQL, con los datos intactos; (4) `write_failed` real, deteniendo PostgreSQL mientras el proveedor generaba: respuesta 200 con el contenido del proveedor intacto y nada guardado. ❓ Con la memoria detenida el 503 tardó 4,0 s y no una fracción de segundo; sigue por debajo del tope de 5 s, pero la causa no se confirmó (hipótesis: el DNS interno de Docker tarda en descartar el nombre de un contenedor detenido). ⚠️ **No verificado:** una caída de PostgreSQL a mitad de una transacción de guardado; la recuperación tras caídas largas; el cierre del cliente de memoria al apagar el orquestador dentro de Docker; y que los logs de `httpx2` no escriban URLs dentro del contenedor con todo tipo de tráfico: con tests, y en vivo solo con el tráfico de la prueba del 2026-11-24 (abajo), con `LOG_LEVEL=INFO` y 0 líneas de `httpx2`/`httpcore2`.

Errores del proveedor, recorte por tokens y logging (2026-11-24) ✅, de punta a punta en Docker, con el orquestador, la memoria, PostgreSQL y llama-local reales con GPU (tope de contexto de 2527 en esa corrida; `context_tokens` declarado: 1800):
- **Proveedor cargando:** un pedido mientras el proveedor cargaba (`/health` 503 `loading`) recibió `503 provider_unavailable` con `x_provider.name`.
- **Conversación:** 14 turnos con mensajes de usuario de 417 a 600 caracteres y `max_tokens=150` dieron 14/14 con `200` y `x_memory.status = "saved"`.
- **Recorte:** desde el turno 5 se recortaron 2 mensajes por turno.
- **Prompt real:** el máximo fue de 1191 tokens; en todos los turnos, el prompt real + 150 quedó ≤ 1800.
- **Estimador:** el cociente estimado/real quedó entre 1,19 y 1,30, siempre ≥ 1, así que fue conservador.
- **Pedido que no entra:** un pedido de unos 2800 tokens estimados recibió `400 context_length_exceeded` del orquestador, sin llegar al proveedor y con 0 mensajes guardados.
- **Logs:** 10 líneas "Historial recortado" con timestamp y nivel.

❓ No verificado:
- Fue una sola corrida, con un solo tope de contexto.
- El texto era sintético y repetitivo, en español.
- El orden relativo de las líneas de uvicorn (stdout) y del orquestador (stderr).
