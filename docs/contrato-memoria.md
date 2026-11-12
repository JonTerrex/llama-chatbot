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
- El **valor** de `limit` lo decide el orquestador (sección 4.2). Es fijo por ahora; el presupuesto de tokens según el modelo es una etapa posterior (sección 6).

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
- Un cliente que mande `x_user_id` (o cualquier otro campo que no sea de la lista) lo tiene descartado como cualquier extra, con warning en el log. El `user_id` nunca se toma del cuerpo.
- El `x_conversation_id` **no se reenvía** al proveedor.

### 4.2 Flujo con memoria

1. Validar la petición y resolver el modelo (como hoy). Si hay un error, no se toca la memoria.
2. Leer el historial: `GET .../messages?user_id=<el del orquestador>&limit=<MEMORY_HISTORY_LIMIT>`. Si falla → `503 memory_unavailable` (4.4).
3. Armar `messages` para el proveedor: `[system del orquestador] + historial + mensaje nuevo`.
4. Reenviar al proveedor (sin cambios respecto de proveedores). Si el proveedor falla, **no se guarda nada**: ni el mensaje del usuario. Se evita dejar un mensaje colgado sin respuesta, y un reintento no lo duplica. ⚠️
5. Si el proveedor respondió, guardar el turno: `[mensaje del usuario, respuesta del asistente]` en una sola llamada, con un `turn_id` (UUID v4) generado una vez por pedido. Si falla → se registra y la respuesta se devuelve igual con `x_memory.status = "write_failed"` (4.4).
6. Devolver la respuesta del proveedor más `x_memory` (4.3).

Configuración nueva del orquestador (nombres ⚠️): `MEMORY_URL`, `DEV_USER_ID`, `MEMORY_HISTORY_LIMIT` (valor inicial propuesto: 20 ⚠️), `MEMORY_TIMEOUT_S` (propuesto: 5 ⚠️, **independiente** de `PROVIDER_TIMEOUT_S` para que una memoria colgada no congele el chat).

### 4.3 Respuesta

Se agrega a la respuesta del proveedor un objeto opcional `x_memory`, **solo si el cliente mandó `x_conversation_id`**:

```json
{"x_memory": {"status": "saved"}}
```

Valores de `status` en una respuesta exitosa: `saved` o `write_failed`. Quitar el campo no cambia nada de lo estándar (principio 2 del contrato de proveedores).

### 4.4 Fallos de memoria

| Cuándo falla | Qué ve el cliente | Qué hace el orquestador |
|---|---|---|
| **Al leer** (antes de generar) | `503`, `code: "memory_unavailable"`, `type: "memory_error"` ⚠️, con un mensaje claro | No llama al proveedor. No se genera ni se guarda nada. |
| **Al guardar** (después de generar) | La respuesta normal, con `x_memory.status = "write_failed"` | Registra el error y no falla el pedido |

- Se considera fallo cualquier cosa que impida usar la memoria: no conecta, timeout, respuesta no JSON, estado no 2xx. Un timeout se informa como `memory_unavailable`, igual que una conexión caída ⚠️.
- **Por qué `503` al leer y no degradar:** responder sin historial produce una respuesta que parece normal y está desconectada de la conversación; además ese turno quedaría guardado en un hilo con un hueco.
- **Log:** `conversation_id`, operación (`read` o `write`) y clase de la excepción. **Nunca** el contenido de los mensajes ni la URL interna.
- El mensaje de error al cliente no incluye el host de la memoria (misma regla que con los proveedores).

## 5. Fuera de alcance de esta fase

Recuerdos duraderos y entre chats · listar y borrar conversaciones · resúmenes · presupuesto de tokens · login y tokens · retención de datos.

## 6. Pendientes

- ❓ Reintento automático del guardado con el mismo `turn_id`: el esquema ya lo soporta, el orquestador todavía no lo usa.
- ❓ Mismo `turn_id` con contenido distinto: hoy se ignora en silencio; podría ser un `409`.
- ❓ Tope de tamaño de `content`, a nivel del servicio.
- ❓ Concurrencia en lectura: dos pedidos simultáneos a la misma conversación no ven el turno del otro. El guardado sí está serializado.
- ❓ Presupuesto de tokens: reemplazar el `limit` fijo por un recorte según el modelo (lo hace el orquestador).
- ❓ Streaming: con `stream: true` el turno se guardaría al terminar el flujo. Qué pasa si el cliente se desconecta a mitad.
- ❓ `/health` del orquestador: ¿depende de la memoria? Hoy no depende de los proveedores.
- ❓ Flexibilizar la regla de "exactamente un mensaje `user`" cuando haya clientes que la necesiten.
- ❓ Retención y borrado de conversaciones.

## 7. Evidencia consultada

Ninguna fuente externa. Todo este documento es **diseño propio** (⚠️). Lo único heredado es el formato de errores y las extensiones `x_`, que vienen del contrato de proveedores y tienen allí su evidencia. El comportamiento de PostgreSQL que se da por sentado en 3.6 (`ON CONFLICT`, bloqueo de fila con `FOR UPDATE`, claves foráneas compuestas) es conocimiento general y ⚠️ **no está verificado** contra la documentación: se contrasta al implementar el servicio.
