# Fase 1: agente con tool calling

Loop agéntico implementado a mano, sin framework, como base para el grafo de la Fase 2.

Código: [agent.py](../src/agentic/agent.py) · [tools.py](../src/agentic/tools.py) · [policy.py](../src/agentic/domain/policy.py)

## Mecanismo

El LLM no ejecuta nada: devuelve un `AIMessage` con `tool_calls` (nombre y argumentos JSON). La aplicación ejecuta la
tool y devuelve el resultado como `ToolMessage` con el mismo `tool_call_id`.

```
[system, historial, usuario]
        │
        ▼
   LLM.invoke ──► tool_calls ── no ──► respuesta final
        ▲               │ sí
        │               ▼
        └──── ToolMessage(resultado) ◄── ejecutar tool
```

| Elemento | Implementación |
|---|---|
| Límite de iteraciones | `max_iterations=6`; al alcanzarlo, mensaje de límite |
| Errores de tools | Tool inexistente, argumentos inválidos o excepción vuelven al LLM como `{"error": ...}`; el turno no falla |
| Reglas de negocio | La elegibilidad la decide `policy.py`; el LLM solo la consulta |
| Validación redundante | `create_refund_request` vuelve a evaluar la política aunque el agente ya la haya consultado |

## Tools

| Tool | Función |
|---|---|
| `get_order` | Detalle de un pedido por ID (normaliza mayúsculas y espacios) |
| `check_refund_eligibility` | Elegibilidad según la política |
| `create_refund_request` | Crea la solicitud; rechaza si la política no lo permite |

## Datos de prueba

Fecha de referencia fija (`APP_TODAY=2026-09-24`) para resultados reproducibles.

| Pedido | Caso | Resultado esperado |
|---|---|---|
| A1001 | Entregado hace 14 días, 45.90 USD | Reembolso aprobado |
| A1002 | Entregado hace 85 días | Rechazo por plazo |
| A1003 | En tránsito | Rechazo: no entregado |
| A1004 | E-book | Rechazo: producto digital |
| A1005 | Monitor de 349 USD | Pendiente de aprobación humana |
| A1006 | Ya reembolsado | Rechazo |

## Ejecución

```powershell
docker compose run --rm app python scripts/fase1_agent.py
docker compose run --rm app python scripts/fase1_agent.py "Quiero devolver el pedido A1005, llegó con un pixel muerto"
docker compose run --rm app python scripts/fase1_eval.py              # 15 casos contra el modelo real
docker compose run --rm app python scripts/fase1_eval.py 01 --reasoning
docker compose run --rm app pytest -q tests/test_agent_loop.py tests/test_policy_and_tools.py
```

Los tests usan un LLM guionado: verifican el loop y las tools sin Ollama. `fase1_eval.py` muestra el flujo paso a paso
(decisión del LLM, tool calls, resultados, latencia y tokens) y guarda el detalle en `reports/`.

## Trazas en LangSmith

`ToolCallingAgent.run` tiene `@traceable`, por lo que cada ejecución genera un árbol:

```
shopassist_agent (chain)
├── ChatOllama (llm)            -> check_refund_eligibility
├── check_refund_eligibility    (tool)
├── ChatOllama (llm)            -> create_refund_request
├── create_refund_request       (tool)
└── ChatOllama (llm)            -> respuesta final
```

## Resultados

`qwen3.5:4b` sin reasoning: **11/15 casos**.

| Hallazgo | Causa | Acción |
|---|---|---|
| 4 casos fallan: el agente pide confirmación o motivo en vez de crear el reembolso | El prompt no define si se debe confirmar antes de una acción irreversible | Política explícita de confirmación en la Fase 2 |

## Extensiones

| Extensión | Descripción |
|---|---|
| Comparar modelos | `qwen3.5:4b` vs `granite4.1:3b` y `OLLAMA_REASONING=true`: pasos y latencia por consulta |
| Nueva tool | `get_customer_orders(customer)` con su test |
| Límite de iteraciones | Efecto de `max_iterations=1` |
