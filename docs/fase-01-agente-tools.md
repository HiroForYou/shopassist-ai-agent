# Fase 1 — Agente con tool calling

## Objetivo

Implementar a mano el loop agentico para entender que hace un framework como LangGraph por debajo.

## Conceptos

**Tool calling.** El LLM no ejecuta nada: devuelve un `AIMessage` con `tool_calls` (nombre + argumentos JSON).
La aplicacion ejecuta la tool y devuelve el resultado como `ToolMessage` con el mismo `tool_call_id`.

**Loop agentico** ([agent.py](../src/agentic/agent.py)):

```
[system, historial, usuario]
        │
        ▼
   LLM.invoke ──► ¿tool_calls? ── no ──► respuesta final
        ▲               │ si
        │               ▼
        └──── ToolMessage(resultado) ◄── ejecutar tool
```

**Limite de iteraciones.** Evita loops infinitos y costo descontrolado (`max_iterations`).

**Errores como datos.** Tool inexistente, argumentos invalidos o excepciones se devuelven al LLM como `{"error": ...}`
en lugar de romper el proceso. El modelo puede corregirse (pedir el dato faltante, reintentar).

**Logica de negocio fuera del LLM.** La elegibilidad la decide [policy.py](../src/agentic/domain/policy.py), no el modelo.
`create_refund_request` re-evalua la politica aunque el agente la haya consultado: primer guardrail (se amplia en Fase 6).

## Dominio

| Pedido | Caso | Resultado esperado |
|---|---|---|
| A1001 | Entregado hace 14 dias, 45.90 USD | Reembolso aprobado |
| A1002 | Entregado hace 85 dias | Rechazo: fuera de plazo |
| A1003 | En transito | Rechazo: no entregado |
| A1004 | E-book | Rechazo: producto digital |
| A1005 | Monitor 349 USD | Pendiente de aprobacion humana |
| A1006 | Ya reembolsado | Rechazo |

Fecha de referencia fija (`APP_TODAY`) para que los resultados sean reproducibles.

## Ejecucion

```powershell
docker compose run --rm app python scripts/fase1_agent.py
docker compose run --rm app python scripts/fase1_agent.py "Quiero devolver el pedido A1005, llego con un pixel muerto"
docker compose run --rm app pytest -q
```

Los tests de [test_agent_loop.py](../tests/test_agent_loop.py) usan un LLM guionado: prueban el loop sin depender de Ollama.

Evaluacion contra el modelo real ([casos](../evals/fase1_cases.json), reporte en `reports/`):
```powershell
docker compose run --rm app python scripts/fase1_eval.py            # 15 casos
docker compose run --rm app python scripts/fase1_eval.py 01 --reasoning
```
Resultado de referencia (qwen3.5:4b, sin reasoning): 11/15 casos PASS. Los 4 FAIL comparten causa: el agente
pide confirmacion o motivo en lugar de crear el reembolso, porque el prompt no define esa politica. Se resuelve en Fase 2.

## Trazas en LangSmith

`ToolCallingAgent.run` esta decorado con `@traceable`, por lo que cada ejecucion genera un arbol:

```
shopassist_agent (chain)
├── ChatOllama (llm)            -> decide llamar check_refund_eligibility
├── check_refund_eligibility    (tool)
├── ChatOllama (llm)            -> decide llamar create_refund_request
├── create_refund_request       (tool)
└── ChatOllama (llm)            -> respuesta final
```

## Ejercicios

1. Pedir un reembolso sin numero de pedido. ¿El agente lo solicita o inventa uno?
2. Probar A1004 y A1002. Comparar en LangSmith si el modelo llama `create_refund_request` a pesar de la inelegibilidad.
3. Comparar `qwen3.5:4b` vs `granite4.1:3b` (y `OLLAMA_REASONING=true`) en numero de pasos y latencia para la misma consulta.
4. Agregar una tool `get_customer_orders(customer)` y un test para ella.
5. Bajar `max_iterations` a 1 y observar el comportamiento.

## Checklist

- [ ] Explicar el ciclo `AIMessage.tool_calls` → `ToolMessage` y el rol de `tool_call_id`.
- [ ] Justificar por que las reglas de negocio no se delegan al LLM.
- [ ] Explicar como se testea un agente sin llamar al modelo.
- [ ] Leer una traza en LangSmith e identificar latencia por paso y tokens por llamada.
