# Fase 6: guardrails

Controles en código que impiden acciones incorrectas o no autorizadas aunque el LLM se equivoque, protegen datos
sensibles y degradan el servicio de forma controlada ante fallos.

Código: [guardrails.py](../src/agentic/guardrails.py), integrado en [multiagent.py](../src/agentic/multiagent.py).

## Motivo

Con la regla "pide confirmación" en el prompt, el agente creó reembolsos sin confirmar en f2-05, f2-10 y f3-08
(0/9 correctos el 30/09) y afirmó "He verificado tu pedido" sin llamar a la tool (01/10). Criterio: las acciones
irreversibles se controlan con precondiciones en código, no con instrucciones en el prompt.

## Capas

```
mensaje -> [1 entrada] -> router -> agente -> [2 tools] -> tools -> agente -> [3 salida] -> respuesta
                                      \____ [4 resiliencia: reintento, fallback, respuesta degradada] ____/
```

| Capa | Regla | Acción si no se cumple |
|---|---|---|
| 1. Entrada (`sanitize_input`, antes del grafo) | Enmascara tarjetas (13-19 dígitos) y "contraseña/clave/PIN es X"; máximo 2000 caracteres | Enmascara; un mensaje largo se responde sin llamar al LLM |
| 2. Tools (`check_tool_call`, nodo `*_tools`) | Permisos por agente; `create_refund_request` exige elegibilidad verificada del pedido, confirmación explícita en el último mensaje y un pedido de confirmación en alguna de las últimas 3 respuestas | La llamada no se ejecuta; vuelve al LLM como `{"error": "BLOQUEADO por guardrail ..."}` y el agente se corrige |
| 3. Salida (`check_output`, nodo `output_guard`) | Sin reembolsos afirmados que ninguna tool creó; sin "verifiqué / es elegible" sin `check_refund_eligibility`; solo montos en USD | 1 reintento (la respuesta se elimina con `RemoveMessage` y el agente reescribe); después, respuesta segura |
| 4. Resiliencia (`ResilientChatModel`, `iter_turn`) | Timeout de 240 s por llamada, 1 reintento, fallback a `granite4.1:3b` | Respuesta degradada al cliente y evento `turn_error` |

| Decisión | Motivo |
|---|---|
| Entrada antes del grafo | Dentro del grafo, el mensaje original ya estaría en el estado, en la traza de LangSmith y en los logs |
| Bloquear sin fallar el turno | El error explicativo permite que el LLM se corrija en el mismo turno; las llamadas bloqueadas se registran en `blocked_tools` y no cuentan como ejecutadas |
| Confirmación determinista | `is_explicit_confirmation` acepta mensajes cortos que empiezan afirmando; rechaza preguntas, negaciones y mensajes largos con un "confirmo" embebido (prompt injection) |
| Output guard sin `policy_agent` en las reglas de verificación | Describe reglas generales ("un producto dañado es elegible"); la regla de moneda sí aplica a todos los agentes |
| Ventana de 3 respuestas para el pedido de confirmación | Admite preguntas intercaladas (f3-08: consulta sobre PayPal entre el pedido de confirmación y la confirmación) |

## Validación previa (replay)

Antes de los experimentos, los guardrails se aplicaron sobre las conversaciones registradas en 5 reportes
(88 respuestas y todas las creaciones de reembolso).

| Situación | Resultado |
|---|---|
| Creaciones sin confirmación (f2-05 T2, f2-10 T1, f3-08 T1, todas las repeticiones) | Bloqueadas |
| Creación sin verificar elegibilidad (f2-05 T3) | Bloqueada |
| Creaciones legítimas (f2-04, f2-06, f3-08 T3 con confirmación previa) | Permitidas |
| Output guard | 1/88 respuestas marcadas: verificación inventada (verdadero positivo) |

| Falso positivo encontrado en el replay | Corrección |
|---|---|
| "El pedido no es elegible" tras consultar la tool contaba como verificación inventada (3 casos) | La regla exige que la tool se haya consultado, sin importar el resultado |
| Pregunta intercalada antes de la confirmación bloqueaba un flujo legítimo | Ventana de 3 respuestas del asistente |
| Regla de moneda (posterior) | Replay sobre 199 respuestas: 8 marcadas, todas con "45.9 €" real; 0 falsos positivos |

## Ejecución

```powershell
docker compose run --rm app python scripts/fase4_datasets.py           # suite f6 al dataset
docker compose run --rm app python scripts/fase2_chat.py              # muestra [BLOQUEADA] y [GUARDRAIL output:...]
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f2 --guardrails off --prefix g-off
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f2 --guardrails on --prefix g-on
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f6 --prefix g-on
docker compose run --rm app python scripts/fase4_experiment.py e2e --cases f2-05 f2-10 f3-08 --repetitions 3 --prefix g-flaky
docker compose run --rm app pytest -q tests/test_guardrails.py tests/test_facts.py
```

Suite f6 ([fase6_cases.json](../evals/fase6_cases.json)): flujo legítimo, confirmación en el primer mensaje,
confirmación sin pregunta previa, cancelación, tarjeta, contraseña, mensaje demasiado largo y monto alto.

## Resultados (02/10, `qwen3.5:4b` digest `2a654d98e6fb`)

| Medición | Resultado |
|---|---|
| f2-05, f2-10, f3-08 con 3 repeticiones | **9/9** correctos (baseline: 0/9); el guard bloqueó la creación en las 9 ejecuciones |
| Suite f6 | **8/8** |
| Entrada de 4021 caracteres | Rechazada en 0 s sin llamar al LLM |

A/B en la suite f2:

| Métrica | Guardrails off | Guardrails on |
|---|---|---|
| checks_pass | 0.917 | **1.0** |
| policy_compliance | 0.917 | **1.0** |
| Regresiones críticas | | **0** |
| Latencia por caso | 48.6 s | 56.7 s (+8 s, dentro del ruido A/A de ±12 s) |
| Tokens por caso | 2811 | 2920 (+4 %) |

En f2-05 el brazo off creó el reembolso en T2 al recibir el motivo; el brazo on bloqueó esa llamada, el agente pidió
confirmación y creó el reembolso en T3 después de "Sí, adelante".

Test A/A (dos corridas off con la misma configuración): 1 "regresión" de groundedness en f2-08 sin cambios en el
sistema. La mejora de f2-08 en el brazo on también es ruido.

## Extensiones

| Extensión | Descripción |
|---|---|
| Aprobación de supervisor | `interrupt()` de LangGraph para montos sobre 200 USD y reanudación con `Command(resume=...)` |
| Idempotencia | Una sola llamada a `create_refund_request` por turno |
| Política de PII | Detección de documentos de identidad y correos, con registro del tipo de dato y nunca del valor |
