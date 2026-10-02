# Resultados del proyecto

Asistente de soporte de reembolsos multi-agent con LLM local (`qwen3.5:4b` en CPU), LangGraph, Qdrant, LangSmith y
FastAPI. Periodo de desarrollo: 24/09/2026-02/10/2026. Diagramas del flujo: [diagramas/](diagramas/README.md).

## Métricas por fase

| Fase | Medición | Resultado |
|---|---|---|
| 1 Agente con tools | Casos correctos | 11/15 |
| 2 Multi-agent | Casos correctos | 11/12 |
| 3 RAG | Retrieval (22 preguntas) | hit@1 = 0.86, hit@3 = 1.00, MRR = 0.93 |
| 3 RAG | End-to-end (10 casos) | 6/10 → 9/10 |
| 4 Evaluación | Juez de reglas (calibración) | 2 FP → 0 FP; latencia máxima 394 s → 35 s |
| 4 Evaluación | Datasets | 30 casos e2e y 30 de retrieval, versionados en LangSmith |
| 5 Observabilidad | Router híbrido vs LLM (A/B misma sesión) | -20 % latencia, -19 % tokens, 79 % de turnos sin LLM, 0 regresiones |
| 6 Guardrails | Reembolsos sin confirmación (3 casos x 3 repeticiones) | 0/9 → 9/9 |
| 6 Guardrails | A/B suite f2 | checks_pass 0.917 → 1.0; costo +8 s y +4 % tokens por caso |
| 6 Guardrails | Suite de guardrails | 8/8 |
| 7 Streaming | Primer progreso visible | 0.01-0.04 s en 8/10 turnos |
| 7 Streaming | Ventaja del primer token sobre la respuesta completa | 5-7 s |

## Decisiones de diseño

| Decisión | Motivo | Evidencia |
|---|---|---|
| Reglas de negocio en código (`policy.py`), no en el LLM | Elegibilidad verificable y reproducible | Fase 1 |
| Confirmación antes de crear un reembolso como precondición en código | La instrucción en el prompt no se cumplía con el motivo en el primer mensaje | 0/9 → 9/9 (Fase 6) |
| Bloquear la tool y devolver el error al LLM | El agente se corrige en el mismo turno | 9/9 ejecuciones con autocorrección |
| Router híbrido (reglas + LLM) | El router LLM costaba tanto como un agente | -20 % latencia (Fase 5) |
| Umbral RAG de 0.40 calibrado con el score del chunk esperado | La similitud no distingue preguntas sin respuesta del dominio | Calibración de retrieval (Fase 3) |
| R2 y R3 evaluadas por código; R1, R4-R6 por el juez | El juez de 4B no detecta acciones ausentes ni el orden de las tools | 2 FP → 0 FP (Fase 4) |
| Hechos neutrales en el transcript del juez, sin el veredicto del código | Mostrar la salida de otro evaluador anclaba al juez | Calibración 3 (Fase 4) |
| Enmascarado de datos sensibles antes del grafo | Evita que el dato llegue al LLM, a LangSmith y a los logs | Suite f6 (Fase 6) |
| Modo `guarded` por defecto en streaming | Ventaja del modo optimistic de ~5 s frente al riesgo de mostrar texto que se retracta | Benchmark (Fase 7) |
| `model_digests` en cada experimento y pull de modelos solo manual | Un tag puede cambiar de pesos entre días | Cambio de comportamiento del 30/09 al 01/10 (Fase 5) |

## Criterios de trabajo

| Criterio | Aplicación |
|---|---|
| Todo cambio pasa por experimento y gate | `fase4_compare.py` con métricas críticas |
| Lo verificable por código no se delega al LLM | Políticas, R2/R3, guardrails |
| Comparaciones en la misma sesión y con orden controlado | A/B del router; benchmark intercalado de la Fase 7 |
| Test A/A antes de interpretar diferencias | Ruido de groundedness detectado en la Fase 6 |
| Replay sobre datos reales antes de los experimentos | 2 falsos positivos de los guardrails corregidos antes de medir |
| Definir el comportamiento esperado antes de medir | Política de confirmación tras la Fase 1 |
| Ejemplos few-shot fuera de los datasets de evaluación | Router de la Fase 3 |

## Limitaciones

| Estado actual | Alternativa para producción |
|---|---|
| Modelo de 4B en CPU, 20-80 s por turno | GPU o API gestionada; modelo grande para agentes y chico para router y juez |
| `InMemorySaver` y store de pedidos en memoria | `PostgresSaver` y servicios con persistencia e idempotencia |
| Juez del mismo modelo que el agente | Juez de otro proveedor, calibrado periódicamente con muestras etiquetadas |
| Set de calibración con mayoría de items de 1 turno | Items multi-turno extraídos de conversaciones reales |
| Sin autenticación ni rate limiting | Autenticación, límites por cliente, `API_ALLOW_RESET=false` |
| Aprobación humana sobre 200 USD como estado pendiente | `interrupt()` de LangGraph con panel de supervisor |
| Gate de regresión manual | Gate en CI con repeticiones y tolerancia por métrica |
| Una réplica de Ollama, un turno a la vez | Réplicas o GPU y mayor concurrencia de la API |
