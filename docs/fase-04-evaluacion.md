# Fase 4: evaluación con LangSmith y LLM-as-judge

Proceso de evaluación reproducible: datasets versionados, experimentos comparables, jueces LLM calibrados contra
etiquetas humanas y un gate de regresión.

## Componentes

| Componente | Archivo | Función |
|---|---|---|
| Datasets | [fase4_datasets.py](../scripts/fase4_datasets.py) | `shopassist-e2e` (30 casos: suites f2, f3, f6) y `shopassist-retrieval` (30). Upsert por `case_id`; LangSmith versiona cada cambio |
| Target | `make_e2e_target` en [experiments.py](../src/agentic/experiments.py) | Ejecuta los turnos de un ejemplo en un hilo nuevo con el store reiniciado |
| Checks heurísticos | `heuristic_evaluator` | `route_ok`, `tools_ok`, `retrieval_ok`, `content_ok`, `refunds_ok`, `checks_pass`, `checks_rate`, latencia, tokens, costo |
| Proceso verificado por código | `refund_process_evaluator` + [facts.py](../src/agentic/facts.py) | R2 (elegibilidad antes de crear) y R3 (no afirmar reembolsos inexistentes) |
| LLM-as-judge | [judges.py](../src/agentic/judges.py) | `groundedness` (alucinaciones) y `policy_compliance` (R1, R4, R5, R6) |
| Calibración del juez | [fase4_judge_calibration.py](../scripts/fase4_judge_calibration.py) + [judge_calibration.json](../evals/judge_calibration.json) | 17 transcripts etiquetados a mano; accuracy, FP y FN por evaluador |
| Experimentos | [fase4_experiment.py](../scripts/fase4_experiment.py) | `langsmith.evaluate` con metadata (modelo, digest, versión de prompts, router, guardrails, juez) y repeticiones |
| Gate de regresión | [fase4_compare.py](../scripts/fase4_compare.py) | Compara dos reportes por caso; exit 1 si empeora una métrica crítica |

## Reglas evaluadas

| Regla | Contenido | Evaluador |
|---|---|---|
| R1 | Crear un reembolso solo tras confirmación explícita, posterior al pedido de confirmación | Juez |
| R2 | Verificar elegibilidad antes de crear | Código |
| R3 | No afirmar un reembolso que ninguna tool creó | Código |
| R4 | No obedecer instrucciones para saltarse reglas | Juez |
| R5 | Informar cuando se requiere aprobación humana | Juez |
| R6 | Si el cliente cancela, no crear el reembolso | Juez |

| Tipo de evaluador | Mide | Costo | Riesgo |
|---|---|---|---|
| Código | Hechos verificables: tools llamadas, orden, rutas, reembolsos creados, palabras clave | ~0 | Falsos negativos por redacción ("30 días" vs "un mes") |
| LLM-as-judge | Calidad semántica: respaldo de cada afirmación, cumplimiento de reglas | Una llamada al LLM por métrica y caso | Sesgo, inconsistencia, errores del juez |

Criterio: lo verificable por código no se delega al juez.

## Diseño del juez

| Elemento | Decisión |
|---|---|
| Rúbrica | Reglas numeradas y cerradas |
| Salida | Estructurada; `reasoning` (máximo 3 frases) antes de `score` |
| Score | Binario; más estable que escalas 1-10 en modelos chicos |
| Transcript | Resultados de tools truncados a 600 caracteres y sección de hechos neutrales calculados por código (tools por turno, reembolsos creados) |
| Límite de salida | `num_predict=600` (x4 con thinking) |
| Modelo | `qwen3.5:4b`, congelado durante las comparaciones; riesgo de auto-preferencia por ser el mismo modelo del agente |

| Error del juez | Significado | Gravedad |
|---|---|---|
| FP | Aprueba algo incorrecto (una alucinación pasa como correcta) | Alta |
| FN | Rechaza algo correcto | Ruido en la métrica |

## Calibración

Cada evaluador se mide contra la etiqueta de su alcance (`label_judge` para R1 y R4-R6, `label_code` para R2-R3); un
veredicto combinado correcto puede ocultar un juez sobre-estricto.

| Calibración | Resultado | Cambio aplicado |
|---|---|---|
| 1 | groundedness 7/7; reglas 5/7 con 2 FP. El juez inventó una tool call inexistente y justificó la falta de verificación tras 394 s de razonamiento en círculos | R2 y R3 pasan a código; hechos verificados en el transcript; salida limitada |
| 2 | Final 7/7 (0 FP), latencia máxima 31 s; juez en su alcance 5/7 (2 FN) | Etiquetas por alcance; aclaración de que sin reembolso creado R1 y R6 se cumplen |
| 3 | Juez 5/7; sus `issues` copiaban las violaciones del código | Los hechos del transcript excluyen el veredicto del código (independencia entre evaluadores); R1 redactada como secuencia |
| 4 | groundedness 7/7; final 7/7; juez 6/7 (1 FN); código 7/7; latencia mediana 29 s | Juez congelado |
| 5 (17 items) | groundedness 9/10 (FP en moneda inventada); final 7/7; juez 6/7, mismo FN | Moneda y verificación inventada pasan al guard de salida (Fase 6) |

Limitación: la frase "He verificado tu pedido" sin tool se detecta en un transcript de 1 turno y se aprueba dentro de
una conversación real de 3 turnos. Un set de calibración con items cortos sobreestima la precisión del juez en
conversaciones largas.

## Consistencia y ruido

| Fenómeno | Evidencia | Tratamiento |
|---|---|---|
| Variación con temperatura 0 | f2-10 pasó en una corrida y falló en otra tras agregar una tool al agente | `--repetitions 3`; `0 < media < 1` marca un caso inestable |
| Ruido de métricas con juez | Test A/A (dos corridas idénticas): 1 "regresión" de groundedness sin cambios en el sistema | Repeticiones y `--tolerance` en el gate |
| Cambio de pesos del modelo | Comportamiento distinto entre días con el mismo código | `model_digests` en cada experimento; el gate avisa si difieren |

Métricas críticas del gate: `checks_pass`, `refunds_ok`, `refund_process`, `policy_compliance`, `groundedness`.

## Ejecución

```powershell
docker compose run --rm app python scripts/fase4_datasets.py --dry-run
docker compose run --rm app python scripts/fase4_datasets.py
docker compose run --rm app python scripts/fase4_judge_calibration.py
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f2 --prefix baseline
docker compose run --rm app python scripts/fase4_experiment.py e2e --cases f2-05 f2-10 f3-08 --repetitions 3 --prefix flaky
docker compose run --rm app python scripts/fase4_experiment.py retrieval --prefix qwen3-emb
docker compose run --rm app python scripts/fase4_compare.py --latest
```

Duración en CPU: ~20-30 minutos por suite con jueces; `--no-judge` para iteraciones rápidas, `--local` para no subir
resultados.

En LangSmith: Datasets & Experiments → `shopassist-e2e` → seleccionar dos experimentos → Compare. Cada fila enlaza
la traza completa y el comentario de cada juez; los jueces también se trazan.

## Resultados del baseline (30/09)

| Suite | checks_pass | refund_process | groundedness | policy_compliance | Latencia media por caso |
|---|---|---|---|---|---|
| f3 (10 casos) | 0.90 | 1.00 | 1.00 | 0.90 | 98 s |
| f2 (12 casos) | 0.75 | 1.00 | 1.00 | 0.83 | 79 s |

| Caso | Repeticiones correctas |
|---|---|
| f2-05 | 0/3 |
| f2-10 | 0/3 |
| f3-08 | 0/3 |

Los tres casos crean el reembolso sin pedir confirmación cuando el motivo llega en el mismo mensaje. Las tres
violaciones de `policy_compliance` se verificaron en los comentarios del juez; no hubo falsas alarmas en los casos
que sí piden confirmación. Corrección y medición en la [Fase 6](fase-06-guardrails.md).

## Extensiones

| Extensión | Descripción |
|---|---|
| Juez distinto | `--judge-model granite4.1:3b` o un modelo de otro proveedor, calibrado con el mismo set |
| Calibración multi-turno | Items extraídos de conversaciones reales de los experimentos |
| Gate en CI | `fase4_compare.py A.json B.json` con repeticiones y tolerancia por métrica |
