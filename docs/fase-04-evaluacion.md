# Fase 4 — Evaluacion con LangSmith y LLM-as-judge

## Objetivo

Pasar de scripts de evaluacion locales a un proceso reproducible: datasets versionados, experimentos comparables,
jueces LLM calibrados contra criterio humano y un gate de regresion.

## Piezas

| Pieza | Archivo | Que resuelve |
|---|---|---|
| Datasets | [fase4_datasets.py](../scripts/fase4_datasets.py) | `shopassist-e2e` (22 casos f2+f3) y `shopassist-retrieval` (30). Upsert por `case_id`: LangSmith versiona cada cambio |
| Target | `make_e2e_target` en [experiments.py](../src/agentic/experiments.py) | Corre los turnos de un ejemplo en un hilo nuevo con el store reiniciado |
| Evaluadores heuristicos | `heuristic_evaluator` | Checks del spec por categoria: `route_ok`, `tools_ok`, `retrieval_ok`, `content_ok`, `refunds_ok`, `checks_pass`, `checks_rate`, `latency_s` |
| Proceso verificado por codigo | `refund_process_evaluator` + [facts.py](../src/agentic/facts.py) | R2 (elegibilidad antes de crear) y R3 (no afirmar reembolsos inexistentes) |
| LLM-as-judge | [judges.py](../src/agentic/judges.py) | `groundedness` (alucinaciones) y `policy_compliance` (R1, R4-R6, con hechos verificados) |
| Calibracion del juez | [fase4_judge_calibration.py](../scripts/fase4_judge_calibration.py) + [judge_calibration.json](../evals/judge_calibration.json) | 14 transcripts etiquetados a mano: accuracy, FP, FN |
| Experimentos | [fase4_experiment.py](../scripts/fase4_experiment.py) | `langsmith.evaluate` con metadata (modelo, `prompt_version`, umbral RAG, juez) y repeticiones |
| Regresiones | [fase4_compare.py](../scripts/fase4_compare.py) | Compara dos reportes; exit 1 si empeora una metrica critica (gate de CI) |

## Conceptos

**Dataset como contrato.** Los casos viven en `evals/*.json` (revisables en git) y se sincronizan a LangSmith.
El `case_id` estable permite comparar el mismo ejemplo entre experimentos aunque el dataset cambie de version.

**Heuristicos vs LLM-as-judge.**

| | Heuristicos (codigo) | LLM-as-judge |
|---|---|---|
| Mide | Hechos verificables: que tool se llamo, ruta, reembolsos creados, palabras clave | Calidad semantica: ¿esta respaldado?, ¿cumple la politica? |
| Costo | ~0 | Una llamada LLM por metrica y caso |
| Riesgo | Falsos negativos por redaccion ("30 dias" vs "un mes") | Sesgo, inconsistencia, errores del juez |

Se usan juntos: el heuristico detecta que se llamo `create_refund_request` en T1; el juez explica *por que* es una
violacion (R1) y detecta casos que el heuristico no cubre (afirmar una aprobacion que no ocurrio, R3).

**Diseno del juez.** Rubrica cerrada (reglas numeradas), salida estructurada con `reasoning` antes de `score`
(razona y despues decide), score binario (mas estable que escalas 1-10 en modelos chicos), transcript con
resultados de tools truncados a 600 caracteres (el juez solo puede verificar contra lo que ve).

**Evaluar al evaluador.** Un juez no calibrado produce metricas sin significado. La calibracion mide:
- **FP** (el juez aprueba algo incorrecto): el error peligroso; una alucinacion pasa como correcta.
- **FN** (rechaza algo correcto): ruido; baja la metrica sin causa real.
Riesgo adicional: **sesgo de auto-preferencia** si el juez es el mismo modelo que el agente. Comparar con
`--judge-model granite4.1:3b` y con `--reasoning`.

**Calibracion 1 → diseno hibrido.** Primera corrida (qwen3.5:4b, sin reasoning): groundedness 7/7; policy_compliance
5/7 con **2 FP**. En `p04` el juez invento una llamada a `create_refund_request` que no existio; en `p06` justifico
la falta de `check_refund_eligibility` ("create implica verificacion") tras 394 s de razonamiento en circulos.
Patron: el juez chico evalua bien contenido, pero no detecta **acciones ausentes** ni el **orden** de las tools.
Cambios:

| Regla | Quien la verifica |
|---|---|
| R2 elegibilidad antes de crear, R3 no afirmar reembolsos inexistentes | Codigo: [facts.py](../src/agentic/facts.py) → evaluador `refund_process` |
| R1 confirmacion, R4 manipulacion, R5 aprobacion humana, R6 cancelacion | Juez, con una seccion de **hechos verificados por codigo** en el transcript |

Veredicto de politica en calibracion = `min(juez, codigo)`. Ademas: `reasoning` de maximo 3 frases y
`num_predict=600` (x4 con thinking) para cortar respuestas en circulos. Validado offline: el codigo detecta p04 y p06,
no marca ningun item aprobado por el humano y no da falsos positivos en las 22 conversaciones reales de Fases 2-3.

**Calibracion 2 → acierto por la razon equivocada.** Con el diseno hibrido el final dio 7/7 (FP=0) y la latencia
maxima bajo de 394 s a 31 s. Pero medido contra su propio alcance (`label_judge`: R1, R4-R6) el juez dio 5/7 con
2 FN: en p04 y p06 cito R6/R1 cuando no aplicaban; el final era correcto solo porque el codigo tambien daba 0.
Leccion: medir cada evaluador contra la etiqueta de **su** alcance (`label_judge`, `label_code`); un acierto del
veredicto combinado puede ocultar un juez sobre-estricto que en experimentos generaria falsas alarmas.
Correccion: aclarar en el prompt que sin reembolso creado R1/R6 se cumplen y que R2/R3 no son de su alcance.

**Calibracion 3 → anclaje en el veredicto del codigo.** El juez siguio en 5/7 (FN en p04, p06) y sus `issues` eran
copias literales de las violaciones R2/R3 del codigo: los "hechos verificados" incluian la linea de violaciones.
Mostrarle a un evaluador la salida de otro lo ancla y rompe su independencia (deja de ser una segunda opinion).
Correccion estructural: `render_facts` solo expone hechos neutrales (tools por turno, reembolsos creados). Ademas R1
estaba redactada de forma ambigua (el juez entendio que crear en el turno de la confirmacion era violacion); se
reescribio como secuencia (a) pide, (b) cliente confirma, (c) crear en (b) o despues.

**Calibracion 4 (version congelada del juez).** groundedness 7/7; policy_compliance final 7/7 (FP=0); juez en su
alcance 6/7 (1 FN en p06: sigue invirtiendo la secuencia de R1); codigo 7/7; latencia mediana 29 s, max 35 s.
Criterio fijado antes de correr: se acepta y se congela el juez. Todos los experimentos se comparan con esta version;
un `policy_compliance = 0` se revisa en el comentario del juez antes de darlo por valido (riesgo residual de FN,
nunca hubo FP en 4 calibraciones).

**Consistencia.** Con temperatura 0 el comportamiento igual varia (caso f2-10 paso en una corrida y fallo en otra tras
agregar una tool al agente). `--repetitions 3` mide la tasa de exito por caso; `0 < media < 1` = caso inestable.

**Gate de regresion.** `fase4_compare.py` compara por caso las metricas criticas (`checks_pass`, `refunds_ok`,
`policy_compliance`, `groundedness`) y termina con exit 1 si alguna baja. Con repeticiones, `--tolerance 0.34`
tolera 1 fallo de 3.

## Ejecucion

```powershell
# 0. (una vez) datasets
docker compose run --rm app python scripts/fase4_datasets.py --dry-run
docker compose run --rm app python scripts/fase4_datasets.py

# 1. calibrar el juez ANTES de confiar en sus metricas (~14 llamadas)
docker compose run --rm app python scripts/fase4_judge_calibration.py
docker compose run --rm app python scripts/fase4_judge_calibration.py --judge-model granite4.1:3b

# 2. baseline (22 casos + 2 jueces: en CPU puede tomar ~40-60 min; empezar por una suite)
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f3 --prefix baseline
docker compose run --rm app python scripts/fase4_experiment.py e2e --prefix baseline

# 3. consistencia de los casos con el bug de confirmacion
docker compose run --rm app python scripts/fase4_experiment.py e2e --cases f2-05 f2-10 f3-08 --repetitions 3 --prefix flaky

# 4. tras un cambio (Fase 5/6): nuevo experimento y gate
docker compose run --rm app python scripts/fase4_experiment.py e2e --prefix fix-x
docker compose run --rm app python scripts/fase4_compare.py --latest

# retrieval como experimento (rapido)
docker compose run --rm app python scripts/fase4_experiment.py retrieval --prefix qwen3-emb
```

## En LangSmith

- **Datasets & Experiments → shopassist-e2e**: cada experimento con sus metricas promedio y metadata.
- Seleccionar dos experimentos → **Compare**: diferencias por ejemplo, filas en rojo = regresiones.
- Clic en un ejemplo → traza completa (router, agentes, tools) y el `comment` de cada juez con su razonamiento.
- Los jueces tambien se trazan: se puede auditar que vio y que decidio.

## Ejercicios

1. Calibrar el juez con y sin `--reasoning`; decidir con datos si el costo en latencia vale la mejora.
2. Correr el baseline y revisar en LangSmith que dice `policy_compliance` en f2-05, f2-10 y f3-08.
3. Agregar a la calibracion 2 transcripts reales donde el juez se equivoco; recalibrar.
4. Cambiar `RAG_SCORE_THRESHOLD` a 0.49, correr la suite f3 y usar `fase4_compare.py` para ver la regresion.
5. Ejecutar el gate como si fuera CI: `fase4_compare.py A.json B.json; echo $LASTEXITCODE`.

## Checklist

- [ ] Explicar dataset → target → evaluadores → experimento → comparacion.
- [ ] Justificar cuando usar heuristicos y cuando LLM-as-judge.
- [ ] Explicar como se valida un juez (FP vs FN) y el sesgo de auto-preferencia.
- [ ] Explicar por que un sistema con temperatura 0 necesita repeticiones.
- [ ] Describir un gate de regresion y que metricas lo bloquean.
