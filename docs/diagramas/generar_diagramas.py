"""Genera los candidatos JSON de Archify (github.com/tt-a1i/archify) para los diagramas de ShopAssist.

Uso, desde ai-agentic/ (ARCHIFY_DIR = carpeta del paquete archify que contiene bin/archify.mjs):
    python docs/diagramas/generar_diagramas.py .                        # todos los candidatos
    python docs/diagramas/generar_diagramas.py . arquitectura-runtime   # uno
Luego, por diagrama (tipos: architecture, sequence, lifecycle, workflow):
    node <ARCHIFY_DIR>/bin/archify.mjs finalize <tipo> docs/diagramas/<slug>/candidate.json
        docs/diagramas/<slug>/<slug>.html --quality showcase --json
Los textos de la interfaz en espanol salen de examples/locales/es.json de Archify.
"""
import json
import os
import sys
from pathlib import Path

ARCHIFY = Path(os.environ["ARCHIFY_DIR"])
ROOT = Path(sys.argv[1])  # ai-agentic/
ES = json.loads((ARCHIFY / "examples" / "locales" / "es.json").read_text(encoding="utf-8"))
OUT = ROOT / "docs" / "diagramas"


def write(slug: str, diagram: dict) -> None:
    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    diagram["meta"]["output"] = f"docs/diagramas/{slug}/{slug}.html"
    diagram["meta"]["locale"] = "es"
    diagram["meta"]["translations"] = ES
    diagram["meta"]["quality_profile"] = "showcase"
    (folder / "candidate.json").write_text(json.dumps(diagram, ensure_ascii=False, indent=2), encoding="utf-8")
    print("escrito", folder / "candidate.json")


W, H = 160, 64
X = [40, 290, 540, 790, 1040, 1290]
YA, YB, YC = 100, 320, 520


def node(id_, type_, label, sub, x, y, w=W, h=H, tag=None):
    n = {"id": id_, "type": type_, "label": label, "sublabel": sub, "pos": [x, y], "size": [w, h]}
    if tag:
        n["tag"] = tag
    return n


arquitectura = {
    "schema_version": 1,
    "diagram_type": "architecture",
    "meta": {"title": "ShopAssist: arquitectura en ejecucion"},
    "components": [
        node("cliente", "external", "Cliente", "fase7_client / curl", X[0], YA),
        node("api", "backend", "FastAPI", "POST /chat/stream :8000", X[1], YA),
        node("guardin", "security", "Guard de entrada", "enmascara tarjeta y clave", X[2], YA),
        node("router", "backend", "Router hibrido", "reglas o LLM (JSON)", X[3], YA),
        node("ollama", "cloud", "Ollama", "qwen3.5:4b + granite4.1:3b", X[4], YA, tag="fallback"),
        node("agentes", "backend", "Agentes LangGraph", "orders / refunds / policy", X[3], YB),
        node("guardout", "security", "Guard de salida", "sin afirmar lo no hecho", X[2], YB),
        node("guardtools", "security", "Guard de tools", "confirmacion + elegibilidad", X[4], YB),
        node("tools", "backend", "Tools de dominio", "get_order, check, create", X[5], YB),
        node("store", "database", "Store de pedidos", "en memoria (demo)", 1180, YC, w=150),
        node("qdrant", "database", "Qdrant", "politicas, embeddings", 1500, YC, w=150),
        node("langsmith", "external", "LangSmith", "trazas y experimentos", X[3], YC),
        node("prometheus", "cloud", "Prometheus", "scrape cada 5 s", X[0], YB),
        node("grafana", "frontend", "Grafana", "dashboard :3000", X[0], YC),
    ],
    "boundaries": [
        {"kind": "region", "label": "Docker Compose (WSL2, 8 GB, CPU)",
         "wraps": ["api", "guardin", "router", "ollama", "agentes", "guardout", "guardtools", "tools", "store",
                   "qdrant", "prometheus", "grafana"]},
        {"kind": "security-group", "label": "Proceso shopassist-api",
         "wraps": ["api", "guardin", "router", "agentes", "guardout", "guardtools", "tools", "store"]},
    ],
    "connections": [
        {"id": "c-cliente-api", "from": "cliente", "to": "api", "label": "POST + SSE", "variant": "emphasis"},
        {"id": "c-api-guardin", "from": "api", "to": "guardin", "label": "texto", "variant": "emphasis"},
        {"id": "c-guardin-router", "from": "guardin", "to": "router", "label": "sin PII", "variant": "emphasis"},
        {"id": "c-router-agentes", "from": "router", "to": "agentes", "label": "ruta", "variant": "emphasis",
         "fromSide": "bottom", "toSide": "top"},
        {"id": "c-router-ollama", "from": "router", "to": "ollama", "label": "si ambiguo"},
        {"id": "c-agentes-ollama", "from": "agentes", "to": "ollama", "label": "chat + tools",
         "fromSide": "top", "toSide": "bottom"},
        {"id": "c-agentes-guardtools", "from": "agentes", "to": "guardtools", "label": "tool call",
         "variant": "security"},
        {"id": "c-guardtools-tools", "from": "guardtools", "to": "tools", "label": "si pasa", "variant": "security"},
        {"id": "c-tools-store", "from": "tools", "to": "store", "label": "lee / escribe", "fromSide": "bottom",
         "toSide": "top"},
        {"id": "c-tools-qdrant", "from": "tools", "to": "qdrant", "label": "busqueda RAG", "fromSide": "bottom",
         "toSide": "top"},
        {"id": "c-agentes-guardout", "from": "agentes", "to": "guardout", "label": "respuesta", "variant": "emphasis"},
        {"id": "c-guardout-api", "from": "guardout", "to": "api", "label": "validada", "variant": "emphasis",
         "fromSide": "left", "toSide": "bottom"},
        {"id": "c-agentes-langsmith", "from": "agentes", "to": "langsmith", "label": "trazas", "variant": "dashed",
         "fromSide": "bottom", "toSide": "top"},
        {"id": "c-prometheus-api", "from": "prometheus", "to": "api", "label": "/metrics", "variant": "dashed",
         "fromSide": "right", "toSide": "bottom"},
        {"id": "c-grafana-prometheus", "from": "grafana", "to": "prometheus", "label": "PromQL", "variant": "dashed",
         "fromSide": "top", "toSide": "bottom"},
    ],
    "cards": [
        {"dot": "rose", "title": "Guardrails (Fase 6)", "items": [
            "Entrada: tarjetas y claves se enmascaran antes del grafo: no llegan al LLM, a LangSmith ni a los logs",
            "Tools: create_refund_request exige elegibilidad verificada y confirmacion explicita; si no, la llamada "
            "vuelve al LLM como error y el agente pide confirmacion",
            "Salida: bloquea reembolsos o verificaciones afirmados sin tool y monedas distintas de USD (1 reintento)",
            "Resiliencia: timeout 240 s, 1 reintento y fallback a granite4.1:3b; si todo falla, respuesta degradada"]},
        {"dot": "emerald", "title": "Latencia (Fases 5 y 7)", "items": [
            "Router hibrido: 79 % de los turnos sin llamar al LLM; -20 % de latencia y -19 % de tokens",
            "Primer progreso SSE en 0.01-0.04 s cuando decide una regla",
            "Ollama en CPU atiende un turno a la vez: semaforo con la cola expuesta como metrica"]},
        {"dot": "violet", "title": "Observabilidad", "items": [
            "Callback por nodo: eventos JSON en logs/shopassist.jsonl",
            "LangSmith: trazas por turno, datasets y experimentos de evaluacion",
            "Prometheus lee /metrics de la API; Grafana muestra latencia, bloqueos y cola"]},
        {"dot": "slate", "title": "Alcance del diagrama", "items": [
            "Basado en el codigo de ai-agentic/src/agentic (proyecto local sin git: sin anclas de commit)",
            "Store de pedidos en memoria: en produccion seria un servicio con persistencia"]},
    ],
}

# ---------- secuencias (un turno por diagrama: legibles en desktop sin scroll horizontal) ----------

PARTICIPANTES = [
    {"id": "cliente", "type": "external", "label": "Cliente", "sublabel": "SSE"},
    {"id": "api", "type": "backend", "label": "FastAPI", "sublabel": "guard de entrada"},
    {"id": "router", "type": "backend", "label": "Router", "sublabel": "hibrido"},
    {"id": "refunds", "type": "backend", "label": "refunds_agent", "sublabel": "LangGraph"},
    {"id": "ollama", "type": "cloud", "label": "Ollama", "sublabel": "qwen3.5:4b"},
    {"id": "guard", "type": "security", "label": "Guard de tools", "sublabel": "precondiciones"},
    {"id": "tools", "type": "backend", "label": "Tools", "sublabel": "store de pedidos"},
    {"id": "salida", "type": "security", "label": "Guard de salida", "sublabel": "respaldo"},
]
START, STEP, WIDTH = 160, 28, 1080  # 28 px: separacion minima entre mensajes que comparten espacio


def sequence(title: str, messages: list[tuple], segments: list[tuple], activations: list[tuple],
             cards: list[dict]) -> dict:
    """messages: (id, from, to, label, variant[, note]); segments/activations referencian ids de mensajes."""
    ys = {m[0]: START + i * STEP for i, m in enumerate(messages)}
    msgs = []
    for m in messages:
        item = {"id": m[0], "from": m[1], "to": m[2], "y": ys[m[0]], "label": m[3], "variant": m[4]}
        if len(m) > 5:
            item["note"] = m[5]
        msgs.append(item)
    last = max(ys.values())
    return {
        "schema_version": 1,
        "diagram_type": "sequence",
        "meta": {"title": title, "viewBox": [WIDTH, last + 108], "column_fit": "spread"},
        "participants": PARTICIPANTES,
        "segments": [{"from": max(150, ys[a] - 10), "to": ys[b] + 10, "label": label} for a, b, label in segments],
        "messages": msgs,
        "activations": [{"participant": p, "from": ys[a] - 4, "to": ys[b] + 4, "type": t} for p, a, b, t in activations],
        "cards": cards,
    }


turno1 = sequence(
    "ShopAssist turno 1: el guard bloquea el reembolso sin confirmar",
    [
        ("post", "cliente", "api", "devolver A1001, llego rayado", "emphasis"),
        ("route", "api", "router", "texto sin PII", "default"),
        ("rule", "router", "refunds", "regla: ID + devolver", "emphasis"),
        ("llm1", "refunds", "ollama", "prompt + esquema de tools", "default"),
        ("tc1", "ollama", "refunds", "tool call: check", "return"),
        ("check", "refunds", "guard", "check_refund_eligibility", "security"),
        ("checkok", "guard", "tools", "permitida", "security"),
        ("eligible", "tools", "refunds", "eligible: true", "return"),
        ("llm2", "refunds", "ollama", "resultado de la tool", "default"),
        ("tc2", "ollama", "refunds", "tool call: create", "return"),
        ("create", "refunds", "guard", "create_refund_request", "security"),
        ("block", "guard", "refunds", "BLOQUEADO: falta confirmacion", "return",
         "la llamada no se ejecuta; vuelve al LLM como error"),
        ("llm3", "refunds", "ollama", "error del guard", "default"),
        ("ask", "ollama", "refunds", "texto: ¿confirmas?", "return"),
        ("out", "refunds", "salida", "respuesta final", "security"),
        ("answer", "salida", "cliente", "SSE answer (validada)", "return"),
    ],
    [("rule", "rule", "Ruteo"), ("llm1", "eligible", "Elegibilidad"), ("llm2", "ask", "Intento bloqueado"),
     ("out", "answer", "Respuesta")],
    [("api", "post", "route", "backend"), ("router", "route", "rule", "backend"),
     ("refunds", "rule", "out", "backend"), ("guard", "check", "checkok", "security"),
     ("guard", "create", "block", "security"), ("tools", "checkok", "eligible", "backend"),
     ("salida", "out", "answer", "security")],
    [
        {"dot": "rose", "title": "Por que el bloqueo", "items": [
            "Con el motivo en el primer mensaje, el modelo intenta crear el reembolso de inmediato: 9 de 9 veces",
            "El guard exige confirmacion explicita en el ultimo mensaje y un pedido de confirmacion previo",
            "El error vuelve al LLM y el agente se autocorrige pidiendo confirmacion"]},
        {"dot": "cyan", "title": "Streaming (modo guarded)", "items": [
            "El cliente recibe el evento route en ~0.02 s porque decide una regla",
            "La respuesta llega despues del guard de salida: nunca se muestra texto que luego se descarte"]},
    ],
)

turno2 = sequence(
    "ShopAssist turno 2: confirmacion explicita y creacion del reembolso",
    [
        ("post", "cliente", "api", "Si, confirmo", "emphasis"),
        ("route", "api", "router", "texto sin PII", "default"),
        ("rule", "router", "refunds", "regla: respuesta corta", "emphasis"),
        ("llm1", "refunds", "ollama", "historial + tools", "default"),
        ("tc", "ollama", "refunds", "tool call: create", "return"),
        ("create", "refunds", "guard", "create_refund_request", "security"),
        ("allow", "guard", "tools", "confirmado: permitida", "security"),
        ("created", "tools", "refunds", "R-0001 approved", "return"),
        ("llm2", "refunds", "ollama", "resultado de la tool", "default"),
        ("text", "ollama", "refunds", "texto con R-0001", "return"),
        ("out", "refunds", "salida", "respuesta final", "security"),
        ("answer", "salida", "cliente", "SSE answer (validada)", "return"),
    ],
    [("rule", "rule", "Ruteo"), ("llm1", "created", "Creacion autorizada"), ("llm2", "answer", "Respuesta")],
    [("api", "post", "route", "backend"), ("router", "route", "rule", "backend"),
     ("refunds", "rule", "out", "backend"), ("guard", "create", "allow", "security"),
     ("tools", "allow", "created", "backend"), ("salida", "out", "answer", "security")],
    [
        {"dot": "emerald", "title": "Que verifica el guard", "items": [
            "Elegibilidad del pedido A1001 consultada en el turno 1 (check_refund_eligibility)",
            "Ultimo mensaje = confirmacion explicita y corta; el turno anterior pidio confirmacion"]},
        {"dot": "violet", "title": "Resultado medido (Fase 6)", "items": [
            "Reembolso sin confirmacion: de 0/9 a 9/9 casos correctos",
            "Sin regresiones; costo +8 s y +4 % de tokens por caso"]},
    ],
)

# ---------- ciclo de vida de una solicitud de reembolso ----------

def state(id_, type_, label, sub, lane, col, tag=None, step=None):
    s = {"id": id_, "type": type_, "label": label, "sublabel": sub, "lane": lane, "col": col}
    if tag:
        s["tag"] = tag
    if step:
        s["step"] = step
    return s


ciclo = {
    "schema_version": 2,
    "diagram_type": "lifecycle",
    "meta": {"title": "ShopAssist: ciclo de vida de una solicitud de reembolso"},
    "lanes": [
        {"id": "main", "label": "Flujo del agente"},
        {"id": "waiting", "label": "Esperas"},
        {"id": "terminal", "label": "Salidas"},
    ],
    "states": [
        state("solicitud", "start", "Solicitud", "pedido + motivo", "main", 0, step="01"),
        state("elegibilidad", "decision", "Elegibilidad", "check_refund_eligibility", "main", 1, "policy.py", "02"),
        state("pedir", "active", "Pide confirmacion", "resume pedido y motivo", "main", 2, step="03"),
        state("respuesta", "waiting", "Espera respuesta", "del cliente", "main", 3, step="04"),
        state("guard", "decision", "Guard de tools", "confirmacion explicita", "main", 4, "Fase 6", "05"),
        state("motivo", "waiting", "Espera motivo", "falta la razon", "waiting", 2),
        state("supervisor", "waiting", "Aprobacion humana", "monto > 200 USD", "waiting", 4,
              "pending_human_approval"),
        state("rechazado", "failure", "Rechazado", "plazo, digital, duplicado", "terminal", 1),
        state("cancelado", "neutral", "Cancelado", "el cliente desiste", "terminal", 3),
        state("creado", "success", "Reembolso creado", "approved, ID R-0001", "terminal", 4),
    ],
    "transitions": [
        {"from": "solicitud", "to": "elegibilidad", "label": "verificar"},
        {"from": "elegibilidad", "to": "pedir", "label": "elegible"},
        {"from": "pedir", "to": "respuesta", "label": "¿confirmas?"},
        {"from": "respuesta", "to": "guard", "label": "responde"},
        {"from": "elegibilidad", "to": "motivo", "label": "sin motivo", "variant": "dashed"},
        {"from": "motivo", "to": "pedir", "label": "da motivo"},
        {"from": "elegibilidad", "to": "rechazado", "label": "no elegible", "variant": "security"},
        {"from": "respuesta", "to": "cancelado", "label": "mejor no", "variant": "dashed"},
        {"from": "guard", "to": "creado", "label": "<= 200 USD", "variant": "emphasis"},
        {"from": "guard", "to": "supervisor", "label": "> 200 USD"},
        {"from": "guard", "to": "pedir", "label": "bloqueado", "variant": "security", "route": "top-channel"},
    ],
    "cards": [
        {"dot": "amber", "title": "Reglas deterministas (policy.py)", "items": [
            "Plazo de 30 dias desde la entrega; productos digitales no reembolsables; un reembolso por pedido",
            "Mas de 200 USD: la solicitud queda pendiente de un supervisor"]},
        {"dot": "rose", "title": "Guard de tools (Fase 6)", "items": [
            "Crear solo con elegibilidad verificada y confirmacion explicita tras un pedido de confirmacion",
            "Si no se cumple, la llamada se bloquea y el flujo vuelve a pedir confirmacion",
            "Medido: reembolsos sin confirmacion de 0/9 a 9/9 casos correctos"]},
        {"dot": "slate", "title": "Defensa en profundidad", "items": [
            "create_refund_request vuelve a validar la politica aunque el agente ya la haya consultado",
            "Un segundo intento sobre el mismo pedido termina en Rechazado (ya reembolsado)"]},
    ],
}

# ---------- workflow: ciclo de evaluacion y mejora (Fases 4-6) ----------

def wnode(id_, lane, col, type_, label, sub, tag=None):
    n = {"id": id_, "lane": lane, "col": col, "type": type_, "label": label, "sublabel": sub, "width": 140}
    if tag:
        n["tag"] = tag
    return n


evaluacion = {
    "schema_version": 2,
    "diagram_type": "workflow",
    "meta": {"title": "ShopAssist: ciclo de evaluacion y mejora", "viewBox": [1060, 660]},
    "lanes": [
        {"id": "datos", "label": "Casos y datasets"},
        {"id": "ejecucion", "label": "Experimento y decision"},
        {"id": "evaluadores", "label": "Evaluadores"},
        {"id": "control", "label": "Control de calidad", "variant": "exception"},
    ],
    "phases": [
        {"id": "preparar", "label": "Preparar", "fromCol": 0, "toCol": 1},
        {"id": "ejecutar", "label": "Ejecutar", "fromCol": 2, "toCol": 2, "variant": "emphasis"},
        {"id": "evaluar", "label": "Evaluar", "fromCol": 3, "toCol": 4},
        {"id": "decidir", "label": "Decidir", "fromCol": 5, "toCol": 5, "variant": "security"},
    ],
    "groups": [
        {"id": "g-evaluadores", "label": "Codigo + juez", "lane": "evaluadores", "fromCol": 3, "toCol": 4,
         "variant": "emphasis"},
    ],
    "mainPath": ["casos", "dataset", "experimento", "codigo", "jueces", "reporte", "gate", "adoptar"],
    "nodes": [
        wnode("casos", "datos", 0, "database", "evals/*.json", "casos con spec"),
        wnode("dataset", "datos", 1, "database", "Dataset LangSmith", "upsert por case_id", "versionado"),
        wnode("adoptar", "datos", 5, "backend", "Adoptar cambio", "nuevo baseline"),
        wnode("experimento", "ejecucion", 2, "backend", "Experimento", "target + repeticiones", "digest + prompts"),
        wnode("reporte", "ejecucion", 4, "database", "Reporte local", "reports/fase4_*.json"),
        wnode("gate", "ejecucion", 5, "security", "Gate de regresion", "fase4_compare", "exit 1"),
        wnode("codigo", "evaluadores", 3, "backend", "Checks de codigo", "tools, rutas, R2/R3"),
        wnode("jueces", "evaluadores", 4, "backend", "Jueces LLM", "groundedness, R1 R4-R6"),
        wnode("calibracion", "control", 4, "security", "Calibracion", "17 etiquetas humanas"),
        wnode("revisar", "control", 5, "external", "Revisar cambio", "o revertir"),
    ],
    "edges": [
        {"id": "e-sync", "from": "casos", "to": "dataset", "label": "sync", "variant": "emphasis"},
        {"id": "e-run", "from": "dataset", "to": "experimento", "label": "ejemplos", "variant": "emphasis"},
        {"id": "e-eval", "from": "experimento", "to": "codigo", "label": "salidas", "variant": "emphasis"},
        {"id": "e-juez", "from": "codigo", "to": "jueces", "variant": "emphasis"},
        {"id": "e-reporte", "from": "jueces", "to": "reporte", "label": "scores", "variant": "emphasis"},
        {"id": "e-gate", "from": "reporte", "to": "gate", "label": "vs baseline", "variant": "emphasis"},
        {"id": "e-adoptar", "from": "gate", "to": "adoptar", "label": "sin regresion", "variant": "emphasis"},
        {"id": "e-calib", "from": "calibracion", "to": "jueces", "label": "valida (FP=0)", "variant": "dashed", "labelDy": -22,
         "role": "branch"},
        {"id": "e-regresion", "from": "gate", "to": "revisar", "label": "regresion", "variant": "security",
         "role": "error"},
        {"id": "e-reintento", "from": "revisar", "to": "experimento", "label": "nuevo intento", "variant": "dashed",
         "role": "return"},
    ],
    "cards": [
        {"dot": "emerald", "title": "Cambios adoptados con este ciclo", "items": [
            "Router hibrido: -20 % latencia, -19 % tokens, 0 regresiones (A/B misma sesion)",
            "Guardrails: reembolso sin confirmacion de 0/9 a 9/9, sin regresiones"]},
        {"dot": "rose", "title": "Evaluar al evaluador", "items": [
            "Juez de reglas: de 2 falsos positivos a 0 en 4 calibraciones; lo verificable paso a codigo",
            "Juez congelado durante las comparaciones: si cambia, los experimentos dejan de ser comparables"]},
        {"dot": "amber", "title": "Controlar el ruido", "items": [
            "Test A/A: la misma configuracion dos veces marco una regresion de groundedness sin cambios",
            "Metricas con juez: repeticiones y tolerancia; digest del modelo en cada experimento"]},
    ],
}

if __name__ == "__main__":
    targets = {"arquitectura-runtime": arquitectura, "secuencia-turno1-bloqueo": turno1,
               "secuencia-turno2-confirmacion": turno2, "ciclo-vida-reembolso": ciclo,
               "workflow-evaluacion": evaluacion}
    for slug in (sys.argv[2:] or targets):
        write(slug, targets[slug])

