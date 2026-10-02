"""Fase 7: cliente de la API con streaming (SSE).

Uso:
    python scripts/fase7_client.py chat                         # chat en vivo (modo guarded)
    python scripts/fase7_client.py chat --mode optimistic       # tokens en vivo
    python scripts/fase7_client.py bench                        # mide TTFT/latencia en ambos modos
    python scripts/fase7_client.py bench --modes guarded --reps 2

Variables: API_URL (default http://localhost:8000; dentro de docker compose: http://api:8000)
Los tiempos se miden en el CLIENTE (incluyen red y serializacion), complementan los del servidor (evento done).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")
REPORTS = Path(__file__).resolve().parents[1] / "reports"
TIMEOUT = httpx.Timeout(connect=10, read=600, write=30, pool=600)  # turnos de varios minutos en CPU

# Escenarios del benchmark: un thread nuevo por escenario; el store se reinicia antes de cada uno.
SCENARIOS = [
    ("saludo", ["Hola, buenas tardes"]),
    ("estado-pedido", ["Cual es el estado de mi pedido A1003?"]),
    ("politica-rag", ["Cuantos dias tengo para devolver un producto?"]),
    ("reembolso-2-turnos", ["Quiero devolver el pedido A1001, llego rayado", "Si, confirmo"]),
]


def iter_sse(response: httpx.Response):
    event, data = None, []
    for line in response.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
        elif line == "" and event:
            yield event, json.loads("\n".join(data))
            event, data = None, []


def stream_turn(client: httpx.Client, thread_id: str | None, message: str, mode: str, on_event=None) -> dict:
    """Envia un turno y mide en el cliente: primer byte, primer evento de progreso, primer token, respuesta, fin."""
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    answer, done, retracts = "", {}, 0
    payload = {"message": message, "mode": mode, **({"thread_id": thread_id} if thread_id else {})}
    with client.stream("POST", f"{API_URL}/chat/stream", json=payload) as r:
        r.raise_for_status()
        for event, data in iter_sse(r):
            now = round(time.perf_counter() - t0, 3)
            timing.setdefault("first_byte_s", now)
            if event in ("route", "tool", "guardrail"):
                timing.setdefault("first_progress_s", now)
            elif event == "token":
                timing.setdefault("first_token_s", now)
            elif event == "retract":
                retracts += 1
            elif event == "answer":
                timing.setdefault("answer_s", now)
                answer = data["text"]
            elif event == "done":
                done = data
            elif event == "meta":
                thread_id = data["thread_id"]
            if on_event:
                on_event(event, data)
    timing["total_s"] = round(time.perf_counter() - t0, 3)
    return {"thread_id": thread_id, "answer": answer, "timing": timing, "retracts": retracts, "done": done}


def chat(mode: str) -> None:
    def show(event, data):
        if event == "token":
            print(data["text"], end="", flush=True)
        elif event == "retract":
            print("\n  [retract: texto descartado]", flush=True)
        elif event == "route":
            print(f"  [router:{data['source']}] -> {data['route']}", flush=True)
        elif event == "tool":
            print(f"  [{'BLOQUEADA' if data['blocked'] else 'tool'}] {data['name']}", flush=True)
        elif event == "guardrail":
            print(f"  [guardrail {data['layer']}:{data['action']}]", flush=True)

    thread = None
    print(f"ShopAssist via API ({API_URL}) | modo {mode} | 'salir' para terminar\n")
    with httpx.Client(timeout=TIMEOUT) as client:
        while (user := input("Tu: ").strip()).lower() not in {"salir", "exit"}:
            if not user:
                continue
            res = stream_turn(client, thread, user, mode, show)
            thread = res["thread_id"]
            t = res["timing"]
            print(f"\nShopAssist: {res['answer']}")
            print(f"  (progreso {t.get('first_progress_s', '-')}s | primer token {t.get('first_token_s', '-')}s | "
                  f"respuesta {t.get('answer_s')}s | total {t['total_s']}s)\n")


def pct(values: list[float], p: float):
    values = sorted(v for v in values if v is not None)
    return round(values[min(len(values) - 1, int(p * len(values)))], 2) if values else "-"


def bench(modes: list[str], reps: int) -> None:
    """Intercala los modos por escenario y alterna cual va primero (rep y escenario): la cache de prompts de Ollama
    acelera el segundo envio de un prompt identico, y correr todos los guarded antes que los optimistic sesgaba la
    comparacion (bench 02/10: router de la misma pregunta 22.2 s -> 11.9 s la segunda vez)."""
    rows = []
    with httpx.Client(timeout=TIMEOUT) as client:
        print("health:", client.get(f"{API_URL}/health").json())
        for rep in range(reps):
            for s_idx, (name, turns) in enumerate(SCENARIOS):
                order = modes if (rep + s_idx) % 2 == 0 else list(reversed(modes))
                for position, mode in enumerate(order, 1):
                    client.post(f"{API_URL}/admin/reset").raise_for_status()
                    thread = None
                    for i, message in enumerate(turns, 1):
                        res = stream_turn(client, thread, message, mode)
                        thread = res["thread_id"]
                        t = res["timing"]
                        rows.append({"rep": rep + 1, "mode": mode, "position": position, "scenario": name, "turn": i,
                                     **t, "retracts": res["retracts"], "server": res["done"].get("timing", {}),
                                     "server_latency_s": res["done"].get("latency_s"),
                                     "tokens_out": res["done"].get("tokens_out"), "answer": res["answer"][:200]})
                        print(f"[{mode:<10} #{position}] {name:<20} T{i} progreso={t.get('first_progress_s', '-'):<7} "
                              f"token={t.get('first_token_s', '-'):<7} respuesta={t.get('answer_s')}s "
                              f"total={t['total_s']}s retracts={res['retracts']}", flush=True)

    def summary(label: str, rs: list[dict]) -> None:
        print(f"{label:<16}{len(rs):<8}{pct([r.get('first_progress_s') for r in rs], .5):<14}"
              f"{pct([r.get('first_token_s') for r in rs], .5):<15}{pct([r.get('answer_s') for r in rs], .5):<15}"
              f"{pct([r.get('answer_s') for r in rs], .95):<15}{sum(r['retracts'] for r in rs)}")

    print(f"\n{'grupo':<16}{'turnos':<8}{'progreso p50':<14}{'1er token p50':<15}{'respuesta p50':<15}"
          f"{'respuesta p95':<15}retracts")
    for mode in modes:
        summary(mode, [r for r in rows if r["mode"] == mode])
    # mismo prompt enviado 1.o vs 2.o: aisla el efecto de la cache de prompts, independiente del modo
    for position in (1, 2):
        summary(f"posicion {position}", [r for r in rows if r["position"] == position])
    ttft_gain = [r["answer_s"] - r["first_token_s"] for r in rows if r.get("first_token_s") and r.get("answer_s")]
    if ttft_gain:
        print(f"\nOptimistic: el primer token llega {pct(ttft_gain, .5)} s (p50) antes que la respuesta completa")

    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"fase7_bench_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"api_url": API_URL, "modes": modes, "reps": reps, "rows": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: reports/{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente de la API ShopAssist (Fase 7)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--mode", choices=["guarded", "optimistic"], default="guarded")
    b = sub.add_parser("bench")
    b.add_argument("--modes", nargs="+", choices=["guarded", "optimistic"], default=["guarded", "optimistic"])
    b.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()
    try:
        chat(args.mode) if args.cmd == "chat" else bench(args.modes, args.reps)
    except httpx.ConnectError:
        sys.exit(f"No se pudo conectar a {API_URL}. ¿Esta corriendo el servicio api? (docker compose up -d api)")


if __name__ == "__main__":
    main()
