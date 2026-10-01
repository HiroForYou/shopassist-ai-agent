"""Fase 5: metricas agregadas desde los logs estructurados (logs/shopassist.jsonl), sin LangSmith.

Uso:
    python scripts/fase5_metrics.py                       # todo el archivo
    python scripts/fase5_metrics.py --since 2026-10-01    # desde una fecha (UTC, ISO)
    python scripts/fase5_metrics.py --last 50             # ultimos 50 turnos
    python scripts/fase5_metrics.py --thread 3f9a1c2b     # linea de tiempo de una conversacion (debug)
    python scripts/fase5_metrics.py --threads             # lista de conversaciones recientes
"""

import argparse
import json
from collections import defaultdict

from agentic.observability import LOG_FILE


def load(since: str | None) -> list[dict]:
    if not LOG_FILE.exists():
        raise SystemExit(f"No hay logs en {LOG_FILE}. Ejecuta el chat o un experimento primero.")
    events = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not since or e.get("ts", "") >= since:
            events.append(e)
    return events


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return round(values[min(len(values) - 1, int(p * len(values)))], 2)


def table(title: str, header: list[str], rows: list[list]) -> None:
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) + 2 for col in zip(header, *rows)] if rows else [len(h) + 2 for h in header]
    print("".join(f"{h:<{w}}" for h, w in zip(header, widths)))
    for r in rows:
        print("".join(f"{str(x):<{w}}" for x, w in zip(r, widths)))


def summary(events: list[dict]) -> None:
    turns = [e for e in events if e["event"] == "turn_end"]
    llm = [e for e in events if e["event"] == "llm_call"]
    tools = [e for e in events if e["event"] == "tool_call"]
    retrieval = [e for e in events if e["event"] == "retrieval"]
    errors = [e for e in events if e["event"] == "llm_error"]
    print(f"Eventos: {len(events)} | turnos: {len(turns)} | llamadas LLM: {len(llm)} | tools: {len(tools)} | "
          f"retrievals: {len(retrieval)} | errores LLM: {len(errors)}")
    if not turns:
        return

    lat = [t["latency_s"] for t in turns]
    cost = [t.get("cost_usd", 0.0) for t in turns]
    tok = [t.get("tokens_in", 0) + t.get("tokens_out", 0) for t in turns]
    print(f"\nTurno: latencia p50={pct(lat, .5)}s p95={pct(lat, .95)}s max={max(lat)}s | tokens/turno p50={pct(tok, .5)} | "
          f"costo/turno medio=${sum(cost) / len(cost):.5f} total=${sum(cost):.4f} (tarifa hipotetica)")
    sources = defaultdict(int)
    for t in turns:
        sources[t.get("route_source") or "?"] += 1
    print("Origen de la decision del router: " + " | ".join(f"{k}={v} ({v / len(turns):.0%})" for k, v in sources.items()))
    print(f"Turnos que alcanzaron el limite de pasos: {sum(t.get('hit_limit', False) for t in turns)} | "
          f"handoffs: {sum(len(t.get('handoffs', [])) for t in turns)}")

    by_node = defaultdict(list)
    for e in llm:
        by_node[e.get("node") or "?"].append(e)
    rows = []
    for node, es in sorted(by_node.items(), key=lambda kv: -sum(x["latency_s"] for x in kv[1])):
        lats = [x["latency_s"] for x in es]
        rows.append([node, len(es), pct(lats, .5), pct(lats, .95), round(sum(lats), 1),
                     round(sum(x.get("tokens_in", 0) for x in es) / len(es)),
                     round(sum(x.get("tokens_out", 0) for x in es) / len(es))])
    table("LLM por nodo", ["nodo", "llamadas", "p50 s", "p95 s", "total s", "tok in", "tok out"], rows)

    by_tool = defaultdict(list)
    for e in tools:
        by_tool[e.get("tool")].append(e)
    rows = []
    for name, es in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
        errs = sum(not x.get("ok", True) for x in es)
        rows.append([name, len(es), f"{errs / len(es):.0%}", pct([x["latency_s"] for x in es], .95)])
    table("Tools", ["tool", "llamadas", "% error", "p95 s"], rows)

    if retrieval:
        scores = [r["top_score"] for r in retrieval if r.get("top_score") is not None]
        empty = sum(r.get("results", 0) == 0 for r in retrieval)
        print(f"\nRetrieval: {len(retrieval)} busquedas | p95={pct([r['latency_s'] for r in retrieval], .95)}s | "
              f"sin resultados (bajo umbral)={empty} | top_score medio={sum(scores) / len(scores):.3f}" if scores else "")


def list_threads(events: list[dict], n: int = 20) -> None:
    turns = [e for e in events if e["event"] == "turn_end"]
    by_thread: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        by_thread[t["thread_id"]].append(t)
    recent = sorted(by_thread.items(), key=lambda kv: kv[1][-1]["ts"], reverse=True)[:n]
    rows = [[tid, ts[-1]["ts"][:19], len(ts), round(sum(x["latency_s"] for x in ts), 1),
             ",".join(dict.fromkeys(a for x in ts for a in x.get("agents", []))) or "-",
             "si" if any("create_refund_request" in x.get("tools", []) for x in ts) else ""]
            for tid, ts in recent]
    table("Conversaciones recientes", ["thread_id", "ultimo ts (UTC)", "turnos", "latencia s", "agentes", "crea reembolso"], rows)


def timeline(events: list[dict], thread: str) -> None:
    es = [e for e in events if e.get("thread_id") == thread]
    if not es:
        raise SystemExit(f"No hay eventos para thread_id={thread}")
    print(f"Linea de tiempo thread_id={thread}\n")
    for e in es:
        ts = e["ts"][11:23]
        if e["event"] == "llm_call":
            print(f"{ts}  T{e['turn']} LLM   {e.get('node'):<20} {e['latency_s']:>7}s  tokens {e.get('tokens_in')}->{e.get('tokens_out')}")
        elif e["event"] == "tool_call":
            status = "OK" if e.get("ok") else f"ERROR {e.get('error', '')[:60]}"
            print(f"{ts}  T{e['turn']} TOOL  {e.get('tool'):<20} {e['latency_s']:>7}s  {status}")
        elif e["event"] == "llm_error":
            print(f"{ts}  T{e['turn']} ERROR {e.get('node')}: {e.get('error')}")
        elif e["event"] == "turn_end":
            print(f"{ts}  T{e['turn']} FIN   route={e.get('route')} ({e.get('route_source')}) | {e['latency_s']}s | "
                  f"tools={e.get('tools')} | ${e.get('cost_usd', 0):.5f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Metricas desde logs estructurados (Fase 5)")
    parser.add_argument("--since", help="ISO UTC, p. ej. 2026-10-01 o 2026-10-01T15:00")
    parser.add_argument("--last", type=int, help="solo los ultimos N turnos (y sus eventos)")
    parser.add_argument("--thread", help="linea de tiempo de un thread_id")
    parser.add_argument("--threads", action="store_true", help="lista conversaciones recientes")
    args = parser.parse_args()

    events = load(args.since)
    if args.last:
        turn_ends = [e for e in events if e["event"] == "turn_end"][-args.last:]
        keep = {(t["thread_id"], t["turn"]) for t in turn_ends}
        events = [e for e in events if (e.get("thread_id"), e.get("turn")) in keep]
    if args.thread:
        timeline(events, args.thread)
    elif args.threads:
        list_threads(events)
    else:
        summary(events)


if __name__ == "__main__":
    main()
