"""Tests de Fase 4 sin Ollama ni LangSmith: datasets, target, evaluadores, jueces, agregacion y regresiones."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from test_multiagent import FakeKB, ScriptedLLM, call, route

from agentic.evalkit import turn_checks
from agentic.experiments import (
    aggregate,
    case_to_example,
    collect_rows,
    compare_reports,
    flaky_cases,
    heuristic_evaluator,
    make_e2e_target,
    refund_process_evaluator,
    retrieval_evaluator,
    sync_dataset,
)
from agentic.judges import POLICY_PROMPT, TOOL_OUTPUT_LIMIT, JudgeVerdict, make_judge_evaluator, render_transcript
from agentic.multiagent import ShopAssistChat, build_graph, prompt_version

CASE = {
    "id": "04-reembolso",
    "description": "confirma y crea",
    "turns": [
        {"user": "Quiero devolver el A1001, llego rayado", "route_any": ["refunds"],
         "must_call": ["check_refund_eligibility"], "must_not_call": ["create_refund_request"]},
        {"user": "Si, confirmo", "must_call": ["create_refund_request"], "mention_any": ["R-0001"]},
    ],
    "refunds": [["approved"]],
}


def turn(user, answer, route="refunds", tools=(), outputs=(), elapsed=1.0):
    return {"user": user, "answer": answer, "route": route, "agents": [], "tools": list(tools),
            "tool_outputs": [{"name": n, "output": o} for n, o in outputs], "hit_limit": False, "elapsed_s": elapsed}


# ---------- datasets ----------

def test_case_to_example_structure():
    ex = case_to_example(CASE, "f2")
    assert ex["inputs"] == {"turns": ["Quiero devolver el A1001, llego rayado", "Si, confirmo"]}
    assert ex["outputs"]["refunds_allowed"] == [["approved"]]
    assert ex["metadata"] == {"case_id": "f2-04-reembolso", "suite": "f2"}


class FakeClient:
    def __init__(self, existing):
        self.examples = {e.metadata["case_id"]: e for e in existing}
        self.created, self.updated, self.deleted = [], [], []

    def has_dataset(self, dataset_name):
        return True

    def read_dataset(self, dataset_name):
        return SimpleNamespace(id="ds")

    def list_examples(self, dataset_id):
        return list(self.examples.values())

    def create_examples(self, dataset_id, examples):
        self.created += examples

    def update_example(self, example_id, **kwargs):
        self.updated.append(example_id)
        self.last_update = kwargs

    def delete_example(self, example_id):
        self.deleted.append(example_id)


def test_sync_dataset_ignores_langsmith_metadata_keys():
    ex = case_to_example(CASE, "f2")
    remote = SimpleNamespace(id="e1", inputs=ex["inputs"], outputs=ex["outputs"],
                             metadata={**ex["metadata"], "dataset_split": ["base"]})
    client = FakeClient([remote])
    r = sync_dataset(client, "ds", "desc", [ex])

    assert (r["created"], r["updated"], r["deleted"]) == (0, 0, 0)


def test_sync_dataset_update_preserves_langsmith_metadata():
    ex = case_to_example({**CASE, "description": "nueva"}, "f2")
    remote = SimpleNamespace(id="e1", inputs=ex["inputs"], outputs={**ex["outputs"], "description": "vieja"},
                             metadata={**ex["metadata"], "dataset_split": ["base"]})
    client = FakeClient([remote])
    sync_dataset(client, "ds", "desc", [ex])

    assert client.updated == ["e1"]
    assert client.last_update["metadata"]["dataset_split"] == ["base"]


def test_sync_dataset_upserts_by_case_id():
    same = case_to_example(CASE, "f2")
    changed = case_to_example({**CASE, "id": "05-x", "description": "nueva descripcion"}, "f2")
    existing = [
        SimpleNamespace(id="e1", **same),
        SimpleNamespace(id="e2", inputs=changed["inputs"], outputs={**changed["outputs"], "description": "vieja"},
                        metadata=changed["metadata"]),
        SimpleNamespace(id="e3", inputs={}, outputs={}, metadata={"case_id": "f2-99-borrado"}),
    ]
    client = FakeClient(existing)
    new = case_to_example({**CASE, "id": "06-nuevo"}, "f2")
    r = sync_dataset(client, "ds", "desc", [same, changed, new])

    assert (r["created"], r["updated"], r["deleted"], r["total"]) == (1, 1, 1, 3)
    assert client.updated == ["e2"] and client.deleted == ["e3"]


# ---------- target + heuristicos ----------

def test_e2e_target_runs_turns_in_fresh_thread():
    llm = ScriptedLLM([
        route("refunds"), call("check_refund_eligibility", {"order_id": "A1001"}, "c1"),
        AIMessage(content="Es elegible. Confirmas?"),
        route("refunds"), call("create_refund_request", {"order_id": "A1001", "reason": "rayado"}, "c2"),
        AIMessage(content="Creado R-0001."),
    ])
    target = make_e2e_target(ShopAssistChat(build_graph(llm, kb=FakeKB())))
    out = target({"turns": ["Quiero devolver el A1001, llego rayado", "Si, confirmo"]})

    assert [t["route"] for t in out["turns"]] == ["refunds", "refunds"]
    assert out["turns"][1]["tools"] == ["create_refund_request"]
    assert out["refunds"] == ["approved"]
    assert out["answer"] == "Creado R-0001."


def test_heuristic_evaluator_pass_and_categories():
    outputs = {"turns": [
        turn("t1", "Es elegible. Confirmas?", tools=["check_refund_eligibility"]),
        turn("t2", "Creado R-0001.", tools=["create_refund_request"], elapsed=2.0),
    ], "refunds": ["approved"]}
    ref = case_to_example(CASE, "f2")["outputs"]
    scores = {r["key"]: r["score"] for r in heuristic_evaluator({}, outputs, ref)["results"]}

    assert scores["checks_pass"] == 1 and scores["checks_rate"] == 1.0
    assert scores["route_ok"] == scores["tools_ok"] == scores["refunds_ok"] == 1
    assert scores["latency_s"] == 3.0


def test_heuristic_evaluator_detects_refund_without_confirmation():
    outputs = {"turns": [
        turn("t1", "Aprobado R-0001", tools=["check_refund_eligibility", "create_refund_request"]),
        turn("t2", "Ya estaba creado", tools=[]),
    ], "refunds": ["approved"]}
    results = heuristic_evaluator({}, outputs, case_to_example(CASE, "f2")["outputs"])["results"]
    scores = {r["key"]: r["score"] for r in results}

    assert scores["checks_pass"] == 0 and scores["tools_ok"] == 0
    assert "T1 no llama create_refund_request" in results[0]["comment"]


def test_heuristic_evaluator_handles_missing_outputs():
    assert heuristic_evaluator({}, None, {})["results"][0]["score"] == 0


def test_turn_checks_categories():
    cats = {c for c, _, _ in turn_checks({"route_any": ["policies"], "must_call": ["search_policies"],
                                          "retrieves_any": ["a#b"], "cites_any": ["a#b"]},
                                         "ver [a#b]", ["search_policies"], "policies", "a#b")}
    assert cats == {"content", "route", "tools", "retrieval"}


def test_retrieval_evaluator():
    ok = retrieval_evaluator({}, {"retrieved": ["x", "a"], "scores": [0.7, 0.6]}, {"expected": ["a"]})["results"]
    assert {r["key"]: r["score"] for r in ok} == {"hit@1": 0, "hit@k": 1, "reciprocal_rank": 0.5}
    none = retrieval_evaluator({}, {"retrieved": ["x"], "scores": [0.45]}, {"expected": []})["results"]
    assert none == [{"key": "no_answer_top_score", "score": 0.45}]


# ---------- jueces ----------

class FakeJudgeLLM:
    def __init__(self, verdict):
        self.verdict = verdict
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


def test_render_transcript_truncates_tool_outputs():
    text = render_transcript([turn("hola", "resp", outputs=[("search_policies", "x" * 2000)])])
    assert "Cliente: hola" in text and "Asistente: resp" in text
    assert "x" * TOOL_OUTPUT_LIMIT + "..." in text and "x" * (TOOL_OUTPUT_LIMIT + 1) not in text


def test_policy_judge_receives_verified_facts():
    llm = FakeJudgeLLM(JudgeVerdict(reasoning="creo sin confirmar", issues=["R1: sin confirmacion"], score=0))
    evaluator = make_judge_evaluator("policy_compliance", llm)
    r = evaluator({}, {"turns": [turn("a", "b")]}, {})

    assert r["key"] == "policy_compliance" and r["score"] == 0
    assert "R1" in r["comment"]
    assert llm.messages[0].content == POLICY_PROMPT
    assert "Hechos verificados por codigo" in llm.messages[1].content


def test_groundedness_judge_has_no_facts_section():
    llm = FakeJudgeLLM(JudgeVerdict(reasoning="ok", score=1))
    make_judge_evaluator("groundedness", llm)({}, {"turns": [turn("a", "b")]}, {})
    assert "Hechos verificados" not in llm.messages[1].content


def test_judge_evaluator_invalid_json_gives_none():
    r = make_judge_evaluator("groundedness", FakeJudgeLLM(ValueError("bad")))({}, {"turns": [turn("a", "b")]}, {})
    assert r["score"] is None


def test_refund_process_evaluator():
    bad = {"turns": [turn("t1", "Tu reembolso fue aprobado", tools=[])], "refunds": []}
    r = refund_process_evaluator({}, bad, {})
    assert r["score"] == 0 and "R3" in r["comment"]


# ---------- resultados y regresiones ----------

def test_collect_and_aggregate_with_repetitions():
    def fake(case_id, score):
        return {"example": SimpleNamespace(metadata={"case_id": case_id}, id="x"),
                "run": SimpleNamespace(id="r", outputs={}),
                "evaluation_results": {"results": [SimpleNamespace(key="checks_pass", score=score, comment=None)]}}

    rows = collect_rows([fake("f2-10", 1), fake("f2-10", 0), fake("f2-10", 1), fake("f2-04", 1)])
    per_case = aggregate(rows)

    assert per_case["f2-10"]["checks_pass"] == 0.667
    assert flaky_cases(per_case) == {"f2-10": ["checks_pass"]}


def test_compare_reports_flags_regressions():
    base = {"per_case": {"a": {"checks_pass": 1.0, "policy_compliance": 1.0}, "b": {"checks_pass": 0.0}}}
    cand = {"per_case": {"a": {"checks_pass": 1.0, "policy_compliance": 0.0}, "b": {"checks_pass": 1.0}, "c": {}}}
    diff = compare_reports(base, cand)

    assert diff["regressions"] == [{"case_id": "a", "metric": "policy_compliance", "baseline": 1.0, "candidate": 0.0}]
    assert diff["improvements"][0]["case_id"] == "b"
    assert diff["only_candidate"] == ["c"]
    assert compare_reports(base, cand, tolerance=1.0)["regressions"] == []


def test_prompt_version_is_stable():
    assert prompt_version() == prompt_version() and len(prompt_version()) == 8
