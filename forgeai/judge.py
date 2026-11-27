"""Hybrid judge: deterministic checks first, LLM-as-judge only as a second gate.

Defences against known LLM-judge failure modes:
- position bias: we never ask the judge to compare two answers; each answer is graded alone.
- verbosity bias: the judge prompt says length/style are irrelevant, and a case can only pass if the
  deterministic key-fact check also passes (a long fluent answer without the facts fails).
- self-preference: the judge cannot pass a case on its own - it can only veto. Refusal correctness and
  key-fact coverage are decided by string/number matching, not by the model.
"""
import re

from .llm import fill


def norm(s):
    return re.sub(r"[^a-z0-9.%]+", " ", str(s).lower()).strip()


def fact_present(fact, answer):
    f, a = norm(fact), norm(answer)
    if f and f in a:
        return True
    nums = re.findall(r"\d+(?:\.\d+)?", f)
    words = [w for w in f.split() if not re.match(r"^\d", w) and len(w) > 3]
    if nums and all(n in re.findall(r"\d+(?:\.\d+)?", a) for n in nums):
        return not words or sum(w in a for w in words) / len(words) >= 0.5
    if not nums and words:
        return sum(w in a for w in words) / len(words) >= 0.8
    return False


def judge_case(case, result, llm):
    """Returns verdict dict with pass/fail, sub-scores and failure category."""
    v = {"id": case["id"], "type": case["type"], "question": case["question"],
         "answer": result["answer"], "refused": result["refused"], "sources": result["sources"],
         "retrieved_docs": result["retrieved_docs"], "tool_used": result.get("tool_used")}
    expected = set(case.get("expected_sources", []))
    source_retrieved = bool(expected & set(result["retrieved_docs"])) or not expected
    v["source_retrieved"] = source_retrieved
    v["cited_expected_source"] = bool(expected & set(result["sources"])) if expected else None

    if case["should_refuse"]:
        v["passed"] = result["refused"]
        v["category"] = None if v["passed"] else "prompt_defect"
        v["reason"] = "correctly refused" if v["passed"] else "answered a question it must refuse"
        return v

    if result["refused"]:
        v["passed"] = False
        v["category"] = "wrong_refusal" if source_retrieved else "retrieval_miss"
        v["reason"] = "refused although the documents answer it" + ("" if source_retrieved else
                                                                     " (expected source was not retrieved)")
        return v

    facts = case.get("key_facts", [])
    hits = [f for f in facts if fact_present(f, result["answer"])]
    v["fact_coverage"] = round(len(hits) / len(facts), 2) if facts else 1.0
    v["missing_facts"] = [f for f in facts if f not in hits]

    if v["fact_coverage"] < 0.5:
        v["passed"] = False
        v["category"] = "retrieval_miss" if not source_retrieved else "prompt_defect"
        v["reason"] = f"missing key facts {v['missing_facts']}"
        # still ask the judge whether the answer invented things (hallucination vs incomplete)
        j = llm.generate(fill("judge", question=case["question"], key_facts=facts,
                              context=result["context"][:6000], answer=result["answer"]), stage="judge")
        v["judge"] = j
        if source_retrieved and j.get("grounded") is False:
            v["category"] = "hallucination"
        return v

    j = llm.generate(fill("judge", question=case["question"], key_facts=facts,
                          context=result["context"][:6000], answer=result["answer"]), stage="judge")
    v["judge"] = j
    grounded = j.get("grounded", True) is not False
    correct = j.get("correct", True) is not False
    v["passed"] = grounded and correct
    v["category"] = None if v["passed"] else ("hallucination" if not grounded else "prompt_defect")
    v["reason"] = j.get("reason", "") if not v["passed"] else "facts present, judge agrees"
    return v
