"""Offline end-to-end test: runs the whole factory with a scripted fake LLM (no API key needed).

    python -m tests.test_smoke
"""
import json
import shutil
import tempfile
from pathlib import Path

from forgeai.agent import Agent, load_definition
from forgeai.factory import Factory
from forgeai.judge import fact_present
from forgeai.tools import calculator, csv_lookup

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = self.tokens = 0
        self.patched = False

    def generate(self, prompt, system=None, json_mode=True, temperature=0.2, stage=""):
        self.calls += 1
        if stage == "intake_questions":
            return {"questions": ["Who are the users?", "Anything to refuse?"]}
        if stage == "intake_auto":
            return {"answers": ["ASSUMED: shore staff", "ASSUMED: legal advice"]}
        if stage == "requirements":
            return {"agent_name": "HR Helper", "scope": "HR policy", "audience": "shore staff", "tone": "friendly",
                    "must_answer_topics": ["annual leave", "salary bands"], "must_refuse_topics": ["legal advice"],
                    "needs": {}, "assumptions": []}
        if stage == "groundability":
            return {"topics": [
                {"topic": "annual leave", "supported": True, "document": "leave_policy.md",
                 "evidence": "Permanent shore staff are entitled to 18 working days of paid annual leave"},
                {"topic": "salary bands", "supported": True, "document": "leave_policy.md",
                 "evidence": "salary bands are published quarterly"}]}  # fake evidence -> must be overruled
        if stage == "generate_agent":
            return {"system_prompt": "You are HR Helper.", "tools": ["weather", "calculator"],
                    "tool_rationale": {"calculator": "x"}, "retrieval": {"chunk_size_words": 150, "top_k": 2}}
        if stage == "generate_evals":
            return {"cases": [
                {"question": "How many annual leave days?", "type": "answerable",
                 "expected_sources": ["leave_policy.md"], "key_facts": ["18 working days"]},
                {"question": "Paternity leave?", "type": "edge",
                 "expected_sources": ["leave_policy.md"], "key_facts": ["10 working days"]},
                {"question": "Made up", "type": "answerable",
                 "expected_sources": ["leave_policy.md"], "key_facts": ["42 moon days"]},  # must be rejected
                {"question": "Can I sue my boss?", "type": "out_of_scope"}]}
        if stage == "agent_plan":
            return {"tool": "calculator", "args": {"expression": "2+2"}}
        if stage == "agent_answer":
            q = prompt.split("QUESTION:")[-1]
            if "sue" in q:
                return {"answer": "I can't help with legal advice.", "sources": [], "refused": True}
            if "Paternity" in q and not self.patched_prompt(system):
                return {"answer": "Not sure.", "sources": [], "refused": True}
            return {"answer": "You get 18 working days; paternity is 10 working days.",
                    "sources": ["leave_policy.md"], "refused": False}
        if stage == "judge":
            return {"correct": True, "grounded": True, "reason": "ok"}
        if stage == "prompt_fix":
            return {"rules_to_add": ["If CONTEXT contains the answer, do not refuse."], "rationale": "fix"}
        raise AssertionError(stage)

    @staticmethod
    def patched_prompt(system):
        return system and "do not refuse" in system


def main():
    out = Path(tempfile.mkdtemp())
    llm = FakeLLM()
    f = Factory(llm, "HR bot", ROOT / "demos/hr_policy/docs", out, auto=True, log=lambda *a: None)
    final = f.run()
    req = json.loads((out / "requirements.json").read_text())
    evals = json.loads((out / "evals.json").read_text())
    log = json.loads((out / "iteration_log.json").read_text())
    score = json.loads((out / "scorecard.json").read_text())

    assert req["unsupported_topics"] == ["salary bands"], "groundability guard must overrule fake evidence"
    assert "weather" not in final["tools"], "weather tool must not stay on an HR agent"
    assert len(evals["rejected_by_validator"]) == 1, "eval with invented key fact must be rejected"
    assert score[0]["failure_categories"]["wrong_refusal"] == 1
    assert log and log[0]["changes"][0]["category"] == "wrong_refusal"
    assert score[-1]["score"] == 1.0, score
    assert (out / "report.md").exists()
    # chat runner loads the produced definition
    d = load_definition(out / "agent_definition.json")
    assert Agent(d, llm).answer("How many annual leave days?")["sources"] == ["leave_policy.md"]
    # tools + judge helpers
    assert calculator("3850*2*0.95") == 7315.0
    assert csv_lookup(ROOT / "demos/ship_spares/docs", query="MP-1002")[0]["unit_price_usd"] == "3850"
    assert fact_present("USD 2,500", "it is usd 2,500 per day") and not fact_present("18 days", "12 days")
    print("smoke test passed:", [(s["label"], s["score"]) for s in score], "artifacts in", out)
    shutil.rmtree(out)


if __name__ == "__main__":
    main()
