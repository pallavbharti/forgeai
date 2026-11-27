"""Rebuild report.md + scorecard.json from a run folder's saved files (no API calls).

    python make_report.py runs/ship_spares demos/ship_spares/brief.txt demos/ship_spares/docs
"""
import json
import sys
from pathlib import Path

from forgeai.factory import MAX_ITERATIONS, Factory

out, brief, docs = Path(sys.argv[1]), Path(sys.argv[2]).read_text(), sys.argv[3]


class NoLLM:
    model, calls, tokens = "see agent_definition.json", 0, 0


load = lambda n: json.loads((out / n).read_text(encoding="utf-8"))
f = Factory(NoLLM(), brief, docs, out, auto=True, log=print)
req, final = load("requirements.json"), load("agent_definition.json")
log = load("iteration_log.json") if (out / "iteration_log.json").exists() else []
cards = [load(p.name) for p in sorted(out.glob("results_v*.json"), key=lambda p: int(p.stem.split("_v")[1]))]
final["tool_rationale"] = final.get("tool_rationale", {})
best = f"v{final['version']}"
stop = "all eval cases pass" if cards[-1]["score"] == 1 else f"reached the maximum of {MAX_ITERATIONS} iterations"
f.report(req, final, cards, log, best, stop)
f.save("scorecard.json", [{k: c[k] for k in ["label", "score", "by_type", "failure_categories"]} for c in cards])
print(f"written {out / 'report.md'} and {out / 'scorecard.json'}")
