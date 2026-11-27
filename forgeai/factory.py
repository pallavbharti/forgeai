"""ForgeAI pipeline: intake -> groundability -> generate agent -> generate evals -> judge -> optimize -> report."""
import copy
import json
import os
import random
from pathlib import Path

from .agent import Agent
from .judge import judge_case, norm
from .llm import ROOT, BudgetExceeded, fill
from .retrieval import full_text, inventory, load_docs
from .tools import REGISTRY, describe

MAX_ITERATIONS = 3
CATEGORIES = ["retrieval_miss", "prompt_defect", "hallucination", "wrong_refusal"]


def quote_found(quote, corpus_norm, threshold=0.8):
    """True if the quote appears in the documents, allowing small wording/punctuation drift:
    at least 80% of the quote's words must match one contiguous stretch of a document, in order."""
    q = norm(quote)
    if len(q) <= 10:
        return False
    if q in corpus_norm:
        return True
    from difflib import SequenceMatcher
    qt, ct = q.split(), corpus_norm.split()
    anchors = {i for i, w in enumerate(ct) if w in set(qt[:3]) | set(qt[-3:])}
    n = len(qt)
    for i in anchors:
        for start in (max(0, i - n), max(0, i - 2)):
            window = ct[start:start + n + 6]
            m = SequenceMatcher(None, qt, window, autojunk=False)
            if sum(b.size for b in m.get_matching_blocks()) / n >= threshold:
                return True
    return False


def rel_to_root(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


class Factory:
    def __init__(self, llm, brief, docs_dir, out_dir, auto=False, log=print):
        self.llm, self.brief, self.auto, self.log = llm, brief.strip(), auto, log
        self.docs_dir = str(Path(docs_dir).resolve())
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.docs = load_docs(docs_dir)
        if not self.docs:
            raise SystemExit(f"No .txt/.md/.pdf/.csv documents found in {docs_dir}")
        self.inv = inventory(self.docs)
        self.corpus = full_text(self.docs)

    def save(self, name, obj):
        p = self.out / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    # ---------- 1. intake interviewer ----------
    def intake(self):
        qs = self.llm.generate(fill("intake_questions", brief=self.brief, inventory=self.inv),
                               stage="intake_questions").get("questions", [])[:5]
        if self.auto:
            ans = self.llm.generate(fill("intake_auto_answer", brief=self.brief, questions=qs),
                                    stage="intake_auto").get("answers", [])
            ans = (ans + ["ASSUMED: no preference"] * len(qs))[:len(qs)]
            for q, a in zip(qs, ans):
                self.log(f"  Q: {q}\n  A (auto): {a}")
        else:
            print("\nForgeAI has a few questions (press Enter to skip one):")
            ans = [input(f"  Q{i + 1}. {q}\n  > ").strip() or "no preference" for i, q in enumerate(qs)]
        qa = [{"question": q, "answer": a} for q, a in zip(qs, ans)]
        req = self.llm.generate(fill("intake_requirements", brief=self.brief, inventory=self.inv, qa=qa),
                                stage="requirements")
        req["clarifications"] = qa
        req["brief"] = self.brief
        return req

    # ---------- 2. groundability guard ----------
    def groundability(self, req):
        topics = req.get("must_answer_topics", [])
        res = self.llm.generate(fill("groundability", topics=topics, documents=self.corpus),
                                stage="groundability").get("topics", [])
        corpus_norm = norm(self.corpus)
        checked, unsupported = [], []
        for t in res:
            ev = t.get("evidence", "")
            verified = bool(t.get("supported")) and quote_found(ev, corpus_norm)
            # distrust the model: a 'supported' claim with an evidence quote we can't find is overruled
            t["evidence_verified"] = verified
            t["supported_final"] = verified
            checked.append(t)
            if not verified:
                unsupported.append(t.get("topic"))
        req["groundability"] = checked
        req["unsupported_topics"] = unsupported
        req["must_answer_topics"] = [t for t in topics if t not in unsupported]
        req["must_refuse_topics"] = list(dict.fromkeys(req.get("must_refuse_topics", []) +
                                                       [f"{u} (not covered by the documents)" for u in unsupported]))
        return req

    def checkpoint(self, req):
        p = self.save("requirements.json", req)
        if req["unsupported_topics"]:
            self.log(f"  ! GROUNDABILITY FLAG: documents cannot support {req['unsupported_topics']} -> "
                     f"these will be refused, not answered.")
        if self.auto:
            return req
        print(f"\nRequirements written to {p}. Review/edit the file now if you want.")
        input("Press Enter to approve and continue... ")
        return json.loads(p.read_text())

    # ---------- 3. agent generator ----------
    def generate_agent(self, req):
        g = self.llm.generate(fill("generate_agent", requirements=req, inventory=self.inv, tools=describe()),
                              stage="generate_agent")
        r = g.get("retrieval", {})
        proposed = [t for t in g.get("tools", []) if t in REGISTRY]
        tools, removed = self.check_tools(proposed, req)
        rationale = g.get("tool_rationale", {})
        for t, why in removed.items():
            rationale[t] = f"REMOVED by factory check: {why}"
        return {
            "name": req.get("agent_name", "Generated Agent"),
            "version": 1,
            "system_prompt": g.get("system_prompt", "Answer only from CONTEXT."),
            "tools": tools,
            "tool_rationale": rationale,
            "retrieval": {"chunk_size_words": int(min(400, max(60, r.get("chunk_size_words", 200)))),
                          "chunk_overlap_words": int(min(100, max(0, r.get("chunk_overlap_words", 40)))),
                          "top_k": int(min(10, max(2, r.get("top_k", 5))))},
            "output_contract": {"answer": "string", "sources": "list[str]", "refused": "bool"},
            "docs_dir": rel_to_root(self.docs_dir),
            "model": self.llm.model,
        }

    TOOL_SIGNALS = {
        "csv_lookup": ("table_lookup", ["catalogue", "catalog", "stock", "part", "inventory", "price", "lookup", "list"]),
        "calculator": ("arithmetic", ["total", "calculate", "sum", "discount", "cost", "price", "how much", "convert"]),
        "weather": ("live_weather", ["weather", "forecast", "wind", "sea state", "temperature", "storm"]),
        "current_date": ("todays_date", ["today", "date", "deadline", "how long until", "days left", "expiry"]),
    }

    def check_tools(self, proposed, req):
        """Deterministic second opinion: a tool survives only if the requirements flag the need
        or the brief itself mentions it. Guards against e.g. an HR agent with a weather tool."""
        brief = self.brief.lower()
        needs = req.get("needs", {}) or {}
        keep, removed = [], {}
        for t in proposed:
            flag, words = self.TOOL_SIGNALS[t]
            if t == "csv_lookup" and not any(d["type"] == "csv" for d in self.docs):
                removed[t] = "no CSV file in the document folder"
            elif needs.get(flag) or any(w in brief for w in words):
                keep.append(t)
            else:
                removed[t] = "neither the brief nor the requirements need it"
        return keep, removed

    # ---------- 4. eval generator ----------
    def generate_evals(self, req, target=22):
        names = {d["name"] for d in self.docs}
        texts = {d["name"]: norm(d["text"]) for d in self.docs}
        cases, rejected = [], []
        for attempt in range(3):
            need_a = max(0, 12 - sum(c["type"] == "answerable" for c in cases))
            need_o = max(0, 5 - sum(c["type"] == "out_of_scope" for c in cases))
            need_e = max(0, 5 - sum(c["type"] == "edge" for c in cases))
            if need_a + need_o + need_e == 0:
                break
            raw = self.llm.generate(fill("generate_evals", requirements=req, documents=self.corpus,
                                         n_answerable=str(need_a), n_refuse=str(need_o), n_edge=str(need_e),
                                         existing=[c["question"] for c in cases]),
                                    stage="generate_evals", temperature=0.6).get("cases", [])
            seen = {norm(c["question"]) for c in cases + rejected}
            for c in raw:
                if not c.get("question") or norm(c["question"]) in seen:
                    continue
                seen.add(norm(c["question"]))
                c["type"] = c.get("type", "answerable")
                c["should_refuse"] = c["type"] == "out_of_scope"
                c["expected_sources"] = [s for s in c.get("expected_sources", []) if s in names]
                if c["should_refuse"]:
                    c["key_facts"] = []
                    cases.append(c)
                    continue
                # validation: key facts must literally occur in a document - otherwise the eval itself hallucinated
                good = [f for f in c.get("key_facts", []) if len(norm(f)) > 1 and any(norm(f) in t for t in texts.values())]
                if not good or not c["expected_sources"]:
                    rejected.append({**c, "reject_reason": "key facts not found verbatim in documents"})
                    continue
                c["key_facts"] = good
                cases.append(c)
        for i, c in enumerate(cases):
            c["id"] = f"case_{i + 1:02d}"
        self.save("evals.json", {"cases": cases, "rejected_by_validator": rejected})
        self.log(f"  generated {len(cases)} eval cases ({len(rejected)} rejected by validator)")
        return cases

    # ---------- 5. run + judge ----------
    def evaluate(self, definition, cases, label):
        agent = Agent(definition, self.llm)
        verdicts = []
        for c in cases:
            try:
                res = agent.answer(c["question"])
            except BudgetExceeded:
                raise
            except Exception as e:  # one broken case shouldn't kill the run
                res = {"answer": f"ERROR {e}", "sources": [], "refused": False, "retrieved_docs": [], "context": ""}
            verdicts.append(judge_case(c, res, self.llm))
        for v in verdicts:
            v.pop("context", None)
        score = sum(v["passed"] for v in verdicts) / max(1, len(verdicts))
        by_type = {}
        for t in ["answerable", "edge", "out_of_scope"]:
            vs = [v for v in verdicts if v["type"] == t]
            by_type[t] = f"{sum(v['passed'] for v in vs)}/{len(vs)}"
        cats = {k: sum(v.get("category") == k for v in verdicts) for k in CATEGORIES}
        card = {"label": label, "score": round(score, 3), "by_type": by_type, "failure_categories": cats,
                "verdicts": verdicts}
        self.save(f"results_{label}.json", card)
        self.log(f"  [{label}] score {score:.0%}  {by_type}  failures {cats}")
        return card

    # ---------- 6. critic-optimizer ----------
    def optimize(self, definition, card):
        d = copy.deepcopy(definition)
        fails = [v for v in card["verdicts"] if not v["passed"]]
        changes = []
        groups = {k: [v for v in fails if v.get("category") == k] for k in CATEGORIES}

        if groups["retrieval_miss"]:
            r = d["retrieval"]
            before = dict(r)
            if r["top_k"] < 8:
                r["top_k"] = min(10, r["top_k"] + 2)
            else:
                r["chunk_size_words"] = max(60, int(r["chunk_size_words"] * 0.75))
            changes.append({"category": "retrieval_miss", "change": f"retrieval {before} -> {dict(r)}",
                            "why": "expected source document was not in the retrieved chunks",
                            "addresses": [v["id"] for v in groups["retrieval_miss"]]})

        for cat in ["prompt_defect", "hallucination", "wrong_refusal"]:
            if not groups[cat]:
                continue
            examples = [{k: v.get(k) for k in ["question", "answer", "refused", "reason", "missing_facts"]}
                        for v in groups[cat][:6]]
            fix = self.llm.generate(fill("prompt_fix", system_prompt=d["system_prompt"], category=cat,
                                         failures=examples), stage="prompt_fix")
            rules = [r for r in fix.get("rules_to_add", []) if isinstance(r, str)][:3]
            if not rules:
                continue
            marker = "\n\n## Additional rules (added by optimizer)\n"
            if marker not in d["system_prompt"]:
                d["system_prompt"] += marker
            d["system_prompt"] += "".join(f"- {r}\n" for r in rules)
            changes.append({"category": cat, "change": f"appended rules to system prompt: {rules}",
                            "why": fix.get("rationale", ""), "addresses": [v["id"] for v in groups[cat]]})
        return d, changes

    # ---------- 7. report ----------
    def report(self, req, final, cards, log, best_label, stop_reason):
        final_card = next(c for c in cards if c["label"] == best_label)
        lines = [f"# ForgeAI report — {final['name']}", "",
                 f"**Brief:** {self.brief}", "",
                 f"**Final agent:** version {final['version']} (`agent_definition.json`), "
                 f"score **{final_card['score']:.0%}** on {len(final_card['verdicts'])} generated cases.",
                 f"**Stopped because:** {stop_reason}",
                 f"**LLM usage:** {self.llm.calls} calls, ~{self.llm.tokens:,} tokens, model `{self.llm.model}`.", "",
                 "## Groundability", ""]
        if req["unsupported_topics"]:
            lines.append("The brief asked for topics the documents cannot support. They were turned into "
                         "refusals instead of being answered:")
            lines += [f"- {u}" for u in req["unsupported_topics"]]
        else:
            lines.append("All must-answer topics were backed by a verified quote from the documents.")
        lines += ["", "## Tools attached", ""]
        lines += [f"- `{t}` — {final['tool_rationale'].get(t, '')}" for t in final["tools"]] or ["- none (documents only)"]
        lines += [f"- ~~`{t}`~~ — {why}" for t, why in final["tool_rationale"].items() if str(why).startswith("REMOVED")]
        lines += ["", "## Scorecard per iteration", "",
                  "| iteration | score | answerable | edge | out-of-scope | retrieval miss | prompt defect | hallucination | wrong refusal |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in cards:
            f = c["failure_categories"]
            lines.append(f"| {c['label']} | {c['score']:.0%} | {c['by_type']['answerable']} | {c['by_type']['edge']} | "
                         f"{c['by_type']['out_of_scope']} | {f['retrieval_miss']} | {f['prompt_defect']} | "
                         f"{f['hallucination']} | {f['wrong_refusal']} |")
        lines += ["", "## Changes made", ""]
        for e in log:
            lines.append(f"**{e['from']} → {e['to']}** (score {e['score_before']:.0%} → {e.get('score_after', 0):.0%})"
                         + (" — reverted, it made things worse" if e.get("reverted") else ""))
            lines += [f"- [{c['category']}] {c['change']} — *{c['why']}* — targets {', '.join(c['addresses'])}"
                      for c in e["changes"]]
            lines.append("")
        fails = [v for v in final_card["verdicts"] if not v["passed"]]
        lines += ["## What still fails and why", ""]
        if not fails:
            lines.append("Nothing failed on this eval set. Treat this with suspicion: the eval set was generated by "
                         "the same model family that answers, so it may be too easy. Spot-check with real users.")
        for v in fails:
            lines.append(f"- **{v['id']}** ({v['type']}, {v['category']}): {v['question']}  \n  "
                         f"→ {v['reason']}")
        lines += ["", "## Known limitations of this evaluation", "",
                  "- The eval set is generated from the documents by an LLM; key facts are verified to appear "
                  "verbatim in the documents, but question quality is not human-checked.",
                  "- Key-fact matching is string/number based and can miss correct paraphrases (false fails).",
                  "- The LLM judge uses the same model as the agent; it can only veto a pass, never grant one, "
                  "to limit self-preference.",
                  "- The optimizer tunes against the same eval set it is scored on, so the final score is optimistic."]
        self.save("report.md", "\n".join(lines))

    # ---------- orchestration ----------
    def run(self):
        self.log("1/6 intake interview")
        req = self.intake()
        self.log("2/6 groundability guard")
        req = self.checkpoint(self.groundability(req))
        self.log("3/6 generating agent definition")
        definition = self.generate_agent(req)
        self.save("agent_v1.json", definition)
        self.log(f"  tools attached: {definition['tools'] or 'none'}")
        self.log("4/6 generating eval set")
        cases = self.generate_evals(req)
        self.log("5/6 evaluate + optimize")
        cards, log = [], []
        best, best_card = definition, None
        stop_reason = f"reached the maximum of {MAX_ITERATIONS} iterations"
        try:
            card = self.evaluate(definition, cases, "v1")
            cards.append(card)
            best_card = card
            next_version = 2
            for it in range(1, MAX_ITERATIONS):
                if best_card["score"] == 1.0:
                    stop_reason = "all eval cases pass"
                    break
                new_def, changes = self.optimize(best, best_card)
                if not changes:
                    stop_reason = "optimizer had no targeted fix to apply"
                    break
                new_def["version"] = next_version
                next_version += 1
                label = f"v{new_def['version']}"
                self.save(f"agent_{label}.json", new_def)
                entry = {"from": f"v{best['version']}", "to": label, "changes": changes,
                         "score_before": best_card["score"]}
                new_card = self.evaluate(new_def, cases, label)
                cards.append(new_card)
                entry["score_after"] = new_card["score"]
                if new_card["score"] >= best_card["score"]:
                    gain = new_card["score"] - best_card["score"]
                    best, best_card = new_def, new_card
                    if gain == 0:
                        entry["note"] = "no improvement"
                else:
                    entry["reverted"] = True
                log.append(entry)
                self.save("iteration_log.json", log)
        except BudgetExceeded as e:
            stop_reason = f"budget: {e}"
        self.save("iteration_log.json", log)
        best_label = best_card["label"] if best_card else "v1"
        final = copy.deepcopy(best)
        self.save("agent_definition.json", final)
        self.log("6/6 writing report")
        if best_card:
            self.report(req, final, cards, log, best_label, stop_reason)
        self.save("scorecard.json", [{k: c[k] for k in ["label", "score", "by_type", "failure_categories"]}
                                     for c in cards])
        self.log(f"done -> {self.out}/report.md")
        return final
