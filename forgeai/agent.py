"""Generic runtime that executes ANY agent definition produced by the factory."""
import json
from pathlib import Path

from .retrieval import BM25, chunk_docs, load_docs
from .tools import REGISTRY, describe, run_tool

OUTPUT_CONTRACT = {
    "answer": "string - the reply to the user (a short polite refusal if refused)",
    "sources": "list of document names actually used, [] if none",
    "refused": "boolean - true if the question is out of scope or not answerable from CONTEXT/tool results",
}


class Agent:
    def __init__(self, definition, llm):
        self.d = definition
        self.llm = llm
        r = definition["retrieval"]
        from .llm import ROOT
        self.docs_dir = str(ROOT / definition["docs_dir"])  # absolute paths survive the join
        self.docs = load_docs(self.docs_dir)
        self.index = BM25(chunk_docs(self.docs, r["chunk_size_words"], r["chunk_overlap_words"]))
        self.tools = [t for t in definition.get("tools", []) if t in REGISTRY]

    def _plan_tool(self, question):
        if not self.tools:
            return None, None
        plan = self.llm.generate(
            "You may call ONE tool before answering, or none.\nTOOLS:\n" + describe(self.tools) +
            f"\n\nQUESTION: {question}\n\nReturn JSON {{\"tool\": \"<tool name or none>\", \"args\": {{...}}}}",
            stage="agent_plan")
        name = str(plan.get("tool", "none"))
        if name not in self.tools:
            return None, None
        return name, plan.get("args", {})

    def answer(self, question, history=None):
        hits = self.index.search(question, self.d["retrieval"]["top_k"])
        context = "\n\n".join(f"[source: {h['doc']}]\n{h['text']}" for h in hits) or "(no relevant context found)"
        tool, args = self._plan_tool(question)
        tool_block = ""
        if tool:
            result = run_tool(tool, args, self.docs_dir)
            tool_block = f"\nTOOL RESULT ({tool} {json.dumps(args)}):\n{json.dumps(result, default=str)[:3000]}\n"
        hist = ""
        if history:
            hist = "CONVERSATION SO FAR:\n" + "\n".join(f"{r}: {t}" for r, t in history[-6:]) + "\n\n"
        prompt = (f"{hist}CONTEXT:\n{context}\n{tool_block}\nQUESTION: {question}\n\n"
                  f"Respond ONLY with JSON matching this contract: {json.dumps(OUTPUT_CONTRACT)}")
        out = self.llm.generate(prompt, system=self.d["system_prompt"], stage="agent_answer")
        return {
            "answer": str(out.get("answer", "")),
            "sources": [s for s in out.get("sources", []) if isinstance(s, str)],
            "refused": bool(out.get("refused", False)),
            "tool_used": tool,
            "retrieved_docs": sorted({h["doc"] for h in hits}),
            "context": context,
        }


def load_definition(path):
    from .llm import ROOT
    d = json.loads(Path(path).read_text())
    if not Path(d["docs_dir"]).is_absolute():
        d["docs_dir"] = str(ROOT / d["docs_dir"])
    return d
