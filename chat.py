"""Generic chat runner: loads ANY agent definition produced by ForgeAI.

    python chat.py runs/hr_policy/agent_definition.json
"""
import sys

from forgeai.agent import Agent, load_definition
from forgeai.llm import GeminiLLM

if len(sys.argv) < 2:
    sys.exit("usage: python chat.py <agent_definition.json>")
d = load_definition(sys.argv[1])
agent = Agent(d, GeminiLLM(d.get("model")))
print(f"{d['name']} (v{d['version']}, tools: {d['tools'] or 'none'}) - type 'exit' to quit\n")
history = []
while True:
    try:
        q = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in {"exit", "quit"}:
        break
    if not q:
        continue
    r = agent.answer(q, history)
    print(f"\nagent> {r['answer']}")
    if r["sources"]:
        print(f"       sources: {', '.join(r['sources'])}")
    if r["tool_used"]:
        print(f"       tool: {r['tool_used']}")
    print()
    history += [("user", q), ("assistant", r["answer"])]
