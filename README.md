# ForgeAI — the agent that builds agents

Give it a one-paragraph brief and a folder of documents. It interviews you, checks the documents can
actually support the brief, generates a declarative agent (system prompt, retrieval settings, tools,
output contract), generates its own test set, judges the agent, fixes failures in up to 3 iterations,
and writes an honest report. Runs on Google Gemini.

## Setup (2 minutes)

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your key from https://aistudio.google.com/apikey
```

## Run the factory

```bash
# interactive: asks you up to 5 questions and lets you approve requirements.json
python forge.py --brief demos/hr_policy/brief.txt --docs demos/hr_policy/docs --out runs/hr_policy

# unattended (what the review uses): answers its own questions, marked ASSUMED
python forge.py --brief demos/hr_policy/brief.txt   --docs demos/hr_policy/docs   --out runs/hr_policy   --auto
python forge.py --brief demos/ship_spares/brief.txt --docs demos/ship_spares/docs --out runs/ship_spares --auto
```

`--brief` also accepts the brief text directly. A full run is roughly 150–250 Gemini calls; on the free tier
calls are spaced by `GEMINI_MIN_INTERVAL` seconds, so expect 15–20 minutes per run.

## Chat with a generated agent

```bash
python chat.py runs/hr_policy/agent_definition.json
```

## Offline test (no key needed)

```bash
python -m tests.test_smoke
```

## How the 9 requirements are met

| # | Requirement | Where |
|---|---|---|
| 1 | Intake interviewer, ≤5 questions, requirements doc | `Factory.intake`, `prompts/intake_*.txt` → `requirements.json` |
| 2 | Groundability guard | `Factory.groundability`: the model must quote evidence; quotes are checked verbatim against the documents and unsupported topics become refusals |
| 3 | Agent generator + tool selection | `Factory.generate_agent` + `check_tools` (a deterministic check removes tools the brief doesn't need) → `agent_v1.json` |
| 4 | Eval generator (≥20 cases) | `Factory.generate_evals`, `prompts/generate_evals.txt`; key facts must appear verbatim in a document or the case is rejected |
| 5 | Judge with bias defences | `forgeai/judge.py`: deterministic refusal + key-fact checks; LLM judge grades one answer alone and can only veto |
| 6 | Critic–optimizer loop | `Factory.optimize`: failures classified into retrieval miss / prompt defect / hallucination / wrong refusal; targeted fix per category; max 3 iterations; reverts a change that lowers the score → `iteration_log.json` |
| 7 | Honest reporting | `report.md`: scorecard per iteration, every change, everything still failing, limitations |
| 8 | Runnable product | `chat.py` loads any `agent_definition.json` |
| 9 | Two demos | `demos/hr_policy` (3 HR policies; brief deliberately asks for salary bands, which aren't in the docs) and `demos/ship_spares` (CSV catalogue + 2 supplier terms; needs csv_lookup + calculator) |

Stretch goals included: human checkpoint (interactive mode) and a call budget (`--max-calls`).

## Artifacts per run (`runs/<name>/`)

`requirements.json`, `agent_v1.json … agent_vN.json`, `agent_definition.json` (best version), `evals.json`
(including cases the validator rejected), `results_vN.json` (per-case verdicts), `iteration_log.json`,
`scorecard.json`, `report.md`.

## Layout

```
forge.py            factory CLI
chat.py             generic chat runner
forgeai/llm.py      Gemini REST client, budget, prompt templating
forgeai/retrieval.py loaders (.txt/.md/.pdf/.csv), chunking, BM25
forgeai/tools.py    tool registry: csv_lookup, calculator, weather (open-meteo), current_date
forgeai/agent.py    runtime for any agent definition
forgeai/judge.py    hybrid judge
forgeai/factory.py  the pipeline
prompts/            every generation prompt
demos/              two briefs + document sets (+ SYNTHESIS_PROMPT.md)
```
