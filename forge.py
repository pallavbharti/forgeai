"""ForgeAI CLI: build, test and improve an agent from a brief + a document folder.

    python forge.py --brief demos/hr_policy/brief.txt --docs demos/hr_policy/docs --out runs/hr_policy
    add --auto to run unattended (the factory answers its own clarifying questions and marks them ASSUMED)
"""
import argparse
from pathlib import Path

from forgeai.factory import Factory
from forgeai.llm import GeminiLLM

ap = argparse.ArgumentParser()
ap.add_argument("--brief", required=True, help="path to a text file with the one-paragraph brief")
ap.add_argument("--docs", required=True, help="folder with .txt/.md/.pdf/.csv documents")
ap.add_argument("--out", required=True, help="output folder for all artifacts")
ap.add_argument("--auto", action="store_true", help="unattended: no questions to the human, no checkpoint")
ap.add_argument("--model", default=None, help="Gemini model (default from GEMINI_MODEL or gemini-3.5-flash-lite)")
ap.add_argument("--max-calls", type=int, default=None, help="LLM call budget for the whole run")
a = ap.parse_args()

brief = Path(a.brief).read_text() if Path(a.brief).exists() else a.brief
Factory(GeminiLLM(a.model, a.max_calls), brief, a.docs, a.out, auto=a.auto).run()
