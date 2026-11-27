"""Thin Gemini client (REST, no SDK) with JSON mode, retries and a call/token budget."""
import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Minimal .env loader so we don't need python-dotenv."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class BudgetExceeded(Exception):
    pass


class GeminiLLM:
    def __init__(self, model=None, max_calls=None):
        load_env()
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key or self.api_key.startswith("your-"):
            raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and paste your key.")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.max_calls = int(max_calls or os.environ.get("FORGE_MAX_CALLS", 400))
        self.min_interval = float(os.environ.get("GEMINI_MIN_INTERVAL", 4.5))  # free tier ~15 RPM
        self.calls = 0
        self.tokens = 0
        self._last = 0.0

    def generate(self, prompt, system=None, json_mode=True, temperature=0.2, stage=""):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget of {self.max_calls} exhausted")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = requests.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=120)
            except (requests.ConnectionError, requests.Timeout) as e:  # internet blip: wait and retry
                print(f"  (network error, retrying in {15 * (attempt + 1)}s: {type(e).__name__})")
                time.sleep(15 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 503):
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 404:
                raise SystemExit(f"Model '{self.model}' not found for this key. Set GEMINI_MODEL in .env to a model your key can use.")
            r.raise_for_status()
            data = r.json()
            self.calls += 1
            self.tokens += data.get("usageMetadata", {}).get("totalTokenCount", 0)
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                text = "{}" if json_mode else ""
            return as_dict(parse_json(text)) if json_mode else text
        raise RuntimeError("Gemini kept failing after 6 attempts (check internet / API quota)")


def as_dict(obj):
    """Models sometimes wrap the object in a list ([{...}]) - always hand callers a dict."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        dicts = [o for o in obj if isinstance(o, dict)]
        if len(dicts) == 1:
            return dicts[0]
        if dicts:  # a bare list of items, e.g. eval cases or topics
            return {"cases": dicts, "topics": dicts, "items": dicts}
    return {}


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def fill(template_name, **kw):
    """Load prompts/<name>.txt and replace {{key}} placeholders."""
    t = (ROOT / "prompts" / f"{template_name}.txt").read_text()
    for k, v in kw.items():
        if not isinstance(v, str):
            v = json.dumps(v, indent=2, ensure_ascii=False)
        t = t.replace("{{" + k + "}}", v)
    return t
