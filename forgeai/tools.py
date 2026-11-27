"""Tool registry the factory chooses from. Each tool: description, args schema, function."""
import ast
import csv
import datetime as dt
import operator
from pathlib import Path

import requests


def csv_lookup(docs_dir, file="", query="", **_):
    files = [Path(docs_dir) / file] if file else list(Path(docs_dir).glob("*.csv"))
    terms = [t for t in str(query).lower().split() if t]
    hits = []
    for f in files:
        if not f.exists():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                line = " ".join(str(v) for v in row.values()).lower()
                if terms and all(t in line for t in terms):
                    hits.append({"file": f.name, **row})
    return hits[:10] or "no matching rows"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression="", **_):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    try:
        return round(ev(ast.parse(str(expression), mode="eval").body), 6)
    except Exception as e:
        return f"error: {e}"


def weather(city="", **_):
    try:
        g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 1}, timeout=15).json()
        loc = g["results"][0]
        w = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                                 "current": "temperature_2m,wind_speed_10m,weather_code"}, timeout=15).json()
        return {"city": loc["name"], "country": loc.get("country"), **w.get("current", {})}
    except Exception as e:
        return f"error: {e}"


def current_date(**_):
    return dt.date.today().isoformat()


REGISTRY = {
    "csv_lookup": {"description": "Find rows in a CSV file of the document folder whose cells contain all query keywords "
                                  "(e.g. part numbers, item names). Use for catalogue / table lookups.",
                   "args": {"file": "optional csv file name", "query": "keywords"}, "fn": csv_lookup},
    "calculator": {"description": "Evaluate an arithmetic expression, e.g. '12 * 4.5 * 0.9'. Use for totals, discounts, "
                                  "unit conversions.", "args": {"expression": "arithmetic expression"}, "fn": calculator},
    "weather": {"description": "Current weather for a city via the keyless open-meteo API.",
                "args": {"city": "city name"}, "fn": weather},
    "current_date": {"description": "Today's date (ISO). Use when answers depend on today's date, deadlines or ages.",
                     "args": {}, "fn": current_date},
}


def describe(names=None):
    return "\n".join(f"- {n}: {t['description']} args={t['args']}"
                     for n, t in REGISTRY.items() if names is None or n in names)


def run_tool(name, args, docs_dir):
    if name not in REGISTRY:
        return f"error: unknown tool {name}"
    args = args if isinstance(args, dict) else {}
    if name == "csv_lookup":
        return csv_lookup(docs_dir, **args)
    return REGISTRY[name]["fn"](**args)
