"""Document loading, word-window chunking and a small BM25 retriever (no external deps)."""
import csv
import math
import re
from collections import Counter
from pathlib import Path

TEXT_EXT = {".txt", ".md"}


def load_docs(folder):
    docs = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in TEXT_EXT:
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
        elif p.suffix.lower() == ".csv":
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            header = rows[0] if rows else []
            text = "\n".join("; ".join(f"{h}: {v}" for h, v in zip(header, r)) for r in rows[1:])
        else:
            continue
        docs.append({"name": p.name, "type": p.suffix.lower().lstrip("."), "text": text})
    return docs


def inventory(docs):
    return "\n".join(f"- {d['name']} ({d['type']}, {len(d['text'].split())} words): "
                     f"{' '.join(d['text'].split()[:40])} ..." for d in docs)


def full_text(docs, limit=150_000):
    out = "\n\n".join(f"=== DOCUMENT: {d['name']} ===\n{d['text']}" for d in docs)
    return out[:limit]


def tokenize(s):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower())


def chunk_docs(docs, size=200, overlap=40):
    chunks = []
    for d in docs:
        if d["type"] == "csv":  # keep rows intact: one chunk per few rows
            lines = d["text"].splitlines()
            step = max(1, size // 25)
            for i in range(0, len(lines), step):
                chunks.append({"doc": d["name"], "text": "\n".join(lines[i:i + step])})
            continue
        words = d["text"].split()
        step = max(1, size - overlap)
        for i in range(0, max(1, len(words)), step):
            piece = words[i:i + size]
            if piece:
                chunks.append({"doc": d["name"], "text": " ".join(piece)})
            if i + size >= len(words):
                break
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.toks = [tokenize(c["text"]) for c in chunks]
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for toks in self.toks for t in set(toks))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(t) for t in self.toks]
        self.k1, self.b = k1, b

    def search(self, query, k=5):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for s, i in scores[:k] if s > 0]
