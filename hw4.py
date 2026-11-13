"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# Words that carry no discriminating power in these questions. Removing them was worth about
# 15 points of recall@5 in the ablation: without this, "in"/"of"/"usd"/"millions" appear in almost
# every chunk and flatten the score differences that separate a balance sheet from a risk factor.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of for in on at to from by with as is are
was were be been being it its their they them we our you your he she his her i me my not no do does
did done have has had having held hold holds holding will would can could may might must should
shall about into over under again further once here there when where why how all any both each few
more most other some such only own same so too very just now what which who whom whose much many
percent percentage usd million millions billion year years fy fiscal period ended end per versus vs
please tell higher high lower low large larger largest small smaller big greater less least increase
increased decrease decreased grew grow growth
""".split())

# Question scaffolding: verbs and generic nouns that say how the question is phrased, not what it
# asks for. They are kept in the BM25 text (they can still help rank) but must not decide whether
# the corpus can answer, otherwise "which audit firm..." is rejected just for the word "firm".
FRAME_WORDS = frozenset("""
firm firms company companies announce announced announces generate generated generates forecast
forecasts forecasted set sets report reported reports record recorded post posted earn earned
achieve achieved deliver delivered say said come came give gave make made take took get got
amount amounts level levels figure figures number numbers value values item items line lines
provide provided include included remain remained continue continued expect expected
""".split())

# The question asks for an attribute; the corpus may name it differently. Only needed so that the
# answerability check does not reject a question just because it used a synonym.
ALIASES = {
    "city": ("headquarter", "headquarters", "headquartered", "based"),
    "headquartered": ("headquarter", "headquarters"),
    "headcount": ("employee", "employees", "staff"),
    "staff": ("employee", "employees"),
    "workforce": ("employee", "employees"),
    "sales": ("revenue", "revenues"),
    "turnover": ("revenue", "revenues"),
    "debt": ("borrowing", "borrowings"),
    "leverage": ("borrowing", "borrowings"),
    "buyback": ("repurchase", "repurchases"),
    "capex": ("capital", "expenditure"),
}

# Words that start a question and must never be taken for the first word of a company name, so that
# "Did Orchid Bay Hotels or Sablewood Retail..." yields two companies and not three.
QUESTION_STARTERS = frozenset("""
did do does is are was were which what who whom whose how when where why in on at for to of the a an
can could will would should has have had if and or but
""".split())

# Words that never name a company under review: brokers, audit firms, titles, calendar words.
NOT_A_COMPANY = re.compile(
    r"\b(equity research|securities|capital|llp|audit|auditors?|partners|associates|bank|trust|"
    r"chief financial officer|chief executive officer|general counsel|head of investor relations|"
    r"fourth quarter|full year|annual report|earnings call|conference call|results review|"
    r"board of directors|audit committee|usd|fy\d{4}|q[1-4])\b", re.I)

# Company names seen by chunk_document, keyed by the grader's own company_key(). parse_query runs
# before build_index and only receives the question, so this registry is how it learns which
# companies the corpus holds.
CORPUS_COMPANIES: dict[str, str] = {}


# ----------------------------------------------------------------------------- small utilities

def _stem(word: str) -> str:
    """Crude plural stripping, enough to match 'borrowing' with 'borrowings'."""
    word = word.strip(".,()%-")
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: Any) -> list[str]:
    """Lower-case word tokens with stop words removed. Numbers keep their commas and parentheses.

    Hyphens are separators: "fourth-quarter" must match the heading "Fourth Quarter", and
    "long-term" must match "Long-term borrowings" the same way on both sides.
    """
    out = []
    for raw in re.sub(r"[^a-z0-9.,%()]+", " ", str(text).lower()).split():
        word = raw.strip(".,()")
        if word and word not in STOPWORDS:
            out.append(word)
    return out


def _variants(term: str) -> set[str]:
    return {_stem(term)} | {_stem(a) for a in ALIASES.get(term, ())}


def _split_text(text: str, limit: int) -> list[str]:
    """Split on line boundaries, then on spaces, so no value is ever cut in half."""
    limit = max(120, limit)
    pieces, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:cut].rstrip())
            line = line[cut:].lstrip()
        if current and len(current) + len(line) + 1 > limit:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


# ----------------------------------------------------------------------------- Task 1.1

def _table_to_text(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table") or soup

    caption_tag = table.find("caption")
    caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""

    head = table.find("thead")
    header_row = head.find("tr") if head else None
    if header_row is None:                       # no <thead>: use the first row if it is all <th>
        first = table.find("tr")
        if first is not None:
            cells = first.find_all(["th", "td"], recursive=False)
            if cells and len(first.find_all("th")) == len(cells):
                header_row = first
    headers = ([c.get_text(" ", strip=True) for c in header_row.find_all(["th", "td"], recursive=False)]
               if header_row is not None else [])

    lines = [caption] if caption else []
    for row in table.find_all("tr"):
        if row is header_row:
            continue
        values = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"], recursive=False)]
        if not values or values == headers:
            continue
        if len(values) == 1:                     # section sub-header, e.g. <th colspan="3">Assets</th>
            lines.append(values[0])
            continue
        label = values[0]
        pairs = []
        for i, value in enumerate(values[1:], 1):
            header = headers[i] if i < len(headers) and headers[i] else f"column {i}"
            pairs.append(f"{header}: {value}")
        line = f"{label} | " + " | ".join(pairs)
        if len(line) <= MAX_LINE_CHARS:
            lines.append(line)
            continue
        current = label                          # very wide row: repeat the label on each part
        for pair in pairs:
            if current != label and len(current) + len(pair) + 3 > MAX_LINE_CHARS:
                lines.append(current)
                current = label
            current = f"{current} | {pair}"
        lines.append(current)
    return "\n".join(lines)


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    return _table_to_text(table_html)


# ----------------------------------------------------------------------------- Task 1.2

def _clean_company(title: str) -> str:
    """'Northgate Robotics Inc. (TSX: NGR)' -> 'Northgate Robotics Inc.'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def _doc_type_of(head_text: str) -> str:
    low = head_text.lower()
    if "earnings conference call" in low or "earnings call" in low:
        return "earnings_call"
    if "equity research" in low or "results review" in low:
        return "research_report"
    return "annual_report"


def _fiscal_year_of(head_text: str) -> int:
    for pattern in (r"annual report\s+(\d{4})", r"fiscal year ended 31 december\s+(\d{4})",
                    r"full year\s+(\d{4})", r"fy\s*(\d{4})\s+results review", r"\bfy\s*(\d{4})"):
        found = re.search(pattern, head_text, re.I)
        if found:
            return int(found.group(1))
    found = re.search(r"\b(20\d\d)\b", head_text)
    return int(found.group(1)) if found else 0


def _blocks(body) -> list[tuple[tuple[str, ...], str, str]]:
    """The document as (section path, kind, text) blocks, in reading order."""
    blocks: list[tuple[tuple[str, ...], str, str]] = []
    section: list[str] = []
    speaker = None
    for element in body.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):
        if element.name != "table" and element.find_parent("table") is not None:
            continue
        if element.name in ("p", "ul", "ol") and element.find_parent(["ul", "ol"]) is not None:
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name == "h1":
            section = [text]
            continue
        if element.name == "h2":
            section = section[:1] + [text]
            continue
        if element.name == "h3":
            section = section[:2] + [text]
            continue
        if element.name == "table":
            blocks.append((tuple(section), "table", _table_to_text(str(element))))
            speaker = None
            continue
        if element.name in ("ul", "ol"):
            items = [li.get_text(" ", strip=True) for li in element.find_all("li", recursive=False)]
            blocks.append((tuple(section), "list", "\n".join(f"- {i}" for i in items if i)))
            speaker = None
            continue
        if "speaker" in " ".join(element.get("class") or []):
            speaker = text
            continue
        blocks.append((tuple(section), "para", f"{speaker}: {text}" if speaker else text))
        speaker = None
    return blocks


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    for tag in body.find_all(["nav", "script", "style", "footer"]):
        tag.decompose()

    head_text = body.get_text(" ", strip=True)[:1500]
    title = body.find("h1")
    company = _clean_company(title.get_text(" ", strip=True) if title else "")
    if not company:                              # fall back to the <title>, which repeats the name
        company = _clean_company(soup.title.get_text(" ", strip=True) if soup.title else "")
    doc_type = _doc_type_of(head_text)
    fiscal_year = _fiscal_year_of(head_text)

    key = company_key(company)
    if key:
        CORPUS_COMPANIES[key] = company

    base = {"source": file_name, "company": company, "fiscal_year": int(fiscal_year),
            "doc_type": doc_type}

    def header(section: tuple[str, ...]) -> str:
        # The <h1> is the company name again: leave it out so the name appears once, not twice.
        parts = [company, f"FY{fiscal_year}", doc_type.replace("_", " ")]
        if len(section) > 1:
            parts.append(" > ".join(section[1:]))
        return " | ".join(p for p in parts if p)

    chunks: list[Document] = []
    buffer: list[str] = []
    buffer_len = 0
    buffer_section: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal buffer, buffer_len, buffer_section
        if buffer:
            head = header(buffer_section)
            text = f"{head}\n" + "\n".join(buffer) if head else "\n".join(buffer)
            chunks.append(Document(page_content=text, metadata=dict(
                base, section=" > ".join(buffer_section[1:]), chunk_ix=len(chunks))))
        buffer, buffer_len, buffer_section = [], 0, ()

    for section, kind, text in _blocks(body):
        if kind == "table":
            flush()                              # a table is never glued to the prose around it
            lines = text.split("\n")
            caption = lines[0] if len(lines) > 1 else ""
            rows = "\n".join(lines[1:]) if caption else text
            budget = MAX_CHUNK_CHARS - len(header(section)) - len(caption) - 2
            pieces = [f"{caption}\n{p}" if caption else p for p in _split_text(rows, budget)]
        else:
            pieces = _split_text(text, MAX_CHUNK_CHARS - len(header(section)) - 2)

        for piece in pieces:
            active = buffer_section if buffer else section
            budget = MAX_CHUNK_CHARS - len(header(active)) - 1
            if buffer and buffer_len + len(piece) + 1 > budget:
                flush()
            if not buffer:
                buffer_section = section
                budget = MAX_CHUNK_CHARS - len(header(section)) - 1
            if len(piece) + 1 > budget:           # only if a header is enormous
                for part in _split_text(piece, budget):
                    buffer, buffer_len, buffer_section = [part], len(part) + 1, section
                    flush()
                continue
            buffer.append(piece)
            buffer_len += len(piece) + 1
    flush()

    # Fold stray one-liners (footers, a lone sentence) into a neighbour: a 60-character chunk wins
    # BM25 on length normalisation and pushes real tables out of the top 5.
    merged: list[Document] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        if (previous is not None and len(chunk.page_content) < 250
                and len(previous.page_content) + len(chunk.page_content) + 1 <= MAX_CHUNK_CHARS):
            previous.page_content += "\n" + chunk.page_content
        else:
            merged.append(chunk)

    # Hard guarantee of the character cap, whatever the headers did.
    final: list[Document] = []
    for chunk in merged:
        if len(chunk.page_content) <= MAX_CHUNK_CHARS:
            final.append(chunk)
            continue
        head, _, rest = chunk.page_content.partition("\n")
        for part in _split_text(rest, MAX_CHUNK_CHARS - len(head) - 1):
            final.append(Document(page_content=f"{head}\n{part}",
                                  metadata=dict(chunk.metadata, chunk_ix=len(final))))
    for i, chunk in enumerate(final):
        chunk.metadata["chunk_ix"] = i
    return final


# ----------------------------------------------------------------------------- Task 2.2

def _known_companies(question: str) -> list[str]:
    """Corpus companies named in the question, in the order the question names them."""
    normalised = " " + " ".join(re.sub(r"[^a-z0-9&. -]+", " ", question.lower()).split()) + " "
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for key in sorted(CORPUS_COMPANIES, key=len, reverse=True):   # longest first: no partial hits
        start = normalised.find(f" {key} ")
        while start != -1:
            if not any(s <= start < e for s, e in taken):
                taken.append((start, start + len(key) + 1))
                found.append((start, CORPUS_COMPANIES[key]))
                break
            start = normalised.find(f" {key} ", start + 1)
    return [name for _, name in sorted(found)]


def _unknown_companies(question: str, known: list[str]) -> list[str]:
    """Companies the question names that the corpus does not hold, e.g. 'Northgate Robotic Systems'.

    They look like a real company with one word changed, so they are found as a capitalised phrase
    that shares a word with a corpus company without being one.
    """
    vocabulary = {w for key in CORPUS_COMPANIES for w in key.split()}
    known_keys = {company_key(c) for c in known}
    out: list[str] = []
    for match in re.finditer(r"\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){1,3})", question):
        words = match.group(1).strip().split()
        while words and words[0].lower() in QUESTION_STARTERS:
            words.pop(0)
        if len(words) < 2:                       # one capitalised word is not a company name
            continue
        phrase = " ".join(words)
        if NOT_A_COMPANY.search(phrase):
            continue
        key = company_key(phrase)
        if not key or key in CORPUS_COMPANIES or key in known_keys:
            continue
        if any(key.startswith(k) or k.startswith(key) for k in CORPUS_COMPANIES):
            continue
        if any(f" {k} " in f" {key} " for k in CORPUS_COMPANIES):
            continue                             # "the Northgate Robotics" is not a new company
        if set(key.split()) & vocabulary and key not in {company_key(c) for c in out}:
            out.append(phrase)
    return out


def _llm_companies(question: str) -> list[str]:
    """Last resort when the rules find no company at all."""
    try:
        llm = make_llm(temperature=0.0, max_tokens=80)
        reply = str(llm.invoke(
            "Name the companies this question asks about, one per line, as named. "
            "Do not list brokers, audit firms or people. Reply NONE if there are none.\n\n"
            f"Question: {question}").content)
    except Exception:
        return []
    names = []
    for line in reply.splitlines():
        name = re.sub(r"^[\s\-*\d.)]+", "", line).strip(" .\"'")
        if name and name.upper() != "NONE" and not NOT_A_COMPANY.search(name):
            names.append(name)
    return names[:3]


def _fiscal_year_in(question: str) -> int | None:
    flat = re.sub(r"[^a-z0-9 ]+", " ", question.lower())
    found = (re.search(r"(?:fy|fiscal|december|year ended)\s*(\d{4})", flat)
             or re.search(r"\b(20[2-3]\d)\b", flat))
    return int(found.group(1)) if found else None


def _doc_type_in(question: str) -> str:
    low = question.lower()
    if re.search(r"\bearnings call\b|\bconference call\b|\btranscript\b|\bon the call\b", low):
        return "earnings_call"
    if re.search(r"\brating\b|target price|\bbroker\b|\bforecast\b|results review|\bestimate\b"
                 r"|\banalyst\b|\bupside\b|\bcoverage\b", low):
        return "research_report"
    return "annual_report"


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    question = str(question or "")
    try:
        companies = _known_companies(question)
        companies += _unknown_companies(question, companies)
        if not companies:
            companies = _llm_companies(question)
        return {"companies": companies,
                "fiscal_year": _fiscal_year_in(question),
                "doc_type": _doc_type_in(question)}
    except Exception:
        return {"companies": [], "fiscal_year": None, "doc_type": "annual_report"}


# ----------------------------------------------------------------------------- Task 2

def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    chunks = list(chunks)
    tokenised = [_tokens(c.page_content) or ["__empty__"] for c in chunks]
    # b=0.3 instead of the default 0.75: with 0.75 a short risk-factor paragraph outranked the
    # income statement that actually holds the figure, purely because it is shorter.
    bm25 = BM25Okapi(tokenised, b=0.3)

    document_frequency: dict[str, int] = {}
    for tokens in tokenised:
        for term in {_stem(t) for t in tokens}:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    # Four-character prefixes, so "guide" counts as known when the corpus only says "guidance".
    prefixes = {term[:4] for term in document_frequency if len(term) >= 4}

    by_company: dict[str, list[Document]] = {}
    years: dict[str, set] = {}
    for chunk in chunks:
        meta = chunk.metadata or {}
        key = company_key(meta.get("company", ""))
        if not key:
            continue
        by_company.setdefault(key, []).append(chunk)
        years.setdefault(key, set()).add(meta.get("fiscal_year"))

    return {"chunks": chunks, "bm25": bm25, "df": document_frequency, "prefixes": prefixes,
            "by_company": by_company, "years": years,
            "companies": sorted({(c.metadata or {}).get("company", "") for c in chunks} - {""})}


def _attribute_missing(question: str, index: dict) -> bool:
    """True when the corpus cannot have the figure: the question asks for something it never mentions."""
    vocabulary = {w for key in index.get("by_company", {}) for w in key.split()}
    terms = []
    for term in _tokens(question):
        if term in vocabulary or term in FRAME_WORDS or re.fullmatch(r"(fy)?20\d\d", term):
            continue
        if len(term) <= 2:
            continue
        terms.append(term)
    if not terms:
        return False
    df = index["df"]
    prefixes = index.get("prefixes", set())
    for term in terms:
        if max(df.get(v, 0) for v in _variants(term)):
            continue
        if len(term) >= 4 and term[:4] in prefixes:
            continue                             # "guide" for "guidance", "generate" for "generated"
        return True                              # e.g. "lobbying", "churn", "carbon", "single"
    return False


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    if not index:
        return []
    try:
        parsed = parse_query(question)
        wanted = [company_key(c) for c in parsed["companies"]]
        known = [k for k in wanted if k in index["by_company"]]
        if not known:
            return []                            # every company named is absent from the corpus

        year = parsed["fiscal_year"]
        doc_type = parsed["doc_type"]

        if doc_type == "annual_report" and year is not None:
            covered = set()
            for key in known:
                covered |= {y for y in index["years"].get(key, set()) if isinstance(y, int)}
            if covered and (year > max(covered) or year < min(covered) - 1):
                return []                        # FY2023 lives in the FY2024 report, FY2021 nowhere

        # Expand the query with the words the corpus actually uses: a question about "sales" has to
        # reach the revenue line, and one about "headcount" the employees table.
        query_tokens = _tokens(question)
        expanded = list(query_tokens)
        for term in query_tokens:
            for alias in ALIASES.get(term, ()):
                expanded.extend(_tokens(alias))
        scores = index["bm25"].get_scores(expanded)
        rank = {id(chunk): score for chunk, score in zip(index["chunks"], scores)}

        pools: list[list[Document]] = []
        for key in known:
            pool = index["by_company"][key]
            typed = [c for c in pool if (c.metadata or {}).get("doc_type") == doc_type]
            if typed:
                pool = typed
            if doc_type == "annual_report" and year is not None:
                in_year = [c for c in pool if (c.metadata or {}).get("fiscal_year") in (year, year + 1)]
                if in_year:
                    pool = in_year
            pools.append(pool)

        # A comparison needs a chunk for each company; 5 chunks of one company are a miss.
        # Rank inside each company's pool, but take at most one chunk per top-level section: a
        # comparison's slot should not be spent on a second risk-factor paragraph or the cover page.
        # The report of the year itself comes first: the FY2025 report names both the old and the new
        # CFO, so it answers "who was CFO at the end of FY2024" ambiguously, the FY2024 report not.
        same_year = doc_type == "annual_report" and year is not None

        def order(chunk: Document):
            matches = (chunk.metadata or {}).get("fiscal_year") == year if same_year else True
            return (matches, rank.get(id(chunk), 0.0))

        queues: list[list[Document]] = []
        for pool in pools:
            fresh, repeated = [], []
            used_sections: set[str] = set()
            for chunk in sorted(pool, key=order, reverse=True):
                section = str((chunk.metadata or {}).get("section", "")).split(" > ")[0]
                if section in used_sections:
                    repeated.append(chunk)
                else:
                    used_sections.add(section)
                    fresh.append(chunk)
            queues.append(fresh + repeated)

        # Interleave the companies, so neither is starved when the five slots run out.
        picked: list[Document] = []
        seen: set[str] = set()
        depth = 0
        while len(picked) < MAX_RETRIEVED and any(depth < len(q) for q in queues):
            for queue in queues:
                if len(picked) >= MAX_RETRIEVED:
                    break
                if depth < len(queue) and queue[depth].page_content not in seen:
                    seen.add(queue[depth].page_content)
                    picked.append(queue[depth])
            depth += 1

        if _attribute_missing(question, index):
            return []
        return picked[:MAX_RETRIEVED]
    except Exception:
        traceback.print_exc()
        return []


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
