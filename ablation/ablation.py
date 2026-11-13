"""FTEC5660 Homework 4, Task 5: the ablation study.

One naive baseline (B0, as in Tutorial 7 Section 2) and ten variants that each change exactly one
component relative to the row above it. Every row is scored with the *provided* grader functions
imported from hw4.py, so the numbers are the same ones the runner prints.

    python ablation/ablation.py                         # all variants, answer accuracy included
    python ablation/ablation.py --answers final         # generator only for the shipped config
    python ablation/ablation.py --answers none          # retrieval metrics only, no API cost
    python ablation/ablation.py --variants B0,V1,V2     # a subset
    python ablation/ablation.py --render-only           # rebuild {out}_table.md from {out}.csv, no re-run

Run it from the repository root; it finds hw4.py and public_test/ on its own, and writes its
artifacts next to this file.

Naming: B0 is the baseline, V1..V10 are the variants ("V" for variant). Each one adds or changes a
single component relative to the row above, and V10 is the shipped configuration (verified
byte-identical against hw4.py). Every component the shipped code runs is introduced by exactly one
row; the CONFIGS list below is the authoritative description, and every description is also written
to the "change" column of {out}.csv and to the "What each variant changes" table in {out}_table.md.

LLM reranking is not part of that chain, because hw4.py does not rerank. It is measured separately:

    python ablation/ablation.py --rerank-side           # {out}_rerank.csv, ~120 LLM calls, no generator

Outputs
    ablation.csv          one row per variant (all metrics)
    ablation_table.md     the same as a markdown table, to paste into README.md
    ablation_errors.csv   the shipped config's per-question failures (error analysis, 2 points)

The last row (V10) is the shipped configuration: it is verified against hw4.chunk_document and
hw4.retrieve on every document and every question, so the table provably matches results.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# this file lives in ablation/, next to hw4.py one level up: make `import hw4` work from any cwd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import hw4
from hw4 import (
    MAX_CHUNK_CHARS,
    MAX_RETRIEVED,
    POINTS,
    RECALL_FULL,
    RECALL_ZERO,
    answer_correct,
    clip,
    company_key,
    document_problems,
    generate,
    load_corpus,
    load_json,
    parse_score,
    retrieval_hit,
    table_score,
)

# =============================================================================
# The configurations: one component changes per row
# =============================================================================


@dataclass(frozen=True)
class Config:
    name: str
    change: str          # what changed relative to the row above (this is the report's table text)
    # ---- Task 1: parsing and chunking
    table: bool = False          # table_to_text inside the chunker
    chunker: str = "naive"       # "naive" (get_text + RecursiveCharacterTextSplitter) | "structured"
    header: bool = False         # contextual chunk header (company | FY | doc_type | section)
    # ---- Task 2: index and query
    tokeniser: str = "whitespace"  # "whitespace" (lower-cased split) | "stopword" (hw4._tokens)
    b: float = 0.75                # BM25 length normalisation
    parse: bool = False            # parse_query + metadata filtering
    aliases: bool = False          # query expansion with hw4.ALIASES
    # ---- Task 2/3: ranking and abstention
    same_year: bool = False        # prefer the report of the year asked about
    per_company: bool = False      # one queue per company, interleaved (comparisons)
    sections: bool = False         # at most one chunk per top-level section per company
    abstain: bool = False          # the lexical answerability rule (hw4._attribute_missing)
    rerank: bool = False           # reorder the top RERANK_DEPTH candidates with make_llm()
    shipped: bool = False          # the configuration that hw4.py actually implements


CONFIGS: list[Config] = [
    Config(
        name="B0",
        change="naive RAG, Tutorial 7 Section 2: get_text(), RecursiveCharacterTextSplitter, "
               "lower-cased whitespace tokens, BM25 top-5 over the whole corpus, no query parsing, never []",
    ),
    Config(
        name="V1",
        change="+ structured parsing: walk h1/h2/h3/p/ul/table as blocks and convert every <table> "
               "with table_to_text (row label + column header kept), tables never glued to prose",
        table=True, chunker="structured",
    ),
    Config(
        name="V2",
        change="+ contextual chunk header: prepend 'company | FY | doc_type | section' to every chunk",
        table=True, chunker="structured", header=True,
    ),
    Config(
        name="V3",
        change="+ better tokeniser: lower-case, strip stopwords and question scaffolding, "
               "light stemming instead of raw whitespace tokens",
        table=True, chunker="structured", header=True, tokeniser="stopword",
    ),
    Config(
        name="V4",
        change="+ BM25 b=0.3 instead of the default 0.75 (less length normalisation, so a short "
               "risk-factor paragraph stops outranking the income statement)",
        table=True, chunker="structured", header=True, tokeniser="stopword", b=0.3,
    ),
    Config(
        name="V5",
        change="+ parse_query and metadata filtering: rank only the chunks of the asked company, "
               "document type and fiscal year",
        table=True, chunker="structured", header=True, tokeniser="stopword", b=0.3, parse=True,
    ),
    Config(
        name="V6",
        change="+ query expansion: add the words the corpus uses for the asked attribute "
               "(sales -> revenue, headcount -> employees, ...)",
        table=True, chunker="structured", header=True, tokeniser="stopword", b=0.3, parse=True,
        aliases=True,
    ),
    Config(
        name="V7",
        change="+ same-year preference: when the year is asked about, the report of that year is "
               "ranked before the other one (it names the CFO in office then)",
        table=True, chunker="structured", header=True, tokeniser="stopword", b=0.3, parse=True,
        aliases=True, same_year=True,
    ),
    Config(
        name="V8",
        change="+ per-company retrieval: rank inside each company's pool and interleave the pools, "
               "so a comparison gets a chunk for each company instead of five for one",
        table=True, chunker="structured", header=True, tokeniser="stopword", b=0.3, parse=True,
        aliases=True, same_year=True, per_company=True,
    ),
    Config(
        name="V9",
        change="+ section diversity: at most one chunk per top-level section in each company's queue",
        table=True, chunker="structured", header=True, tokeniser="stopword", b=0.3, parse=True,
        aliases=True, same_year=True, per_company=True, sections=True,
    ),
    Config(
        name="V10",
        change="+ abstention: return [] when the question asks for an attribute the corpus never "
               "mentions (lexical document-frequency check) -- the shipped configuration",
        table=True, chunker="structured", header=True, tokeniser="stopword", b=0.3, parse=True,
        aliases=True, same_year=True, per_company=True, sections=True, abstain=True, shipped=True,
    ),
]

SHIPPED: Config = next(c for c in CONFIGS if c.shipped)


# =============================================================================
# Task 1: the chunkers
# =============================================================================


def _metadata_of(soup: BeautifulSoup, file_name: str) -> dict:
    """source / company / fiscal_year / doc_type, read from the document head, as in hw4."""
    body = soup.body or soup
    head_text = body.get_text(" ", strip=True)[:1500]
    title = body.find("h1")
    company = hw4._clean_company(title.get_text(" ", strip=True) if title else "")
    if not company:
        company = hw4._clean_company(soup.title.get_text(" ", strip=True) if soup.title else "")
    key = company_key(company)
    if key:
        hw4.CORPUS_COMPANIES[key] = company
    return {"source": file_name, "company": company,
            "fiscal_year": int(hw4._fiscal_year_of(head_text)),
            "doc_type": hw4._doc_type_of(head_text)}


def chunk_naive(file_name: str, html: str) -> list[Document]:
    """B0: the whole document as one string, cut every 1500 characters, no table awareness."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["nav", "script", "style", "footer"]):
        tag.decompose()
    base = _metadata_of(soup, file_name)
    text = (soup.body or soup).get_text("\n", strip=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=0,
                                              separators=["\n\n", "\n", " ", ""])
    return [Document(page_content=piece, metadata=dict(base, section="", chunk_ix=i))
            for i, piece in enumerate(p for p in splitter.split_text(text) if p.strip())]


def chunk_structured(file_name: str, html: str, header_on: bool) -> list[Document]:
    """V1 onward: a copy of hw4.chunk_document with the contextual header switchable.

    With header_on=True the output is byte-identical to hw4.chunk_document (asserted in verify()).
    """
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    for tag in body.find_all(["nav", "script", "style", "footer"]):
        tag.decompose()

    base = _metadata_of(soup, file_name)
    company = base["company"]
    fiscal_year, doc_type = base["fiscal_year"], base["doc_type"]

    def header(section: tuple[str, ...]) -> str:
        if not header_on:
            return ""
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

    for section, kind, text in hw4._blocks(body):
        if kind == "table":
            flush()
            lines = text.split("\n")
            caption = lines[0] if len(lines) > 1 else ""
            rows = "\n".join(lines[1:]) if caption else text
            budget = MAX_CHUNK_CHARS - len(header(section)) - len(caption) - 2
            pieces = [f"{caption}\n{p}" if caption else p for p in hw4._split_text(rows, budget)]
        else:
            pieces = hw4._split_text(text, MAX_CHUNK_CHARS - len(header(section)) - 2)

        for piece in pieces:
            active = buffer_section if buffer else section
            budget = MAX_CHUNK_CHARS - len(header(active)) - 1
            if buffer and buffer_len + len(piece) + 1 > budget:
                flush()
            if not buffer:
                buffer_section = section
                budget = MAX_CHUNK_CHARS - len(header(section)) - 1
            if len(piece) + 1 > budget:
                for part in hw4._split_text(piece, budget):
                    buffer, buffer_len, buffer_section = [part], len(part) + 1, section
                    flush()
                continue
            buffer.append(piece)
            buffer_len += len(piece) + 1
    flush()

    merged: list[Document] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        if (previous is not None and len(chunk.page_content) < 250
                and len(previous.page_content) + len(chunk.page_content) + 1 <= MAX_CHUNK_CHARS):
            previous.page_content += "\n" + chunk.page_content
        else:
            merged.append(chunk)

    final: list[Document] = []
    for chunk in merged:
        if len(chunk.page_content) <= MAX_CHUNK_CHARS:
            final.append(chunk)
            continue
        head, _, rest = chunk.page_content.partition("\n")
        for part in hw4._split_text(rest, MAX_CHUNK_CHARS - len(head) - 1):
            final.append(Document(page_content=f"{head}\n{part}",
                                  metadata=dict(chunk.metadata, chunk_ix=len(final))))
    for i, chunk in enumerate(final):
        chunk.metadata["chunk_ix"] = i
    return final


def chunker_of(cfg: Config) -> Callable[[str, str], list[Document]]:
    if cfg.chunker == "naive":
        return chunk_naive
    return lambda file_name, html: chunk_structured(file_name, html, cfg.header)


# =============================================================================
# Tasks 2 and 3: index, query parsing, retrieval
# =============================================================================


def tokeniser_of(cfg: Config) -> Callable[[str], list[str]]:
    return hw4._tokens if cfg.tokeniser == "stopword" else (lambda text: text.lower().split())


def build_index_cfg(cfg: Config, chunks: list[Document]) -> dict:
    """hw4.build_index with the tokeniser and b switchable."""
    tokenise = tokeniser_of(cfg)
    chunks = list(chunks)
    tokenised = [tokenise(c.page_content) or ["__empty__"] for c in chunks]
    bm25 = BM25Okapi(tokenised, b=cfg.b)

    document_frequency: dict[str, int] = {}
    for tokens in tokenised:
        for term in {hw4._stem(t) for t in tokens}:
            document_frequency[term] = document_frequency.get(term, 0) + 1
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


NO_PARSE = {"companies": [], "fiscal_year": None, "doc_type": None}

# ---- V11: LLM reranking

RERANK_DEPTH = 20       # candidates shown to the reranker before it picks 5
RERANK_CHARS = 1200     # characters of each candidate it sees: nearly a whole chunk, so the
                        # figure it has to judge is not cut off before it can read it

RERANK_PROMPT = """You rank passages for a question about company reports. A passage is useful when
it is about the company and the fiscal year the question asks about and contains the figure or the
name the question asks for.

Question: {question}

Passages:
{passages}

Reply with the numbers of the {n} most useful passages, most useful first, as a comma-separated
list and nothing else. Example: 3, 1, 12, 7, 2"""


def _rerank(question: str, candidates: list[Document]) -> list[Document]:
    """Reorder candidates with make_llm(). Falls back to the BM25 order if the reply is unusable."""
    if len(candidates) <= MAX_RETRIEVED:
        return candidates[:MAX_RETRIEVED]
    lines = []
    for i, chunk in enumerate(candidates, 1):
        head, _, body = chunk.page_content.partition("\n")
        lines.append(f"[{i}] {head}\n{body[:RERANK_CHARS]}")
    prompt = RERANK_PROMPT.format(question=question, passages="\n\n".join(lines), n=MAX_RETRIEVED)
    try:
        reply = str(hw4.make_llm(temperature=0.0).invoke(prompt).content)
    except Exception:
        traceback.print_exc()
        return candidates[:MAX_RETRIEVED]

    picked: list[Document] = []
    seen: set[int] = set()
    for number in (int(n) for n in re.findall(r"\d+", reply)):
        if 1 <= number <= len(candidates) and number not in seen:
            seen.add(number)
            picked.append(candidates[number - 1])
        if len(picked) >= MAX_RETRIEVED:
            break
    for i, chunk in enumerate(candidates, 1):        # a short reply must not shrink the result
        if len(picked) >= MAX_RETRIEVED:
            break
        if i not in seen:
            seen.add(i)
            picked.append(chunk)
    return picked[:MAX_RETRIEVED]


def parse_of(cfg: Config) -> Callable[[str], dict]:
    return hw4.parse_query if cfg.parse else (lambda question: dict(NO_PARSE))


def retrieve_cfg(cfg: Config, index: dict, question: str) -> list[Document]:
    """hw4.retrieve with every later component switchable off."""
    if not index or not index["chunks"]:
        return []
    tokenise = tokeniser_of(cfg)

    query_tokens = tokenise(question)
    if cfg.aliases:
        expanded = list(query_tokens)
        for term in query_tokens:
            for alias in hw4.ALIASES.get(term, ()):
                expanded.extend(tokenise(alias))
        query_tokens = expanded
    scores = index["bm25"].get_scores(query_tokens)
    rank = {id(chunk): score for chunk, score in zip(index["chunks"], scores)}

    # The reranker needs something to reorder: it sees RERANK_DEPTH candidates and returns 5.
    depth = RERANK_DEPTH if cfg.rerank else MAX_RETRIEVED

    def finish(candidates: list[Document]) -> list[Document]:
        if cfg.abstain and hw4._attribute_missing(question, index):
            return []
        return _rerank(question, candidates) if cfg.rerank else candidates[:MAX_RETRIEVED]

    # ---- no query parsing: one BM25 ranking over the whole corpus
    if not cfg.parse:
        ordered = sorted(index["chunks"], key=lambda c: rank.get(id(c), 0.0), reverse=True)
        return finish(ordered[:depth])

    parsed = parse_of(cfg)(question)
    wanted = [company_key(c) for c in parsed["companies"]]
    known = [key for key in wanted if key in index["by_company"]]
    if not known:
        return []                                # no company of the question is in the corpus
    year, doc_type = parsed["fiscal_year"], parsed["doc_type"]

    if doc_type == "annual_report" and year is not None:
        covered: set[int] = set()
        for key in known:
            covered |= {y for y in index["years"].get(key, set()) if isinstance(y, int)}
        if covered and (year > max(covered) or year < min(covered) - 1):
            return []                            # the year is not covered by any document

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

    same_year = cfg.same_year and doc_type == "annual_report" and year is not None

    def order(chunk: Document):
        matches = (chunk.metadata or {}).get("fiscal_year") == year if same_year else True
        return (matches, rank.get(id(chunk), 0.0))

    picked: list[Document] = []
    seen: set[str] = set()
    if cfg.per_company:
        queues: list[list[Document]] = []
        for pool in pools:
            if cfg.sections:
                fresh, repeated, used = [], [], set()
                for chunk in sorted(pool, key=order, reverse=True):
                    section = str((chunk.metadata or {}).get("section", "")).split(" > ")[0]
                    (repeated if section in used else fresh).append(chunk)
                    used.add(section)
                queues.append(fresh + repeated)
            else:
                queues.append(sorted(pool, key=order, reverse=True))
        depth_index = 0
        while len(picked) < depth and any(depth_index < len(q) for q in queues):
            for queue in queues:
                if len(picked) >= depth:
                    break
                if depth_index < len(queue) and queue[depth_index].page_content not in seen:
                    seen.add(queue[depth_index].page_content)
                    picked.append(queue[depth_index])
            depth_index += 1
    else:
        for chunk in sorted([c for pool in pools for c in pool], key=order, reverse=True):
            if chunk.page_content in seen:
                continue
            seen.add(chunk.page_content)
            picked.append(chunk)
            if len(picked) >= depth:
                break

    return finish(picked)


# =============================================================================
# Scoring one configuration with the provided graders
# =============================================================================


def evaluate(cfg: Config, data: dict, answers: str) -> tuple[dict, list[dict]]:
    """Run one configuration end to end and return its metrics (and its per-question records)."""
    start = time.perf_counter()
    corpus, questions = data["corpus"], data["questions"]
    ground_truth, docs_meta, tables = data["ground_truth"], data["docs_meta"], data["tables"]

    hw4.CORPUS_COMPANIES.clear()                 # parse_query learns the corpus from chunking
    chunk_fn = chunker_of(cfg)
    chunks_by_file: dict[str, list[Document]] = {}
    for file_name, html in corpus.items():
        chunks_by_file[file_name] = list(chunk_fn(file_name, html))

    metrics: dict[str, Any] = {"variant": cfg.name, "change": cfg.change}

    # Task 1.1: only the configurations that have a table_to_text
    if tables and cfg.table:
        scores = [table_score(hw4.table_to_text(t["html"]), t) for t in tables]
        metrics["t1_1"] = sum(scores) / len(scores)
    elif tables:
        metrics["t1_1"] = 0.0
    else:
        metrics["t1_1"] = ""

    # Task 1.2
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), None, m, f) for f, m in docs_meta.items()}
        metrics["t1_2"] = sum(not p for p in problems.values()) / len(problems)

    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    metrics["chunks"] = len(all_chunks)
    by_text: dict[str, list[Document]] = {}
    for chunk in all_chunks:
        by_text.setdefault(chunk.page_content, []).append(chunk)

    # Task 2.2
    parse_fn = parse_of(cfg)
    parsed = {qid: parse_fn(question) for qid, question in questions.items()}

    # Tasks 2.1, 2.3, 3
    index = build_index_cfg(cfg, all_chunks)
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve_cfg(cfg, index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in result or []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:                  # retrieve() invented a chunk: Task 2.1 = 0
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    metrics["index_seconds"] = round(time.perf_counter() - start, 1)

    # Task 4: the fixed generator, optionally
    do_answers = answers == "all" or (answers == "final" and cfg.shipped)
    if do_answers:
        with ThreadPoolExecutor(max_workers=8) as pool:
            listed = pool.map(lambda qid: generate(questions[qid], retrieved[qid]), list(questions))
            answers_by_id = dict(zip(questions, listed))
    else:
        answers_by_id = {qid: "" for qid in questions}

    records = []
    for qid, question in questions.items():
        record = {"id": qid, "category": "", "parsed": json.dumps(parsed[qid], default=str),
                  "parse_score": "", "retrieved": ";".join(sorted({c.metadata.get("source", "")
                                                                    for c in retrieved[qid]})),
                  "valid": valid[qid], "retrieval_hit": "", "answer": answers_by_id[qid],
                  "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            record["category"] = gold["category"]
            record["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            record["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            if do_answers:
                record["correct"] = answer_correct(answers_by_id[qid], gold)
                if not record["correct"]:
                    record["expected"] = gold["answer"]
        records.append(record)

    if ground_truth:
        graded = [r for r in records if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        metrics["parse"] = sum(r["parse_score"] for r in graded) / len(graded)
        metrics["non_empty"] = (sum(r["valid"] and bool(r["retrieved"]) for r in answerable)
                                / max(len(answerable), 1))
        metrics["recall"] = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        metrics["kept"], metrics["rejected"] = kept, rejected
        points = {"1.1": POINTS["1.1"] * float(metrics["t1_1"] or 0.0),
                  "1.2": POINTS["1.2"] * float(metrics.get("t1_2", 0.0)),
                  "2.1": POINTS["2.1"] * metrics["non_empty"],
                  "2.2": POINTS["2.2"] * metrics["parse"],
                  "2.3": POINTS["2.3"] * clip((metrics["recall"] - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO)),
                  "3": POINTS["3"] * clip(kept + rejected - 1)}
        if do_answers:
            metrics["accuracy"] = sum(r["correct"] for r in graded) / len(graded)
            points["4"] = POINTS["4"] * metrics["accuracy"]
        metrics["points"] = sum(points.values())

    metrics["seconds"] = round(time.perf_counter() - start, 1)
    return metrics, records


def verify(data: dict) -> bool:
    """The shipped row must be the shipped code: compare chunks and retrievals with hw4 itself."""
    print("\nVerifying the last variant against the shipped functions in hw4.py ...")
    ok = True
    hw4.CORPUS_COMPANIES.clear()
    chunks = {f: hw4.chunk_document(f, html) for f, html in data["corpus"].items()}
    for f, html in data["corpus"].items():
        mine = chunk_structured(f, html, True)
        theirs = chunks[f]
        if [(c.page_content, c.metadata) for c in mine] != [(c.page_content, c.metadata) for c in theirs]:
            ok = False
            print(f"  chunk_document differs on {f}")
            break
    all_chunks = [c for f in sorted(chunks) for c in chunks[f]]
    index = hw4.build_index(all_chunks)
    for qid, question in data["questions"].items():
        mine = [c.page_content for c in retrieve_cfg(SHIPPED, index, question)]
        theirs = [c.page_content for c in hw4.retrieve(index, question)]
        if mine != theirs:
            ok = False
            print(f"  retrieve differs on {qid}")
            break
    print("  PASS: the last variant is byte-identical to hw4.py" if ok
          else "  FAIL: fix the harness before trusting the table")
    return ok


# =============================================================================
# Output
# =============================================================================

COLUMNS = ["variant", "chunks", "t1_1", "t1_2", "parse", "non_empty", "recall", "kept",
           "rejected", "accuracy", "points", "llm_calls", "seconds"]
HEADERS = ["Variant", "Chunks", "T1.1", "T1.2", "T2.2 parse", "T2.1 non-empty", "T2.3 recall@5",
           "T3 kept", "T3 rejected", "T4 accuracy", "Points", "LLM calls", "Time (s)"]


def _fmt(key: str, value: Any) -> str:
    if value in ("", None):
        return "-"
    if key in ("chunks", "seconds", "llm_calls"):
        return f"{value:g}" if isinstance(value, (int, float)) else str(value)
    if key == "points":
        return f"{value:.2f}"
    return f"{100 * value:.1f}%"


def write_outputs(rows: list[dict], out: str) -> None:
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS + ["change"],
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    lines = ["| " + " | ".join(HEADERS) + " |",
             "|" + "|".join("---" for _ in HEADERS) + "|"]
    for row in rows:
        lines.append("| " + " | ".join([row["variant"]] + [_fmt(k, row.get(k)) for k in COLUMNS[1:]]) + " |")
    table = "\n".join(lines)

    # The report needs the "what changed" column too: one row per variant, one change per row.
    changes = ["", "### What each variant changes", "",
               "| Variant | Change relative to the row above |", "|---|---|"]
    for row in rows:
        change = str(row.get("change", "")).replace("|", "\\|")
        changes.append(f"| {row['variant']} | {change} |")

    with open(f"{out}_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n" + "\n".join(changes) + "\n")
    print("\n" + table + "\n" + "\n".join(changes))


def write_errors(by_variant: dict[str, list[dict]], out: str) -> None:
    """Every question a variant got wrong or could not retrieve, for the error analysis."""
    rows = []
    for variant, records in by_variant.items():
        for record in records:
            if record["correct"] is False:
                why = "wrong answer"
            elif record["retrieval_hit"] is False:
                why = "retrieval miss"
            else:
                continue
            rows.append({"variant": variant, "why": why, **record})
    if not rows:
        print(f"no errors on {out}: every variant got every answerable question right")
        return
    with open(f"{out}_errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["variant"]] = counts.get(row["variant"], 0) + 1
    print(f"{len(rows)} failures written to {out}_errors.csv "
          f"({', '.join(f'{v}:{n}' for v, n in counts.items())})")


def rerank_side(data: dict, out: str) -> None:
    """Does an LLM reranker replace metadata filtering? Measured on V4, before any filtering.

    Deliberately generator-free: the question is whether the evidence reaches the top 5, which is a
    recall question, and this keeps the experiment at ~120 calls instead of ~300.
    """
    global RERANK_DEPTH
    base = next(c for c in CONFIGS if c.name == "V4")
    rows = []
    for depth in (0, 20, 40):
        RERANK_DEPTH = depth or 20
        cfg = replace(base, name=("V4, no rerank" if depth == 0 else f"V4 + rerank top-{depth}"),
                      change="", rerank=depth > 0)
        before = hw4.USAGE.llm_calls
        metrics, _ = evaluate(cfg, data, "none")
        metrics["llm_calls"] = hw4.USAGE.llm_calls - before
        rows.append(metrics)
        print(f"  {metrics['variant']:22s} recall={metrics['recall']:6.1%}  "
              f"kept={metrics['kept']:4.0%}  rejected={metrics['rejected']:5.0%}  "
              f"calls={metrics['llm_calls']:3d}  {metrics['seconds']:5.1f}s")

    columns = ["variant", "recall", "kept", "rejected", "llm_calls", "seconds"]
    with open(f"{out}_rerank.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                         for r in rows)

    lines = ["| Configuration | recall@5 | kept | rejected | LLM calls | Time (s) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {_fmt('recall', r['recall'])} | {_fmt('kept', r['kept'])} "
                     f"| {_fmt('rejected', r['rejected'])} | {r['llm_calls']} | {r['seconds']} |")
    table = "\n".join(lines)
    with open(f"{out}_rerank.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5 ablation study")
    parser.add_argument("--data-folder", default=str(ROOT / "public_test"))
    parser.add_argument("--out", default=str(HERE / "ablation"),
                        help="path prefix for the artifacts (default: ablation/ablation)")
    parser.add_argument("--answers", choices=("none", "final", "all"), default="all",
                        help="call the generator never / only for the shipped config / for every row")
    parser.add_argument("--variants", default="", help="comma-separated subset, e.g. B0,V1,V2")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--render-only", action="store_true",
                        help="rewrite {out}_table.md from an existing {out}.csv, no re-run, no API cost")
    parser.add_argument("--rerank-side", action="store_true",
                        help="run only the separate LLM-reranking experiment on V4")
    args = parser.parse_args()

    if args.render_only:
        with open(f"{args.out}.csv", newline="", encoding="utf-8") as f:
            rows = [{k: (float(v) if k not in ("variant", "change") and v not in ("", None) else v)
                     for k, v in row.items()} for row in csv.DictReader(f)]
        write_outputs(rows, args.out)
        return

    folder = Path(args.data_folder)
    data = {"corpus": load_corpus(folder / "corpus"),
            "questions": load_json(folder / "questions.json"),
            "ground_truth": load_json(folder / "ground_truth.json"),
            "docs_meta": load_json(folder / "docs.json"),
            "tables": load_json(folder / "tables.json")}

    wanted = {name.strip() for name in args.variants.split(",") if name.strip()}
    configs = [c for c in CONFIGS if not wanted or c.name in wanted]

    if args.rerank_side:
        print("LLM reranking on its own, on top of V4 (before any metadata filtering):")
        rerank_side(data, args.out)
        return

    print(f"{len(data['corpus'])} documents, {len(data['questions'])} questions, "
          f"{len(configs)} configurations")

    rows, by_variant = [], {}
    for cfg in configs:
        print(f"\n=== {cfg.name}: {cfg.change}")
        calls_before = hw4.USAGE.llm_calls
        metrics, records = evaluate(cfg, data, args.answers)
        metrics["llm_calls"] = hw4.USAGE.llm_calls - calls_before
        rows.append(metrics)
        by_variant[cfg.name] = records
        print("  " + " | ".join(f"{k}={_fmt(k, metrics.get(k))}" for k in COLUMNS[1:]))

    write_outputs(rows, args.out)
    if args.answers != "none":
        write_errors(by_variant, args.out)
    if not args.no_verify:
        verify(data)


if __name__ == "__main__":
    main()
