| Variant | Chunks | T1.1 | T1.2 | T2.2 parse | T2.1 non-empty | T2.3 recall@5 | T3 kept | T3 rejected | T4 accuracy | Points | LLM calls | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 211 | 0.0% | 100.0% | 0.6% | 100.0% | 26.1% | 100.0% | 0.0% | 38.3% | 19.72 | 0 | 7.5 |
| V1 | 317 | 100.0% | 100.0% | 0.6% | 100.0% | 23.9% | 100.0% | 0.0% | 36.7% | 27.39 | 0 | 8 |
| V2 | 406 | 100.0% | 100.0% | 0.6% | 100.0% | 67.4% | 100.0% | 0.0% | 71.7% | 40.91 | 0 | 10.6 |
| V3 | 406 | 100.0% | 100.0% | 0.6% | 100.0% | 84.8% | 100.0% | 0.0% | 83.3% | 49.77 | 0 | 9.3 |
| V4 | 406 | 100.0% | 100.0% | 0.6% | 100.0% | 87.0% | 100.0% | 0.0% | 83.3% | 50.58 | 0 | 8.4 |
| V5 | 406 | 100.0% | 100.0% | 100.0% | 100.0% | 91.3% | 100.0% | 71.4% | 95.0% | 71.14 | 0 | 7.8 |
| V6 | 406 | 100.0% | 100.0% | 100.0% | 100.0% | 91.3% | 100.0% | 71.4% | 95.0% | 71.14 | 0 | 7.3 |
| V7 | 406 | 100.0% | 100.0% | 100.0% | 100.0% | 97.8% | 100.0% | 71.4% | 98.3% | 71.81 | 0 | 7.2 |
| V8 | 406 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 71.4% | 100.0% | 72.14 | 0 | 7 |
| V9 | 406 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 71.4% | 100.0% | 72.14 | 0 | 7 |
| V10 | 406 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 75.00 | 0 | 6.5 |

### What each variant changes

| Variant | Change relative to the row above |
|---|---|
| B0 | naive RAG, Tutorial 7 Section 2: get_text(), RecursiveCharacterTextSplitter, lower-cased whitespace tokens, BM25 top-5 over the whole corpus, no query parsing, never [] |
| V1 | + structured parsing: walk h1/h2/h3/p/ul/table as blocks and convert every <table> with table_to_text (row label + column header kept), tables never glued to prose |
| V2 | + contextual chunk header: prepend 'company \| FY \| doc_type \| section' to every chunk |
| V3 | + better tokeniser: lower-case, strip stopwords and question scaffolding, light stemming instead of raw whitespace tokens |
| V4 | + BM25 b=0.3 instead of the default 0.75 (less length normalisation, so a short risk-factor paragraph stops outranking the income statement) |
| V5 | + parse_query and metadata filtering: rank only the chunks of the asked company, document type and fiscal year |
| V6 | + query expansion: add the words the corpus uses for the asked attribute (sales -> revenue, headcount -> employees, ...) |
| V7 | + same-year preference: when the year is asked about, the report of that year is ranked before the other one (it names the CFO in office then) |
| V8 | + per-company retrieval: rank inside each company's pool and interleave the pools, so a comparison gets a chunk for each company instead of five for one |
| V9 | + section diversity: at most one chunk per top-level section in each company's queue |
| V10 | + abstention: return [] when the question asks for an attribute the corpus never mentions (lexical document-frequency check) -- the shipped configuration |
