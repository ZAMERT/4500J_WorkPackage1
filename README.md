# RAPID RAG — ABB Robot Code Generation with Retrieval-Augmented Generation

## Overview
RAG pipeline that retrieves relevant ABB RAPID manual knowledge and generates
valid RAPID code via an LLM. Two retrieval modes are supported: a baseline
single-collection vector search, and an improved hybrid search that combines
segmented section retrieval with instruction-level canonical knowledge cards.

---

## Pipeline

### Step 1 — Extract Documentation

The source manual is packed as a `.rspak` file (ZIP format). Unzip it, then
extract the CHM help files into HTML using `7z`.

```bash
unzip ABB.RobotWareDoc.OmniCore-7.10.rspak -d rapid_docs

7z x rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation/en/3HAC065038_TRM_RAPID_RW_7-en.chm \
    -o rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation/en/rapid_manual_html
```

This produces 669 HTML files, one per RAPID instruction or data type.

---

### Step 2 — Build the Index

#### Chunking Logic

Each HTML file is parsed using BeautifulSoup. The ABB manual uses a consistent
`<span class="blocklabel">` structure to divide each instruction page into named
sections (e.g. `Usage`, `Arguments`, `Basic examples`, `Syntax`). The parser
extracts each section as an independent text block.

Long sections are then split into overlapping chunks:
- Maximum **1800 characters** per chunk
- **250 character overlap** between adjacent chunks
- Split boundaries follow natural breaks: newline → period → semicolon (in priority order)
- Chunks shorter than 50 characters are discarded

Each chunk is stored with metadata: instruction title, section name, source file,
language, and document version.

#### Option A — Original (single collection)

All sections from all instruction pages are indexed into one flat collection,
regardless of section type.

```bash
python build_rapid_index.py \
    --manual-root rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation \
    --languages en
```

Output: `rapid_chroma_db/` — collection `rapid_manual`

#### Option B — Segmented + canonical cards (recommended)

Before indexing, each section is classified by its `blocklabel` type and routed
to a dedicated collection. This allows the retriever to query the right segment
depending on what information is needed.

In addition, each RAPID instruction, function, or data type page is aggregated
into a canonical JSON card. A card keeps the API-level facts together:

```json
{
  "instruction": "MoveL",
  "type": "instruction",
  "syntax": "...",
  "arguments": ["..."],
  "required_context": ["robtarget", "tooldata", "wobjdata"],
  "examples": ["..."],
  "related": ["MoveJ", "MoveAbsJ"],
  "common_errors": ["..."]
}
```

This is more useful for code generation than returning disconnected paragraph
chunks, because syntax, arguments, examples, and error notes from the same API
page remain bound together.

```bash
python build_rapid_index_segmented.py \
    --manual-root rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation \
    --languages en
```

Output: `rapid_chroma_db_segmented/` — four collections:

| Collection | Blocklabel sections indexed |
|---|---|
| `rapid_definitions` | Usage, Arguments, Program execution, Error handling, Return value, Description, Limitations |
| `rapid_syntax` | Syntax, Predefined data |
| `rapid_examples` | Basic examples, Basic example, More examples, Examples |
| `rapid_cards` | Full instruction/data-type cards aggregated from each HTML page |

Sections such as `Related information` are skipped entirely as they contain no
actionable content for chunk retrieval. Related instruction names are still
captured inside cards when present.

To build only the three legacy segmented collections, pass `--no-cards`.

---

### Step 3 — Generate RAPID Code

#### Original pipeline (single collection, vector only)

```bash
python generate_rapid.py "move to pick position and close gripper"
```

#### Hybrid pipeline (segmented vector + BM25 keyword + cards)

```bash
python generate_rapid_hybrid.py "move to pick position and close gripper"
```

By default, the hybrid retriever:

1. Retrieves candidate sections from `rapid_definitions`, `rapid_syntax`, and
   `rapid_examples`.
2. Retrieves keyword candidates from BM25 over the local HTML manual.
3. Merges candidates with Reciprocal Rank Fusion.
4. Uses the winning source files to pull complete cards from `rapid_cards`.
5. Sends card context to the LLM, with chunk fallback if the card collection is
   missing.

Optional arguments:

```
--vector-weight   weight of vector retrieval in RRF merge (default: 1.0)
--bm25-weight     weight of BM25 retrieval in RRF merge (default: 1.0)
--no-cards        disable card expansion and return chunk context only
--top-k           number of context items sent to LLM (default: 6)
--candidate-k     candidates retrieved per retriever before merging (default: 12)
--language        preferred manual language (default: en)
```

---

## Architecture

```
User query
    │
    ├─► SegmentedRetriever (vector search across 3 collections)
    │       bge-m3 embedding → ChromaDB cosine similarity → top-12 candidates
    │
    ├─► BM25Retriever (keyword search, built from HTML at startup, no DB required)
    │       exact term matching → top-12 candidates
    │
    └─► RRF Fusion (Reciprocal Rank Fusion, rank-based merge)
            │
            ├─► candidate source files
            │
            └─► rapid_cards expansion → top-6 cards/chunks → DeepSeek LLM → RAPID module output
```

RRF score formula: `score(d) = Σ weight / (60 + rank(d))` across all retrievers.
Equal weights by default; `--bm25-weight` can be raised for queries containing
exact RAPID instruction names (e.g. `MoveL`, `WaitDI`, `SetDO`).

---

## File Structure

```
build_rapid_index.py            Build original single-collection vector index
build_rapid_index_segmented.py  Build segmented vector index plus rapid_cards
generate_rapid.py               Generate RAPID code using original pipeline
generate_rapid_hybrid.py        Generate RAPID code using hybrid card retrieval

rapid_rag/
  loaders.py          Discover HTML manual files from extracted CHM directories
  parser.py           Parse ABB HTML manual structure (blocklabel sections)
  chunker.py          Text splitting, stable ID generation, segment classification
  cards.py            Build canonical instruction/data-type cards from HTML pages
  embeddings.py       Load BAAI/bge-m3 multilingual embedding model
  vectorstore.py      ChromaDB collection creation and batch write
  retriever.py        RapidRetriever (single collection) + SegmentedRetriever (3 collections)
  bm25_retriever.py   BM25 keyword retriever built on-the-fly from HTML files
  hybrid_retriever.py RRF fusion plus card expansion from rapid_cards
  prompts.py          LLM prompt construction from retrieved context
```

---

## Environment Setup

```bash
conda create -n abb python=3.11 -y
conda activate abb
pip install sentence-transformers chromadb beautifulsoup4 lxml python-dotenv openai rank-bm25
```

Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your_key_here
```
