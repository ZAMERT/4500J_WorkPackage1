# RAPID RAG — ABB Robot Code Generation with Retrieval-Augmented Generation

## Overview
RAG pipeline that retrieves relevant ABB RAPID manual sections and generates
valid RAPID code via an LLM. Two retrieval modes are supported: a baseline
single-collection vector search, and an improved hybrid search over three
semantically segmented collections.

---

## Pipeline

### Step 1 — Extract Documentation

The source manual is packed as a `.rspak` file (ZIP format). Unzip it, then
extract the CHM help files into HTML using `7z`.

```bash
unzip ABB.RobotWareDoc.OmniCore-7.10.rspak -d rapid_docs

7z x rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation/en/3HAC065038_TRM_RAPID_RW_7-en.chm \
    -orapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation/en/rapid_manual_html
```

This produces 669 HTML files, one per RAPID instruction or data type. The
extracted RobotWare documentation also includes `rapid_kernel` and
`rapid_overview`, which are indexed by default in the segmented pipeline.

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
- Split boundaries follow natural breaks: newline -> period -> semicolon (in priority order)
- Chunks shorter than 50 characters are discarded

Each chunk is stored with metadata: instruction title, section name, source file,
language, source manual directory, and document version.

#### Option A — Original (single collection)

All sections from all instruction pages are indexed into one flat collection,
regardless of section type.

```bash
python build_rapid_index.py \
    --manual-root rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation \
    --languages en
```

Output: `rapid_chroma_db/` — collection `rapid_manual`

#### Option B — Segmented (three collections, recommended)

Before indexing, each section is classified by its `blocklabel` type and routed
to a dedicated collection. This includes API reference pages, RAPID language
kernel pages, and RAPID overview/programming principle pages.

```bash
python build_rapid_index_segmented.py \
    --manual-root rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation \
    --languages en
```

Output: `rapid_chroma_db_segmented/` — three collections:

| Collection | Blocklabel sections indexed |
|---|---|
| `rapid_definitions` | Usage, Arguments, Program execution, Error handling, Return value, Description, Definition, Introduction, Programming principles, Parameters, Instructions, Data, General, Limitations |
| `rapid_syntax` | Syntax, Syntax rules, Predefined data |
| `rapid_examples` | Basic examples, Basic example, More examples, Examples, Example |

Sections such as `Related information`, `References`, and `About this manual`
are skipped entirely as they contain little actionable code generation content.

Default manual directories:

- `rapid_manual_html` — RAPID instruction/function/data type reference
- `rapid_kernel` — RAPID language core semantics
- `rapid_overview` — programming principles, motion concepts, coordinate systems, and execution model

---

### Step 3 — Generate RAPID Code

#### Original pipeline (single collection, vector only)

```bash
python generate_rapid.py "move to pick position and close gripper"
```

#### Hybrid pipeline (segmented vector + BM25 keyword, merged via RRF)

```bash
python generate_rapid_hybrid.py "move to pick position and close gripper"
```

Optional arguments:

```
--vector-weight   weight of vector retrieval in RRF merge (default: 1.0)
--bm25-weight     weight of BM25 retrieval in RRF merge (default: 1.0)
--top-k           number of chunks sent to LLM (default: 8)
--candidate-k     candidates retrieved per retriever before merging (default: 18)
--language        preferred manual language (default: en)
```

---

## Architecture

```
User query
    |
    |---> SegmentedRetriever (vector search across 3 collections)
    |       bge-m3 embedding -> ChromaDB cosine similarity -> top candidates
    |
    |---> BM25Retriever (keyword search, built from HTML at startup, no DB required)
    |       exact term matching -> top candidates
    |
    `---> RRF Fusion (Reciprocal Rank Fusion, rank-based merge)
            |
            `---> top chunks -> DeepSeek LLM -> RAPID module output
```

RRF score formula: `score(d) = sum(weight / (60 + rank(d)))` across all
retrievers. Equal weights by default; `--bm25-weight` can be raised for queries
containing exact RAPID instruction names (e.g. `MoveL`, `WaitDI`, `SetDO`).

---

## File Structure

```
build_rapid_index.py            Build original single-collection vector index
build_rapid_index_segmented.py  Build segmented three-collection vector index
generate_rapid.py               Generate RAPID code using original pipeline
generate_rapid_hybrid.py        Generate RAPID code using hybrid retrieval

rapid_rag/
  loaders.py          Discover HTML manual files from extracted CHM directories
  parser.py           Parse ABB HTML manual structure (blocklabel sections)
  chunker.py          Text splitting, stable ID generation, segment classification
  embeddings.py       Load BAAI/bge-m3 multilingual embedding model
  vectorstore.py      ChromaDB collection creation and batch write
  retriever.py        RapidRetriever (single collection) + SegmentedRetriever (3 collections)
  bm25_retriever.py   BM25 keyword retriever built on-the-fly from HTML files
  hybrid_retriever.py RRF fusion of vector and BM25 results
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
