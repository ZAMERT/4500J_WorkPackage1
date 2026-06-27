# RAPID RAG - ABB Robot Code Generation with Hybrid Retrieval

## Overview

This project builds a Retrieval-Augmented Generation (RAG) pipeline for
generating ABB RAPID robot programs from natural-language requirements.

The current recommended workflow is:

1. Extract ABB RobotWare RAPID documentation from CHM/HTML files.
2. Build a segmented ChromaDB vector index with `BAAI/bge-m3`.
3. Retrieve relevant manual sections using hybrid retrieval:
   - vector search over segmented collections
   - BM25 keyword search over the extracted HTML manuals
   - Reciprocal Rank Fusion (RRF) to merge both result lists
4. Send the retrieved evidence to a DeepSeek-compatible OpenAI client to
   generate RAPID code.

The older single-collection vector pipeline is still kept as a baseline.

## Project Layout

```text
build_rapid_index.py
    - Build the baseline single-collection ChromaDB index.

build_rapid_index_segmented.py
    - Build the recommended segmented ChromaDB index.

generate_rapid.py
    - Generate RAPID code with the baseline vector-only retriever.

generate_rapid_hybrid.py
    - Generate RAPID code with segmented vector retrieval + BM25 + RRF.

rapid_rag/
    loaders.py
        - Locate extracted manual HTML directories.
    parser.py
        - Parse ABB HTML pages into blocklabel-based sections.
    chunker.py
        - Split sections into chunks, classify sections into retrieval segments, and generate stable document IDs.
    code_detector.py
        - Detect RAPID-like code blocks so chunking avoids splitting code when practical.
    embeddings.py
        - Load the SentenceTransformers embedding model.
    vectorstore.py
        - Create/open ChromaDB collections and batch-write embeddings.
    retriever.py
        - Baseline vector retriever and segmented vector retriever.
    bm25_retriever.py
        - Build an in-memory BM25 keyword index directly from extracted HTML.
    hybrid_retriever.py
        - Merge vector and BM25 results with Reciprocal Rank Fusion.
    reranker.py
        - Optional CrossEncoder reranking after hybrid retrieval.
    prompts.py
        - Build the RAPID generation prompt from retrieved manual context.
```

Generated/local data is intentionally ignored by git:

```text
rapid_docs/
rapid_chroma_db/
rapid_chroma_db_segmented/
.env
```

## Environment Setup

Recommended Python version: 3.11.

```bash
conda create -n abb python=3.11 -y
conda activate abb
pip install sentence-transformers chromadb beautifulsoup4 lxml python-dotenv openai rank-bm25
```

Create `.env` in this repo's directory:

```text
DEEPSEEK_API_KEY=your_key_here
```

The generator uses:

```text
model: deepseek-chat
base_url: https://api.deepseek.com
```

You can override the LLM model with `--model`.

## Documentation Extraction

The source ABB RobotWare documentation package is a `.rspak` file, which is a
ZIP archive. Extract it first:

```bash
unzip ABB.RobotWareDoc.OmniCore-7.10.rspak -d rapid_docs
```

Then extract the RAPID CHM help file to HTML:

```bash
7z x rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation/en/3HAC065038_TRM_RAPID_RW_7-en.chm \
  -orapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation/en/rapid_manual_html
```

By default, the index builder looks under:

```text
rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation
```

and tries these manual directories for each language:

```text
rapid_manual_html
rapid_kernel
rapid_overview
```

Missing directories are skipped with a warning.

## Build the Index

### Recommended: segmented index

```bash
python3 build_rapid_index_segmented.py \
  --manual-root rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation \
  --languages en
```

Default output:

```text
rapid_chroma_db_segmented/
```

The segmented pipeline creates three ChromaDB collections:

| Collection | Segment | Main content |
|---|---:|---|
| `rapid_definitions` | `s1` | Usage, arguments, descriptions, definitions, limitations, programming principles, and other explanatory sections |
| `rapid_syntax` | `s2` | Syntax, syntax rules, predefined data |
| `rapid_examples` | `s3` | Basic examples, more examples, type examples |

Sections such as `Related information`, `References`, and `About this manual`
are skipped because they are usually not useful for code generation.

Useful options:

```text
--manual-root       Documentation root containing language folders
--manual-dir        Index one extracted HTML directory directly; overrides manual-root/languages/doc-dirs
--languages         Language folders to index, default: en
--doc-dirs          Manual directory names to scan under each language
--db-dir            Output ChromaDB directory, default: rapid_chroma_db_segmented
--embedding-model   SentenceTransformers model, default: BAAI/bge-m3
--chunk-chars       Maximum characters per chunk, default: 1800
--overlap           Character overlap between chunks, default: 250
--batch-size        Embedding/write batch size, default: 64
--no-reset          Keep existing collections instead of deleting before indexing
```

### Baseline: single collection

```bash
python3 build_rapid_index.py \
  --manual-root rapid_docs/ABB.RobotWareDoc.OmniCore-7.10/Documentation \
  --languages en
```

Default output:

```text
rapid_chroma_db/
collection: rapid_manual
```

## Generate RAPID Code

### Recommended hybrid generator

```bash
python3 generate_rapid_hybrid.py "move from home to pPick, close doGrip, move to pPlace, open doGrip, return home"
```

If no task is passed, the script runs a built-in pick-and-place example.

The script prints two sections:

```text
===== Generated RAPID =====
...

===== Retrieved Sources =====
...
```

The retrieved source list is useful for debugging whether the generated code was
grounded in the correct manual pages and sections.

### Optional reranking

Enable CrossEncoder reranking after RRF fusion:

```bash
python3 generate_rapid_hybrid.py "use MoveL and WaitDI for a guarded pick motion" --rerank
```

Default reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Hybrid generator options

```text
--db-dir             Segmented ChromaDB directory, default: rapid_chroma_db_segmented
--manual-root        Documentation root used by BM25 manual loading
--manual-dir         Use one extracted manual directory for BM25 loading
--languages          Manual language folders to load for BM25, default: en
--doc-dirs           Manual directory names to load for BM25
--embedding-model    Must match the model used to build the vector index
--language           Preferred retrieval language, default: en
--fallback-language  Fallback language for vector retrieval, default: en
--top-k              Final chunks sent to the LLM, default: 8
--candidate-k        Candidates retrieved per retriever before fusion, default: 18
--vector-weight      RRF weight for vector retrieval, default: 1.0
--bm25-weight        RRF weight for BM25 retrieval, default: 1.0
--rerank             Enable CrossEncoder reranking
--rerank-model       CrossEncoder model name
--model              LLM model, default: deepseek-chat
```

For queries that include exact RAPID instruction names such as `MoveL`,
`WaitDI`, or `SetDO`, increasing `--bm25-weight` can make exact manual matches
more influential:

```bash
python3 generate_rapid_hybrid.py "generate code using WaitDI then MoveL" --bm25-weight 1.5
```

### Baseline vector-only generator

```bash
python3 generate_rapid.py "move to pick position and close gripper"
```

Useful options:

```text
--db-dir             Baseline ChromaDB directory, default: rapid_chroma_db
--collection         ChromaDB collection name, default: rapid_manual
--embedding-model    Must match the model used to build the index
--language           Preferred manual language, default: en
--fallback-language  Fallback manual language, default: en
--top-k              Final chunks sent to the LLM, default: 6
--candidate-k        Vector candidates retrieved before trimming, default: 12
--model              LLM model, default: deepseek-chat
```

## How the Hybrid Pipeline Works

```text
User requirement
    |
    |--> SegmentedRetriever
    |      BAAI/bge-m3 embedding
    |      ChromaDB cosine search over:
    |        - rapid_definitions
    |        - rapid_syntax
    |        - rapid_examples
    |
    |--> BM25Retriever
    |      Reads extracted HTML manuals at startup
    |      Uses keyword/token matching
    |
    |--> Reciprocal Rank Fusion
    |      Combines vector and BM25 rankings
    |      score = sum(weight / (60 + rank + 1))
    |
    |--> Optional CrossEncoder reranker
    |
    |--> Prompt construction
    |      Retrieved manual context + RAPID generation rules
    |
    |--> DeepSeek LLM
    |
    |--> RAPID module output
```

The generation prompt tells the model to:

- use retrieved manual context as the source of truth
- avoid hallucinating RAPID APIs, argument order, or data types
- avoid inventing coordinates, targets, tools, workobjects, payloads, or I/O names
- output a complete `MODULE ... ENDMODULE` with `PROC main() ... ENDPROC`
- output RAPID code only, without Markdown fences

## Chunking and Metadata

Each HTML manual page is parsed with BeautifulSoup. ABB documentation commonly
uses `span.blocklabel` headings such as `Usage`, `Arguments`, `Syntax`, and
`Basic examples`; these sections become independent text blocks.

Long sections are split into overlapping chunks:

```text
max chunk size: 1800 characters
overlap:        250 characters
minimum chunk:  100 characters for split sections
```

The splitter prefers natural boundaries in this order:

```text
newline -> ". " -> ";"
```

RAPID code-like blocks are detected so boundaries can be moved away from the
middle of code when practical.

Each chunk stores metadata:

```text
language
manual
title
section
file
path
section_instance
chunk_id
segment
doc_version
```

## Notes and Limitations

- The hybrid generator needs both the segmented vector DB and the extracted HTML
  manuals. The vector DB powers semantic retrieval; the HTML manuals power BM25.
- BM25 is built in memory every time `generate_rapid_hybrid.py` starts.
- The embedding model used at generation time must match the model used when the
  index was built.
- If the retrieved context does not support a required instruction or site
  symbol, the prompt asks the model to leave a RAPID `TODO` comment instead of
  fabricating data.
- Local manual files, ChromaDB indexes, `.env`, caches, and tests are ignored by
  `.gitignore` in this working copy.
