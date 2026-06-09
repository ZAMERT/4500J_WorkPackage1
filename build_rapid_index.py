import argparse

from rapid_rag.chunker import build_records
from rapid_rag.embeddings import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from rapid_rag.loaders import DEFAULT_DOC_DIRS, DEFAULT_MANUAL_ROOT, discover_manual_dirs
from rapid_rag.vectorstore import add_in_batches, create_collection


def parse_args():
    parser = argparse.ArgumentParser(description="Build a multilingual ABB RAPID RAG index")
    parser.add_argument(
        "--manual-root",
        type=str,
        default=DEFAULT_MANUAL_ROOT,
        help="Root Documentation directory containing language folders, e.g. Documentation/en",
    )
    parser.add_argument(
        "--manual-dir",
        type=str,
        default=None,
        help="Optional single extracted HTML manual directory. Overrides --manual-root/--languages.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Language folders to index, e.g. en zh-CN de. Only extracted HTML dirs are indexed.",
    )
    parser.add_argument(
        "--doc-dirs",
        nargs="+",
        default=list(DEFAULT_DOC_DIRS),
        help="Candidate extracted manual directory names inside each language folder.",
    )
    parser.add_argument("--db-dir", type=str, default="rapid_chroma_db", help="Output Chroma database directory")
    parser.add_argument("--collection", type=str, default="rapid_manual", help="Chroma collection name")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformers embedding model. Use a multilingual model for multilingual manuals.",
    )
    parser.add_argument("--chunk-chars", type=int, default=1800, help="Maximum characters per chunk")
    parser.add_argument("--overlap", type=int, default=250, help="Character overlap when splitting long sections")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/add batch size")
    parser.add_argument("--no-reset", action="store_true", help="Do not delete the existing collection before adding documents")
    return parser.parse_args()


def main():
    args = parse_args()
    manuals = discover_manual_dirs(
        manual_root=args.manual_root,
        languages=args.languages,
        doc_dirs=args.doc_dirs,
        manual_dir=args.manual_dir,
    )
    if not manuals:
        raise SystemExit("No extracted HTML manual directories found. Extract CHM files first or pass --manual-dir.")

    ids, docs, metadatas = build_records(manuals, chunk_chars=args.chunk_chars, overlap=args.overlap)
    print(f"Total chunks: {len(docs)}")
    if not docs:
        raise SystemExit("No valid chunks were produced.")

    collection = create_collection(
        db_dir=args.db_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        reset=not args.no_reset,
    )

    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = load_embedding_model(args.embedding_model)
    add_in_batches(collection, embed_model, ids, docs, metadatas, batch_size=args.batch_size)
    print("Done. RAPID manual index built.")


if __name__ == "__main__":
    main()
