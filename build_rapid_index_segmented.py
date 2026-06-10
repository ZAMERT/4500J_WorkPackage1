import argparse

from rapid_rag.chunker import build_records_segmented
from rapid_rag.embeddings import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from rapid_rag.loaders import DEFAULT_DOC_DIRS, DEFAULT_MANUAL_ROOT, discover_manual_dirs
from rapid_rag.vectorstore import add_in_batches, create_collection

COLLECTION_NAMES = {
    "s1": "rapid_definitions",
    "s2": "rapid_syntax",
    "s3": "rapid_examples",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build segmented ABB RAPID RAG index (3 collections)")
    parser.add_argument("--manual-root", type=str, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--manual-dir", type=str, default=None)
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--doc-dirs", nargs="+", default=list(DEFAULT_DOC_DIRS))
    parser.add_argument("--db-dir", type=str, default="rapid_chroma_db_segmented", help="Output Chroma database directory")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-reset", action="store_true")
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
        raise SystemExit("No extracted HTML manual directories found.")

    segments = build_records_segmented(manuals, chunk_chars=args.chunk_chars, overlap=args.overlap)

    print(f"\nLoading embedding model: {args.embedding_model}")
    embed_model = load_embedding_model(args.embedding_model)

    for seg_key, (ids, docs, metadatas) in segments.items():
        collection_name = COLLECTION_NAMES[seg_key]
        print(f"\n[{collection_name}] {len(docs)} chunks")
        if not docs:
            print("  Skipping empty segment.")
            continue
        collection = create_collection(
            db_dir=args.db_dir,
            collection_name=collection_name,
            embedding_model=args.embedding_model,
            reset=not args.no_reset,
        )
        add_in_batches(collection, embed_model, ids, docs, metadatas, batch_size=args.batch_size)
        print(f"  Done indexing {collection_name}.")

    print("\nSegmented RAPID index built.")


if __name__ == "__main__":
    main()
