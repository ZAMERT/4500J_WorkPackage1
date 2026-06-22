import argparse
import time

from rapid_rag.cards import CARD_COLLECTION, build_card_embedding_document, build_instruction_cards
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
    parser = argparse.ArgumentParser(description="Build segmented ABB RAPID RAG index plus canonical instruction cards")
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
    parser.add_argument("--no-cards", action="store_true", help="Skip canonical instruction/data-type card collection")
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

    build_start = time.monotonic()
    print(f"Discovered {len(manuals)} manual directories.", flush=True)

    chunk_start = time.monotonic()
    print("Building segmented chunk records...", flush=True)
    segments = build_records_segmented(manuals, chunk_chars=args.chunk_chars, overlap=args.overlap)
    print(f"Segmented chunk records ready in {time.monotonic() - chunk_start:.1f}s", flush=True)

    if args.no_cards:
        print("Skipping rapid_cards because --no-cards was passed.", flush=True)
        card_records = ([], [], [])
    else:
        cards_start = time.monotonic()
        print("Building rapid_cards records...", flush=True)
        card_records = build_instruction_cards(manuals)
        print(f"rapid_cards records ready in {time.monotonic() - cards_start:.1f}s", flush=True)

    model_start = time.monotonic()
    print(f"\nLoading embedding model: {args.embedding_model}", flush=True)
    embed_model = load_embedding_model(args.embedding_model)
    print(f"Embedding model loaded in {time.monotonic() - model_start:.1f}s", flush=True)

    for seg_key, (ids, docs, metadatas) in segments.items():
        collection_name = COLLECTION_NAMES[seg_key]
        print(f"\n[{collection_name}] {len(docs)} chunks", flush=True)
        if not docs:
            print("  Skipping empty segment.", flush=True)
            continue
        collection = create_collection(
            db_dir=args.db_dir,
            collection_name=collection_name,
            embedding_model=args.embedding_model,
            reset=not args.no_reset,
        )
        add_in_batches(collection, embed_model, ids, docs, metadatas, batch_size=args.batch_size)
        print(f"  Done indexing {collection_name}.", flush=True)

    card_ids, card_docs, card_metadatas = card_records
    if card_docs:
        print(f"\n[{CARD_COLLECTION}] {len(card_docs)} canonical cards", flush=True)
        card_embedding_docs = [build_card_embedding_document(doc) for doc in card_docs]
        average_embedding_chars = sum(len(doc) for doc in card_embedding_docs) // len(card_embedding_docs)
        print(
            f"Using compact card embedding text "
            f"(avg {average_embedding_chars} chars, max {max(len(doc) for doc in card_embedding_docs)} chars)",
            flush=True,
        )
        card_collection = create_collection(
            db_dir=args.db_dir,
            collection_name=CARD_COLLECTION,
            embedding_model=args.embedding_model,
            reset=not args.no_reset,
        )
        add_in_batches(
            card_collection,
            embed_model,
            card_ids,
            card_docs,
            card_metadatas,
            batch_size=args.batch_size,
            embedding_docs=card_embedding_docs,
        )
        print(f"  Done indexing {CARD_COLLECTION}.", flush=True)

    print(f"\nSegmented RAPID index built in {time.monotonic() - build_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
