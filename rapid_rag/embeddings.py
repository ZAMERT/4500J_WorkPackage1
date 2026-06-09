DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
