from sentence_transformers import CrossEncoder
def init_rerank_model():
    cache_path = 'models/ms-marco-MiniLM-L6-v2'
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2",cache_dir=cache_path)
    return model




