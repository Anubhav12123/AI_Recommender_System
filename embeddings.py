from sentence_transformers import SentenceTransformer

def get_encoder(model_name: str):
    return SentenceTransformer(model_name)
