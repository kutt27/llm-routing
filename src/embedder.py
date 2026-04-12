from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts, show_progress=False):
        """Converts a list of strings into a numpy array of embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=show_progress)
