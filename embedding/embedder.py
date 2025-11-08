'''Text embedding placeholder'''
import torch
from sentence_transformers import SentenceTransformer
from config.settings import settings
import numpy as np

class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=device)

    def embed_text(self, texts):
        return np.array(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False),dtype=np.float32)
    
    def embed_query(self, query):
        return np.array(self.model.encode([query], convert_to_numpy=True, show_progress_bar=False),dtype=np.float32)[0]  # Load environment variables