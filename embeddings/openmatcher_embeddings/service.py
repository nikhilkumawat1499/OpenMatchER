from functools import lru_cache


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    @lru_cache(maxsize=1)
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def encode(self, values: list[str]) -> list[list[float]]:
        return self._model().encode(values, normalize_embeddings=True).tolist()

