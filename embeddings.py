"""
Embedding providers for ChromaDB.

Usage:
    fn = get_embedding_function("huggingface", model="all-MiniLM-L6-v2")
    fn = get_embedding_function("jina", model="jina-embeddings-v3",
                                base_url="http://localhost:8080", api_key="...")

Adding a new provider:
    1. Write a class with  self.name  and  __call__(self, texts) -> list[list[float]]
    2. Add it to the REGISTRY dict at the bottom
"""

from typing import List




class HuggingFaceEmbedding:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.name   = model
        self._model = SentenceTransformer(model)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._model.encode(input, convert_to_numpy=True).tolist()



class JinaEmbedding:
    def __init__(self, model: str, base_url: str, api_key: str = None, batch_size: int = 32):
        import requests
        self.name        = model
        self._model      = model
        self._url        = base_url.rstrip("/") + "/embeddings"
        self._batch_size = batch_size
        self._requests   = requests
        self._headers    = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(input), self._batch_size):
            batch = input[i : i + self._batch_size]
            resp  = self._requests.post(
                self._url,
                json={"model": self._model, "input": batch},
                headers=self._headers
            )
            resp.raise_for_status()
            items = sorted(resp.json()["data"], key=lambda x: x["index"])
            results.extend(item["embedding"] for item in items)
        return results




REGISTRY = {
    "huggingface": HuggingFaceEmbedding,
    "jina":        JinaEmbedding,
}


def get_embedding_function(provider: str, **kwargs):
    """
    Get an embedding function by provider name.

    Args:
        provider:  "huggingface" | "jina"
        **kwargs:  passed straight to the provider class

    Examples:
        get_embedding_function("huggingface", model="all-MiniLM-L6-v2")
        get_embedding_function("jina", model="jina-embeddings-v3",
                               base_url="http://localhost:8080")
        get_embedding_function("jina", model="jina-embeddings-v3",
                               base_url="https://api.jina.ai/v1", api_key="jina_...")
    """
    provider = provider.lower()
    if provider not in REGISTRY:
        raise ValueError(f"Unknown embedding provider '{provider}'. "
                         f"Available: {list(REGISTRY)}")
    return REGISTRY[provider](**kwargs)
