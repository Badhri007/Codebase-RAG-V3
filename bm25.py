import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from config import BM25_DIR


class BM25Retriever:
    """BM25 keyword-based retriever with disk persistence."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[List[str]] = []
        self.doc_map: List[str] = []  # index → chunk_id
        self.chunk_index: Dict[str, object] = {}  # chunk_id → Chunk object

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase and extract words
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def index(self, chunks):
        """Build BM25 index from chunks."""
        print(f"  Building BM25 index from {len(chunks)} chunks...")

        self.documents = []
        self.doc_map = []
        self.chunk_index = {}

        for chunk in chunks:
            tokens = self._tokenize(chunk.embedding_text())
            self.documents.append(tokens)
            self.doc_map.append(chunk.id)
            self.chunk_index[chunk.id] = chunk

        # Build BM25 index
        self.bm25 = BM25Okapi(self.documents)

        print(f"  ✓ BM25 index built with {len(self.documents)} documents")

    def search(self, query: str, k: int = 20) -> List[Dict]:
        """
        Search using BM25 and return chunk information.

        Returns:
            List of dicts with chunk info and scores, similar to VectorDB format
        """
        if self.bm25 is None:
            print("  ⚠️ BM25 index not initialized, returning empty results")
            return []

        # Tokenize query
        tokens = self._tokenize(query)

        if not tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokens)

        # Rank and return top-k with chunk information
        ranked = sorted(
            zip(self.doc_map, scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Format results similar to VectorDB
        results = []
        for chunk_id, score in ranked:
            if score > 0:  # Only include non-zero scores
                chunk = self.chunk_index.get(chunk_id)
                if chunk:
                    results.append({
                        'id': chunk.id,
                        'name': chunk.name,
                        'type': chunk.type,
                        'file': chunk.file,
                        'start': chunk.start,
                        'end': chunk.end,
                        'language': chunk.language,
                        'signature': chunk.signature or '',
                        'score': float(score),
                    })

        return results

    def get_scores_only(self, query: str, k: int = 20) -> Dict[str, float]:
        """
        Search using BM25 and return only scores (for hybrid retrieval).

        Returns:
            Dict mapping chunk_id to score
        """
        if self.bm25 is None:
            return {}

        tokens = self._tokenize(query)
        if not tokens:
            return {}

        scores = self.bm25.get_scores(tokens)

        # Rank and return top-k
        ranked = sorted(
            zip(self.doc_map, scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        return {chunk_id: float(score) for chunk_id, score in ranked if score > 0}

    def save(self, repo_name: str):
        """Save BM25 index to disk."""
        bm25_dir = Path(BM25_DIR)
        bm25_dir.mkdir(parents=True, exist_ok=True)

        # Don't save the full chunk objects, just the data needed for searching
        data = {
            'documents': self.documents,
            'doc_map': self.doc_map,
            'bm25': self.bm25,
            # Store minimal chunk info for reconstruction
            'chunk_metadata': {
                chunk_id: {
                    'id': chunk.id,
                    'name': chunk.name,
                    'type': chunk.type,
                    'file': chunk.file,
                    'start': chunk.start,
                    'end': chunk.end,
                    'language': chunk.language,
                    'signature': chunk.signature,
                }
                for chunk_id, chunk in self.chunk_index.items()
            }
        }

        path = bm25_dir / f"{repo_name}.pkl"

        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  ✓ BM25 index saved to {path}")
        except Exception as e:
            print(f"  ⚠️ Failed to save BM25 index: {e}")

    def load(self, repo_name: str) -> bool:
        """Load BM25 index from disk."""
        path = Path(BM25_DIR) / f"{repo_name}.pkl"

        if not path.exists():
            print(f"  ⚠️ BM25 index not found for {repo_name}")
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.documents = data['documents']
            self.doc_map = data['doc_map']
            self.bm25 = data['bm25']

            # Reconstruct chunk_index from metadata
            # Note: We store simple dicts, not full Chunk objects
            self.chunk_index = data.get('chunk_metadata', {})

            print(f"  ✓ BM25 index loaded ({len(self.doc_map)} documents)")
            return True

        except Exception as e:
            print(f"  ⚠️ Failed to load BM25 index: {e}")
            return False

    def delete(self, repo_name: str):
        """Delete saved BM25 index."""
        path = Path(BM25_DIR) / f"{repo_name}.pkl"
        if path.exists():
            path.unlink()
            print(f"  ✓ Deleted BM25 index for {repo_name}")
