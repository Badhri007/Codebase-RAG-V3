"""Vector database with embeddings."""
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config import PERSIST_DIR, EMBEDDING_MODEL
from chunk import Chunk


class VectorDB:
    """ChromaDB wrapper for semantic search over code chunks."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._model is None:
            print(f"  Loading embedding model: {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()

    def _sanitize_name(self, name: str) -> str:
        """Sanitize collection name for ChromaDB."""
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        if not safe[0].isalpha():
            safe = 'r_' + safe
        return safe[:63] if len(safe) > 63 else (safe + '_db' if len(safe) < 3 else safe)

    def index(self, repo_name: str, chunks: List[Chunk]):
        """Index chunks into ChromaDB."""
        coll_name = self._sanitize_name(repo_name)

        # Delete existing
        try:
            self.client.delete_collection(coll_name)
        except:
            pass

        collection = self.client.create_collection(
            name=coll_name,
            metadata={"repo": repo_name}
        )

        # Prepare data
        ids, documents, metadatas = [], [], []

        for c in chunks:
            ids.append(c.id)
            # Use rich context for embedding
            documents.append(c.embedding_text())
            metadatas.append({
                'name': c.name,
                'type': c.type,
                'file': c.file,
                'start': c.start,
                'end': c.end,
                'language': c.language,
                'signature': c.signature or '',
                'has_docstring': bool(c.docstring),
            })

        # Generate embeddings
        print(f"  Embedding {len(chunks)} chunks...")
        embeddings = self.embed(documents)

        # Add in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end]
            )

        print(f"  Indexed {len(chunks)} chunks")

    def search(self, repo_name: str, query: str, n: int = 10,
               filter_type: str = None, filter_file: str = None) -> List[Dict]:
        """
        Search for relevant chunks.

        Args:
            repo_name: Repository to search
            query: Search query
            n: Number of results
            filter_type: Filter by chunk type (function, class, etc.)
            filter_file: Filter by file path

        Returns:
            List of chunk info dicts with scores
        """
        coll_name = self._sanitize_name(repo_name)

        try:
            collection = self.client.get_collection(coll_name)
        except:
            return []

        # Build filter
        where = {}
        if filter_type:
            where['type'] = filter_type
        if filter_file:
            where['file'] = filter_file

        # Query
        query_embedding = self.embed([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where if where else None,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        chunks = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i]

                chunks.append({
                    'id': chunk_id,
                    'name': meta['name'],
                    'type': meta['type'],
                    'file': meta['file'],
                    'start': meta['start'],
                    'end': meta['end'],
                    'language': meta['language'],
                    'signature': meta.get('signature', ''),
                    'score': 1 - dist,  # Convert distance to similarity
                })

        return chunks

    def list_repos(self) -> List[str]:
        """List all indexed repositories."""
        return [c.name for c in self.client.list_collections()]

    def has_repo(self, repo_name: str) -> bool:
        """Check if repository is indexed."""
        coll_name = self._sanitize_name(repo_name)
        return coll_name in self.list_repos()

    def delete_repo(self, repo_name: str):
        """Delete repository from index."""
        coll_name = self._sanitize_name(repo_name)
        try:
            self.client.delete_collection(coll_name)
        except:
            pass

    def get_stats(self, repo_name: str) -> Dict:
        """Get stats for a repository."""
        coll_name = self._sanitize_name(repo_name)
        try:
            collection = self.client.get_collection(coll_name)
            return {
                'repo': repo_name,
                'chunks': collection.count(),
            }
        except:
            return {'repo': repo_name, 'chunks': 0}
