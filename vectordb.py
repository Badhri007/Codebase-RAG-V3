from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from embeddings import get_embedding_function
from config import EMBEDDING_MODEL


class VectorDB:
    def __init__(self, path: str = "./chroma_data",
                 embedding_provider: str = "huggingface",
                 **embedding_kwargs):
        self.path = path
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        # embedding_kwargs e.g. model="all-MiniLM-L6-v2"
        #                    or model="jina-embeddings-v3", base_url="...", api_key="..."
        if "model" not in embedding_kwargs:
            embedding_kwargs["model"] = EMBEDDING_MODEL
        self.embedding_function = get_embedding_function(embedding_provider, **embedding_kwargs)
        print(f"  ✓ ChromaDB ready | provider={embedding_provider} model={self.embedding_function.name}")



    def index_batch(self, repo_name: str, data: List[Dict[str, Any]]):
        col_name = self._get_collection_name(repo_name)
        try:
            self.client.get_collection(col_name)
            self.client.delete_collection(col_name)
        except Exception:
            pass

        collection = self.client.create_collection(col_name)

        ids       = [d["id"]       for d in data]
        documents = [d["text"]     for d in data]
        metadatas = [d["metadata"] for d in data]

        for i in range(0, len(ids), 100):
            end = min(i + 100, len(ids))
            try:
                collection.add(ids=ids[i:end], documents=documents[i:end], metadatas=metadatas[i:end])
            except Exception as e:
                print(f"  ⚠️ Batch {i}-{end} error: {e}")
                for j in range(i, end):
                    try:
                        collection.add(ids=[ids[j]], documents=[documents[j]], metadatas=[metadatas[j]])
                    except Exception as e2:
                        print(f"    ⚠️ Skipping {ids[j]}: {e2}")

        print(f"  ✓ Indexed {len(ids)} entities")



    def search(self, repo_name: str, query: str,
               k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        col_name = self._get_collection_name(repo_name)
        try:
            col = self.client.get_collection(col_name)
        except Exception as e:
            print(f"  Collection not found: {e}")
            return []

        where = dict(filters) if filters else None
        try:
            results = col.query(query_texts=[query], n_results=k, where=where)
        except Exception as e:
            print(f"  ⚠️ Search error: {e}")
            return []

        out = []
        if results["ids"] and results["ids"][0]:
            for i, eid in enumerate(results["ids"][0]):
                dist = results["distances"][0][i] if results["distances"] else 1.0
                out.append({
                    "id":       eid,
                    "score":    max(0.0, 1.0 - dist / 2.0),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })
        return out



    def has_repo(self, repo_name: str) -> bool:
        try:
            col = self.client.get_collection(
                self._get_collection_name(repo_name),
            )
            return col.count() > 0
        except Exception:
            return False

    def list_repos(self) -> List[str]:
        cols  = self.client.list_collections()
        repos = sorted(c.name[:-9] for c in cols if c.name.endswith("_entities"))
        print(f"  [DEBUG] repos: {repos}")
        return repos

    def delete_repo(self, repo_name: str):
        try:
            self.client.delete_collection(self._get_collection_name(repo_name))
            print(f"  ✓ Deleted {repo_name}")
        except Exception as e:
            print(f"  ⚠️ {e}")

    def get_stats(self, repo_name: str) -> Dict[str, Any]:
        try:
            col = self.client.get_collection(
                self._get_collection_name(repo_name),
                embedding_function=self.embedding_function
            )
            return {"chunks": col.count(), "name": repo_name}
        except Exception:
            return {"chunks": 0, "name": repo_name}

    def _get_collection_name(self, repo_name: str) -> str:
        safe = repo_name.replace("/", "_").replace(".", "_").lower()[:50]
        if len(safe) < 3:
            safe += "_db"
        return f"{safe}_entities"
