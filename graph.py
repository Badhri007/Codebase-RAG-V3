"""Neo4j-based dependency graph with enhanced traversal."""
import os
from typing import List, Dict, Optional
from collections import deque
from chunk import Chunk
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j package not installed. Install with: pip install neo4j")





class Neo4jDependencyGraph:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize Neo4j connection."""

        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD

        self.driver = None
        self.chunks: Dict[str, Chunk] = {}
        self.current_repo: Optional[str] = None

        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                self.driver.verify_connectivity()
                print(f"  ✓ Neo4j connected at {self.uri}")
            except Exception as e:
                print(f"  ⚠️ Neo4j connection failed: {e}")
                print("  Falling back to in-memory cache")
                self.driver = None

    def build(self, chunks: List[Chunk], repo_name: str):
        """Build graph from chunks with hierarchical structure."""
        self.current_repo = repo_name
        self.chunks = {c.id: c for c in chunks}

        if not self.driver:
            print("  Using in-memory graph (Neo4j unavailable)")
            return

        print(f"  Building Neo4j graph for {repo_name}...")

        with self.driver.session() as session:
            # Clear existing repo data
            session.run(
                "MATCH (n) WHERE n.repo = $repo DETACH DELETE n",
                repo=repo_name
            )

            file_info = {}
            for chunk in chunks:
                if chunk.file not in file_info:
                    file_info[chunk.file] = {
                        'language': chunk.language,
                        'chunk_count': 0
                    }
                file_info[chunk.file]['chunk_count'] += 1

            # Create file nodes

            for file_path, info in file_info.items():
                session.run("""
                    MERGE (f:File {id: $id, repo: $repo})
                    SET f.path = $path,
                        f.language = $language,
                        f.name = $name,
                        f.chunk_count = $chunk_count
                """,
                    id=f"{repo_name}::{file_path}",
                    repo=repo_name,
                    path=file_path,
                    language=info['language'],
                    name=file_path.split('/')[-1],
                    chunk_count=info['chunk_count']
                )

            # Create chunk nodes in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                session.run("""
                    UNWIND $chunks as chunk
                    CREATE (c:Chunk {
                        id: chunk.id,
                        repo: chunk.repo,
                        name: chunk.name,
                        type: chunk.type,
                        file: chunk.file,
                        start: chunk.start,
                        end: chunk.end,
                        language: chunk.language,
                        signature: chunk.signature,
                        docstring: chunk.docstring,
                        code_length: chunk.code_length,
                        has_context: chunk.has_context
                    })
                """, chunks=[{
                    'id': c.id,
                    'repo': repo_name,
                    'name': c.name,
                    'type': c.type,
                    'file': c.file,
                    'start': c.start,
                    'end': c.end,
                    'language': c.language,
                    'signature': c.signature or '',
                    'docstring': (c.docstring or '')[:500],
                    'code_length': len(c.code),
                    'has_context': bool(c.situating_context)
                } for c in batch])

            # Create indexes
            session.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            session.run("CREATE INDEX chunk_repo IF NOT EXISTS FOR (c:Chunk) ON (c.repo)")
            session.run("CREATE INDEX chunk_name IF NOT EXISTS FOR (c:Chunk) ON (c.name)")
            session.run("CREATE INDEX file_id IF NOT EXISTS FOR (f:File) ON (f.id)")

            # Build relationships
            self._build_relationships(session, chunks, repo_name)

        print(f"  ✓ Neo4j graph built with {len(chunks)} chunks")

    def _build_relationships(self, session, chunks: List[Chunk], repo_name: str):
        """Build all relationship types."""

        # File → Chunk (CONTAINS)
        file_rels = []
        for c in chunks:
            file_rels.append({
                'file_id': f"{repo_name}::{c.file}",
                'chunk_id': c.id
            })

        if file_rels:
            session.run("""
                UNWIND $rels as rel
                MATCH (f:File {id: rel.file_id})
                MATCH (c:Chunk {id: rel.chunk_id})
                MERGE (f)-[:CONTAINS]->(c)
            """, rels=file_rels)

        # Parent → Child (HAS_MEMBER)
        parent_rels = []
        for c in chunks:
            if c.parent and c.parent in self.chunks:
                parent_rels.append({
                    'parent_id': c.parent,
                    'child_id': c.id
                })

        if parent_rels:
            session.run("""
                UNWIND $rels as rel
                MATCH (p:Chunk {id: rel.parent_id})
                MATCH (c:Chunk {id: rel.child_id})
                MERGE (p)-[:HAS_MEMBER]->(c)
            """, rels=parent_rels)

        # Call relationships
        call_rels = []
        for c in chunks:
            for call_name in c.calls:
                targets = self._resolve_calls(c, call_name)
                for target_id in targets:
                    if target_id != c.id:
                        call_rels.append({
                            'caller': c.id,
                            'callee': target_id,
                            'function_name': call_name
                        })

        if call_rels:
            session.run("""
                UNWIND $rels as rel
                MATCH (caller:Chunk {id: rel.caller})
                MATCH (callee:Chunk {id: rel.callee})
                MERGE (caller)-[r:CALLS]->(callee)
                SET r.function_name = rel.function_name
            """, rels=call_rels)

        # Import relationships
        import_rels = []
        for c in chunks:
            for imp in c.imports:
                targets = self._resolve_import(imp)
                for target_id in targets:
                    import_rels.append({
                        'importer': c.id,
                        'imported': target_id,
                        'import_name': imp
                    })

        if import_rels:
            session.run("""
                UNWIND $rels as rel
                MATCH (importer:Chunk {id: rel.importer})
                MATCH (imported:Chunk {id: rel.imported})
                MERGE (importer)-[r:IMPORTS]->(imported)
                SET r.import_name = rel.import_name
            """, rels=import_rels)

    def _resolve_calls(self, caller: Chunk, call_name: str) -> List[str]:
        """Resolve function call to chunk IDs."""
        candidates = [cid for cid, c in self.chunks.items() if c.name == call_name]

        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates

        # Prefer same file
        same_file = [c for c in candidates if self.chunks[c].file == caller.file]
        if same_file:
            return same_file[:3]

        # Prefer same parent
        if caller.parent:
            siblings = [c for c in candidates if self.chunks[c].parent == caller.parent]
            if siblings:
                return siblings

        return candidates

    def _resolve_import(self, import_name: str) -> List[str]:
        """Resolve import to chunk IDs."""
        parts = import_name.split('.')
        name = parts[-1]

        return [cid for cid, c in self.chunks.items()
                if c.name == name and c.type in ('class', 'interface', 'function')]

    def traverse(self, seed_ids: List[str], max_chunks: int = 50,
                 max_tokens: int = 8000) -> List[Chunk]:
        if not self.driver:
            return self._traverse_in_memory(seed_ids, max_chunks, max_tokens)

        with self.driver.session() as session:
          return self._traverse_graph(session, seed_ids, max_chunks, max_tokens)

    def _traverse_graph(self, session, seed_ids: List[str],
                       max_chunks: int, max_tokens: int) -> List[Chunk]:
        """Smart traversal with multi-criteria scoring."""

        query = """
        MATCH (seed:Chunk)
        WHERE seed.id IN $seed_ids AND seed.repo = $repo

        CALL {
            WITH seed
            MATCH (seed)-[:CALLS]->(hop1:Chunk)
            RETURN hop1 as chunk, 10 as score, 1 as depth

            UNION

            WITH seed
            MATCH (seed)<-[:CONTAINS]-(f:File)-[:CONTAINS]->(hop1:Chunk)
            WHERE seed.id <> hop1.id
            RETURN hop1 as chunk, 8 as score, 1 as depth

            UNION

            WITH seed
            MATCH (seed)<-[:HAS_MEMBER]-(parent:Chunk)
            RETURN parent as chunk, 9 as score, 1 as depth

            UNION

            WITH seed
            MATCH (seed)-[:CALLS*1..2]->(hop2:Chunk)
            RETURN hop2 as chunk, 5 as score, 2 as depth

            UNION

            WITH seed
            MATCH (hop2:Chunk)-[:CALLS]->(seed)
            RETURN hop2 as chunk, 6 as score, 1 as depth
        }

        WITH DISTINCT chunk, MAX(score) as final_score, MIN(depth) as min_depth
        WHERE chunk.code_length <= $max_chunk_size

        RETURN chunk.id as id, final_score, min_depth
        ORDER BY final_score DESC, min_depth ASC
        LIMIT $limit
        """

        result = session.run(
            query,
            seed_ids=seed_ids,
            repo=self.current_repo,
            max_chunk_size=max_tokens * 4,
            limit=max_chunks * 2
        )

        return self._collect_chunks(result, max_chunks, max_tokens)



    def _collect_chunks(self, result, max_chunks: int, max_tokens: int) -> List[Chunk]:
        """Helper to collect chunks from query result."""
        chunks = []
        tokens = 0

        for record in result:
            chunk_id = record['id']
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                chunk_tokens = len(chunk.code) // 4

                if tokens + chunk_tokens <= max_tokens and len(chunks) < max_chunks:
                    chunks.append(chunk)
                    tokens += chunk_tokens

        return chunks

    def _traverse_in_memory(self, seed_ids: List[str],
                           max_chunks: int, max_tokens: int) -> List[Chunk]:
        """Fallback in-memory BFS traversal."""
        visited = set()
        result = []
        tokens = 0
        queue = deque(seed_ids)

        while queue and len(result) < max_chunks:
            chunk_id = queue.popleft()

            if chunk_id in visited or chunk_id not in self.chunks:
                continue

            visited.add(chunk_id)
            chunk = self.chunks[chunk_id]
            chunk_tokens = len(chunk.code) // 4

            if tokens + chunk_tokens > max_tokens:
                continue

            result.append(chunk)
            tokens += chunk_tokens

            # Add related chunks
            for call in chunk.calls:
                for cid, c in self.chunks.items():
                    if c.name == call and cid not in visited:
                        queue.append(cid)

        return result


    def load(self, repo_name: str) -> bool:
        """Load chunks into memory cache."""
        self.current_repo = repo_name

        if not self.driver:
            return False

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk {repo: $repo})
                RETURN count(c) as count
            """, repo=repo_name)

            record = result.single()
            if record and record['count'] > 0:
                print(f"  ✓ Loaded graph for {repo_name}")
                return True

        return False

    def delete(self, repo_name: str):
        """Delete repository from Neo4j."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run(
                "MATCH (n {repo: $repo}) DETACH DELETE n",
                repo=repo_name
            )

        if self.current_repo == repo_name:
            self.chunks.clear()
            self.current_repo = None

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
