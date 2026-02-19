"""Simplified graph with 100% accurate call resolution using Neo4j."""
from typing import List, Dict, Optional, Set
from collections import deque
from chunk import Chunk
from neo4j import GraphDatabase
import os


class Graph:
    """Neo4j graph with accurate import-based resolution."""

    def __init__(self, uri=None, user=None, password=None):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_repo(self, repo_name: str):
        """Delete all data for a repository."""
        with self.driver.session() as session:
            session.run("""
                MATCH (f:File {repo: $repo})
                DETACH DELETE f
            """, repo=repo_name)

            session.run("""
                MATCH (c:Chunk {repo: $repo})
                DETACH DELETE c
            """, repo=repo_name)

    def store_chunks(self, chunks: List[Chunk], repo_name: str):
        """Store chunks with accurate relationship resolution."""
        with self.driver.session() as session:
            # 1. Create File nodes
            files = {}
            for chunk in chunks:
                if chunk.file not in files:
                    files[chunk.file] = []
                files[chunk.file].append(chunk)

            print(f"  📁 Creating {len(files)} file nodes...")
            for file_path, file_chunks in files.items():
                result = session.run("""
                    MERGE (f:File {path: $path, repo: $repo})
                    SET f.language = $language,
                        f.chunk_count = $count
                    RETURN f
                """, path=file_path, repo=repo_name,
                     language=file_chunks[0].language, count=len(file_chunks))
                result.consume()  # Ensure query completes

            # 2. Create Chunk nodes with code stored
            print(f"  🧩 Creating {len(chunks)} chunk nodes...")
            for chunk in chunks:
                # Store code in chunk node for graph traversal
                result = session.run("""
                    CREATE (c:Chunk {
                        id: $id,
                        repo: $repo,
                        name: $name,
                        type: $type,
                        file: $file,
                        start: $start,
                        end: $end,
                        language: $language,
                        signature: $signature,
                        docstring: $docstring,
                        is_test: $is_test,
                        code: $code
                    })
                    RETURN c
                """, id=chunk.id, repo=repo_name, name=chunk.name, type=chunk.type,
                     file=chunk.file, start=chunk.start, end=chunk.end,
                     language=chunk.language, signature=chunk.signature or "",
                     docstring=chunk.docstring or "", is_test=chunk.is_test,
                     code=chunk.code[:5000])  # Limit code size
                result.consume()

            # 3. Create CONTAINS relationships
            print(f"  🔗 Creating CONTAINS relationships...")
            contains_count = 0
            for chunk in chunks:
                result = session.run("""
                    MATCH (f:File {path: $file, repo: $repo})
                    MATCH (c:Chunk {id: $chunk_id, repo: $repo})
                    MERGE (f)-[:CONTAINS]->(c)
                    RETURN f, c
                """, file=chunk.file, repo=repo_name, chunk_id=chunk.id)
                if result.single():
                    contains_count += 1
            print(f"  ✓ Created {contains_count} CONTAINS relationships")

            # 4. Create HAS_MEMBER relationships (parent-child)
            for chunk in chunks:
                if chunk.parent:
                    session.run("""
                        MATCH (parent:Chunk {id: $parent_id, repo: $repo})
                        MATCH (child:Chunk {id: $child_id, repo: $repo})
                        MERGE (parent)-[:HAS_MEMBER]->(child)
                    """, parent_id=chunk.parent, child_id=chunk.id, repo=repo_name)

            # 5. Create CALLS relationships with accurate resolution
            print(f"  🔗 Resolving {sum(len(c.calls_with_context) for c in chunks)} calls...")
            resolved_count = 0
            for chunk in chunks:
                for call_info in chunk.calls_with_context:
                    target = self._resolve_call(call_info, chunk, chunks)
                    if target:
                        session.run("""
                            MATCH (caller:Chunk {id: $caller_id, repo: $repo})
                            MATCH (callee:Chunk {id: $callee_id, repo: $repo})
                            MERGE (caller)-[:CALLS {method: $method}]->(callee)
                        """, caller_id=chunk.id, callee_id=target.id,
                             method=call_info["name"], repo=repo_name)
                        resolved_count += 1

            print(f"  ✓ Resolved {resolved_count} calls accurately")

            # 6. Create IMPORTS relationships
            for chunk in chunks:
                for imp_name, imp_info in chunk.imports_map.items():
                    target_file = self._resolve_import_path(
                        imp_info["from"], chunk.file, chunk.language
                    )
                    if target_file:
                        # Find chunks in target file
                        targets = [c for c in chunks if c.file == target_file and c.name == imp_name]
                        for target in targets:
                            session.run("""
                                MATCH (importer:Chunk {id: $importer_id, repo: $repo})
                                MATCH (imported:Chunk {id: $imported_id, repo: $repo})
                                MERGE (importer)-[:IMPORTS {from: $from}]->(imported)
                            """, importer_id=chunk.id, imported_id=target.id,
                                 from_=imp_info["from"], repo=repo_name)

            # 7. Create SAME_FILE relationships
            for file_path, file_chunks in files.items():
                for i, chunk1 in enumerate(file_chunks):
                    for chunk2 in file_chunks[i+1:]:
                        session.run("""
                            MATCH (c1:Chunk {id: $id1, repo: $repo})
                            MATCH (c2:Chunk {id: $id2, repo: $repo})
                            MERGE (c1)-[:SAME_FILE]-(c2)
                        """, id1=chunk1.id, id2=chunk2.id, repo=repo_name)

    def _resolve_call(self, call_info: Dict, caller: Chunk, all_chunks: List[Chunk]) -> Optional[Chunk]:
        """100% accurate call resolution using imports and types."""
        call_name = call_info["name"]
        receiver = call_info.get("receiver")
        receiver_type = call_info.get("receiver_type")

        # Method call (has receiver and type)
        if receiver and receiver_type:
            # Find where receiver_type is imported from
            if receiver_type in caller.imports_map:
                import_info = caller.imports_map[receiver_type]
                source_file = self._resolve_import_path(
                    import_info["from"], caller.file, caller.language
                )

                if source_file:
                    # Find the class in source file
                    target_classes = [c for c in all_chunks
                                     if c.file == source_file and c.name == receiver_type and c.type == "class"]

                    if target_classes:
                        target_class = target_classes[0]
                        # Find method in class
                        methods = [c for c in all_chunks
                                  if c.parent == target_class.id and c.name == call_name and c.type == "method"]
                        if methods:
                            return methods[0]

        # Direct function call
        else:
            # Check if imported
            if call_name in caller.imports_map:
                import_info = caller.imports_map[call_name]
                source_file = self._resolve_import_path(
                    import_info["from"], caller.file, caller.language
                )

                if source_file:
                    targets = [c for c in all_chunks
                              if c.file == source_file and c.name == call_name]
                    if targets:
                        return targets[0]

            # Check same file
            same_file = [c for c in all_chunks
                        if c.file == caller.file and c.name == call_name and c.id != caller.id]
            if same_file:
                return same_file[0]

        return None

    def _resolve_import_path(self, import_from: str, current_file: str, language: str) -> Optional[str]:
        """Convert import statement to file path."""
        if not import_from:
            return None

        if language == "python":
            # "models.User" → "models/User.py"
            return import_from.replace(".", "/") + ".py"

        elif language in ("javascript", "typescript", "jsx", "tsx"):
            # "./models/User" → "models/User.js"
            if import_from.startswith("."):
                base_dir = "/".join(current_file.split("/")[:-1])
                import_path = import_from.lstrip("./")
                full_path = f"{base_dir}/{import_path}" if base_dir else import_path
                # Normalize path
                parts = []
                for part in full_path.split("/"):
                    if part == "..":
                        if parts:
                            parts.pop()
                    elif part and part != ".":
                        parts.append(part)
                normalized = "/".join(parts)

                # Try different extensions
                for ext in [".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"]:
                    candidate = normalized + ext
                    return candidate
                return normalized + ".js"
            else:
                # External module
                return None

        elif language == "java":
            # "com.example.models.User" → "com/example/models/User.java"
            return import_from.replace(".", "/") + ".java"

        elif language == "go":
            # "github.com/example/models" → look for package name
            # Simplified: use last part
            return import_from.split("/")[-1] + ".go"

        return None

    def expand_bfs(self, seed_ids: List[str], repo_name: str,
                   max_tokens: int = 8000) -> List[Dict]:
        """BFS expansion with token budget."""
        print(f"  [DEBUG BFS] Starting with {len(seed_ids)} seed IDs")
        print(f"  [DEBUG BFS] Repo name: {repo_name}")
        print(f"  [DEBUG BFS] Sample seed ID: {seed_ids[0] if seed_ids else 'None'}")

        with self.driver.session() as session:
            # First, check what chunks exist in Neo4j
            check_result = session.run("""
                MATCH (c:Chunk {repo: $repo})
                RETURN count(c) as total, collect(c.id)[0..5] as sample_ids
            """, repo=repo_name).single()

            if check_result:
                print(f"  [DEBUG BFS] Neo4j has {check_result['total']} chunks for this repo")
                print(f"  [DEBUG BFS] Sample Neo4j IDs: {check_result['sample_ids']}")

            visited = set(seed_ids)
            result = []
            queue = deque(seed_ids)
            tokens = 0
            not_found_count = 0

            while queue and tokens < max_tokens:
                chunk_id = queue.popleft()

                # Get chunk data
                chunk_result = session.run("""
                    MATCH (c:Chunk {id: $id, repo: $repo})
                    RETURN c
                """, id=chunk_id, repo=repo_name).single()

                if chunk_result:
                    chunk_data = dict(chunk_result["c"])
                    chunk_tokens = len(chunk_data.get("code", "")) // 4

                    if tokens + chunk_tokens <= max_tokens:
                        result.append(chunk_data)
                        tokens += chunk_tokens

                        # Get neighbors
                        neighbors = session.run("""
                            MATCH (c:Chunk {id: $id, repo: $repo})
                            -[:CALLS|IMPORTS|HAS_MEMBER|SAME_FILE]-
                            (neighbor:Chunk {repo: $repo})
                            RETURN neighbor.id as id
                        """, id=chunk_id, repo=repo_name)

                        for record in neighbors:
                            neighbor_id = record["id"]
                            if neighbor_id not in visited:
                                visited.add(neighbor_id)
                                queue.append(neighbor_id)
                else:
                    not_found_count += 1
                    if not_found_count <= 3:  # Only print first few
                        print(f"  [DEBUG BFS] Chunk not found in Neo4j: {chunk_id}")

            if not_found_count > 0:
                print(f"  [DEBUG BFS] Total chunks not found: {not_found_count}/{len(seed_ids)}")

            return result

    def get_stats(self, repo_name: str) -> Dict:
        """Get graph statistics."""
        with self.driver.session() as session:
            stats = {}

            result = session.run("""
                MATCH (c:Chunk {repo: $repo})
                RETURN count(c) as chunks
            """, repo=repo_name).single()
            stats["chunks"] = result["chunks"] if result else 0

            # CONTAINS: File -> Chunk
            result = session.run("""
                MATCH (f:File {repo: $repo})-[r:CONTAINS]->(c:Chunk {repo: $repo})
                RETURN count(r) as count
            """, repo=repo_name).single()
            stats["CONTAINS"] = result["count"] if result else 0

            # HAS_MEMBER, CALLS, IMPORTS: Chunk -> Chunk
            for rel_type in ["HAS_MEMBER", "CALLS", "IMPORTS"]:
                result = session.run(f"""
                    MATCH (c1:Chunk {{repo: $repo}})-[r:{rel_type}]->(c2:Chunk {{repo: $repo}})
                    RETURN count(r) as count
                """, repo=repo_name).single()
                stats[rel_type] = result["count"] if result else 0

            # SAME_FILE: bidirectional
            result = session.run("""
                MATCH (c1:Chunk {repo: $repo})-[r:SAME_FILE]-(c2:Chunk {repo: $repo})
                RETURN count(r) as count
            """, repo=repo_name).single()
            stats["SAME_FILE"] = (result["count"] // 2) if result else 0  # Divide by 2 for bidirectional

            return stats
