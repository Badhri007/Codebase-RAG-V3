"""
graph.py

Change from original:
  store_chunks() passes receiver_type and resolved flag from
  calls_with_context to go_package_map.resolve_call().
  Calls with resolved=False and no receiver_type are still attempted
  (same-package direct calls are valid unresolved cases) but calls
  with a receiver and resolved=False are skipped to avoid wrong edges.
"""

from collections import deque
from typing import List, Dict, Optional
from chunk import Chunk
from neo4j import GraphDatabase
from parsers.go_resolver import GoPackageMap
import os

BATCH = 500

EXTENSION_CANDIDATES = {
    "python":     [".py", "/__init__.py"],
    "javascript": [".js", ".jsx", "/index.js"],
    "typescript": [".ts", ".tsx", "/index.ts", ".js", "/index.js"],
    "java":       [".java"],
    "go":         [".go"],
}

EXTERNAL_JAVA = (
    "java.", "javax.", "android.", "kotlin.", "scala.",
    "org.junit", "org.springframework", "org.hibernate",
    "com.google", "com.fasterxml", "com.amazonaws",
    "org.slf4j", "ch.qos", "io.netty",
)


class Graph:

    def __init__(self, uri=None, user=None, password=None):
        uri      = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        user     = user     or os.getenv("NEO4J_USER",     "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def has_repo(self, repo_name: str) -> bool:
        with self.driver.session() as session:
            r = session.run(
                "MATCH (c:Chunk {repo: $repo}) RETURN count(c) AS n",
                repo=repo_name
            ).single()
            return bool(r and r["n"] > 0)

    def clear_repo(self, repo_name: str):
        with self.driver.session() as session:
            session.run(
                "MATCH (f:File {repo: $repo}) DETACH DELETE f",
                repo=repo_name
            )
            session.run(
                "MATCH (c:Chunk {repo: $repo}) DETACH DELETE c",
                repo=repo_name
            )

    def store_chunks(self, chunks: List[Chunk], repo_name: str,
                     go_package_map: Optional[GoPackageMap] = None):
        with self.driver.session() as session:

            # ── 1. File nodes ─────────────────────────────────────────
            files = {}
            for c in chunks:
                files.setdefault(c.file, []).append(c)

            file_rows = [
                {"path": fp, "lang": fc[0].language, "cnt": len(fc)}
                for fp, fc in files.items()
            ]
            for i in range(0, len(file_rows), BATCH):
                session.run("""
                    UNWIND $rows AS r
                    MERGE (f:File {path: r.path, repo: $repo})
                    SET f.language = r.lang, f.chunk_count = r.cnt
                """, rows=file_rows[i:i+BATCH], repo=repo_name)

            # ── 2. Chunk nodes ────────────────────────────────────────
            print(f"  🧩 {len(chunks)} chunk nodes...")
            chunk_rows = [{
                "id":           c.id,
                "repo":         repo_name,
                "name":         c.name,
                "type":         c.type,
                "file":         c.file,
                "start":        c.start,
                "end":          c.end,
                "language":     c.language,
                "signature":    c.signature or "",
                "docstring":    (c.docstring or "")[:500],
                "is_test":      c.is_test,
                "code":         c.code[:5000],
                "package_name": getattr(c, "package_name", ""),
                "is_entry_point":    getattr(c, "is_entry_point",    False),
                "entry_point_type":  getattr(c, "entry_point_type",  "") or "",
                "entry_point_route": getattr(c, "entry_point_route", "") or "",
                "entry_point_method":getattr(c, "entry_point_method","") or "",
                "implements":        getattr(c, "implements",        []),
                "receiver_type":     getattr(c, "receiver_type",     "") or "",
            } for c in chunks]

            for i in range(0, len(chunk_rows), BATCH):
                session.run("""
                    UNWIND $rows AS r
                    CREATE (c:Chunk) SET c = r
                """, rows=chunk_rows[i:i+BATCH])

            # ── 3. CONTAINS ───────────────────────────────────────────
            contains_rows = [{"file": c.file, "cid": c.id} for c in chunks]
            for i in range(0, len(contains_rows), BATCH):
                session.run("""
                    UNWIND $rows AS r
                    MATCH (f:File  {path: r.file, repo: $repo})
                    MATCH (c:Chunk {id:   r.cid,  repo: $repo})
                    MERGE (f)-[:CONTAINS]->(c)
                """, rows=contains_rows[i:i+BATCH], repo=repo_name)

            # ── 4. HAS_MEMBER ─────────────────────────────────────────
            member_rows = [
                {"pid": c.parent, "cid": c.id}
                for c in chunks if c.parent
            ]
            for i in range(0, len(member_rows), BATCH):
                session.run("""
                    UNWIND $rows AS r
                    MATCH (p:Chunk {id: r.pid, repo: $repo})
                    MATCH (c:Chunk {id: r.cid, repo: $repo})
                    MERGE (p)-[:HAS_MEMBER]->(c)
                """, rows=member_rows[i:i+BATCH], repo=repo_name)

            # ── 5. CALLS ──────────────────────────────────────────────
            print("  📞 Resolving CALLS...")

            by_file_name = {}
            for c in chunks:
                by_file_name.setdefault((c.file, c.name), []).append(c)

            known_files = set(c.file for c in chunks)
            call_rows   = []

            go_file_pkg = {}
            if go_package_map:
                for pkg, pkg_chunks in go_package_map.package_map.items():
                    for c in pkg_chunks:
                        go_file_pkg[c.file] = pkg

            for chunk in chunks:
                for call_info in chunk.calls_with_context:

                    if chunk.language == "go" and go_package_map:

                        receiver      = call_info.get("receiver")
                        receiver_type = call_info.get("receiver_type")
                        resolved      = call_info.get("resolved", False)

                        # Skip calls that have a receiver variable but no
                        # resolved type — we don't know what object this is,
                        # so any match would be a guess.
                        # Direct calls (receiver=None) are always attempted
                        # because same-package resolution handles them safely.
                        if receiver and not resolved:
                            continue

                        caller_pkg = go_file_pkg.get(chunk.file, "")
                        target = go_package_map.resolve_call(
                            call_name      = call_info["name"],
                            receiver       = receiver,
                            receiver_type  = receiver_type,
                            caller_file    = chunk.file,
                            caller_package = caller_pkg,
                            resolved       = resolved,
                        )

                    else:
                        target = self._resolve_call(
                            call_info, chunk, by_file_name, known_files
                        )

                    if target:
                        call_rows.append({
                            "caller": chunk.id,
                            "callee": target.id,
                            "method": call_info["name"],
                        })

            print(f"    Resolved {len(call_rows)} calls")
            for i in range(0, len(call_rows), BATCH):
                session.run("""
                    UNWIND $rows AS r
                    MATCH (caller:Chunk {id: r.caller, repo: $repo})
                    MATCH (callee:Chunk {id: r.callee, repo: $repo})
                    MERGE (caller)-[:CALLS {method: r.method}]->(callee)
                """, rows=call_rows[i:i+BATCH], repo=repo_name)

            # ── 6. IMPORTS ────────────────────────────────────────────
            import_rows = []
            for chunk in chunks:
                if chunk.language == "go" and go_package_map:
                    for local_name, imp_info in chunk.imports_map.items():
                        if not go_package_map.is_internal_import(
                            imp_info["from"]
                        ):
                            continue
                        imp_dir = go_package_map.import_path_to_dir(
                            imp_info["from"]
                        )
                        if not imp_dir:
                            continue
                        pkg_name = go_package_map.dir_package_map.get(imp_dir)
                        if not pkg_name:
                            continue
                        for target in go_package_map.package_map.get(
                            pkg_name, []
                        ):
                            if target.type in ("struct", "interface",
                                               "function"):
                                import_rows.append({
                                    "importer": chunk.id,
                                    "imported": target.id,
                                    "from":     imp_info["from"],
                                })
                else:
                    for imp_name, imp_info in chunk.imports_map.items():
                        base = self._resolve_import_path(
                            imp_info["from"], chunk.file, chunk.language
                        )
                        target_file = self._find_file(
                            base, chunk.language, known_files
                        )
                        if target_file:
                            for t in by_file_name.get(
                                (target_file, imp_name), []
                            ):
                                import_rows.append({
                                    "importer": chunk.id,
                                    "imported": t.id,
                                    "from":     imp_info["from"],
                                })

            for i in range(0, len(import_rows), BATCH):
                session.run("""
                    UNWIND $rows AS r
                    MATCH (a:Chunk {id: r.importer, repo: $repo})
                    MATCH (b:Chunk {id: r.imported, repo: $repo})
                    MERGE (a)-[:IMPORTS {from: r.from}]->(b)
                """, rows=import_rows[i:i+BATCH], repo=repo_name)

            print("  ✅ Graph stored")

    def expand_bfs(self, seed_ids: List[str], repo_name: str,
                   max_tokens: int = 8000) -> List[Dict]:
        with self.driver.session() as session:
            visited = set(seed_ids)
            queue   = deque((0, sid) for sid in seed_ids)
            result  = []
            tokens  = 0

            while queue and tokens < max_tokens:
                priority, chunk_id = queue.popleft()

                row = session.run(
                    "MATCH (c:Chunk {id: $id, repo: $repo}) RETURN c",
                    id=chunk_id, repo=repo_name
                ).single()
                if not row:
                    continue

                data         = dict(row["c"])
                chunk_tokens = len(data.get("code", "")) // 4
                if tokens + chunk_tokens > max_tokens:
                    continue

                result.append(data)
                tokens += chunk_tokens

                nbr = session.run("""
                    MATCH (c:Chunk {id: $id, repo: $repo})
                    OPTIONAL MATCH (c)-[:CALLS]->(callee:Chunk   {repo: $repo})
                    OPTIONAL MATCH (c)-[:IMPORTS]->(imp:Chunk    {repo: $repo})
                    OPTIONAL MATCH (c)-[:HAS_MEMBER]->(mem:Chunk {repo: $repo})
                    OPTIONAL MATCH (c)<-[:HAS_MEMBER]-(par:Chunk {repo: $repo})
                    RETURN
                        collect(DISTINCT {id: callee.id, pri: 1}) +
                        collect(DISTINCT {id: imp.id,    pri: 2}) +
                        collect(DISTINCT {id: mem.id,    pri: 3}) +
                        collect(DISTINCT {id: par.id,    pri: 3}) AS neighbours
                """, id=chunk_id, repo=repo_name).single()

                if nbr and nbr["neighbours"]:
                    for n in nbr["neighbours"]:
                        if n["id"] and n["id"] not in visited:
                            visited.add(n["id"])
                            queue.append((n["pri"], n["id"]))

                if tokens < max_tokens * 0.5:
                    sibs = session.run("""
                        MATCH (c:Chunk {id: $id, repo: $repo})
                              <-[:CONTAINS]-(f:File)
                              -[:CONTAINS]->(sib:Chunk {repo: $repo})
                        WHERE sib.id <> $id
                        RETURN sib.id AS id LIMIT 5
                    """, id=chunk_id, repo=repo_name)
                    for rec in sibs:
                        if rec["id"] not in visited:
                            visited.add(rec["id"])
                            queue.append((4, rec["id"]))

            return result

    def _resolve_call(self, call_info, caller, by_file_name, known_files):
        name          = call_info["name"]
        receiver_type = call_info.get("receiver_type")

        if receiver_type and receiver_type in caller.imports_map:
            base = self._resolve_import_path(
                caller.imports_map[receiver_type]["from"],
                caller.file, caller.language
            )
            src = self._find_file(base, caller.language, known_files)
            if src:
                for cls in by_file_name.get((src, receiver_type), []):
                    for m in by_file_name.get((src, name), []):
                        if m.parent == cls.id:
                            return m

        if name in caller.imports_map:
            base = self._resolve_import_path(
                caller.imports_map[name]["from"],
                caller.file, caller.language
            )
            src = self._find_file(base, caller.language, known_files)
            if src:
                candidates = by_file_name.get((src, name), [])
                if candidates:
                    return candidates[0]

        same = [c for c in by_file_name.get((caller.file, name), [])
                if c.id != caller.id]
        return same[0] if same else None

    def _resolve_import_path(self, import_from, current_file, language):
        if not import_from:
            return None
        lang = language.lower()
        if lang == "python":
            if import_from.startswith("."):
                dots  = len(import_from) - len(import_from.lstrip("."))
                rest  = import_from.lstrip(".")
                parts = current_file.replace("\\", "/").split("/")[:-1]
                for _ in range(dots - 1):
                    parts = parts[:-1] if parts else []
                if rest:
                    parts += rest.split(".")
                return "/".join(parts) if parts else None
            return import_from.replace(".", "/")
        elif lang in ("javascript", "typescript", "js", "ts", "jsx", "tsx"):
            if not import_from.startswith("."):
                return None
            base_dir = "/".join(
                current_file.replace("\\", "/").split("/")[:-1]
            )
            raw   = f"{base_dir}/{import_from}" if base_dir else import_from
            parts = []
            for seg in raw.replace("\\", "/").split("/"):
                if seg == "..":
                    if parts: parts.pop()
                elif seg and seg != ".":
                    parts.append(seg)
            return "/".join(parts) if parts else None
        elif lang == "java":
            if import_from.endswith(".*"):
                return None
            if any(import_from.startswith(p) for p in EXTERNAL_JAVA):
                return None
            return import_from.replace(".", "/")
        return None

    def _find_file(self, base_path, language, known_files):
        if not base_path:
            return None
        lang     = language.lower()
        lang_key = {
            "js": "javascript", "jsx": "javascript",
            "ts": "typescript", "tsx": "typescript",
        }.get(lang, lang)
        for ext in EXTENSION_CANDIDATES.get(lang_key, [".py"]):
            candidate = (base_path + ext).lstrip("/")
            if candidate in known_files:
                return candidate
        return None

    def get_stats(self, repo_name: str) -> Dict:
        with self.driver.session() as session:
            stats = {}
            r = session.run(
                "MATCH (c:Chunk {repo:$repo}) RETURN count(c) AS n",
                repo=repo_name
            ).single()
            stats["chunks"] = r["n"] if r else 0
            for rel in ["CONTAINS", "HAS_MEMBER", "CALLS", "IMPORTS"]:
                r = session.run(f"""
                    MATCH (a {{repo:$repo}})-[r:{rel}]->(b {{repo:$repo}})
                    RETURN count(r) AS n
                """, repo=repo_name).single()
                stats[rel] = r["n"] if r else 0
            stats["SAME_FILE"] = 0
            return stats
