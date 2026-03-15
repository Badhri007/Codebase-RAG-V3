"""
summariser.py

Phase 2 — chunk situating context
Phase 3 — file summaries
Phase 4 — flow detection and summaries
Phase 5 — package summaries
Phase 6 — codebase summary
"""

import os
import re
import json
import time
import hashlib
from typing import List, Dict, Optional
from collections import defaultdict

import anthropic
from chunk import Chunk


MODEL      = "claude-sonnet-4-20250514"
RATE_PAUSE = 0.3


# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — Chunk situating context
# ─────────────────────────────────────────────────────────────────────

_CHUNK_SYSTEM = """You write short technical summaries for code chunks.
Rules:
- Only describe what is directly visible in the code and context provided.
- Never infer or guess what is not shown.
- 3 sentences maximum.
- Plain prose, no bullet points, no code reproduction.
- If something is unclear, say "purpose unclear" — do not guess."""


def _chunk_prompt(chunk: Chunk, callers: list,
                   callees: list, parent: dict) -> str:
    caller_lines = "\n".join(
        f"  - {c['name']} ({c['file']})" for c in callers
    ) or "  none found"

    callee_lines = "\n".join(
        f"  - {c['name']} ({c['file']})" for c in callees
    ) or "  none found"

    parent_line = (
        f"{parent['name']} ({parent['type']})"
        if parent else "none (top-level)"
    )

    ep_line = "not an entry point"
    if chunk.is_entry_point:
        ep_line = f"{chunk.entry_point_type}"
        if chunk.entry_point_method:
            ep_line += f" {chunk.entry_point_method}"
        if chunk.entry_point_route:
            ep_line += f" {chunk.entry_point_route}"

    return f"""Summarise this code chunk in 3 sentences.
Only use the information provided. Do not guess.

Name:      {chunk.name}
Type:      {chunk.type}
File:      {chunk.file}
Package:   {chunk.package_name or 'unknown'}
Signature: {chunk.signature or 'none'}
Entry point: {ep_line}
Parent:    {parent_line}

Callers (chunks that call this):
{caller_lines}

Callees (chunks this calls, within this codebase):
{callee_lines}

Docstring: {chunk.docstring or 'none'}

Code:
{chunk.code[:1500]}

Write the 3-sentence summary now."""


def _fetch_chunk_context(chunk: Chunk, repo_name: str,
                          driver) -> Dict:
    with driver.session() as s:
        callers = [dict(r) for r in s.run("""
            MATCH (caller:Chunk {repo:$repo})
                  -[:CALLS]->(c:Chunk {id:$id, repo:$repo})
            RETURN caller.name AS name, caller.file AS file
            LIMIT 6
        """, repo=repo_name, id=chunk.id)]

        callees = [dict(r) for r in s.run("""
            MATCH (c:Chunk {id:$id, repo:$repo})
                  -[:CALLS]->(callee:Chunk {repo:$repo})
            RETURN callee.name AS name, callee.file AS file
            LIMIT 6
        """, repo=repo_name, id=chunk.id)]

        parent = None
        if chunk.parent:
            row = s.run("""
                MATCH (p:Chunk {id:$pid, repo:$repo})
                RETURN p.name AS name, p.type AS type
            """, pid=chunk.parent, repo=repo_name).single()
            if row:
                parent = dict(row)

    return {"callers": callers, "callees": callees, "parent": parent}


def _code_hash(chunk: Chunk) -> str:
    return hashlib.md5(chunk.code.encode()).hexdigest()[:8]


def generate_situating_contexts(chunks: List[Chunk],
                                  repo_name: str,
                                  driver,
                                  vector_db=None,
                                  force: bool = False) -> List[Chunk]:
    client     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    to_process = [c for c in chunks if c.type != 'module']
    updated    = []

    print(f"\n📝 Generating situating context for {len(to_process)} chunks...")

    for i, chunk in enumerate(to_process, 1):
        if not force and chunk.situating_context:
            if chunk.situating_context.endswith(f"[{_code_hash(chunk)}]"):
                continue

        try:
            ctx = _fetch_chunk_context(chunk, repo_name, driver)
        except Exception as e:
            print(f"  Graph fetch failed for {chunk.name}: {e}")
            ctx = {"callers": [], "callees": [], "parent": None}

        try:
            resp = client.messages.create(
                model      = MODEL,
                max_tokens = 300,
                system     = _CHUNK_SYSTEM,
                messages   = [{"role": "user",
                                "content": _chunk_prompt(chunk, **ctx)}],
            )
            text = resp.content[0].text.strip()
            chunk.situating_context = f"{text} [{_code_hash(chunk)}]"
            clean = chunk.situating_context.rsplit(" [", 1)[0]
            chunk.contextual_embedding_text = f"{clean}\n\n{chunk.code}"
            chunk.use_contextual_embedding  = True
            updated.append(chunk)

        except anthropic.RateLimitError:
            print("  Rate limit — waiting 60s...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"  Failed for {chunk.name}: {e}")
            continue

        if i % 20 == 0:
            print(f"  {i}/{len(to_process)} done...")
        time.sleep(RATE_PAUSE)

    print(f"  ✓ {len(updated)} chunks updated.")

    if vector_db and updated:
        print(f"  🔢 Re-embedding {len(updated)} chunks...")
        vector_db.upsert_batch(repo_name, [{
            "id":   c.id,
            "text": c.embedding_text(),
            "metadata": {
                "name":             c.name,
                "type":             c.type,
                "file":             c.file,
                "is_test":          c.is_test,
                "is_entry_point":   c.is_entry_point,
                "entry_point_type": c.entry_point_type or "",
            },
        } for c in updated])

    return chunks


# ─────────────────────────────────────────────────────────────────────
# PHASE 3 — File summaries
# ─────────────────────────────────────────────────────────────────────

_FILE_SYSTEM = """You write short technical summaries for source files.
Rules:
- Only describe what is directly visible in the provided chunk summaries
  and graph data. Never infer or guess.
- 4 sentences maximum.
- Cover: what this file does, what it contains, who calls it,
  what it calls.
- Plain prose. No bullet points. No code."""


def _file_prompt(file_path: str, language: str, package: str,
                  chunk_summaries: list, inbound: list,
                  outbound: list, entry_points: list) -> str:
    chunks_text = "\n".join(
        f"  [{c['type']}] {c['name']}: "
        f"{c.get('situating_context') or c.get('signature','')}"
        for c in chunk_summaries
    ) or "  none"

    ep_text = "\n".join(
        f"  - {e['name']} "
        f"({e['entry_point_type']} {e.get('entry_point_route','')})"
        for e in entry_points
    ) or "  none"

    return f"""Summarise this source file in 4 sentences.
Only use the information provided. Do not guess.

File:     {file_path}
Language: {language}
Package:  {package or 'unknown'}

Entry points in this file:
{ep_text}

Chunks defined in this file:
{chunks_text}

Files that call into this file:
{chr(10).join('  - ' + f for f in inbound) or '  none'}

Files this file calls into:
{chr(10).join('  - ' + f for f in outbound) or '  none'}

Write the 4-sentence summary now."""


def _fetch_file_graph_context(file_path: str,
                               repo_name: str, driver) -> Dict:
    with driver.session() as s:
        inbound = [r["file"] for r in s.run("""
            MATCH (caller:Chunk {repo:$repo})
                  -[:CALLS]->(c:Chunk {repo:$repo, file:$file})
            WHERE caller.file <> $file
            RETURN DISTINCT caller.file AS file LIMIT 10
        """, repo=repo_name, file=file_path)]

        outbound = [r["file"] for r in s.run("""
            MATCH (c:Chunk {repo:$repo, file:$file})
                  -[:CALLS]->(callee:Chunk {repo:$repo})
            WHERE callee.file <> $file
            RETURN DISTINCT callee.file AS file LIMIT 10
        """, repo=repo_name, file=file_path)]

    return {"inbound": inbound, "outbound": outbound}


def generate_file_summaries(chunks: List[Chunk],
                              repo_name: str,
                              driver,
                              vector_db=None) -> Dict[str, str]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    by_file: Dict[str, List[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.type != 'module':
            by_file[chunk.file].append(chunk)

    summaries: Dict[str, str] = {}
    total = len(by_file)
    print(f"\n📄 Generating file summaries for {total} files...")

    for i, (file_path, file_chunks) in enumerate(by_file.items(), 1):
        chunk_summaries = []
        entry_points    = []

        for c in file_chunks:
            ctx = c.situating_context or ""
            if ctx:
                ctx = ctx.rsplit(" [", 1)[0]
            chunk_summaries.append({
                "name":              c.name,
                "type":              c.type,
                "signature":         c.signature or "",
                "situating_context": ctx,
            })
            if c.is_entry_point:
                entry_points.append({
                    "name":             c.name,
                    "entry_point_type": c.entry_point_type or "",
                    "entry_point_route":c.entry_point_route or "",
                })

        try:
            graph_ctx = _fetch_file_graph_context(
                file_path, repo_name, driver
            )
        except Exception as e:
            print(f"  Graph fetch failed for {file_path}: {e}")
            graph_ctx = {"inbound": [], "outbound": []}

        first    = file_chunks[0]
        language = first.language
        package  = first.package_name or ""

        try:
            resp = client.messages.create(
                model      = MODEL,
                max_tokens = 400,
                system     = _FILE_SYSTEM,
                messages   = [{"role": "user", "content": _file_prompt(
                    file_path, language, package,
                    chunk_summaries,
                    graph_ctx["inbound"],
                    graph_ctx["outbound"],
                    entry_points,
                )}],
            )
            summary = resp.content[0].text.strip()
            summaries[file_path] = summary

        except anthropic.RateLimitError:
            print("  Rate limit — waiting 60s...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"  Failed for {file_path}: {e}")
            continue

        if i % 10 == 0:
            print(f"  {i}/{total} files done...")
        time.sleep(RATE_PAUSE)

    print(f"  ✓ {len(summaries)} file summaries generated.")

    # Store on File nodes in Neo4j
    if summaries:
        rows = [{"path": fp, "summary": s}
                for fp, s in summaries.items()]
        with driver.session() as s:
            s.run("""
                UNWIND $rows AS r
                MATCH (f:File {path:r.path, repo:$repo})
                SET f.summary = r.summary
            """, rows=rows, repo=repo_name)

    # Store in vector DB
    if vector_db and summaries:
        vector_db.upsert_batch(repo_name, [{
            "id":   f"file::{fp}",
            "text": summary,
            "metadata": {
                "type":     "file_summary",
                "file":     fp,
                "language": by_file[fp][0].language,
                "package":  by_file[fp][0].package_name or "",
            },
        } for fp, summary in summaries.items()])

    return summaries


# ─────────────────────────────────────────────────────────────────────
# PHASE 4 — Flow detection and summaries
# ─────────────────────────────────────────────────────────────────────

_FLOW_SYSTEM = """You write short summaries for code execution flows.
Rules:
- Only describe what is shown in the provided step data.
- Never guess what happens between steps.
- 4 sentences maximum.
- Cover: what triggers this flow, what it does, which packages
  are involved, what the final outcome is.
- Plain prose. No bullet points."""


def _detect_flows(repo_name: str, driver) -> List[Dict]:
    """
    BFS from every entry point chunk through CALLS edges.
    Returns list of flow dicts:
        {
          name:        str   (entry_point_type + route or function name)
          entry_id:    str   (entry point chunk id)
          entry_name:  str
          entry_type:  str   (http / kafka / cron / etc.)
          route:       str
          steps:       [{id, name, file, signature}]
          files:       [distinct file paths]
          packages:    [distinct package names]
        }
    """
    flows = []

    with driver.session() as s:
        # Fetch all entry point chunks
        entry_points = [dict(r) for r in s.run("""
            MATCH (c:Chunk {repo:$repo, is_entry_point:true})
            RETURN c.id               AS id,
                   c.name             AS name,
                   c.file             AS file,
                   c.entry_point_type AS ep_type,
                   c.entry_point_route AS route,
                   c.package_name     AS package
        """, repo=repo_name)]

        for ep in entry_points:
            # BFS through CALLS edges — max depth 10 to prevent runaway
            visited = set()
            queue   = [ep["id"]]
            steps   = []

            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                # Fetch chunk details
                row = s.run("""
                    MATCH (c:Chunk {id:$id, repo:$repo})
                    RETURN c.id        AS id,
                           c.name      AS name,
                           c.file      AS file,
                           c.signature AS signature,
                           c.package_name AS package
                """, id=current_id, repo=repo_name).single()

                if row:
                    steps.append(dict(row))

                # Stop if we've gone 10 hops deep
                if len(steps) >= 10:
                    break

                # Get next callees
                callees = [r["id"] for r in s.run("""
                    MATCH (c:Chunk {id:$id, repo:$repo})
                          -[:CALLS]->(callee:Chunk {repo:$repo})
                    WHERE NOT callee.is_entry_point
                    RETURN callee.id AS id
                    LIMIT 5
                """, id=current_id, repo=repo_name)]

                queue.extend(callees)

            if not steps:
                continue

            # Build flow name from entry point info
            route = ep.get("route") or ""
            name  = (
                f"{ep['ep_type']} {route}".strip()
                if route
                else ep["name"]
            )

            flows.append({
                "name":       name,
                "entry_id":   ep["id"],
                "entry_name": ep["name"],
                "entry_type": ep.get("ep_type", ""),
                "route":      route,
                "steps":      steps,
                "files":      list({s["file"] for s in steps
                                    if s.get("file")}),
                "packages":   list({s["package"] for s in steps
                                    if s.get("package")}),
            })

    print(f"  Detected {len(flows)} flows from entry points.")
    return flows


def _flow_prompt(flow: Dict) -> str:
    steps_text = "\n".join(
        f"  {i+1}. [{s.get('package','')}] "
        f"{s['name']} ({s.get('file','')})"
        for i, s in enumerate(flow["steps"])
    )

    return f"""Summarise this code execution flow in 4 sentences.
Only use the information provided. Do not guess.

Flow name:   {flow['name']}
Entry type:  {flow['entry_type']}
Route/topic: {flow['route'] or 'none'}
Files:       {', '.join(flow['files'])}
Packages:    {', '.join(flow['packages'])}

Steps (in order):
{steps_text}

Write the 4-sentence summary now."""


def _store_flows(flows: List[Dict], summaries: Dict[str, str],
                  repo_name: str, driver):
    """Store Flow nodes in Neo4j and PART_OF_FLOW edges."""
    with driver.session() as s:
        for flow in flows:
            summary = summaries.get(flow["name"], "")
            steps_json = json.dumps([
                {"name": st["name"], "file": st.get("file", "")}
                for st in flow["steps"]
            ])

            # Create or update Flow node
            s.run("""
                MERGE (f:Flow {name:$name, repo:$repo})
                SET f.entry_type  = $entry_type,
                    f.route       = $route,
                    f.entry_point = $entry_name,
                    f.steps       = $steps,
                    f.files       = $files,
                    f.packages    = $packages,
                    f.summary     = $summary
            """,
                name       = flow["name"],
                repo       = repo_name,
                entry_type = flow["entry_type"],
                route      = flow["route"],
                entry_name = flow["entry_name"],
                steps      = steps_json,
                files      = flow["files"],
                packages   = flow["packages"],
                summary    = summary,
            )

            # Link each step chunk to the flow
            for step in flow["steps"]:
                s.run("""
                    MATCH (f:Flow {name:$name, repo:$repo})
                    MATCH (c:Chunk {id:$cid, repo:$repo})
                    MERGE (c)-[:PART_OF_FLOW]->(f)
                """, name=flow["name"], repo=repo_name,
                     cid=step["id"])


def generate_flow_summaries(repo_name: str,
                              driver,
                              vector_db=None) -> Dict[str, str]:
    """
    Detect all flows from entry points via BFS.
    Generate a summary for each flow.
    Store in Neo4j and vector DB.

    Must be called after graph is fully stored
    (store_chunks must have run first).
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    flows = _detect_flows(repo_name, driver)
    if not flows:
        print("  No flows detected.")
        return {}

    summaries: Dict[str, str] = {}
    print(f"\n🌊 Generating flow summaries for {len(flows)} flows...")

    for i, flow in enumerate(flows, 1):
        try:
            resp = client.messages.create(
                model      = MODEL,
                max_tokens = 400,
                system     = _FLOW_SYSTEM,
                messages   = [{"role": "user",
                                "content": _flow_prompt(flow)}],
            )
            summaries[flow["name"]] = resp.content[0].text.strip()

        except anthropic.RateLimitError:
            print("  Rate limit — waiting 60s...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"  Failed for flow {flow['name']}: {e}")
            continue

        time.sleep(RATE_PAUSE)

    print(f"  ✓ {len(summaries)} flow summaries generated.")

    # Store flows and summaries in Neo4j
    _store_flows(flows, summaries, repo_name, driver)

    # Store in vector DB
    if vector_db and summaries:
        vector_db.upsert_batch(repo_name, [{
            "id":   f"flow::{name}",
            "text": summary,
            "metadata": {
                "type":      "flow_summary",
                "flow_name": name,
                "entry_type": next(
                    (f["entry_type"] for f in flows if f["name"] == name),
                    ""
                ),
            },
        } for name, summary in summaries.items()])

    return summaries


# ─────────────────────────────────────────────────────────────────────
# PHASE 5 — Package summaries
# ─────────────────────────────────────────────────────────────────────

_PKG_SYSTEM = """You write short technical summaries for code packages.
Rules:
- Only describe what is shown in the provided file summaries.
- Never infer or guess.
- 4 sentences maximum.
- Cover: what this package does, its public-facing functions/types,
  what it depends on, what depends on it.
- Plain prose. No bullet points."""


def _pkg_prompt(package: str, language: str,
                 file_summaries: list,
                 entry_points: list,
                 depends_on: list,
                 depended_by: list) -> str:
    files_text = "\n".join(
        f"  {fs['file']}: {fs['summary']}"
        for fs in file_summaries
    ) or "  none"

    ep_text = "\n".join(
        f"  - {e['name']} ({e['type']} {e.get('route','')})"
        for e in entry_points
    ) or "  none"

    return f"""Summarise this code package in 4 sentences.
Only use the information provided. Do not guess.

Package:   {package}
Language:  {language}

Entry points in this package:
{ep_text}

Files in this package and their summaries:
{files_text}

Packages this package depends on:
{chr(10).join('  - ' + p for p in depends_on) or '  none'}

Packages that depend on this package:
{chr(10).join('  - ' + p for p in depended_by) or '  none'}

Write the 4-sentence summary now."""


def _fetch_package_graph_context(package: str,
                                   repo_name: str,
                                   driver) -> Dict:
    """
    Find which packages this package calls into and which call into it.
    Uses CALLS edges between chunks across different packages.
    """
    with driver.session() as s:
        depends_on = [r["pkg"] for r in s.run("""
            MATCH (c:Chunk {repo:$repo, package_name:$pkg})
                  -[:CALLS]->(callee:Chunk {repo:$repo})
            WHERE callee.package_name <> $pkg
              AND callee.package_name IS NOT NULL
              AND callee.package_name <> ''
            RETURN DISTINCT callee.package_name AS pkg
            LIMIT 10
        """, repo=repo_name, pkg=package)]

        depended_by = [r["pkg"] for r in s.run("""
            MATCH (caller:Chunk {repo:$repo})
                  -[:CALLS]->(c:Chunk {repo:$repo, package_name:$pkg})
            WHERE caller.package_name <> $pkg
              AND caller.package_name IS NOT NULL
              AND caller.package_name <> ''
            RETURN DISTINCT caller.package_name AS pkg
            LIMIT 10
        """, repo=repo_name, pkg=package)]

    return {"depends_on": depends_on, "depended_by": depended_by}


def generate_package_summaries(chunks: List[Chunk],
                                 file_summaries: Dict[str, str],
                                 repo_name: str,
                                 driver,
                                 vector_db=None) -> Dict[str, str]:
    """
    Generate one summary per package.
    Must be called after generate_file_summaries().

    Groups files by package_name, uses file summaries as input.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Group by package
    pkg_files:   Dict[str, set]        = defaultdict(set)
    pkg_lang:    Dict[str, str]        = {}
    pkg_eps:     Dict[str, list]       = defaultdict(list)

    for chunk in chunks:
        if not chunk.package_name or chunk.type == 'module':
            continue
        pkg = chunk.package_name
        pkg_files[pkg].add(chunk.file)
        pkg_lang[pkg] = chunk.language

        if chunk.is_entry_point:
            pkg_eps[pkg].append({
                "name":  chunk.name,
                "type":  chunk.entry_point_type or "",
                "route": chunk.entry_point_route or "",
            })

    summaries: Dict[str, str] = {}
    total = len(pkg_files)
    print(f"\n📦 Generating package summaries for {total} packages...")

    for i, (package, files) in enumerate(pkg_files.items(), 1):
        # Collect file summaries for this package
        fs_list = [
            {"file": fp, "summary": file_summaries.get(fp, "")}
            for fp in files
            if file_summaries.get(fp)
        ]

        if not fs_list:
            continue

        try:
            graph_ctx = _fetch_package_graph_context(
                package, repo_name, driver
            )
        except Exception as e:
            print(f"  Graph fetch failed for package {package}: {e}")
            graph_ctx = {"depends_on": [], "depended_by": []}

        try:
            resp = client.messages.create(
                model      = MODEL,
                max_tokens = 400,
                system     = _PKG_SYSTEM,
                messages   = [{"role": "user", "content": _pkg_prompt(
                    package    = package,
                    language   = pkg_lang.get(package, ""),
                    file_summaries = fs_list,
                    entry_points   = pkg_eps.get(package, []),
                    depends_on     = graph_ctx["depends_on"],
                    depended_by    = graph_ctx["depended_by"],
                )}],
            )
            summaries[package] = resp.content[0].text.strip()

        except anthropic.RateLimitError:
            print("  Rate limit — waiting 60s...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"  Failed for package {package}: {e}")
            continue

        time.sleep(RATE_PAUSE)

    print(f"  ✓ {len(summaries)} package summaries generated.")

    # Store Package nodes in Neo4j
    if summaries:
        with driver.session() as s:
            for pkg, summary in summaries.items():
                s.run("""
                    MERGE (p:Package {name:$name, repo:$repo})
                    SET p.summary  = $summary,
                        p.language = $lang,
                        p.files    = $files
                """,
                    name    = pkg,
                    repo    = repo_name,
                    summary = summary,
                    lang    = pkg_lang.get(pkg, ""),
                    files   = list(pkg_files.get(pkg, [])),
                )

    # Store in vector DB
    if vector_db and summaries:
        vector_db.upsert_batch(repo_name, [{
            "id":   f"package::{pkg}",
            "text": summary,
            "metadata": {
                "type":     "package_summary",
                "package":  pkg,
                "language": pkg_lang.get(pkg, ""),
            },
        } for pkg, summary in summaries.items()])

    return summaries


# ─────────────────────────────────────────────────────────────────────
# PHASE 6 — Codebase summary
# ─────────────────────────────────────────────────────────────────────

_REPO_SYSTEM = """You write a high-level technical summary for an entire codebase.
Rules:
- Only describe what is shown in the provided package summaries and data.
- Never guess or infer.
- Cover: what the codebase does, its architecture, all entry points,
  core packages, key flows, external dependencies.
- Plain prose with clear sections. Be thorough but concise.
- This will be read by developers who are completely new to the codebase."""


def _repo_prompt(repo_name: str,
                  languages: List[str],
                  pkg_summaries: Dict[str, str],
                  entry_points: list,
                  flow_summaries: Dict[str, str],
                  external_deps: List[str]) -> str:
    pkg_text = "\n\n".join(
        f"Package [{pkg}]:\n  {summary}"
        for pkg, summary in pkg_summaries.items()
    ) or "  none"

    ep_text = "\n".join(
        f"  [{e.get('type','')}] "
        f"{e.get('method','')} {e.get('route','')} "
        f"→ {e.get('name','')} ({e.get('file','')})"
        for e in entry_points
    ) or "  none"

    flow_text = "\n".join(
        f"  [{name}]: {summary[:200]}"
        for name, summary in flow_summaries.items()
    ) or "  none"

    deps_text = ", ".join(external_deps) or "none detected"

    return f"""Write a comprehensive codebase summary for: {repo_name}

Languages: {', '.join(languages)}

All entry points:
{ep_text}

All flows:
{flow_text}

Package summaries:
{pkg_text}

External dependencies detected:
{deps_text}

Write the comprehensive codebase summary now.
Structure it with these sections:
1. Purpose
2. Architecture
3. Entry points
4. Core packages
5. Key flows
6. External dependencies"""


def _fetch_entry_points_for_repo(repo_name: str, driver) -> List[Dict]:
    with driver.session() as s:
        rows = s.run("""
            MATCH (c:Chunk {repo:$repo, is_entry_point:true})
            RETURN c.name             AS name,
                   c.file             AS file,
                   c.entry_point_type AS type,
                   c.entry_point_route AS route,
                   c.entry_point_method AS method
            ORDER BY c.entry_point_type, c.name
        """, repo=repo_name)
        return [dict(r) for r in rows]


def _fetch_external_deps(chunks: List[Chunk]) -> List[str]:
    """
    Extract likely external dependency names from import paths.
    Heuristic: import paths containing a domain prefix are external.
    """
    external = set()
    external_prefixes = (
        "github.com/", "golang.org/", "gopkg.in/",
        "google.golang.org/", "k8s.io/", "go.uber.org/",
        "npmjs", "pypi",
    )
    for chunk in chunks:
        for imp in chunk.imports:
            if any(imp.startswith(p) for p in external_prefixes):
                # Take the third path segment as the package name
                # e.g. github.com/gin-gonic/gin → gin
                parts = imp.split("/")
                if len(parts) >= 3:
                    external.add(parts[2])
                elif len(parts) == 2:
                    external.add(parts[1])

    return sorted(external)


def generate_codebase_summary(chunks: List[Chunk],
                                pkg_summaries: Dict[str, str],
                                flow_summaries: Dict[str, str],
                                repo_name: str,
                                driver,
                                vector_db=None) -> str:
    """
    Generate the single top-level codebase summary.
    Must be called after generate_package_summaries()
    and generate_flow_summaries().
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"\n🌐 Generating codebase summary...")

    # Collect languages
    languages = list({c.language for c in chunks
                      if c.language and c.type != 'module'})

    # Fetch all entry points
    try:
        entry_points = _fetch_entry_points_for_repo(repo_name, driver)
    except Exception as e:
        print(f"  Failed to fetch entry points: {e}")
        entry_points = []

    # External deps
    external_deps = _fetch_external_deps(chunks)

    try:
        resp = client.messages.create(
            model      = MODEL,
            max_tokens = 1200,
            system     = _REPO_SYSTEM,
            messages   = [{"role": "user", "content": _repo_prompt(
                repo_name      = repo_name,
                languages      = languages,
                pkg_summaries  = pkg_summaries,
                entry_points   = entry_points,
                flow_summaries = flow_summaries,
                external_deps  = external_deps,
            )}],
        )
        summary = resp.content[0].text.strip()

    except Exception as e:
        print(f"  Failed to generate codebase summary: {e}")
        return ""

    print("  ✓ Codebase summary generated.")

    # Store on Repo node in Neo4j
    with driver.session() as s:
        s.run("""
            MERGE (r:Repo {name:$repo})
            SET r.summary   = $summary,
                r.languages = $languages,
                r.entry_point_count = $ep_count
        """,
            repo       = repo_name,
            summary    = summary,
            languages  = languages,
            ep_count   = len(entry_points),
        )

    # Store in vector DB
    if vector_db:
        vector_db.upsert_batch(repo_name, [{
            "id":   f"codebase::{repo_name}",
            "text": summary,
            "metadata": {
                "type":      "codebase_summary",
                "repo":      repo_name,
                "languages": ", ".join(languages),
            },
        }])

    return summary
