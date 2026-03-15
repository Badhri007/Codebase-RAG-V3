import os
import subprocess
from typing import List, Dict, Optional
from chunk import Chunk
from graph import Graph
from vectordb import VectorDB
from parsers import parse_python, parse_javascript, parse_java
from parsers.parse_go import parse_go
from parsers.go_resolver import (
    parse_go_mod, extract_package_name, GoPackageMap
)
from summariser import (
    generate_situating_contexts,
    generate_file_summaries,
    generate_flow_summaries,
    generate_package_summaries,
    generate_codebase_summary,
)


def clone_repo(github_url: str, target_dir: str):
    if os.path.exists(target_dir):
        subprocess.run(['rm', '-rf', target_dir])
    print(f"  Cloning {github_url}...")
    subprocess.run(['git', 'clone', github_url, target_dir],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_all_files(repo_path: str,
                    module_name: Optional[str] = None):
    chunks:           List[Chunk]    = []
    file_package_map: Dict[str, str] = {}

    supported = {
        '.py':   parse_python,
        '.js':   parse_javascript,
        '.ts':   parse_javascript,
        '.jsx':  parse_javascript,
        '.tsx':  parse_javascript,
        '.java': parse_java,
    }

    file_count = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {
            'node_modules', '.git', '__pycache__', 'venv', 'env',
            'build', 'dist', 'target', '.next', 'vendor',
        }]

        for file in files:
            ext       = os.path.splitext(file)[1]
            file_path = os.path.join(root, file)
            rel_path  = os.path.relpath(file_path, repo_path)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"  Error reading {rel_path}: {e}")
                continue

            try:
                if ext == '.go':
                    pkg_name = extract_package_name(content) or ""
                    file_package_map[rel_path] = pkg_name
                    file_chunks = parse_go(content, rel_path,
                                           package_name=pkg_name)
                elif ext in supported:
                    file_chunks = supported[ext](content, rel_path)
                else:
                    continue

                is_test = is_test_file(rel_path)
                for chunk in file_chunks:
                    chunk.is_test = is_test or is_test_chunk(chunk)

                chunks.extend(file_chunks)
                file_count += 1

            except Exception as e:
                print(f"  Error parsing {rel_path}: {e}")

    print(f"  ✓ Parsed {file_count} files → {len(chunks)} chunks")
    return chunks, file_package_map


def is_test_file(file_path: str) -> bool:
    indicators = ['/test/', '/tests/', '/__tests__/', '/spec/',
                  '_test.', '.test.', '.spec.', 'test_', '_spec.']
    return any(i in file_path.lower() for i in indicators)


def is_test_chunk(chunk: Chunk) -> bool:
    if chunk.name.startswith('test_') or chunk.name.startswith('Test'):
        return True
    return any(d in chunk.decorators
               for d in ['@Test', '@test', '@pytest', '@unittest'])


def index_repository(repo_identifier: str,
                      force_reindex: bool = False,
                      embedding_provider: str = "huggingface",
                      **embedding_kwargs) -> str:

    # Resolve repo name and path
    if repo_identifier.startswith('http'):
        parts     = repo_identifier.rstrip('/').split('/')
        repo_name = f"{parts[-2]}__{parts[-1]}".lower()
        repo_path = f"./repos/{repo_name}"
    else:
        repo_path = repo_identifier
        repo_name = os.path.basename(repo_path.rstrip('/')).lower()

    # Early exit if already indexed
    if not force_reindex:
        vector_db = VectorDB(embedding_provider=embedding_provider,
                             **embedding_kwargs)
        graph = Graph()
        if vector_db.has_repo(repo_name) and graph.has_repo(repo_name):
            print(f"✅ '{repo_name}' already indexed.")
            graph.close()
            return repo_name
        graph.close()

    # Clone if needed
    if repo_identifier.startswith('http') and not os.path.exists(repo_path):
        clone_repo(repo_identifier, repo_path)

    print(f"\n{'='*60}\n📦 Indexing: {repo_name}\n{'='*60}")

    # Go: parse go.mod
    module_name = parse_go_mod(repo_path)

    # Parse all files
    print("📖 Parsing code...")
    chunks, file_package_map = parse_all_files(repo_path, module_name)
    if not chunks:
        print("❌ No chunks extracted.")
        return repo_name

    # Build GoPackageMap
    go_package_map = None
    go_chunks = [c for c in chunks if c.language == "go"]
    if go_chunks:
        print(f"  🐹 Building Go package map ({len(go_chunks)} chunks)...")
        go_package_map = GoPackageMap(module_name)
        go_package_map.build(go_chunks, file_package_map)

    # Graph and vector DB
    graph     = Graph()
    vector_db = VectorDB(embedding_provider=embedding_provider,
                         **embedding_kwargs)

    if force_reindex:
        print("🗑️  Clearing existing data...")
        graph.clear_repo(repo_name)
        vector_db.delete_repo(repo_name)

    # Store graph
    print("🔗 Building graph...")
    graph.store_chunks(chunks, repo_name, go_package_map=go_package_map)

    # Initial vector indexing
    print("\n🔢 Initial embedding...")
    vector_db.index_batch(repo_name, [{
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
    } for c in chunks])

    # ── Phase 2: chunk situating context ─────────────────────────────
    generate_situating_contexts(
        chunks    = chunks,
        repo_name = repo_name,
        driver    = graph.driver,
        vector_db = vector_db,
        force     = force_reindex,
    )

    # ── Phase 3: file summaries ───────────────────────────────────────
    file_summaries = generate_file_summaries(
        chunks    = chunks,
        repo_name = repo_name,
        driver    = graph.driver,
        vector_db = vector_db,
    )

    # ── Phase 4: flow detection and summaries ─────────────────────────
    flow_summaries = generate_flow_summaries(
        repo_name = repo_name,
        driver    = graph.driver,
        vector_db = vector_db,
    )

    # ── Phase 5: package summaries ────────────────────────────────────
    pkg_summaries = generate_package_summaries(
        chunks         = chunks,
        file_summaries = file_summaries,
        repo_name      = repo_name,
        driver         = graph.driver,
        vector_db      = vector_db,
    )

    # ── Phase 6: codebase summary ─────────────────────────────────────
    generate_codebase_summary(
        chunks         = chunks,
        pkg_summaries  = pkg_summaries,
        flow_summaries = flow_summaries,
        repo_name      = repo_name,
        driver         = graph.driver,
        vector_db      = vector_db,
    )

    print(f"\n✅ Done — {len(chunks)} chunks indexed.")
    graph.close()
    return repo_name
