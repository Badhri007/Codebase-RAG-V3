"""Simplified indexer with accurate resolution."""
import os
import subprocess
from typing import List
from chunk import Chunk
from config import EMBEDDING_MODEL
from graph import Graph
from vectordb import VectorDB
from parsers import parse_python, parse_javascript, parse_java, parse_go


def clone_repo(github_url: str, target_dir: str):
    """Clone a GitHub repository."""
    if os.path.exists(target_dir):
        subprocess.run(['rm', '-rf', target_dir])

    print(f"  Cloning {github_url}...")
    subprocess.run(['git', 'clone', github_url, target_dir],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_all_files(repo_path: str) -> List[Chunk]:
    """Parse all supported files in repository."""
    chunks = []
    supported_extensions = {
        '.py': parse_python,
        '.js': parse_javascript,
        '.ts': parse_javascript,
        '.jsx': parse_javascript,
        '.tsx': parse_javascript,
        '.java': parse_java,
        '.go': parse_go
    }

    file_count = 0
    for root, dirs, files in os.walk(repo_path):
        # Skip common ignore directories
        dirs[:] = [d for d in dirs if d not in {
            'node_modules', '.git', '__pycache__', 'venv', 'env',
            'build', 'dist', 'target', '.next', 'vendor'
        }]

        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    parser = supported_extensions[ext]
                    language = ext.lstrip('.')
                    file_chunks = parser(content, rel_path)

                    # Mark test files
                    is_test = is_test_file(rel_path)
                    for chunk in file_chunks:
                        chunk.is_test = is_test or is_test_chunk(chunk)

                    chunks.extend(file_chunks)
                    file_count += 1

                except Exception as e:
                    print(f"  Error parsing {rel_path}: {e}")

    print(f"  ✓ Parsed {file_count} files → {len(chunks)} chunks")
    return chunks


def is_test_file(file_path: str) -> bool:
    """Check if file is a test file."""
    test_indicators = [
        '/test/', '/tests/', '/__tests__/', '/spec/', '/specs/',
        '_test.', '.test.', '.spec.', 'test_', '_spec.'
    ]
    return any(indicator in file_path.lower() for indicator in test_indicators)


def is_test_chunk(chunk: Chunk) -> bool:
    """Check if chunk is a test function/method."""
    if chunk.name.startswith('test_') or chunk.name.startswith('Test'):
        return True

    test_decorators = ['@Test', '@test', '@pytest', '@unittest', '@it', '@describe']
    return any(dec in chunk.decorators for dec in test_decorators)


def index_repository(repo_identifier, force_reindex=False,
                     embedding_provider="huggingface", **embedding_kwargs):
    vector_db = VectorDB(embedding_provider=embedding_provider, **embedding_kwargs)

    # Determine if it's a GitHub URL or local path
    if repo_identifier.startswith('http'):
        # GitHub URL
        repo_name = repo_identifier.rstrip('/').split('/')[-2] + '__' + repo_identifier.rstrip('/').split('/')[-1]
        # IMPORTANT: Lowercase for consistency with vector DB
        repo_name = repo_name.lower()
        repo_path = f"./repos/{repo_name}"
        clone_repo(repo_identifier, repo_path)
    else:
        # Local path
        repo_path = repo_identifier
        repo_name = os.path.basename(repo_path.rstrip('/')).lower()

    print(f"\n{'='*60}")
    print(f"📦 Indexing: {repo_name}")
    print(f"{'='*60}")

    # Initialize connections
    graph = Graph()

    # Clear existing data if force reindex
    if force_reindex:
        print("🗑️  Force re-index: Deleting existing data...")
        graph.clear_repo(repo_name)
        vector_db.delete_repo(repo_name)
        print("  ✓ Deleted existing data")

    # Parse all files
    print("📖 Parsing code...")
    chunks = parse_all_files(repo_path)

    if not chunks:
        print("❌ No chunks extracted!")
        return

    # Store in graph with accurate resolution
    print("🔗 Building graph with accurate resolution...")
    graph.store_chunks(chunks, repo_name)

    # Get statistics
    stats = graph.get_stats(repo_name)
    print(f"\n📊 Graph Statistics:")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Relationships:")
    for rel_type in ["CONTAINS", "HAS_MEMBER", "CALLS", "IMPORTS", "SAME_FILE"]:
        print(f"    {rel_type}: {stats[rel_type]}")

    # Store in vector DB
    print("\n🔢 Indexing in vector database...")
    # Convert chunks to vector DB format
    vector_data = []
    for chunk in chunks:
        vector_data.append({
            'id': chunk.id,
            'text': chunk.embedding_text(),
            'metadata': {
                'name': chunk.name,
                'type': chunk.type,
                'file': chunk.file,
                'is_test': chunk.is_test
            }
        })
    vector_db.index_batch(repo_name, vector_data)
    print(f"  ✓ Indexed {len(chunks)} chunks")

    # Validation
    print("\n✅ Indexing complete!")
    print(f"  Repository: {repo_name}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Test chunks: {sum(1 for c in chunks if c.is_test)}")
    print(f"  Relationships: {sum(stats[k] for k in ['CONTAINS', 'HAS_MEMBER', 'CALLS', 'IMPORTS', 'SAME_FILE'])}")

    graph.close()




if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_indexer.py <github_url_or_path> [--force]")
        sys.exit(1)

    repo = sys.argv[1]
    force = '--force' in sys.argv

    index_repository(repo, force)
