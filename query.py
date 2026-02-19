from typing import List, Dict
from config import EMBEDDING_MODEL
from graph import Graph
from vectordb import VectorDB
from llm.providers import get_llm
import os


def answer_query(query, repo_name, llm_provider="claude", llm_model=None,
                 embedding_provider="huggingface", **embedding_kwargs):

    """
    Answer a query about a repository.

    Simple flow:
    1. Vector search for relevant chunks
    2. Filter test files (unless requested)
    3. Graph BFS expansion
    4. LLM synthesis
    """
    print(f"\n{'='*60}")
    print(f"❓ Question: {query}")
    print(f"{'='*60}\n")

    # Initialize
    graph = Graph()

    llm = get_llm(llm_provider, model=llm_model)
    vector_db = VectorDB(embedding_provider=embedding_provider, **embedding_kwargs)

    # Step 1: Vector search
    print("🔍 Vector search...")
    print(f"  Searching in repo: {repo_name}")
    print(f"  Query: {query}")

    seed_chunks = vector_db.search(repo_name, query, k=20)
    print(f"  Found {len(seed_chunks)} seed chunks")

    if seed_chunks:
        print(f"  Sample result: {seed_chunks[0]}")

    if not seed_chunks:
        # Debug: Check if repo exists
        if vector_db.has_repo(repo_name):
            print(f"  ⚠️ Repo exists but no results. Trying broader search...")
            # Try without filters
            seed_chunks = vector_db.search(repo_name, query, k=20, filters=None)
            print(f"  Broader search found: {len(seed_chunks)} chunks")

        if not seed_chunks:
            return "❌ No relevant code found for your query."

    # Step 2: Filter tests (unless query explicitly asks for tests)
    include_tests = any(word in query.lower() for word in ['test', 'testing', 'spec', 'unittest'])

    if not include_tests:
        before = len(seed_chunks)
        seed_chunks = [c for c in seed_chunks if not c.get('is_test', False)]
        if len(seed_chunks) < before:
            print(f"  Filtered {before - len(seed_chunks)} test chunks")

    # Step 3: Graph expansion
    print("🔗 Expanding through graph...")
    seed_ids = [c['id'] for c in seed_chunks]
    expanded_chunks = graph.expand_bfs(seed_ids, repo_name, max_tokens=4000)
    print(f"  Expanded to {len(expanded_chunks)} chunks")

    if not expanded_chunks:
        expanded_chunks = seed_chunks  # Fallback to seeds

    # Step 4: Format context for LLM
    context_parts = []
    for i, chunk_data in enumerate(expanded_chunks[:30], 1):  # Limit to 30 chunks
        context_parts.append(f"""
## Chunk {i}: {chunk_data.get('name', 'Unknown')}
**Type:** {chunk_data.get('type', 'unknown')}
**File:** {chunk_data.get('file', 'unknown')} (lines {chunk_data.get('start', '?')}-{chunk_data.get('end', '?')})
**Signature:** {chunk_data.get('signature', 'N/A')}

```{chunk_data.get('language', 'python')}
{chunk_data.get('code', '')}
```
""")

    context = "\n".join(context_parts)

    # Step 5: LLM synthesis
    print("🤖 Synthesizing answer...")

    prompt = f"""You are a code expert helping developers understand a codebase.

Given the following code context from the repository, answer the user's question accurately and concisely.

**User Question:** {query}

**Code Context:**
{context}

**Instructions:**
- Answer the question directly and accurately
- Reference specific files, functions, or classes when relevant
- Include code snippets if helpful
- If the answer isn't in the provided context, say so

**Answer:**"""

    answer = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

    print("✅ Answer generated\n")

    graph.close()

    return answer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python simple_query.py <repo_name> <query>")
        sys.exit(1)

    repo = sys.argv[1]
    query = " ".join(sys.argv[2:])

    answer = answer_query(query, repo)
    print(f"\n{'='*60}")
    print("📝 Answer:")
    print(f"{'='*60}")
    print(answer)
