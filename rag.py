"""Enhanced RAG pipeline with Neo4j + Hierarchical + Contextual."""
import re
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

from config import REPOS_DIR, MAX_CONTEXT_TOKENS, DIAGRAM_KEYWORDS
from core import GitManager, read_file, get_language, is_test_file
from chunk import Chunk
from parsers import parse_file
from graph import Neo4jDependencyGraph
from vectordb import VectorDB
from llm import get_llm
from bm25 import BM25Retriever
from hierarchical_chunking import (
    HierarchicalChunker,
    ChunkingStrategy
)
from repo_analyzer import RepositoryAnalyzer, RepoContext
from query_analyzer import QueryAnalyzer, QueryType

load_dotenv()


class RAGPipeline:
    def __init__(self,
                  llm_provider: str = "claude",
                  llm_model: str = None,
                  use_neo4j: bool = True,
                  use_contextual_retrieval: bool = True,
                  use_llm_context: bool = False,
                  chunking_strategy: ChunkingStrategy = None):

        print(f"🚀 Initializing Enhanced RAG Pipeline...")

        # Core components
        self.git = GitManager()
        self.db = VectorDB()
        self.bm25 = BM25Retriever()
        self.llm = get_llm(llm_provider, llm_model)
        self.llm_name = llm_provider

        # Graph database
        self.use_neo4j = use_neo4j
        if use_neo4j:
            try:
                self.graph = Neo4jDependencyGraph()
            except Exception as e:
                print(f"⚠️ Neo4j unavailable: {e}")
                self.graph = None
                self.use_neo4j = False
        else:
            self.graph = None

        # Chunking strategy
        if chunking_strategy is None:
            chunking_strategy = ChunkingStrategy(
                use_contextual_retrieval=use_contextual_retrieval,
                use_llm_for_context=use_llm_context,
                llm_provider=llm_provider,
                max_chunk_size=2000,
                create_file_summary=True,
                create_class_summary=True,
            )

        self.chunking_strategy = chunking_strategy
        self.chunker = HierarchicalChunker(chunking_strategy)

        # Repository and Query Analyzers
        self.repo_analyzer = RepositoryAnalyzer()
        self.query_analyzer = QueryAnalyzer(self.llm, use_llm=True)
        self.repo_context: Optional[RepoContext] = None

        # State
        self.current_repo: Optional[str] = None
        self.history: List[Dict] = []

        print(f"✓ Ready!")
        print(f"  LLM: {self.llm.name}")
        print(f"  Graph: {'Neo4j' if self.use_neo4j else 'In-memory'}")
        print(f"  Contextual: {use_contextual_retrieval}")
        print(f"  LLM Context: {use_llm_context}")
        print(f"  Query Analyzer: Enabled")
        print(f"  Repo Analyzer: Enabled")

    def index(self, repo_url: str, force: bool = False):
        repo_name = self.git.extract_repo_name(repo_url)
        print(f"\n{'='*60}")
        print(f"Indexing: {repo_name}")
        print(f"{'='*60}")

        # Check if already indexed
        if not force and self.db.has_repo(repo_name):
            stats = self.db.get_stats(repo_name)
            print(f"✓ Already indexed ({stats['chunks']} chunks)")
            print("  Use force=True to re-index")
            self.current_repo = repo_name
            if self.graph:
                self.graph.load(repo_name)
            return

        # Clone/pull
        print("\n📥 Cloning repository...")
        repo_path = self.git.clone_or_pull(repo_url, REPOS_DIR, force)

        # Get files
        print("\n📂 Scanning files...")
        files = self.git.get_files(repo_path)
        print(f"  Found {len(files)} files")

        # Parse with enhancements
        print("\n Parsing with AST")
        all_chunks: List[Chunk] = []
        stats = {
            'files_processed': 0,
            'files_enhanced': 0,
            'contextual_chunks': 0,
            'regular_chunks': 0,
            'file_summaries': 0,
            'class_summaries': 0,
            'split_functions': 0,
        }

        for file_path in files:
            try:
              full_path = Path(repo_path) / file_path
              content = read_file(str(full_path))

              if not content:
                  continue

              stats['files_processed'] += 1
              language = get_language(file_path)

              # Parse with AST
              chunks = parse_file(content, file_path, language)

              # Mark test chunks
              is_test = is_test_file(file_path)
              for chunk in chunks:
                  chunk.is_test = is_test

              if len(content) > 1000:
                  stats['files_enhanced'] += 1

                  enhanced = self.chunker.enhance_chunks(
                      chunks, file_path, language
                  )

                  # Count chunk types
                  for chunk in enhanced:
                      if chunk.type == 'file_summary':
                          stats['file_summaries'] += 1
                      elif chunk.type == 'class_summary':
                          stats['class_summaries'] += 1


                      if hasattr(chunk, 'contextual_embedding_text') and chunk.contextual_embedding_text:
                          stats['contextual_chunks'] += 1
                      else:
                          stats['regular_chunks'] += 1

                  chunks = enhanced

              all_chunks.extend(chunks)

            except Exception as e:
              print(f"  ⚠️ Failed to parse {file_path}: {e}")
              continue

        if stats['files_processed'] % 50 == 0:
            print(f"  Processed {stats['files_processed']}/{len(files)} files...")

        print(f"\n📊 Parsing Statistics:")
        print(f"  Files: {stats['files_processed']}")
        print(f"  Enhanced: {stats['files_enhanced']}")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"    - Contextual: {stats['contextual_chunks']}")
        print(f"    - Regular: {stats['regular_chunks']}")
        print(f"    - File summaries: {stats['file_summaries']}")
        print(f"    - Class summaries: {stats['class_summaries']}")
        print(f"    - Split parts: {stats['split_functions']}")

        if not all_chunks:
            print("⚠️ No chunks found!")
            return

        print("Chunks : ", all_chunks[:4])

        print("\n📊 Analyzing repository structure...")

        self.repo_context = self.repo_analyzer.analyze(all_chunks, repo_name, all_scanned_files=files)

        # Create overview chunk
        overview_chunk = self.repo_analyzer.create_overview_chunk(self.repo_context)
        all_chunks.append(overview_chunk)

        print(f"  ✓ Repository analysis complete")
        print(f"    - Architecture: {self.repo_context.architecture_pattern}")
        print(f"    - Entry points: {len(self.repo_context.entry_points)}")
        print(f"    - Components: {len(self.repo_context.components)}")

        # Build graph
        print("\n🔗 Building dependency graph...")
        if self.graph:
            self.graph.build(all_chunks, repo_name)

        # Index embeddings
        print("\n Indexing embeddings...")
        self.db.index(repo_name, all_chunks)

        # Index BM25
        print("\n Building BM25 index...")
        self.bm25.index(all_chunks)
        self.bm25.save(repo_name)

        self.current_repo = repo_name
        self.export_repo_context(f"docs/{repo_name}_structure.json")

        print(f"\n✅ Indexing complete!")
        print(f"  Chunks: {len(all_chunks)}")
        print(f"  Contextual: {stats['contextual_chunks']}")

    def ask(self, question: str,
            n_chunks: int = 25,
            explain: bool = False) -> Dict:
        """
        Answer question with enhanced retrieval.

        Args:
            question: User question
            n_chunks: Number of chunks to retrieve
            explain: Show retrieval process

        Returns:
            Dict with 'answer' and optional 'diagram'
        """
        if not self.current_repo:
            raise ValueError("No repository loaded. Call index() first.")

        query_analysis = self.query_analyzer.analyze(question)

        if explain:
            print(f"\n🔍 Query Analysis:")
            print(f"  Type: {query_analysis.query_type.value}")
            print(f"  Confidence: {query_analysis.confidence:.2f}")
            print(f"  Needs repo context: {query_analysis.needs_repo_context}")
            print(f"  Include tests: {query_analysis.include_tests}")

        # Get optimal retrieval config
        retrieval_config = self.query_analyzer.get_retrieval_config(query_analysis)
        n_chunks = retrieval_config['k']

        if explain:
            print(f"\n🔍 Retrieval Process:")
            print(f"  Target: {n_chunks} chunks")

        # 1. Multi-query hybrid search with rewritten queries
        if explain:
            print(f"\n1️⃣ Multi-Query Hybrid Search...")
            print(f"  Using {len(query_analysis.rewritten_queries)} query variations")

        all_search_results = []
        for rewritten_query in query_analysis.rewritten_queries[:3]:
            results = self.hybrid_search(
                self.current_repo,
                rewritten_query,
                k=n_chunks // 2,
                include_tests=retrieval_config['include_tests']
            )
            all_search_results.extend(results)

        # Deduplicate and merge scores
        seen_ids = {}
        for result in all_search_results:
            chunk_id = result['id']
            if chunk_id in seen_ids:
                # Average the scores for duplicate chunks
                seen_ids[chunk_id]['score'] = (seen_ids[chunk_id]['score'] + result['score']) / 2
            else:
                seen_ids[chunk_id] = result

        search_results = sorted(seen_ids.values(), key=lambda x: x['score'], reverse=True)[:n_chunks]

        if not search_results:
            return {
                'answer': "No relevant code found.",
                'diagram': None
            }

        if explain:
            print(f"  Found {len(search_results)} seeds")
            for i, r in enumerate(search_results[:3], 1):
                print(f"    {i}. {r['file']}::{r['name']} ({r['score']:.2f})")

        # 2. Graph traversal
        seed_ids = [r['id'] for r in search_results[:10]]

        if explain:
            print(f"\n2️⃣ Graph Traversal...")

        if self.graph and seed_ids:
            traversed = self.graph.traverse(
                seed_ids,
                max_chunks=n_chunks,
                max_tokens=MAX_CONTEXT_TOKENS // 2,
            )

            if explain:
                print(f"  Retrieved {len(traversed)} chunks")

            # Merge with search results
            seen_ids = {c.id for c in traversed}
            chunks = traversed

            for r in search_results:
                if r['id'] not in seen_ids and r['id'] in self.graph.chunks:
                    chunks.append(self.graph.chunks[r['id']])
                    seen_ids.add(r['id'])
                    if len(chunks) >= n_chunks:
                        break
        else:
            chunks = []
            for r in search_results[:n_chunks]:
                if self.graph and r['id'] in self.graph.chunks:
                    chunks.append(self.graph.chunks[r['id']])

        if explain:
            print(f"  Final: {len(chunks)} chunks")

        # 3. Build context with repo overview if needed
        if explain:
            print(f"\n3️⃣ Building Context...")

        context = self._build_context_smart(chunks)

        # Add repository overview for architectural queries
        if query_analysis.needs_repo_context:
            repo_overview = self._get_repo_overview()
            if repo_overview and explain:
                print(f"  Adding repository overview context")
            context = f"{repo_overview}\n\n{'='*60}\n\n{context}"

        if explain:
            print(f"  Size: ~{len(context) // 4} tokens")

        # 4. Create prompt
        needs_diagram = any(re.search(p, question.lower())
                          for p in DIAGRAM_KEYWORDS)
        prompt = self._build_prompt(context, question, needs_diagram)

        # 5. Query LLM
        if explain:
            print(f"\n4️⃣ Querying LLM...")

        messages = self.history[-6:] + [{"role": "user", "content": prompt}]
        answer = self.llm.chat(messages)

        # 6. Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({
            "role": "assistant",
            "content": answer[:500] + "..." if len(answer) > 500 else answer
        })
        if len(self.history) > 10:
            self.history = self.history[-10:]

        # 7. Parse response
        return self._parse_response(answer)

    def hybrid_search(self, repo_name: str, query: str,
                      k: int = 25, alpha: float = 0.6, include_tests: bool = False) -> List[Dict]:
        """Hybrid search with contextual boost and test filtering."""

        # Vector search
        vector_results = self.db.search(repo_name, query, n=k*2)

        # BM25 search
        bm25_scores = self.bm25.get_scores_only(query, k=k*2)

        # Normalize
        def normalize(scores_dict):
            if not scores_dict:
                return {}
            max_score = max(scores_dict.values())
            if max_score == 0:
                return scores_dict
            return {k: v/max_score for k, v in scores_dict.items()}

        vector_scores = {r['id']: r['score'] for r in vector_results}
        vector_scores = normalize(vector_scores)
        bm25_scores = normalize(bm25_scores)

        # Combine
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined = {}

        for chunk_id in all_ids:
            v_score = vector_scores.get(chunk_id, 0)
            b_score = bm25_scores.get(chunk_id, 0)

            score = alpha * v_score + (1 - alpha) * b_score

            # Boost contextual chunks
            if self.graph and chunk_id in self.graph.chunks:
                chunk = self.graph.chunks[chunk_id]
                if hasattr(chunk, 'contextual_embedding_text') and chunk.contextual_embedding_text:
                    score *= 1.05

                # Boost summaries
                if 'summary' in chunk.type:
                    score *= 1.1

                # Filter/penalize test chunks
                if not include_tests and chunk.is_test:
                    score *= 0.2

            combined[chunk_id] = score

        # Sort
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        # Format
        chunk_map = {r['id']: r for r in vector_results}
        results = []

        for chunk_id, score in ranked:
            if chunk_id in chunk_map:
                result = chunk_map[chunk_id].copy()
                result['score'] = score
                result['vector_score'] = vector_scores.get(chunk_id, 0)
                result['bm25_score'] = bm25_scores.get(chunk_id, 0)
                results.append(result)

        return results

    def _build_context_smart(self, chunks: List[Chunk]) -> str:
        """Build context with smart ordering."""
        parts = []
        tokens = 0

        # Separate by type
        summaries = [c for c in chunks if 'summary' in c.type]
        code_chunks = [c for c in chunks if 'summary' not in c.type]

        # Add summaries first
        for chunk in summaries:
            context = chunk.llm_context()
            chunk_tokens = len(context) // 4

            if tokens + chunk_tokens > MAX_CONTEXT_TOKENS // 4:
                break

            parts.append(context)
            tokens += chunk_tokens

        # Add code
        for chunk in code_chunks:
            context = chunk.llm_context()
            chunk_tokens = len(context) // 4

            if tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
                break

            parts.append(context)
            tokens += chunk_tokens

        return "\n\n".join(parts)

    def _build_prompt(self, context: str, question: str,
                     needs_diagram: bool) -> str:
        """Build prompt for LLM."""
        diagram_instruction = ""
        if needs_diagram:
            diagram_instruction = """

**Create a Mermaid diagram** to visualize:
- Use: flowchart TD, sequenceDiagram, or classDiagram
- Put in ```mermaid code block
- Clear labels and relationships"""

        return f"""Based on this code from the repository, answer the question.

## Code Context
{context}

## Question
{question}

## Instructions
- Answer based on the provided code
- Reference specific files and line numbers
- Explain the approach/flow clearly
- If information is missing, say so{diagram_instruction}"""

    def _parse_response(self, answer: str) -> Dict:
        """Extract diagram from response."""
        pattern = r'```mermaid\n(.*?)```'
        matches = re.findall(pattern, answer, re.DOTALL)

        if matches:
            diagram = matches[0].strip()
            clean_answer = re.sub(pattern, '', answer, flags=re.DOTALL).strip()
            return {'answer': clean_answer, 'diagram': diagram}

        return {'answer': answer, 'diagram': None}

    def switch_llm(self, provider: str, model: str = None):
        """Switch to a different LLM provider."""
        self.llm = get_llm(provider, model)
        self.llm_name = provider
        print(f"✓ Switched to {self.llm.name}")

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def list_repos(self) -> List[str]:
        """List indexed repositories."""
        return self.db.list_repos()

    def delete_repo(self, name: str):
        """Delete a repository."""
        self.db.delete_repo(name)
        if self.graph:
            self.graph.delete(name)
        self.bm25.delete(name)
        if self.current_repo == name:
            self.current_repo = None

    def load(self, repo_name: str) -> bool:
        """Load a previously indexed repository."""
        if not self.db.has_repo(repo_name):
            print(f"Repository not found: {repo_name}")
            return False

        self.current_repo = repo_name

        if self.graph:
            self.graph.load(repo_name)

        self.bm25.load(repo_name)

        print(f"✓ Loaded: {repo_name}")
        return True

    def _get_repo_overview(self) -> str:
        """Get repository overview text."""
        if self.repo_context and self.graph:
            # Look for overview chunk
            for chunk_id, chunk in self.graph.chunks.items():
                if chunk.type == 'repo_overview':
                    return chunk.code
        return ""

    def export_repo_context(self, filepath: str = None) -> str:
        if not self.repo_context:
            raise ValueError("No repository context available. Index a repository first.")

        if filepath:
            self.repo_context.save_to_file(filepath)
            return f"Saved to {filepath}"
        else:
            return self.repo_context.to_json()

    def close(self):
        if self.graph:
            self.graph.close()
