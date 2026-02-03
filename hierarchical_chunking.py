from typing import List, Dict, Optional
from dataclasses import dataclass
from parsers.utils import *

@dataclass
class ChunkingStrategy:
    """Configuration for hierarchical + contextual chunking."""
    max_chunk_size: int = 2000
    min_chunk_size: int = 500
    overlap_size: int = 200
    create_file_summary: bool = True
    create_class_summary: bool = True
    use_contextual_retrieval: bool = True
    use_llm_for_context: bool = False
    llm_provider: str = "claude"


class HierarchicalChunker:
    """Creates hierarchical chunks with contextual retrieval."""

    def __init__(self, strategy: ChunkingStrategy = None):
        self.strategy = strategy or ChunkingStrategy()
        self.llm = None
        if self.strategy.use_llm_for_context:
            try:
                from llm import get_llm
                self.llm = get_llm(self.strategy.llm_provider)
            except Exception as e:
                print(f"  Warning: Could not initialize LLM for context: {e}")

    def enhance_chunks(self, chunks: List[Chunk], file_path: str, language: str) -> List[Chunk]:
        """Enhance chunks with hierarchy + context."""
        enhanced = list(chunks)

        if self.strategy.create_file_summary:
            if file_chunk := self._create_file_chunk(chunks, file_path, language):
                enhanced.append(file_chunk)

        if self.strategy.create_class_summary:
            enhanced.extend(self._create_class_chunks(chunks))

        if self.strategy.use_contextual_retrieval:
            enhanced = self._add_contextual_information(enhanced, file_path, language)

        return enhanced

    def _create_file_chunk(self, chunks: List[Chunk], file_path: str, language: str) -> Optional[Chunk]:
        """Create file-level summary chunk."""
        if not chunks:
            return None

        classes = [c for c in chunks if c.type in ('class', 'interface', 'struct')]
        functions = [c for c in chunks if c.type == 'function']
        methods = [c for c in chunks if c.type == 'method']

        summary_parts = [f"File: {file_path}", f"Language: {language}"]

        if classes:
            summary_parts.append(f"\nClasses ({len(classes)}):")
            for cls in classes:
                sig = cls.signature or f"class {cls.name}"
                doc = cls.docstring.split('\n')[0][:80] if cls.docstring else ""
                summary_parts.append(f"  - {sig}" + (f": {doc}" if doc else ""))

        if functions:
            summary_parts.append(f"\nFunctions ({len(functions)}):")
            for func in functions:
                sig = func.signature or f"function {func.name}"
                doc = func.docstring.split('\n')[0][:80] if func.docstring else ""
                summary_parts.append(f"  - {sig}" + (f": {doc}" if doc else ""))

        all_imports = set()
        for c in chunks:
            all_imports.update(c.imports)

        if all_imports:
            summary_parts.append(f"\nImports: {', '.join(list(all_imports)[:20])}")

        return Chunk(
            id=f"{file_path}::file::summary",
            name=file_path.split('/')[-1],
            type='file_summary',
            file=file_path,
            start=min(c.start for c in chunks),
            end=max(c.end for c in chunks),
            language=language,
            code='\n'.join(summary_parts),
            calls=[],
            imports=list(all_imports),
            docstring=f"File summary: {len(classes)} classes, {len(functions)} functions, {len(methods)} methods"
        )

    def _create_class_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Create class-level summary chunks."""
        class_summaries = []
        classes = [c for c in chunks if c.type in ('class', 'interface', 'struct')]

        for cls in classes:
            methods = [c for c in chunks if c.parent == cls.id]
            if not methods:
                continue

            summary_parts = [f"Class: {cls.name}", f"Type: {cls.type}"]

            if cls.signature:
                summary_parts.append(f"Signature: {cls.signature}")
            if cls.docstring:
                summary_parts.append(f"Description: {cls.docstring[:300]}")
            if cls.imports:
                summary_parts.append(f"Extends/Implements: {', '.join(cls.imports)}")

            summary_parts.append(f"\nMethods ({len(methods)}):")
            for method in methods[:20]:
                sig = method.signature or f"{method.name}()"
                doc = method.docstring.split('\n')[0][:100] if method.docstring else ""
                summary_parts.append(f"  - {sig}" + (f": {doc}" if doc else ""))

            class_summaries.append(Chunk(
                id=f"{cls.id}::summary",
                name=f"{cls.name}_summary",
                type='class_summary',
                file=cls.file,
                start=cls.start,
                end=cls.end,
                language=cls.language,
                code='\n'.join(summary_parts),
                calls=[],
                imports=cls.imports,
                parent=cls.id,
                docstring=f"Summary of {cls.name} with {len(methods)} methods"
            ))

        return class_summaries

    def _add_contextual_information(self, chunks: List[Chunk], file_path: str, language: str) -> List[Chunk]:
        """Add situating context to each chunk."""
        chunk_map = {c.id: c for c in chunks}
        file_summary = next((c for c in chunks if c.type == 'file_summary'), None)
        class_summaries = {c.parent: c for c in chunks if c.type == 'class_summary' and c.parent}

        contextualized = []
        for chunk in chunks:
            context = self._build_situating_context(chunk, file_summary, class_summaries, chunk_map)
            chunk = self._add_context_to_chunk(chunk, context)
            contextualized.append(chunk)

        return contextualized

    def _build_situating_context(self, chunk: Chunk, file_summary: Optional[Chunk],
                                 class_summaries: Dict[str, Chunk], chunk_map: Dict[str, Chunk]) -> str:
        """Build rich situating context for a chunk."""
        context_parts = [
            f"Location: {chunk.file}",
            f"Language: {chunk.language}",
            f"Type: {chunk.type}",
            f"Lines: {chunk.start}-{chunk.end}"
        ]

        if chunk.parent and (parent_chunk := chunk_map.get(chunk.parent)):
            context_parts.append(f"Parent: This is a {chunk.type} inside {parent_chunk.name} {parent_chunk.type}")
            if parent_chunk.docstring:
                context_parts.append(f"Parent purpose: {parent_chunk.docstring.split(chr(10))[0][:100]}")

        if file_summary:
            for line in file_summary.code.split('\n'):
                if 'Classes' in line or 'Functions' in line:
                    context_parts.append(f"File context: This file contains {line.strip()}")
                    break

        if chunk.parent and (class_summary := class_summaries.get(chunk.parent)):
            purpose = class_summary.docstring.split('\n')[0][:100] if class_summary.docstring else "contains related methods"
            context_parts.append(f"Class context: Part of {class_summary.name} which {purpose}")

        if chunk.calls:
            context_parts.append(f"Calls: {', '.join(chunk.calls[:10])}")
        if chunk.imports:
            context_parts.append(f"Uses: {', '.join(chunk.imports[:10])}")

        if chunk.docstring:
            context_parts.append(f"Purpose: {chunk.docstring.split(chr(10))[0][:150]}")
        elif chunk.signature:
            context_parts.append(f"Signature: {chunk.signature}")

        if self.strategy.use_llm_for_context and self.llm:
            if semantic := self._generate_llm_context(chunk, file_summary, class_summaries):
                context_parts.append(f"Semantic role: {semantic}")

        return "\n".join(context_parts)

    def _generate_llm_context(self, chunk: Chunk, file_summary: Optional[Chunk],
                              class_summaries: Dict[str, Chunk]) -> str:
        """Use LLM to generate rich semantic context."""
        if not self.llm:
            return ""

        prompt_parts = []
        if file_summary:
            prompt_parts.append(f"File overview:\n{file_summary.code[:500]}")
        if chunk.parent and (class_sum := class_summaries.get(chunk.parent)):
            prompt_parts.append(f"\nClass overview:\n{class_sum.code}")
        prompt_parts.append(f"\nCode to analyze:\n{chunk.code}")

        prompt = f"""Given this code context, provide a 2-3 sentence description of what this code does and its role in the codebase. Focus on:
- What problem does it solve?
- How does it fit into the larger system?
- What are its key responsibilities?

Keep it concise and factual. No code, just explanation.

{chr(10).join(prompt_parts)}

Description:"""

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
            return response.strip()[:300]
        except Exception as e:
            print(f"  Warning: LLM context generation failed: {e}")
            return ""

    def _add_context_to_chunk(self, chunk: Chunk, context: str) -> Chunk:
        """Add contextual embedding text to chunk."""
        chunk.contextual_embedding_text = f"""<context>
{context}
</context>

<chunk_info>
Name: {chunk.name}
Type: {chunk.type}
File: {chunk.file}
</chunk_info>

<code>
{chunk.code}
</code>"""
        chunk.situating_context = context
        return chunk

