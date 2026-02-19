"""Enhanced Chunk dataclass with contextual retrieval support."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Chunk:
    """
    Enhanced code chunk with contextual retrieval support.

    A chunk represents a code element (function, class, method, etc.)
    with all necessary context for retrieval and understanding.

    New features:
    - situating_context: Rich context for better retrieval
    - contextual_embedding_text: Context + code for embedding
    - use_contextual_embedding: Enable/disable contextual embeddings
    """

    # Core identity
    id: str                          # Unique: file::type::name
    name: str                        # Function/class/method name
    type: str                        # function, method, class, etc.

    # Location
    file: str                        # Relative file path
    start: int                       # Start line (1-indexed)
    end: int                         # End line (1-indexed)
    language: str                    # python, javascript, java, go

    # Code
    code: str                        # Actual source code

    # Relationships (for graph building)
    calls: List[str] = field(default_factory=list)      # Functions called
    imports: List[str] = field(default_factory=list)    # Imports used
    parent: Optional[str] = None                         # Parent class/struct

    # Enhanced resolution data (NEW)
    imports_map: Dict[str, Dict[str, str]] = field(default_factory=dict)  # {name: {from: path, name: actual}}
    type_map: Dict[str, str] = field(default_factory=dict)  # {var_name: type_name}
    calls_with_context: List[Dict[str, str]] = field(default_factory=list)  # [{name, receiver, receiver_type}]

    # Context (for better retrieval)
    docstring: Optional[str] = None                      # Documentation
    signature: Optional[str] = None                      # Function signature
    decorators: List[str] = field(default_factory=list) # Decorators
    params: List[Dict[str, str]] = field(default_factory=list)  # Parameters
    returns: Optional[str] = None                        # Return type

    # Contextual retrieval (NEW)
    situating_context: Optional[str] = None              # Rich context
    contextual_embedding_text: Optional[str] = None      # Context + code
    use_contextual_embedding: bool = True                # Use context?

    # Test file detection
    is_test: bool = False                                 # Is this a test chunk?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'file': self.file,
            'start': self.start,
            'end': self.end,
            'language': self.language,
            'code': self.code,
            'calls': self.calls,
            'imports': self.imports,
            'parent': self.parent,
            'docstring': self.docstring,
            'signature': self.signature,
            'decorators': self.decorators,
            'params': self.params,
            'returns': self.returns,
            'situating_context': self.situating_context,
            'use_contextual_embedding': self.use_contextual_embedding
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Chunk':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    def context_string(self) -> str:
        """
        Rich context for embedding (legacy method).

        Note: For contextual retrieval, use embedding_text() instead.
        """
        parts = [f"{self.type}: {self.name}", f"File: {self.file}"]

        if self.signature:
            parts.append(f"Signature: {self.signature}")
        if self.docstring:
            doc = self.docstring[:300] + "..." if len(self.docstring) > 300 else self.docstring
            parts.append(f"Description: {doc}")
        if self.decorators:
            parts.append(f"Decorators: {', '.join(self.decorators)}")
        if self.params:
            params_str = ', '.join(f"{p['name']}: {p.get('type', 'any')}" for p in self.params)
            parts.append(f"Parameters: {params_str}")
        if self.returns:
            parts.append(f"Returns: {self.returns}")
        if self.calls:
            parts.append(f"Calls: {', '.join(self.calls[:10])}")
        if self.parent:
            parts.append(f"In: {self.parent.split('::')[-1]}")

        return '\n'.join(parts)

    def embedding_text(self) -> str:
        """
        Get text for embedding.

        If contextual retrieval is enabled and context exists,
        returns contextual_embedding_text. Otherwise falls back
        to regular context + code.
        """
        if self.use_contextual_embedding and self.contextual_embedding_text:
            return self.contextual_embedding_text

        # Fallback: regular embedding
        return f"{self.context_string()}\n\n{self.code}"

    def llm_context(self) -> str:
        """
        Format for LLM context window.

        Now includes situating context if available for better
        understanding by the LLM.
        """
        parts = []

        # Add situating context first (if available)
        if self.situating_context:
            parts.append(f"<context>\n{self.situating_context}\n</context>\n")

        # Add standard header
        header = f"File: {self.file} (lines {self.start}-{self.end}) [{self.language}]"
        if self.signature:
            header += f"\n{self.type}: {self.signature}"
        if self.docstring:
            doc_preview = self.docstring[:200]
            header += f"\nDoc: {doc_preview}"

        parts.append(header)
        parts.append(f"```{self.language}\n{self.code}\n```")

        return "\n".join(parts)
