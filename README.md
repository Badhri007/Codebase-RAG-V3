# 🔍 Codebase RAG

AI-powered code understanding with AST parsing and dependency graph traversal.

## Features

- **Full AST Parsing**: Python, JavaScript/JSX, TypeScript/TSX, Java, Go
- **Dependency Graph**: Tracks function calls, class relationships
- **Smart Retrieval**: Vector search + graph traversal for complete context
- **Multi-LLM**: Claude, GPT-4, Gemini, DeepSeek
- **Auto Diagrams**: Mermaid flowcharts, sequence diagrams

## Quick Start

Run this while starting neo4j

sudo chown -R $USER:$(id -g) .

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys

# UI
streamlit run app.py

# CLI
python cli.py index https://github.com/owner/repo
python cli.py ask "How does authentication work?"
python cli.py chat
```

## Project Structure

```
├── app.py              # Streamlit UI
├── cli.py              # Command line
├── config.py           # Settings
├── core.py             # Git operations
├── chunk.py            # Single Chunk dataclass
├── parsers/
│   ├── __init__.py     # Parser factory
│   ├── parse_python.py    # Python AST
│   ├── parse_javascript.py # JS/TS/JSX (tree-sitter)
│   ├── parse_java.py      # Java (tree-sitter)
│   ├── parse_go.py        # Go (tree-sitter)
│   └── generic_parser.py   # Fallback
├── graph.py            # Dependency graph
├── vectordb.py         # ChromaDB + embeddings
├── llm.py              # LLM providers
└── rag.py              # Main pipeline
|__ bm25.py             # Best Matching Index and Retreiver
```

## Core Data Structure

Single `Chunk` dataclass used everywhere:

```python
@dataclass
class Chunk:
    id: str           # file::type::name
    name: str         # function/class name
    type: str         # function, method, class, struct, interface
    file: str         # file path
    start: int        # start line
    end: int          # end line
    language: str     # python, javascript, java, go
    code: str         # source code

    # For graph building
    calls: List[str]      # called function names
    imports: List[str]    # imported modules
    parent: str           # parent class/struct

    # For better retrieval
    docstring: str        # docstring/javadoc
    signature: str        # function signature
    decorators: List[str] # @annotations
    params: List[Dict]    # parameters with types
    returns: str          # return type
```

## AST Parsing

| Language | Parser | Features |
|----------|--------|----------|
| Python | Built-in `ast` | Classes, functions, methods, decorators, type hints |
| JavaScript/JSX | tree-sitter | Classes, functions, arrow functions, JSDoc |
| TypeScript/TSX | tree-sitter | + interfaces, type aliases, generics |
| Java | tree-sitter | Classes, interfaces, methods, Javadoc, annotations |
| Go | tree-sitter | Structs, interfaces, functions, methods, doc comments |
| Others | Regex fallback | Basic function/class detection |

## Retrieval Flow

```
Query → Vector Search (top 10) → Seed Chunks
                                      ↓
                               Graph Traversal
                              (callees + callers)
                                      ↓
                              Merged Context
                                      ↓
                                 LLM Response
```

## Diagram Triggers

Ask questions containing: "flow", "diagram", "architecture", "how does X work", "sequence", "process"

Examples:
- "How does user authentication work?"
- "Show the data flow for API requests"
- "Explain the class structure"
- "What's the sequence for order processing?"

## Environment Variables

```
ANTHROPIC_API_KEY=...   # Required for Claude
OPENAI_API_KEY=...      # For GPT-4
GOOGLE_API_KEY=...      # For Gemini
DEEP