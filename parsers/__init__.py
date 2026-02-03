"""Code parsers for multiple languages."""

from typing import List
from chunk import Chunk
from .parse_go import parse_go
from .parse_javascript import parse_javascript, parse_typescript
from .parse_python import parse_python
from .parse_java import parse_java
from .generic_parser import parse_generic

__all__ = [
    'parse_file',
    'parse_go',
    'parse_javascript',
    'parse_typescript',
    'parse_python',
    'parse_java',
    'parse_generic',
]

LANGUAGE_PARSERS = {
    'go': parse_go,
    'javascript': parse_javascript,
    'js': parse_javascript,
    'jsx': lambda c, f: parse_javascript(c, f, 'jsx'),
    'typescript': parse_typescript,
    'ts': parse_typescript,
    'tsx': lambda c, f: parse_typescript(c, f, 'tsx'),
    'python': parse_python,
    'py': parse_python,
    'java': parse_java,
}


def parse_file(content: str, file_path: str, language: str) -> List[Chunk]:
    """Parse a code file into chunks."""
    language = language.lower()

    if language in LANGUAGE_PARSERS:
        return LANGUAGE_PARSERS[language](content, file_path)

    print(f"  No specific parser for {language}, using generic parser")
    return parse_generic(content, file_path, language)
