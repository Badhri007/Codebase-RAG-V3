import re
from config import CHUNK_SIZE
from parsers.utils import *

SKIP_GENERIC = {'if', 'else', 'for', 'while', 'return', 'function', 'class', 'true', 'false', 'null'}


def parse_generic(content, file_path, language):
    """Simple boundary-based chunking for unsupported languages."""
    lines = content.split('\n')
    make_id = ParserUtils.make_id_generator()
    chunks = []

    # Find function/class boundaries
    patterns = [
        r'^(?:def|function|func|fn|sub)\s+([a-zA-Z_]\w*)',
        r'^(?:class|struct|interface)\s+([a-zA-Z_]\w*)',
    ]

    boundaries = [0]
    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.match(pattern, line.strip(), re.I):
                if i > 0:
                    boundaries.append(i)
                break
    boundaries.append(len(lines))

    # Create chunks from boundaries
    if len(boundaries) > 2:
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            code = '\n'.join(lines[start:end])
            if len(code.strip()) < 50:
                continue

            # Extract name from first line
            name = 'block'
            for pattern in patterns:
                if match := re.search(pattern, lines[start], re.I):
                    name = match.group(1)
                    break

            chunks.append(Chunk(
                id=make_id(f"{file_path}::block::{name}"),
                name=name, type='block', file=file_path,
                start=start + 1, end=end, language=language,
                code=code, calls=[], imports=[]
            ))

    # Fallback: size-based chunking
    if not chunks:
        i = 0
        while i < len(lines):
            chunk_lines, size, start = [], 0, i
            while i < len(lines) and size < CHUNK_SIZE:
                chunk_lines.append(lines[i])
                size += len(lines[i]) + 1
                i += 1

            if chunk_lines and len(code := '\n'.join(chunk_lines).strip()) >= 50:
                chunks.append(Chunk(
                    id=make_id(f"{file_path}::block::L{start + 1}"),
                    name=f"block_L{start + 1}", type='block',
                    file=file_path, start=start + 1, end=i,
                    language=language, code=code, calls=[], imports=[]
                ))

            i -= min(5, len(chunk_lines) // 4)  # Overlap
            if i <= start:
                i = start + len(chunk_lines)

    return chunks
