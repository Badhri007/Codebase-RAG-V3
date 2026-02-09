import ast
from typing import List, Dict, Optional
from parsers.utils import *

def parse_python(content: str, file_path: str) -> List[Chunk]:
    """Parse Python using built-in AST."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"  Syntax error in {file_path}: {e}")
        return []

    lines = content.split('\n')
    chunks = []
    make_id = ParserUtils.make_id_generator()
    all_imports, import_map = ImportExtractor.extract(content, 'python')

    def get_code(s, e):
        return '\n'.join(lines[s-1:e])

    def get_attr_name(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))

    def get_ann(node):
        if not node: return ''
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Constant): return str(node.value)
        if isinstance(node, ast.Subscript): return f"{get_ann(node.value)}[{get_ann(node.slice)}]"
        if isinstance(node, ast.Attribute): return get_attr_name(node)
        if isinstance(node, ast.Tuple): return ', '.join(get_ann(e) for e in node.elts)
        if isinstance(node, ast.BinOp): return f"{get_ann(node.left)} | {get_ann(node.right)}"
        return 'Any'

    def get_params(node):
        params = []
        for arg in node.args.args:
            params.append({'name': arg.arg, 'type': get_ann(arg.annotation)})
        if node.args.vararg:
            params.append({'name': f'*{node.args.vararg.arg}', 'type': get_ann(node.args.vararg.annotation)})
        if node.args.kwarg:
            params.append({'name': f'**{node.args.kwarg.arg}', 'type': get_ann(node.args.kwarg.annotation)})
        return params

    def get_decorators(node):
        decs = []
        for dec in getattr(node, 'decorator_list', []):
            if isinstance(dec, ast.Name):
                decs.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decs.append(get_attr_name(dec))
            elif isinstance(dec, ast.Call) and isinstance(dec.func, (ast.Name, ast.Attribute)):
                decs.append(dec.func.id if isinstance(dec.func, ast.Name) else get_attr_name(dec.func))
        return decs

    def extract_calls(node):
        calls = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    calls.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    calls.add(n.func.attr)
        return [c for c in calls if c not in {'print', 'len', 'str', 'int', 'float', 'list', 'dict',
                'set', 'tuple', 'range', 'enumerate', 'zip', 'isinstance', 'type', 'super'}]

    def visit(node, parent=None):
        if isinstance(node, ast.ClassDef):
            cid = make_id(f"{file_path}::class::{node.name}")
            bases = [b.id if isinstance(b, ast.Name) else get_attr_name(b) for b in node.bases]

            chunks.append(Chunk(
                id=cid,
                name=node.name,
                type='class',
                file=file_path,
                start=node.lineno,
                end=node.end_lineno or node.lineno,
                language='python',
                code=get_code(node.lineno, node.end_lineno),
                calls=extract_calls(node), imports=all_imports, parent=parent,
                docstring=ast.get_docstring(node),
                signature=f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
                decorators=get_decorators(node)
            ))

            for child in ast.iter_child_nodes(node):
                visit(child, cid)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            typ = 'method' if parent else 'function'
            fid = make_id(f"{file_path}::{typ}::{node.name}")
            params = get_params(node)
            ret = get_ann(node.returns)

            params_str = ', '.join(f"{p['name']}: {p['type']}" if p['type'] else p['name'] for p in params)
            prefix = 'async ' if isinstance(node, ast.AsyncFunctionDef) else ''
            sig = f"{prefix}{node.name}({params_str})" + (f" -> {ret}" if ret else "")

            chunks.append(Chunk(
                id=fid,
                name=node.name,
                type=typ,
                file=file_path,
                start=node.lineno,
                end=node.end_lineno or node.lineno,
                language='python',
                code=get_code(node.lineno, node.end_lineno),
                calls=extract_calls(node),
                imports=all_imports if not parent else [],
                parent=parent,
                docstring=ast.get_docstring(node),
                signature=sig,
                decorators=get_decorators(node),
                params=params,
                returns=ret if ret else None
            ))

        elif isinstance(node, ast.Module):
            for child in ast.iter_child_nodes(node):
                visit(child, None)

    visit(tree)

    # If no chunks were extracted but file has content, create a module-level chunk
    if not chunks and content.strip():
        import re

        # Extract imports
        module_imports = []
        module_imports.extend(re.findall(r'import\s+(\w+)', content))
        module_imports.extend(re.findall(r'from\s+(\S+)\s+import', content))

        # Extract function calls
        module_calls = []
        call_matches = re.findall(r'(\w+)\s*\(', content)
        module_calls = [c for c in call_matches if c not in {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict',
            'set', 'tuple', 'range', 'enumerate', 'zip', 'isinstance',
            'type', 'super', 'if', 'for', 'while', 'with'
        }]

        chunk_id = make_id(f"{file_path}::module::main")
        chunks = [Chunk(
            id=chunk_id,
            name=file_path.split('/')[-1],
            type='module',
            file=file_path,
            start=1,
            end=len(lines),
            language='python',
            code=content,
            docstring=f"Module-level code: {file_path}",
            calls=list(set(module_calls)),
            imports=list(set(module_imports)),
        )]

    return chunks
