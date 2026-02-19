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

    def build_imports_map(import_map):
        """Convert import_map to enhanced format."""
        imports_map = {}
        for name, source in import_map.items():
            imports_map[name] = {"from": source, "name": name}
        return imports_map

    def extract_type_map(node):
        """Extract variable type information from assignments and annotations."""
        type_map = {}

        for n in ast.walk(node):
            # From annotations: user: User
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                var_name = n.target.id
                var_type = get_ann(n.annotation)
                if var_type and var_type != 'Any':
                    type_map[var_name] = var_type

            # From assignments: user = User()
            elif isinstance(n, ast.Assign):
                if isinstance(n.value, ast.Call):
                    call_name = None
                    if isinstance(n.value.func, ast.Name):
                        call_name = n.value.func.id
                    elif isinstance(n.value.func, ast.Attribute):
                        call_name = n.value.func.attr

                    if call_name:
                        for target in n.targets:
                            if isinstance(target, ast.Name):
                                type_map[target.id] = call_name

        return type_map

    def extract_calls_with_context(node, type_map):
        """Extract calls with receiver and type information."""
        calls_with_context = []
        builtin_methods = {'print', 'len', 'str', 'int', 'float', 'list', 'dict',
                          'set', 'tuple', 'range', 'enumerate', 'zip', 'isinstance', 'type', 'super'}

        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute):
                    # Method call: obj.method()
                    receiver_name = None
                    if isinstance(n.func.value, ast.Name):
                        receiver_name = n.func.value.id
                    elif isinstance(n.func.value, ast.Attribute):
                        receiver_name = get_attr_name(n.func.value).split('.')[-1]

                    method_name = n.func.attr
                    if method_name not in builtin_methods:
                        receiver_type = type_map.get(receiver_name) if receiver_name else None
                        calls_with_context.append({
                            "name": method_name,
                            "receiver": receiver_name,
                            "receiver_type": receiver_type
                        })

                elif isinstance(n.func, ast.Name):
                    # Direct function call: function()
                    func_name = n.func.id
                    if func_name not in builtin_methods:
                        calls_with_context.append({
                            "name": func_name,
                            "receiver": None,
                            "receiver_type": None
                        })

        return calls_with_context

    def visit(node, parent=None):
        if isinstance(node, ast.ClassDef):
            cid = make_id(f"{file_path}::class::{node.name}")
            bases = [b.id if isinstance(b, ast.Name) else get_attr_name(b) for b in node.bases]

            # Enhanced extraction
            type_map = extract_type_map(node)
            imports_map = build_imports_map(import_map)
            calls_with_context = extract_calls_with_context(node, type_map)

            chunks.append(Chunk(
                id=cid,
                name=node.name,
                type='class',
                file=file_path,
                start=node.lineno,
                end=node.end_lineno or node.lineno,
                language='python',
                code=get_code(node.lineno, node.end_lineno),
                calls=extract_calls(node),
                imports=all_imports,
                parent=parent,
                docstring=ast.get_docstring(node),
                signature=f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
                decorators=get_decorators(node),
                # Enhanced fields
                imports_map=imports_map,
                type_map=type_map,
                calls_with_context=calls_with_context
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

            # Enhanced extraction - include parameter types in type_map
            type_map = extract_type_map(node)
            for param in params:
                if param.get('type') and param['type'] != 'Any':
                    type_map[param['name'].lstrip('*')] = param['type']

            imports_map = build_imports_map(import_map)
            calls_with_context = extract_calls_with_context(node, type_map)

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
                returns=ret if ret else None,
                # Enhanced fields
                imports_map=imports_map,
                type_map=type_map,
                calls_with_context=calls_with_context
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
