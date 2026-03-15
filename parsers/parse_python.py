import ast
import re
from typing import List, Dict, Optional, Any
from parsers.utils import ParserUtils, ImportExtractor
from chunk import Chunk


# ── Decorator → entry point classification ───────────────────────────
# Maps decorator name patterns to (entry_point_type, http_method).
# Checked against the full decorator string (lowercased).
#
# The match is substring-based for flexibility across frameworks:
#   @app.route("/path", methods=["GET"])  → decorator text = "app.route"
#   @router.get("/path")                 → decorator text = "router.get"
#   @api_router.post("/path")            → decorator text = "api_router.post"

_DECORATOR_EP: List[tuple] = [
    # (pattern_to_match_in_decorator_text, entry_point_type, http_method)
    # HTTP — method-specific
    ('.get',      'http', 'GET'),
    ('.post',     'http', 'POST'),
    ('.put',      'http', 'PUT'),
    ('.delete',   'http', 'DELETE'),
    ('.patch',    'http', 'PATCH'),
    ('.head',     'http', 'HEAD'),
    ('.options',  'http', 'OPTIONS'),
    # HTTP — generic route registration
    ('.route',    'http', 'ANY'),
    ('.add_route','http', 'ANY'),
    ('.api_route','http', 'ANY'),
    # Celery / async tasks
    ('.task',     'task', None),
    ('.shared_task', 'task', None),
    # CLI
    ('.command',  'cli',  None),
    # Kafka / messaging
    ('.consumer', 'kafka', None),
    ('kafkaconsumer', 'kafka', None),
    # WebSocket
    ('.websocket','websocket', None),
    ('on_connect','websocket', None),
    # GraphQL
    ('.mutation', 'graphql', None),
    ('.query',    'graphql', None),
    ('.subscription', 'graphql', None),
]

# Decorators that carry a route as their first positional argument.
_ROUTE_BEARING_DECORATORS = {'.route', '.get', '.post', '.put',
                              '.delete', '.patch', '.head', '.options',
                              '.api_route', '.add_route', '.websocket'}


def _classify_decorators(
        decorator_nodes: list,
        source_lines: List[str],
        node_lineno: int,
) -> Dict[str, Any]:
    """
    Inspect decorator AST nodes and return entry point classification.

    Returns dict:
        is_entry_point:     bool
        entry_point_type:   str | None
        entry_point_route:  str | None
        entry_point_method: str | None
    """
    result = {
        "is_entry_point":     False,
        "entry_point_type":   None,
        "entry_point_route":  None,
        "entry_point_method": None,
    }

    for dec in decorator_nodes:
        # Get full decorator text for pattern matching
        dec_text = _get_decorator_text(dec).lower()

        for pattern, ep_type, http_method in _DECORATOR_EP:
            if pattern in dec_text:
                result["is_entry_point"]   = True
                result["entry_point_type"] = ep_type
                if http_method:
                    result["entry_point_method"] = http_method

                # Try to extract route from decorator arguments
                if any(p in dec_text for p in _ROUTE_BEARING_DECORATORS):
                    route = _extract_route_from_decorator(dec)
                    if route:
                        result["entry_point_route"] = route

                    # For .route() with methods=[] kwarg, override http_method
                    methods = _extract_methods_kwarg(dec)
                    if methods:
                        result["entry_point_method"] = ','.join(methods)

                # Also extract topic for kafka decorators
                if ep_type == 'kafka':
                    topic = _extract_route_from_decorator(dec)
                    if topic:
                        result["entry_point_route"] = topic

                return result  # first matching decorator wins

    return result


def _get_decorator_text(dec_node) -> str:
    """
    Get the full decorator text for pattern matching.
    Handles: @name, @module.name, @module.name(args)
    """
    if isinstance(dec_node, ast.Name):
        return dec_node.id
    if isinstance(dec_node, ast.Attribute):
        return f"{_get_decorator_text(dec_node.value)}.{dec_node.attr}"
    if isinstance(dec_node, ast.Call):
        return _get_decorator_text(dec_node.func)
    return ""


def _extract_route_from_decorator(dec_node) -> Optional[str]:
    """
    Extract first string argument from decorator call.
    @app.route("/api/login") → "/api/login"
    @consumer("order.created") → "order.created"
    """
    if not isinstance(dec_node, ast.Call):
        return None
    if dec_node.args:
        first = dec_node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _extract_methods_kwarg(dec_node) -> Optional[List[str]]:
    """
    Extract methods=["GET","POST"] kwarg from @app.route().
    Returns list of method strings or None.
    """
    if not isinstance(dec_node, ast.Call):
        return None
    for kw in dec_node.keywords:
        if kw.arg == 'methods':
            if isinstance(kw.value, ast.List):
                methods = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        methods.append(str(elt.value).upper())
                return methods if methods else None
    return None


# ── Main parser ───────────────────────────────────────────────────────

def parse_python(content: str, file_path: str) -> List[Chunk]:
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"  Syntax error in {file_path}: {e}")
        return []

    lines       = content.split('\n')
    chunks      = []
    make_id     = ParserUtils.make_id_generator()
    all_imports, import_map = ImportExtractor.extract(content, 'python')

    def get_code(s, e):
        return '\n'.join(lines[s - 1:e])

    def get_attr_name(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))

    def get_ann(node):
        if not node:                      return ''
        if isinstance(node, ast.Name):    return node.id
        if isinstance(node, ast.Constant):return str(node.value)
        if isinstance(node, ast.Subscript):
            return f"{get_ann(node.value)}[{get_ann(node.slice)}]"
        if isinstance(node, ast.Attribute): return get_attr_name(node)
        if isinstance(node, ast.Tuple):
            return ', '.join(get_ann(e) for e in node.elts)
        if isinstance(node, ast.BinOp):
            return f"{get_ann(node.left)} | {get_ann(node.right)}"
        return 'Any'

    def get_params(fn_node):
        params = []
        for arg in fn_node.args.args:
            params.append({
                'name': arg.arg,
                'type': get_ann(arg.annotation),
            })
        if fn_node.args.vararg:
            params.append({
                'name': f'*{fn_node.args.vararg.arg}',
                'type': get_ann(fn_node.args.vararg.annotation),
            })
        if fn_node.args.kwarg:
            params.append({
                'name': f'**{fn_node.args.kwarg.arg}',
                'type': get_ann(fn_node.args.kwarg.annotation),
            })
        return params

    def get_decorators_text(node) -> List[str]:
        decs = []
        for dec in getattr(node, 'decorator_list', []):
            if isinstance(dec, ast.Name):
                decs.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decs.append(get_attr_name(dec))
            elif isinstance(dec, ast.Call):
                func = dec.func
                name = (func.id if isinstance(func, ast.Name)
                        else get_attr_name(func)
                        if isinstance(func, ast.Attribute) else "")
                decs.append(name)
        return decs

    def build_imports_map() -> Dict:
        result = {}
        for name, source in import_map.items():
            result[name] = {"from": source, "name": name}
        return result

    def extract_type_map(fn_node) -> Dict[str, str]:
        """
        Build {var_name: type_string} from function/class body.
        Sources:
          - Type-annotated assignments: user: User = ...  → {"user": "User"}
          - Constructor calls: user = User()              → {"user": "User"}
          - Parameter annotations: def f(repo: UserRepo) → {"repo": "UserRepo"}
        Raw type strings stored — no stripping.
        """
        type_map: Dict[str, str] = {}

        for n in ast.walk(fn_node):
            # Annotated assignment: user: User
            if (isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name)):
                t = get_ann(n.annotation)
                if t and t != 'Any':
                    type_map[n.target.id] = t

            # Regular assignment with constructor call: user = User()
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
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

    def extract_calls_with_context(fn_node,
                                    type_map: Dict[str, str]
                                    ) -> List[Dict]:
        """
        Extract calls with receiver and type context.
        Structure matches Go parser for consistency:
            name, receiver, receiver_type, resolved
        """
        _BUILTIN = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict',
            'set', 'tuple', 'range', 'enumerate', 'zip', 'isinstance',
            'type', 'super', 'hasattr', 'getattr', 'setattr',
        }
        results = []
        seen    = set()

        for n in ast.walk(fn_node):
            if not isinstance(n, ast.Call):
                continue

            if isinstance(n.func, ast.Attribute):
                # Method call: obj.method()
                receiver_name = None
                if isinstance(n.func.value, ast.Name):
                    receiver_name = n.func.value.id
                elif isinstance(n.func.value, ast.Attribute):
                    # chained: self.repo.find() → take last "repo"
                    receiver_name = get_attr_name(
                        n.func.value
                    ).split('.')[-1]

                method_name = n.func.attr
                if method_name in _BUILTIN:
                    continue

                key = (f"{receiver_name}.{method_name}"
                       if receiver_name else method_name)
                if key in seen:
                    continue
                seen.add(key)

                receiver_type = (type_map.get(receiver_name)
                                 if receiver_name else None)
                results.append({
                    "name":          method_name,
                    "receiver":      receiver_name,
                    "receiver_type": receiver_type,
                    "resolved":      receiver_type is not None,
                })

            elif isinstance(n.func, ast.Name):
                func_name = n.func.id
                if func_name in _BUILTIN or func_name in seen:
                    continue
                seen.add(func_name)
                results.append({
                    "name":          func_name,
                    "receiver":      None,
                    "receiver_type": None,
                    "resolved":      False,
                })

        return results

    def extract_calls_flat(fn_node) -> List[str]:
        """Flat call list for Chunk.calls field."""
        _BUILTIN = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict',
            'set', 'tuple', 'range', 'enumerate', 'zip', 'isinstance',
            'type', 'super',
        }
        calls = set()
        for n in ast.walk(fn_node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    calls.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    calls.add(n.func.attr)
        return [c for c in calls if c not in _BUILTIN]

    def detect_main_block(module_node) -> bool:
        """Check if module has if __name__ == '__main__': block."""
        for node in ast.iter_child_nodes(module_node):
            if isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Compare)
                        and isinstance(test.left, ast.Name)
                        and test.left.id == '__name__'
                        and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value == '__main__'):
                    return True
        return False

    def visit(node, parent=None):
        if isinstance(node, ast.ClassDef):
            cid   = make_id(f"{file_path}::class::{node.name}")
            bases = [
                (b.id if isinstance(b, ast.Name) else get_attr_name(b))
                for b in node.bases
            ]

            type_map         = extract_type_map(node)
            calls_ctx        = extract_calls_with_context(node, type_map)
            dec_texts        = get_decorators_text(node)

            # Classes themselves are not entry points in Python
            # (their methods are). But we record implements from bases.
            implements = [b for b in bases if b not in ('object',)]

            chunks.append(Chunk(
                id        = cid,
                name      = node.name,
                type      = 'class',
                file      = file_path,
                start     = node.lineno,
                end       = node.end_lineno or node.lineno,
                language  = 'python',
                code      = get_code(node.lineno, node.end_lineno),
                calls     = extract_calls_flat(node),
                imports   = all_imports,
                parent    = parent,
                docstring = ast.get_docstring(node),
                signature = (f"class {node.name}({', '.join(bases)})"
                             if bases else f"class {node.name}"),
                decorators    = dec_texts,
                imports_map   = build_imports_map(),
                type_map      = type_map,
                calls_with_context = calls_ctx,
                implements    = implements,
            ))

            for child in ast.iter_child_nodes(node):
                visit(child, cid)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            typ = 'method' if parent else 'function'
            fid = make_id(f"{file_path}::{typ}::{node.name}")

            params  = get_params(node)
            ret     = get_ann(node.returns)
            prefix  = ('async '
                       if isinstance(node, ast.AsyncFunctionDef) else '')
            params_str = ', '.join(
                f"{p['name']}: {p['type']}" if p['type'] else p['name']
                for p in params
            )
            sig = (f"{prefix}{node.name}({params_str})"
                   + (f" -> {ret}" if ret else ""))

            # Build type_map: body assignments + parameter type annotations
            type_map = extract_type_map(node)
            for p in params:
                if p.get('type') and p['type'] != 'Any':
                    type_map[p['name'].lstrip('*')] = p['type']

            calls_ctx = extract_calls_with_context(node, type_map)
            dec_texts = get_decorators_text(node)

            # Entry point classification from decorators
            ep = _classify_decorators(
                decorator_nodes = node.decorator_list,
                source_lines    = lines,
                node_lineno     = node.lineno,
            )

            # Also detect __main__ entry for module-level functions
            # called directly from the main block
            if (node.name == 'main' and not parent):
                ep["is_entry_point"]   = True
                ep["entry_point_type"] = "main"

            chunks.append(Chunk(
                id        = fid,
                name      = node.name,
                type      = typ,
                file      = file_path,
                start     = node.lineno,
                end       = node.end_lineno or node.lineno,
                language  = 'python',
                code      = get_code(node.lineno, node.end_lineno),
                calls     = extract_calls_flat(node),
                imports   = all_imports if not parent else [],
                parent    = parent,
                docstring = ast.get_docstring(node),
                signature = sig,
                decorators    = dec_texts,
                params        = params,
                returns       = ret if ret else None,
                imports_map   = build_imports_map(),
                type_map      = type_map,
                calls_with_context = calls_ctx,
                is_entry_point     = ep["is_entry_point"],
                entry_point_type   = ep["entry_point_type"],
                entry_point_route  = ep["entry_point_route"],
                entry_point_method = ep["entry_point_method"],
            ))

        elif isinstance(node, ast.Module):
            # Check for __main__ block and create a synthetic entry chunk
            if detect_main_block(node):
                mid = make_id(f"{file_path}::function::__main__")
                chunks.append(Chunk(
                    id       = mid,
                    name     = '__main__',
                    type     = 'function',
                    file     = file_path,
                    start    = 1,
                    end      = len(lines),
                    language = 'python',
                    code     = content,
                    calls    = [],
                    imports  = all_imports,
                    is_entry_point   = True,
                    entry_point_type = 'main',
                ))

            for child in ast.iter_child_nodes(node):
                visit(child, None)

    visit(tree)

    if not chunks and content.strip():
        chunk_id = make_id(f"{file_path}::module::main")
        chunks = [Chunk(
            id       = chunk_id,
            name     = file_path.split('/')[-1],
            type     = 'module',
            file     = file_path,
            start    = 1,
            end      = len(lines),
            language = 'python',
            code     = content,
            docstring= f"Module-level code: {file_path}",
            calls    = [],
            imports  = all_imports,
        )]

    return chunks
