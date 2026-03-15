from parsers.utils import *

SKIP_JS = {
    'if', 'else', 'for', 'while', 'switch', 'case', 'return', 'throw',
    'new', 'async', 'await', 'const', 'let', 'var', 'function', 'class',
    'true', 'false', 'null', 'undefined',
}
BUILTIN_JS = {
    'console', 'log', 'error', 'warn', 'require', 'parseInt', 'parseFloat',
    'Promise', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    'fetch', 'JSON', 'Math', 'Object', 'Array', 'String', 'Number',
}

# ── HTTP route registration method names ─────────────────────────────
_HTTP_ROUTE_METHODS = {
    'get', 'post', 'put', 'delete', 'patch', 'head', 'options',
    'all', 'use', 'route', 'handle',
}
# Receiver names that are router/app objects (heuristic)
_ROUTER_NAMES = {
    'app', 'router', 'server', 'api', 'route', 'routes',
    'express', 'fastify', 'koa', 'hapi',
}

# ── NestJS / TypeScript decorator → entry point map ──────────────────
_TS_DECORATOR_EP = [
    # (decorator_name_lower, entry_point_type, http_method)
    ('get',          'http', 'GET'),
    ('post',         'http', 'POST'),
    ('put',          'http', 'PUT'),
    ('delete',       'http', 'DELETE'),
    ('patch',        'http', 'PATCH'),
    ('head',         'http', 'HEAD'),
    ('options',      'http', 'OPTIONS'),
    ('all',          'http', 'ANY'),
    ('messagpattern','kafka', None),
    ('eventpattern', 'event', None),
    ('grpcmethod',   'grpc',  None),
    ('websocketgateway', 'websocket', None),
    ('subscribemessageof', 'event', None),
    ('cronexpression', 'cron', None),
]

try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript as ts_js
    import tree_sitter_typescript as ts_ts
    TS_JS = True
except Exception:
    TS_JS = False


class JSTreeSitterParser(TreeSitterBase):

    def __init__(self, content, file_path, language):
        super().__init__(content, file_path, language, SKIP_JS)
        self.parser = Parser()
        if language in ('typescript', 'ts'):
            self.parser.language = Language(ts_ts.language_typescript())
        elif language in ('tsx', 'jsx'):
            self.parser.language = Language(ts_ts.language_tsx())
        else:
            self.parser.language = Language(ts_js.language())

        # Populated during first pass — same pattern as Go parser
        self._route_registry: Dict[str, Dict[str, str]] = {}
        self._kafka_registry: Dict[str, Dict[str, str]] = {}

    def parse(self):
        tree = self.parser.parse(bytes(self.content, 'utf8'))
        # First pass: collect route/kafka registrations
        self._collect_registrations(tree.root_node)
        # Second pass: create chunks
        self._visit(tree.root_node)
        return self.chunks

    # ── Registration collector ────────────────────────────────────────

    def _collect_registrations(self, root):
        """
        Collect Express/Fastify route registrations and event listeners.

        Patterns:
            app.get('/path', handler)
            router.post('/path', handler)
            app.use('/path', middleware, handler)
            emitter.on('event', handler)
            socket.on('message', handler)
            consumer.subscribe({topic: 'order.created'}, handler)
        """
        def walk(n):
            if not n:
                return
            if n.type == "call_expression":
                self._try_register_route(n)
                self._try_register_event(n)
            for child in n.children:
                walk(child)
        walk(root)

    def _try_register_route(self, call_node):
        func = call_node.child_by_field_name("function")
        if not func or func.type != "member_expression":
            return

        obj    = self.get_text(func.child_by_field_name("object") or func)
        method = self.get_text(
            func.child_by_field_name("property") or func
        ).lower()

        # Only classify if receiver looks like a router/app AND
        # method is an HTTP verb
        if (method not in _HTTP_ROUTE_METHODS
                or obj.lower() not in _ROUTER_NAMES):
            return

        args = call_node.child_by_field_name("arguments")
        if not args:
            return

        arg_list = [c for c in args.children
                    if c.type not in (",", "(", ")")]

        route   = None
        handler = None

        for arg in arg_list:
            if arg.type == "string":
                route = route or self.get_text(arg).strip('"\'`')
            elif arg.type == "identifier":
                handler = self.get_text(arg)
            elif arg.type == "member_expression":
                handler = self.get_text(
                    arg.child_by_field_name("property") or arg
                )

        if handler:
            self._route_registry[handler] = {
                "method": method.upper(),
                "route":  route or "",
            }

    def _try_register_event(self, call_node):
        """
        Detect: emitter.on('event', handler)
                socket.on('connection', handler)
                consumer.subscribe({topic: 'x'}, handler)
        """
        func = call_node.child_by_field_name("function")
        if not func or func.type != "member_expression":
            return

        method = self.get_text(
            func.child_by_field_name("property") or func
        ).lower()

        if method not in ('on', 'subscribe', 'addlistener'):
            return

        args = call_node.child_by_field_name("arguments")
        if not args:
            return

        arg_list = [c for c in args.children
                    if c.type not in (",", "(", ")")]

        topic   = None
        handler = None

        for arg in arg_list:
            if arg.type == "string":
                topic = topic or self.get_text(arg).strip('"\'`')
            elif arg.type == "identifier":
                handler = self.get_text(arg)

        if handler:
            self._kafka_registry[handler] = {
                "topic": topic or "",
                "method": method,
            }

    # ── Type map extraction ───────────────────────────────────────────

    def _extract_type_map(self, node) -> Dict[str, str]:
        """
        Build {var_name: type_string} from TypeScript type annotations
        and constructor call assignments.

        Sources:
            const repo: UserRepository = ...   → {"repo": "UserRepository"}
            let svc: AuthService                → {"svc": "AuthService"}
            const user = new User()             → {"user": "User"}

        For plain JS (no annotations), only constructor calls are tracked.
        """
        type_map: Dict[str, str] = {}

        def walk(n):
            if not n:
                return

            # TypeScript typed declaration:
            # lexical_declaration → variable_declarator
            #   name: identifier
            #   type: type_annotation → type_identifier
            if n.type == "variable_declarator":
                name_n = n.child_by_field_name("name")
                type_n = n.child_by_field_name("type")
                val_n  = n.child_by_field_name("value")

                var_name = self.get_text(name_n).strip() if name_n else None
                if not var_name:
                    pass
                elif type_n:
                    # TypeScript explicit type annotation
                    # type_annotation node wraps the actual type
                    # Get its text and strip the leading ":"
                    raw = self.get_text(type_n).strip().lstrip(':').strip()
                    if raw:
                        type_map[var_name] = raw
                elif val_n and val_n.type == "new_expression":
                    # new User(...) or new models.UserRepository()
                    constructor = val_n.child_by_field_name("constructor")
                    if constructor:
                        raw = self.get_text(constructor).strip()
                        # Take last segment for qualified names
                        type_map[var_name] = (raw.split('.')[-1]
                                              if '.' in raw else raw)

            for child in n.children:
                walk(child)

        walk(node)
        return type_map

    # ── calls_with_context ────────────────────────────────────────────

    def _extract_calls_with_context(self, node,
                                     type_map: Dict[str, str]
                                     ) -> List[Dict]:
        """
        Extract calls with receiver and type context.
        Same structure as Go parser for consistency.
        """
        results = []
        seen    = set()

        def walk(n):
            if not n:
                return
            if n.type == "call_expression":
                func = n.child_by_field_name("function")
                if func:
                    if func.type == "identifier":
                        name = self.get_text(func)
                        if (name not in SKIP_JS
                                and name not in BUILTIN_JS
                                and name not in seen
                                and len(name) > 1):
                            seen.add(name)
                            results.append({
                                "name":          name,
                                "receiver":      None,
                                "receiver_type": None,
                                "resolved":      False,
                            })
                    elif func.type == "member_expression":
                        obj   = func.child_by_field_name("object")
                        prop  = func.child_by_field_name("property")
                        if obj and prop:
                            call_name     = self.get_text(prop)
                            receiver_text = self.get_text(obj)
                            receiver_name = (
                                receiver_text.split('.')[-1]
                                if '.' in receiver_text
                                else receiver_text
                            )
                            receiver_type = type_map.get(receiver_name)
                            key = f"{receiver_name}.{call_name}"
                            if (call_name not in BUILTIN_JS
                                    and key not in seen):
                                seen.add(key)
                                results.append({
                                    "name":          call_name,
                                    "receiver":      receiver_name,
                                    "receiver_type": receiver_type,
                                    "resolved":      receiver_type is not None,
                                })
            for child in n.children:
                walk(child)

        walk(node)
        return results

    # ── Decorator classification (NestJS / TypeScript) ────────────────

    def _classify_ts_decorators(self, node) -> Dict[str, any]:
        """
        Inspect decorator nodes on a method/class and return
        entry point classification for NestJS-style decorators.

        @Get('/path')     → http GET /path
        @MessagePattern() → kafka
        @GrpcMethod()     → grpc
        @Cron()           → cron
        """
        result = {
            "is_entry_point":     False,
            "entry_point_type":   None,
            "entry_point_route":  None,
            "entry_point_method": None,
        }

        for child in node.children:
            if child.type != "decorator":
                continue

            dec_text = self.get_text(child).lstrip('@').lower()

            for pattern, ep_type, http_method in _TS_DECORATOR_EP:
                if dec_text.startswith(pattern):
                    result["is_entry_point"]   = True
                    result["entry_point_type"] = ep_type
                    if http_method:
                        result["entry_point_method"] = http_method

                    # Extract route/pattern from decorator argument
                    for sub in child.children:
                        if sub.type == "call_expression":
                            args = sub.child_by_field_name("arguments")
                            if args:
                                for arg in args.children:
                                    if arg.type == "string":
                                        result["entry_point_route"] = (
                                            self.get_text(arg).strip('"\'`')
                                        )
                    return result

        return result

    # ── React component detection ─────────────────────────────────────

    def _is_react_component(self, node, name: str) -> bool:
        """
        Heuristic detection for React functional components.
        Rules:
          1. Name starts with uppercase letter (React convention)
          2. Body contains JSX (jsx_element or jsx_fragment nodes)
          3. OR return type annotation includes "JSX" or "ReactElement"
        """
        if not name or not name[0].isupper():
            return False

        # Check for JSX in the function body
        def has_jsx(n):
            if not n:
                return False
            if n.type in ("jsx_element", "jsx_fragment",
                          "jsx_self_closing_element"):
                return True
            return any(has_jsx(c) for c in n.children)

        return has_jsx(node)

    # ── Visitor ───────────────────────────────────────────────────────

    def _visit(self, node, parent=None):
        handlers = {
            'class_declaration':              self._handle_class,
            'class':                          self._handle_class,
            'function_declaration':           self._handle_function,
            'function':                       self._handle_function,
            'generator_function_declaration': self._handle_function,
            'lexical_declaration':            self._handle_variable,
            'variable_declaration':           self._handle_variable,
            'method_definition':              self._handle_method,
            'interface_declaration': lambda n, p:
                self._handle_simple(n, p, 'interface'),
            'type_alias_declaration': lambda n, p:
                self._handle_simple(n, p, 'type'),
        }

        if handler := handlers.get(node.type):
            handler(node, parent)
        elif node.type in ('export_statement',
                           'export_default_declaration'):
            for child in node.children:
                self._visit(child, parent)
        else:
            for child in node.children:
                self._visit(child, parent)

    def _handle_class(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        extends = []
        implements_list = []
        for child in node.children:
            if child.type in ('heritage_clause', 'class_heritage'):
                text = self.get_text(child)
                if 'extends' in text:
                    extends.append(
                        text.replace('extends', '').strip().split()[0]
                    )
                if 'implements' in text:
                    impl_part = text.split('implements', 1)[1].strip()
                    implements_list.extend(
                        i.strip() for i in impl_part.split(',')
                    )

        cid = self.make_unique_id(f"{self.file_path}::class::{name}")
        type_map  = self._extract_type_map(node)
        calls_ctx = self._extract_calls_with_context(node, type_map)

        # NestJS @Controller decorator — class-level entry point marker
        ep = self._classify_ts_decorators(node)

        self.chunks.append(self.create_chunk(
            node, cid, name, 'class', parent,
            calls = [
                c["name"] for c in calls_ctx
                if c["name"] not in BUILTIN_JS
            ],
            imports    = extends,
            signature  = (f"class {name}"
                          + (f" extends {', '.join(extends)}"
                             if extends else "")),
            type_map            = type_map,
            calls_with_context  = calls_ctx,
            implements          = implements_list,
            is_entry_point      = ep["is_entry_point"],
            entry_point_type    = ep["entry_point_type"],
            entry_point_route   = ep["entry_point_route"],
            entry_point_method  = ep["entry_point_method"],
        ))

        if body := node.child_by_field_name('body'):
            for child in body.children:
                self._visit(child, cid)

    def _handle_function(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            for child in node.children:
                self._visit(child, parent)
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            for child in node.children:
                self._visit(child, parent)
            return

        params = self.extract_params(node)
        ret = (self.get_text(r).lstrip(': ')
               if (r := node.child_by_field_name('return_type')) else None)
        params_str = ', '.join(
            f"{p['name']}: {p.get('type', 'any')}" for p in params
        )
        sig = f"function {name}({params_str})" + (f": {ret}" if ret else "")

        type_map  = self._extract_type_map(node)
        calls_ctx = self._extract_calls_with_context(node, type_map)

        # Detect component type for React
        chunk_type = ('component'
                      if self._is_react_component(node, name)
                      else 'function')

        # Entry point from route_registry or decorators
        ep = self._classify_ts_decorators(node)
        if not ep["is_entry_point"] and name in self._route_registry:
            info = self._route_registry[name]
            ep = {
                "is_entry_point":     True,
                "entry_point_type":   "http",
                "entry_point_route":  info.get("route"),
                "entry_point_method": info.get("method"),
            }
        if not ep["is_entry_point"] and name in self._kafka_registry:
            info = self._kafka_registry[name]
            ep = {
                "is_entry_point":     True,
                "entry_point_type":   "event",
                "entry_point_route":  info.get("topic"),
                "entry_point_method": None,
            }

        self.chunks.append(self.create_chunk(
            node,
            self.make_unique_id(f"{self.file_path}::{chunk_type}::{name}"),
            name, chunk_type, parent,
            calls = [
                c["name"] for c in calls_ctx
                if c["name"] not in BUILTIN_JS
            ],
            signature           = sig,
            params              = params,
            returns             = ret,
            type_map            = type_map,
            calls_with_context  = calls_ctx,
            is_entry_point      = ep["is_entry_point"],
            entry_point_type    = ep["entry_point_type"],
            entry_point_route   = ep["entry_point_route"],
            entry_point_method  = ep["entry_point_method"],
        ))

    def _handle_variable(self, node, parent):
        type_map = self._extract_type_map(node)

        for decl in node.children:
            if decl.type != 'variable_declarator':
                continue
            name_n = decl.child_by_field_name('name')
            val_n  = decl.child_by_field_name('value')
            if not (name_n and val_n):
                continue
            if val_n.type not in ('arrow_function', 'function',
                                   'function_expression'):
                continue
            if not self.is_valid_name(name := self.get_text(name_n)):
                continue

            params = self.extract_params(val_n)
            params_str = ', '.join(
                f"{p['name']}: {p.get('type', 'any')}" for p in params
            )
            calls_ctx = self._extract_calls_with_context(val_n, type_map)

            chunk_type = ('component'
                          if self._is_react_component(val_n, name)
                          else 'function')

            ep = {"is_entry_point": False, "entry_point_type": None,
                  "entry_point_route": None, "entry_point_method": None}
            if name in self._route_registry:
                info = self._route_registry[name]
                ep = {
                    "is_entry_point":     True,
                    "entry_point_type":   "http",
                    "entry_point_route":  info.get("route"),
                    "entry_point_method": info.get("method"),
                }

            self.chunks.append(self.create_chunk(
                node,
                self.make_unique_id(
                    f"{self.file_path}::{chunk_type}::{name}"
                ),
                name, chunk_type, parent,
                calls = [
                    c["name"] for c in calls_ctx
                    if c["name"] not in BUILTIN_JS
                ],
                signature          = f"const {name} = ({params_str}) =>",
                params             = params,
                type_map           = type_map,
                calls_with_context = calls_ctx,
                is_entry_point     = ep["is_entry_point"],
                entry_point_type   = ep["entry_point_type"],
                entry_point_route  = ep["entry_point_route"],
                entry_point_method = ep["entry_point_method"],
            ))

    def _handle_method(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        params = self.extract_params(node)
        ret = (self.get_text(r).lstrip(': ')
               if (r := node.child_by_field_name('return_type')) else None)
        params_str = ', '.join(
            f"{p['name']}: {p.get('type', 'any')}" for p in params
        )
        sig = f"{name}({params_str})" + (f": {ret}" if ret else "")

        type_map  = self._extract_type_map(node)
        calls_ctx = self._extract_calls_with_context(node, type_map)
        ep        = self._classify_ts_decorators(node)

        self.chunks.append(self.create_chunk(
            node,
            self.make_unique_id(f"{self.file_path}::method::{name}"),
            name, 'method', parent,
            calls = [
                c["name"] for c in calls_ctx
                if c["name"] not in BUILTIN_JS
            ],
            signature          = sig,
            params             = params,
            returns            = ret,
            type_map           = type_map,
            calls_with_context = calls_ctx,
            is_entry_point     = ep["is_entry_point"],
            entry_point_type   = ep["entry_point_type"],
            entry_point_route  = ep["entry_point_route"],
            entry_point_method = ep["entry_point_method"],
        ))

    def _handle_simple(self, node, parent, typ):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        # Collect implements from interface extends
        implements = []
        for child in node.children:
            text = self.get_text(child)
            if 'extends' in text:
                implements.extend(
                    i.strip()
                    for i in text.replace('extends', '').split(',')
                    if i.strip()
                )

        self.chunks.append(self.create_chunk(
            node,
            self.make_unique_id(f"{self.file_path}::{typ}::{name}"),
            name, typ, parent,
            calls      = [],
            signature  = f"{typ} {name}",
            implements = implements,
        ))

    def _extract_single_param(self, node):
        if node.type not in ('required_parameter', 'optional_parameter',
                              'identifier', 'rest_parameter'):
            return []
        pattern = node.child_by_field_name('pattern')
        name    = (self.get_text(pattern) if pattern
                   else self.get_text(node))
        if not self.is_valid_name(name) or name in '(),{}[]':
            return []
        p = {'name': name}
        if type_node := node.child_by_field_name('type'):
            p['type'] = self.get_text(type_node).lstrip(': ')
        return [p]


def parse_javascript(content, file_path, language='javascript'):
    if TS_JS:
        try:
            chunks = JSTreeSitterParser(content, file_path, language).parse()
            if not chunks and content.strip():
                from chunk import Chunk
                import re as _re
                imports = []
                imports.extend(
                    _re.findall(r'require\(["\']([^"\']+)["\']\)', content)
                )
                imports.extend(
                    _re.findall(r'from\s+["\']([^"\']+)["\']', content)
                )
                chunks = [Chunk(
                    id       = f"{file_path}::module::main",
                    name     = file_path.split('/')[-1],
                    type     = 'module',
                    file     = file_path,
                    start    = 1,
                    end      = len(content.split('\n')),
                    language = language,
                    code     = content,
                    docstring= f"Entry point/configuration module: {file_path}",
                    calls    = [],
                    imports  = list(set(imports)),
                )]
            return chunks
        except Exception as e:
            print(f"  Tree-sitter error: {e}")
    return []


def parse_typescript(content, file_path, language='typescript'):
    return parse_javascript(content, file_path, language)
