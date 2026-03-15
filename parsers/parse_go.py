"""
parsers/parse_go.py

Changes in Phase 1:
  - _clean_type() removed entirely.
    type_map now stores raw type strings exactly as tree-sitter produces them.
    All normalization happens in go_resolver.resolve_type() at resolution time.

  - _infer_type_from_expr() stores raw strings with package prefix preserved.
    models.NewUser() → "models.User"  (not "User")

  - calls_with_context entries now include "resolved": bool.

  - Entry point detection added:
    _detect_entry_point() inspects every function/method chunk after creation
    and classifies it as http / kafka / grpc / cron / main / init.

  - implements detection added for struct chunks:
    _detect_implements() reads interface fields to find implemented interfaces.

  - _handle_function / _handle_method updated to call both detectors.

  - _extract_single_param no longer calls _clean_type —
    parameter types stored verbatim for full signature fidelity.
"""

import re
from typing import List, Dict, Optional, Any
from parsers.utils import TreeSitterBase, ParserUtils, ImportExtractor

# ── Filter sets ──────────────────────────────────────────────────────

SKIP_GO = {
    'if', 'else', 'for', 'range', 'switch', 'case', 'return', 'defer', 'go',
    'func', 'type', 'struct', 'interface', 'make', 'new', 'len', 'cap',
    'append', 'copy', 'delete', 'close', 'panic', 'recover', 'print', 'println',
}

BUILTIN_GO = {
    'Printf', 'Println', 'Sprintf', 'Errorf', 'Fprintf',
    'Error', 'String', 'New', 'Unwrap',
}

# ── HTTP router registration patterns ────────────────────────────────
# Matches calls like: router.GET("/path", handler)
# The receiver is any name (router, r, mux, e, g, v1, etc.)
# Methods are the HTTP verbs plus Any/Use/Group/Static
_HTTP_METHODS = {
    'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS',
    'Any', 'Handle', 'HandleFunc', 'Use', 'Group', 'Static',
}

# ── Kafka / messaging patterns ────────────────────────────────────────
# Function/method names that indicate a Kafka consumer handler.
# Checked against chunk name and calls.
_KAFKA_HANDLER_NAMES = {
    'ConsumeClaim', 'ProcessMessage', 'HandleMessage',
    'Setup', 'Cleanup',  # sarama ConsumerGroupHandler interface
}

# Function names used to register consumers — if a chunk calls one of
# these, the chunk being registered is the consumer handler.
_KAFKA_REGISTER_CALLS = {
    'Subscribe', 'Consume', 'AddHandler', 'RegisterHandler',
    'ConsumePartition',
}

# ── gRPC patterns ────────────────────────────────────────────────────
# Structs that implement generated gRPC server interfaces end with
# "Server" by convention (e.g. UserServiceServer, OrderServiceServer).
_GRPC_SERVER_SUFFIX = 'Server'

# ── Cron / scheduler patterns ─────────────────────────────────────────
_CRON_REGISTER_CALLS = {
    'AddFunc', 'Schedule', 'Every', 'Cron', 'NewJob',
    'ScheduleFunc', 'AddJob',
}

# ── tree-sitter availability ─────────────────────────────────────────
try:
    from tree_sitter import Language, Parser
    import tree_sitter_go as ts_go
    TS_GO = True
except Exception:
    TS_GO = False


class GoTreeSitterParser(TreeSitterBase):

    def __init__(self, content: str, file_path: str,
                 package_name: str = ""):
        super().__init__(content, file_path, 'go', SKIP_GO)
        self.package_name = package_name

        # Collect router registration calls found at file scope so that
        # handler functions can be tagged as entry points even when the
        # registration call is in a different function (e.g. SetupRoutes).
        # Structure: {handler_func_name: {"method": "GET", "route": "/path"}}
        self._route_registry: Dict[str, Dict[str, str]] = {}

        # Collect kafka consumer registrations similarly.
        # Structure: {handler_func_name: {"topic": "order.created"}}
        self._kafka_registry: Dict[str, Dict[str, str]] = {}

        # Collect cron job registrations.
        # Structure: {handler_func_name: {"schedule": "every 1h"}}
        self._cron_registry: Dict[str, Dict[str, str]] = {}

        if TS_GO:
            self.parser = Parser(Language(ts_go.language()))

    # ── Public entry ─────────────────────────────────────────────────

    def parse(self) -> List:
        tree = self.parser.parse(bytes(self.content, 'utf8'))

        # First pass: collect all registration calls from the entire file
        # before visiting declarations. This ensures handler functions
        # are tagged correctly even when the registration is below the
        # handler in source order.
        self._collect_registrations(tree.root_node)

        # Second pass: visit declarations and create chunks
        self._visit(tree.root_node)
        return self.chunks

    # ── Registration collector (first pass) ──────────────────────────

    def _collect_registrations(self, root):
        """
        Walk the entire AST looking for route/kafka/cron registration calls.
        Populates self._route_registry, self._kafka_registry,
        self._cron_registry before chunks are created.

        Patterns detected:

        HTTP (gin/echo/chi/gorilla/fiber/stdlib):
            router.GET("/path", HandlerFunc)
            router.POST("/path", HandlerFunc)
            http.HandleFunc("/path", HandlerFunc)
            mux.Handle("/path", handler)

        Kafka (confluent-kafka-go / sarama):
            consumer.Subscribe([]string{"topic"}, handler)
            consumer.Consume(ctx, []string{"topic"}, handler)
            handler.AddHandler("topic", HandlerFunc)

        Cron (robfig/cron / gocron):
            c.AddFunc("@every 1h", HandlerFunc)
            s.Every(1).Hour().Do(HandlerFunc)
        """
        def walk(n):
            if not n:
                return
            if n.type == "call_expression":
                self._try_register_route(n)
                self._try_register_kafka(n)
                self._try_register_cron(n)
            for child in n.children:
                walk(child)

        walk(root)

    def _try_register_route(self, call_node):
        """
        Detect: receiver.METHOD("route", HandlerFunc)
        e.g.    router.GET("/api/login", LoginHandler)
                http.HandleFunc("/", IndexHandler)
        """
        func = call_node.child_by_field_name("function")
        if not func or func.type != "selector_expression":
            return

        method_name = self.get_text(func.child_by_field_name("field") or func)
        if method_name not in _HTTP_METHODS:
            return

        args = call_node.child_by_field_name("arguments")
        if not args:
            return

        arg_nodes = [c for c in args.children
                     if c.type not in (",", "(", ")")]

        # We need at least: route_string, handler_ref
        # Some frameworks: router.GET("/path", middleware, handler)
        # We take the first string arg as route, last identifier as handler.
        route = None
        handler = None

        for arg in arg_nodes:
            if arg.type == "interpreted_string_literal":
                if route is None:
                    # strip surrounding quotes
                    route = self.get_text(arg).strip('"\'')
            elif arg.type == "identifier":
                handler = self.get_text(arg)
            elif arg.type == "selector_expression":
                # pkg.HandlerFunc style
                handler = self.get_text(
                    arg.child_by_field_name("field") or arg
                )

        if handler:
            http_method = method_name.upper()
            if http_method in ('HANDLE', 'HANDLEFUNC', 'USE', 'GROUP',
                               'STATIC', 'ANY'):
                http_method = 'ANY'
            self._route_registry[handler] = {
                "method": http_method,
                "route":  route or "",
            }

    def _try_register_kafka(self, call_node):
        """
        Detect Kafka consumer registrations.
        e.g. consumer.Subscribe([]string{"order.created"}, handler)
             handler.AddHandler("order.created", ProcessOrder)
        """
        func = call_node.child_by_field_name("function")
        if not func:
            return

        # Get the method/function name
        if func.type == "selector_expression":
            method_name = self.get_text(
                func.child_by_field_name("field") or func
            )
        elif func.type == "identifier":
            method_name = self.get_text(func)
        else:
            return

        if method_name not in _KAFKA_REGISTER_CALLS:
            return

        args = call_node.child_by_field_name("arguments")
        if not args:
            return

        arg_nodes = [c for c in args.children
                     if c.type not in (",", "(", ")")]

        topic = None
        handler = None

        for arg in arg_nodes:
            text = self.get_text(arg).strip('"\'')
            if arg.type == "interpreted_string_literal":
                topic = topic or text
            elif arg.type == "composite_literal":
                # []string{"order.created"} — extract string inside
                for sub in arg.children:
                    if sub.type == "literal_value":
                        for elem in sub.children:
                            if elem.type == "interpreted_string_literal":
                                topic = topic or self.get_text(elem).strip('"\'')
            elif arg.type == "identifier":
                handler = self.get_text(arg)

        if handler:
            self._kafka_registry[handler] = {"topic": topic or ""}

    def _try_register_cron(self, call_node):
        """
        Detect cron job registrations.
        e.g. c.AddFunc("@every 1h", CleanupTokens)
             s.Every(1).Hour().Do(CleanupTokens)
        """
        func = call_node.child_by_field_name("function")
        if not func:
            return

        if func.type == "selector_expression":
            method_name = self.get_text(
                func.child_by_field_name("field") or func
            )
        elif func.type == "identifier":
            method_name = self.get_text(func)
        else:
            return

        if method_name not in _CRON_REGISTER_CALLS:
            return

        args = call_node.child_by_field_name("arguments")
        if not args:
            return

        arg_nodes = [c for c in args.children
                     if c.type not in (",", "(", ")")]

        schedule = None
        handler = None

        for arg in arg_nodes:
            if arg.type == "interpreted_string_literal":
                schedule = schedule or self.get_text(arg).strip('"\'')
            elif arg.type == "identifier":
                handler = self.get_text(arg)

        if handler:
            self._cron_registry[handler] = {"schedule": schedule or ""}

    # ── AST visitor (second pass) ─────────────────────────────────────

    def _visit(self, node, parent=None):
        handlers = {
            'function_declaration': self._handle_function,
            'method_declaration':   self._handle_method,
            'type_declaration':     self._handle_type,
        }
        if handler := handlers.get(node.type):
            handler(node, parent)
        else:
            for child in node.children:
                self._visit(child, parent)

    # ── type_map — stores raw strings, no stripping ───────────────────

    def _extract_go_type_map(self, node) -> Dict[str, str]:
        """
        Walk function/method body and build {variable_name: raw_type_string}.

        Raw means exactly what the AST gives us — no pointer stripping,
        no package prefix removal. Examples of what gets stored:
            user := &models.User{}        → {"user": "models.User"}
              (composite literal type node text = "models.User",
               the & is the unary operator, not part of the type node)
            users := []*models.User{}     → {"users": "[]*models.User"}
            var s *UserService            → {"s": "*UserService"}
            repo := NewUserRepo()         → {"repo": "UserRepo"}
              (New heuristic: NewX → X, same package so no prefix)
            repo := models.NewUserRepo()  → {"repo": "models.UserRepo"}
              (New heuristic with package: models.NewX → "models.X")

        The resolver in go_resolver.py strips and interprets these
        raw strings at resolution time when it has the full import_map
        available to cross-reference packages.
        """
        type_map = {}

        def walk(n):
            if not n:
                return

            if n.type == "short_var_declaration":
                left  = n.child_by_field_name("left")
                right = n.child_by_field_name("right")
                if left and right:
                    var_names = self._extract_identifier_list(left)
                    typ       = self._infer_type_from_expr(right)
                    if typ:
                        for name in var_names:
                            type_map[name] = typ

            elif n.type == "var_declaration":
                for spec in n.children:
                    if spec.type == "var_spec":
                        name_node = spec.child_by_field_name("name")
                        type_node = spec.child_by_field_name("type")
                        val_node  = spec.child_by_field_name("value")
                        if name_node:
                            typ = None
                            if type_node:
                                # Store verbatim — e.g. "*models.User"
                                typ = self.get_text(type_node).strip()
                            elif val_node:
                                typ = self._infer_type_from_expr(val_node)
                            if typ:
                                type_map[self.get_text(name_node)] = typ

            for child in n.children:
                walk(child)

        walk(node)
        return type_map

    def _infer_type_from_expr(self, node) -> Optional[str]:
        """
        Infer raw type string from right-hand side expression.

        Returns the type string as it appears in source, with package
        prefix preserved. No stripping at all — caller stores this raw.

        Cases:
            &models.User{}      composite literal type node = "models.User"
                                unary & is NOT part of the type node
                                → returns "models.User"

            &UserRepo{}         type node = "UserRepo"
                                → returns "UserRepo"

            []*models.User{}    this is a composite literal with type = "[]*models.User"
                                → returns "[]*models.User"

            models.NewUser()    package.NewX pattern
                                → returns "models.User"  (package preserved)

            NewUserService()    same-package constructor
                                → returns "UserService"

            getUser()           unknown return type
                                → returns "getUser" (function name as hint)
        """
        if not node:
            return None

        # &models.User{} or &UserRepo{}
        # AST: unary_expression → operand: composite_literal → type: ...
        if node.type == "unary_expression":
            operand = node.child_by_field_name("operand")
            if operand and operand.type == "composite_literal":
                type_node = operand.child_by_field_name("type")
                if type_node:
                    # type_node text is "models.User" or "UserRepo"
                    # The & is the unary operator and NOT included in
                    # the type node text — so this is already clean.
                    return self.get_text(type_node).strip()

        # models.User{} or UserRepo{}
        if node.type == "composite_literal":
            type_node = node.child_by_field_name("type")
            if type_node:
                return self.get_text(type_node).strip()

        # models.NewUser() or NewUserService() or getUser()
        if node.type == "call_expression":
            func = node.child_by_field_name("function")
            if not func:
                return None

            func_text = self.get_text(func).strip()

            if "." in func_text:
                # Package-qualified: "models.NewUser"
                pkg, func_name = func_text.rsplit(".", 1)
                if func_name.startswith("New") and len(func_name) > 3:
                    # models.NewUser → "models.User" (package preserved)
                    return f"{pkg}.{func_name[3:]}"
                # Non-constructor: "models.Parse" — store as-is
                # resolver will see this has a dot and treat pkg as alias
                return func_text
            else:
                # Same-package call
                if func_text.startswith("New") and len(func_text) > 3:
                    # NewUserService → "UserService"
                    return func_text[3:]
                # Unknown return type — store function name as hint
                return func_text

        return None

    # ── calls_with_context ────────────────────────────────────────────

    def _extract_go_calls_with_context(self, node,
                                        type_map: Dict[str, str]
                                        ) -> List[Dict]:
        """
        Walk body and extract every call with receiver context.

        Each entry:
            name:          call name
            receiver:      immediate receiver variable name (or None)
            receiver_type: raw type string from type_map (or None)
            resolved:      True if receiver_type was found in type_map
        """
        results = []
        seen    = set()

        def walk(n):
            if not n:
                return

            if n.type == "call_expression":
                func = n.child_by_field_name("function")
                if not func:
                    for child in n.children:
                        walk(child)
                    return

                # Direct call: FindByUsername(id)
                if func.type == "identifier":
                    name = self.get_text(func)
                    if (name not in SKIP_GO
                            and name not in BUILTIN_GO
                            and name not in seen
                            and len(name) > 1):
                        seen.add(name)
                        results.append({
                            "name":          name,
                            "receiver":      None,
                            "receiver_type": None,
                            "resolved":      False,
                        })

                # Method/package call: x.Method()
                elif func.type == "selector_expression":
                    operand = func.child_by_field_name("operand")
                    field_n = func.child_by_field_name("field")
                    if operand and field_n:
                        call_name     = self.get_text(field_n)
                        receiver_text = self.get_text(operand)

                        # Chained: s.repo.Find() → take last segment "repo"
                        receiver_name = (receiver_text.split(".")[-1]
                                         if "." in receiver_text
                                         else receiver_text)

                        receiver_type = type_map.get(receiver_name)
                        is_resolved   = receiver_type is not None

                        key = f"{receiver_name}.{call_name}"
                        if (call_name not in BUILTIN_GO
                                and call_name not in seen
                                and key not in seen):
                            seen.add(key)
                            results.append({
                                "name":          call_name,
                                "receiver":      receiver_name,
                                "receiver_type": receiver_type,
                                "resolved":      is_resolved,
                            })

            for child in n.children:
                walk(child)

        walk(node)
        return results

    # ── Entry point detection ─────────────────────────────────────────

    def _detect_entry_point(self, chunk_name: str,
                             calls_ctx: List[Dict],
                             is_method: bool,
                             receiver_type: Optional[str]
                             ) -> Dict[str, Any]:
        """
        Determine whether this chunk is an entry point and classify it.

        Detection sources (in priority order):
          1. route_registry   — populated by _collect_registrations()
                                most reliable because it's from actual calls
          2. kafka_registry   — populated by _collect_registrations()
          3. cron_registry    — populated by _collect_registrations()
          4. name heuristics  — main(), init(), ConsumeClaim etc.
          5. receiver_type    — if struct implements gRPC Server interface

        Returns dict with keys:
            is_entry_point:    bool
            entry_point_type:  str | None
            entry_point_route: str | None
            entry_point_method: str | None
        """
        result = {
            "is_entry_point":    False,
            "entry_point_type":  None,
            "entry_point_route": None,
            "entry_point_method": None,
        }

        # ── 1. HTTP from route_registry ───────────────────────────────
        if chunk_name in self._route_registry:
            info = self._route_registry[chunk_name]
            result["is_entry_point"]     = True
            result["entry_point_type"]   = "http"
            result["entry_point_route"]  = info.get("route")
            result["entry_point_method"] = info.get("method")
            return result

        # ── 2. Kafka from kafka_registry ──────────────────────────────
        if chunk_name in self._kafka_registry:
            info = self._kafka_registry[chunk_name]
            result["is_entry_point"]    = True
            result["entry_point_type"]  = "kafka"
            result["entry_point_route"] = info.get("topic")
            return result

        # ── 3. Cron from cron_registry ────────────────────────────────
        if chunk_name in self._cron_registry:
            info = self._cron_registry[chunk_name]
            result["is_entry_point"]    = True
            result["entry_point_type"]  = "cron"
            result["entry_point_route"] = info.get("schedule")
            return result

        # ── 4. Name-based heuristics ──────────────────────────────────

        # main() in package main
        if (chunk_name == "main"
                and self.package_name == "main"
                and not is_method):
            result["is_entry_point"]   = True
            result["entry_point_type"] = "main"
            return result

        # init() — Go package initializer
        if chunk_name == "init" and not is_method:
            result["is_entry_point"]   = True
            result["entry_point_type"] = "init"
            return result

        # Sarama ConsumerGroupHandler interface methods
        if chunk_name in _KAFKA_HANDLER_NAMES and is_method:
            result["is_entry_point"]   = True
            result["entry_point_type"] = "kafka"
            return result

        # ── 5. gRPC — receiver implements XxxServer interface ─────────
        # Go gRPC generated code names server interfaces like
        # UserServiceServer, OrderServiceServer.
        # If the receiver struct name ends with "Server" we flag it,
        # but only for methods (not functions).
        if (is_method
                and receiver_type
                and receiver_type.endswith(_GRPC_SERVER_SUFFIX)):
            result["is_entry_point"]   = True
            result["entry_point_type"] = "grpc"
            result["entry_point_route"] = (
                f"{receiver_type}.{chunk_name}"
            )
            return result

        return result

    def _detect_implements(self, type_node) -> List[str]:
        """
        For a struct type node, detect which interfaces it implements
        by inspecting embedded types in the struct body.

        Go does not have an explicit 'implements' keyword — a struct
        implements an interface by having all required methods. We can
        only detect explicit embedding here (a heuristic, not complete):
            type MyHandler struct {
                BaseHandler        ← embedded, may signal interface
            }

        Full interface satisfaction detection would require type checking
        across all method signatures, which we do not do at parse time.
        We capture embeddings as a starting point — the resolver can
        extend this later using method signature matching.

        Returns list of embedded type names.
        """
        embedded = []
        if not type_node or type_node.type != "struct_type":
            return embedded

        for field in type_node.children:
            if field.type == "field_declaration_list":
                for decl in field.children:
                    if decl.type == "field_declaration":
                        # A field with no name is an embedded type
                        name_node = decl.child_by_field_name("name")
                        type_n    = decl.child_by_field_name("type")
                        if not name_node and type_n:
                            # Embedded: "BaseHandler" or "*models.UserRepository"
                            raw = self.get_text(type_n).strip().lstrip("*")
                            if "." in raw:
                                raw = raw.split(".")[-1]
                            embedded.append(raw)
        return embedded

    # ── Handlers ─────────────────────────────────────────────────────

    def _handle_function(self, node, parent):
        name_node = node.child_by_field_name('name')
        if not name_node:
            return
        name = self.get_text(name_node)
        if not self.is_valid_name(name):
            return

        params   = self.extract_params(node)
        ret_node = node.child_by_field_name('result')
        ret      = self.get_text(ret_node) if ret_node else None
        sig      = (f"func {name}("
                    + ', '.join(
                        f"{p['name']} {p.get('type', '')}" for p in params
                    )
                    + ")")
        if ret:
            sig += f" {ret}"

        body      = node.child_by_field_name('body')
        type_map  = self._extract_go_type_map(body)  if body else {}
        calls_ctx = (self._extract_go_calls_with_context(body, type_map)
                     if body else [])

        # Entry point detection
        ep = self._detect_entry_point(
            chunk_name   = name,
            calls_ctx    = calls_ctx,
            is_method    = False,
            receiver_type= None,
        )

        self.chunks.append(self.create_chunk(
            node,
            self.make_unique_id(f"{self.file_path}::function::{name}"),
            name, 'function', parent,
            calls               = [c["name"] for c in calls_ctx
                                   if c["name"] not in BUILTIN_GO],
            calls_with_context  = calls_ctx,
            type_map            = type_map,
            imports_map         = self._build_imports_map(),
            signature           = sig,
            params              = params,
            returns             = ret,
            package_name        = self.package_name,
            # Entry point fields
            is_entry_point      = ep["is_entry_point"],
            entry_point_type    = ep["entry_point_type"],
            entry_point_route   = ep["entry_point_route"],
            entry_point_method  = ep["entry_point_method"],
        ))

    def _handle_method(self, node, parent):
        name_node = node.child_by_field_name('name')
        if not name_node:
            return
        name = self.get_text(name_node)
        if not self.is_valid_name(name):
            return

        # Extract receiver
        receiver_type = None
        receiver_var  = None
        if receiver := node.child_by_field_name('receiver'):
            for child in receiver.children:
                if child.type == 'parameter_declaration':
                    for sub in child.children:
                        if sub.type == 'identifier' and not receiver_var:
                            receiver_var = self.get_text(sub)
                        elif sub.type in ('type_identifier', 'pointer_type'):
                            raw = self.get_text(sub)
                            # Strip pointer stars from receiver type
                            # (receiver types are always same-package bare names)
                            t = raw
                            while t.startswith('*'):
                                t = t[1:]
                            receiver_type = t

        body     = node.child_by_field_name('body')
        type_map = {}
        if receiver_var and receiver_type:
            type_map[receiver_var] = receiver_type
        if body:
            type_map.update(self._extract_go_type_map(body))

        calls_ctx = (self._extract_go_calls_with_context(body, type_map)
                     if body else [])

        parent_id = (f"{self.file_path}::struct::{receiver_type}"
                     if receiver_type else parent)

        params   = self.extract_params(node)
        ret_node = node.child_by_field_name('result')
        ret      = self.get_text(ret_node) if ret_node else None
        sig      = (f"func ({receiver_type}) {name}("
                    + ', '.join(
                        f"{p['name']} {p.get('type', '')}" for p in params
                    )
                    + ")")
        if ret:
            sig += f" {ret}"

        ep = self._detect_entry_point(
            chunk_name    = name,
            calls_ctx     = calls_ctx,
            is_method     = True,
            receiver_type = receiver_type,
        )

        self.chunks.append(self.create_chunk(
            node,
            self.make_unique_id(f"{self.file_path}::method::{name}"),
            name, 'method', parent_id,
            calls               = [c["name"] for c in calls_ctx
                                   if c["name"] not in BUILTIN_GO],
            calls_with_context  = calls_ctx,
            type_map            = type_map,
            imports_map         = self._build_imports_map(),
            signature           = sig,
            params              = params,
            returns             = ret,
            receiver_type       = receiver_type,
            package_name        = self.package_name,
            is_entry_point      = ep["is_entry_point"],
            entry_point_type    = ep["entry_point_type"],
            entry_point_route   = ep["entry_point_route"],
            entry_point_method  = ep["entry_point_method"],
        ))

    def _handle_type(self, node, parent):
        for child in node.children:
            if child.type == 'type_spec':
                name_node = child.child_by_field_name('name')
                if not name_node:
                    continue
                name = self.get_text(name_node)
                if not self.is_valid_name(name):
                    continue
                type_node = child.child_by_field_name('type')
                if not type_node:
                    continue

                typ = {
                    'struct_type':    'struct',
                    'interface_type': 'interface',
                }.get(type_node.type, 'type')

                # Detect embedded interfaces for struct chunks
                implements = []
                if typ == 'struct':
                    implements = self._detect_implements(type_node)

                self.chunks.append(self.create_chunk(
                    node,
                    self.make_unique_id(f"{self.file_path}::{typ}::{name}"),
                    name, typ, parent,
                    calls              = [],
                    calls_with_context = [],
                    type_map           = {},
                    imports_map        = self._build_imports_map(),
                    signature          = f"type {name} {typ}",
                    package_name       = self.package_name,
                    implements         = implements,
                ))

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_imports_map(self) -> Dict:
        result = {}
        for local_name, source in self.import_map.items():
            result[local_name] = {"from": source, "name": local_name}
        return result

    def _extract_identifier_list(self, node) -> List[str]:
        names = []
        if node.type == "expression_list":
            for child in node.children:
                if child.type == "identifier":
                    names.append(self.get_text(child))
        elif node.type == "identifier":
            names.append(self.get_text(node))
        return names

    def _extract_single_param(self, node):
        """
        Parameter types stored verbatim — no stripping.
        Full type string needed for signature display and LLM context.
        e.g. "context.Context" stays "context.Context".
        """
        if node.type != 'parameter_declaration':
            return []
        names, typ = [], ''
        for sub in node.children:
            if sub.type == 'identifier' and self.is_valid_name(
                    self.get_text(sub)):
                names.append(self.get_text(sub))
            elif sub.type in (
                'type_identifier', 'pointer_type', 'slice_type',
                'array_type', 'map_type', 'interface_type',
                'struct_type', 'qualified_type',
            ):
                typ = self.get_text(sub).strip()
        return [{'name': n, 'type': typ} for n in names]


# ── Public parse function ─────────────────────────────────────────────

def parse_go(content: str, file_path: str,
             package_name: str = "") -> List:
    from parsers.go_resolver import extract_package_name
    if not package_name:
        package_name = extract_package_name(content) or ""
    if TS_GO:
        try:
            chunks = GoTreeSitterParser(
                content, file_path, package_name
            ).parse()
            if not chunks and content.strip():
                from chunk import Chunk
                chunks = [Chunk(
                    id           = f"{file_path}::module::main",
                    name         = file_path.split('/')[-1],
                    type         = 'module',
                    file         = file_path,
                    start        = 1,
                    end          = len(content.splitlines()),
                    language     = 'go',
                    code         = content,
                    docstring    = f"Go module: {file_path}",
                    calls        = [],
                    imports      = [],
                    package_name = package_name,
                )]
            return chunks
        except Exception as e:
            print(f"  Tree-sitter error in {file_path}: {e}")
    return []
