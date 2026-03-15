from parsers.utils import TreeSitterBase
from typing import List, Dict, Optional, Any

SKIP_JAVA = {
    'if', 'else', 'for', 'while', 'switch', 'case', 'return', 'throw',
    'new', 'public', 'private', 'protected', 'static', 'final', 'void',
    'class', 'interface', 'abstract', 'synchronized', 'volatile',
}
BUILTIN_JAVA = {
    'println', 'print', 'toString', 'equals', 'hashCode',
    'length', 'size', 'get', 'set', 'add', 'remove',
}

# ── Annotation → entry point classification ───────────────────────────
# (annotation_name_lower, entry_point_type, http_method)
_ANNOTATION_EP = [
    # Spring MVC / Spring Boot
    ('getmapping',     'http', 'GET'),
    ('postmapping',    'http', 'POST'),
    ('putmapping',     'http', 'PUT'),
    ('deletemapping',  'http', 'DELETE'),
    ('patchmapping',   'http', 'PATCH'),
    ('requestmapping', 'http', 'ANY'),
    # Kafka
    ('kafkalistener',  'kafka', None),
    ('kafkahandler',   'kafka', None),
    # Scheduled
    ('scheduled',      'cron',  None),
    # RabbitMQ
    ('rabbitlistener', 'event', None),
    # SQS
    ('sqslistener',    'event', None),
    # gRPC (grpc-spring-boot-starter)
    ('grpcmethod',     'grpc',  None),
    # GraphQL (graphql-java)
    ('querymapping',   'graphql', None),
    ('mutationmapping','graphql', None),
    ('schemamap',      'graphql', None),
]

try:
    from tree_sitter import Language, Parser
    import tree_sitter_java as ts_java
    TS_JAVA = True
except Exception:
    TS_JAVA = False


class JavaTreeSitterParser(TreeSitterBase):

    def __init__(self, content, file_path):
        super().__init__(content, file_path, 'java', SKIP_JAVA)
        if TS_JAVA:
            self.parser = Parser(Language(ts_java.language()))

        # Class-level @RequestMapping base path — applied to method routes
        # e.g. @RequestMapping("/api/v1/users") on class + @GetMapping("/")
        # → full route = "/api/v1/users/"
        self._class_base_path: Dict[str, str] = {}
        # {class_name: base_path}

    def parse(self):
        tree = self.parser.parse(bytes(self.content, 'utf8'))
        self._visit(tree.root_node)
        return self.chunks

    # ── Annotation helpers ────────────────────────────────────────────

    def _get_annotations(self, node) -> List[Dict[str, Any]]:
        """
        Collect all annotation nodes on a class/method declaration.

        Returns list of dicts:
            {
              "name":  str,              annotation name e.g. "GetMapping"
              "value": str | None,       first string argument if present
              "attrs": {key: value}      key-value attributes
            }

        AST structure for @GetMapping("/path"):
            annotation
              name: identifier → "GetMapping"
              arguments: annotation_argument_list
                string_literal → '"/path"'

        AST structure for @RequestMapping(value="/path", method=GET):
            annotation
              name: identifier → "RequestMapping"
              arguments: annotation_argument_list
                element_value_pair
                  key: "value"
                  value: string_literal
                element_value_pair
                  key: "method"
                  value: identifier → "GET"
        """
        annotations = []

        for child in node.children:
            if child.type != 'annotation':
                continue

            name_node = child.child_by_field_name('name')
            if not name_node:
                continue
            ann_name = self.get_text(name_node)

            value = None
            attrs: Dict[str, str] = {}

            args = child.child_by_field_name('arguments')
            if args:
                for arg in args.children:
                    if arg.type == 'string_literal':
                        value = value or self.get_text(arg).strip('"')
                    elif arg.type == 'element_value_pair':
                        k_node = arg.child_by_field_name('key')
                        v_node = arg.child_by_field_name('value')
                        if k_node and v_node:
                            k = self.get_text(k_node)
                            v = self.get_text(v_node).strip('"')
                            attrs[k] = v
                            if k in ('value', 'path') and not value:
                                value = v
                    elif arg.type in ('identifier', 'scoped_identifier'):
                        value = value or self.get_text(arg).strip('"')

            annotations.append({
                "name":  ann_name,
                "value": value,
                "attrs": attrs,
            })

        return annotations

    def _classify_annotations(
            self,
            annotations: List[Dict],
            class_base: str = "",
    ) -> Dict[str, Any]:
        """
        Given a list of annotation dicts, return entry point classification.

        class_base: the @RequestMapping path on the enclosing class,
                    prepended to method-level routes.
        """
        result = {
            "is_entry_point":     False,
            "entry_point_type":   None,
            "entry_point_route":  None,
            "entry_point_method": None,
        }

        for ann in annotations:
            ann_lower = ann["name"].lower()

            for pattern, ep_type, http_method in _ANNOTATION_EP:
                if ann_lower == pattern:
                    result["is_entry_point"]   = True
                    result["entry_point_type"] = ep_type

                    if http_method:
                        # May be overridden by method= attribute
                        m = ann["attrs"].get("method", http_method)
                        result["entry_point_method"] = (
                            m.upper() if m else http_method
                        )
                    else:
                        result["entry_point_method"] = None

                    # Construct full route
                    route = ann.get("value") or ann["attrs"].get("value", "")
                    full_route = (class_base.rstrip('/') + '/' +
                                  route.lstrip('/')) if class_base else route
                    result["entry_point_route"] = full_route or None

                    # For @KafkaListener, topics attribute
                    if ep_type == 'kafka':
                        topics = ann["attrs"].get("topics", route)
                        result["entry_point_route"] = topics or None

                    return result

        return result

    # ── Type map from class fields ─────────────────────────────────────

    def _extract_field_type_map(self, class_node) -> Dict[str, str]:
        """
        Extract field-level type map from a class body.
        Captures @Autowired / @Inject fields which are the primary
        way Java dependencies are declared.

        e.g.:
            @Autowired
            private UserRepository userRepo;
            → {"userRepo": "UserRepository"}

            private final AuthService authService;
            → {"authService": "AuthService"}

        These fields are used by _extract_calls_with_context so that
        calls like userRepo.findById() can resolve receiver_type.
        """
        type_map: Dict[str, str] = {}

        body = class_node.child_by_field_name('body')
        if not body:
            return type_map

        for child in body.children:
            if child.type != 'field_declaration':
                continue

            type_node = child.child_by_field_name('type')
            if not type_node:
                continue
            type_str = self.get_text(type_node).strip()

            # Extract all declared variable names in this field
            for sub in child.children:
                if sub.type == 'variable_declarator':
                    name_node = sub.child_by_field_name('name')
                    if name_node:
                        field_name = self.get_text(name_node).strip()
                        if field_name:
                            type_map[field_name] = type_str

        return type_map

    def _extract_local_type_map(self, method_node) -> Dict[str, str]:
        """
        Extract local variable types from a method body.
        Handles:
            UserService svc = new UserServiceImpl();  → {"svc": "UserService"}
            var user = userRepo.findById(id);          → skipped (unknown)
        """
        type_map: Dict[str, str] = {}

        def walk(n):
            if not n:
                return
            if n.type == 'local_variable_declaration':
                type_node = n.child_by_field_name('type')
                if type_node:
                    type_str = self.get_text(type_node).strip()
                    for sub in n.children:
                        if sub.type == 'variable_declarator':
                            name_n = sub.child_by_field_name('name')
                            if name_n:
                                type_map[self.get_text(name_n)] = type_str
            for child in n.children:
                walk(child)

        walk(method_node)
        return type_map

    # ── calls_with_context ────────────────────────────────────────────

    def _extract_calls_with_context(self, node,
                                     type_map: Dict[str, str]
                                     ) -> List[Dict]:
        """
        Extract method calls with receiver and type context.
        Same structure as Go and Python parsers.
        """
        results = []
        seen    = set()

        def walk(n):
            if not n:
                return
            if n.type == 'method_invocation':
                name_node = n.child_by_field_name('name')
                obj_node  = n.child_by_field_name('object')
                if not name_node:
                    for child in n.children:
                        walk(child)
                    return

                call_name = self.get_text(name_node)
                if call_name in BUILTIN_JAVA:
                    for child in n.children:
                        walk(child)
                    return

                if obj_node:
                    receiver_text = self.get_text(obj_node).strip()
                    # Handle chained: this.repo.find() → "repo"
                    receiver_name = (
                        receiver_text.split('.')[-1]
                        if '.' in receiver_text
                        else receiver_text
                    )
                    # Strip 'this.' prefix
                    if receiver_name == 'this':
                        receiver_name = None

                    receiver_type = (type_map.get(receiver_name)
                                     if receiver_name else None)
                    key = (f"{receiver_name}.{call_name}"
                           if receiver_name else call_name)
                else:
                    receiver_name = None
                    receiver_type = None
                    key           = call_name

                if key not in seen:
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

    # ── Visitor ───────────────────────────────────────────────────────

    def _visit(self, node, parent=None):
        if node.type in ('class_declaration', 'interface_declaration',
                          'enum_declaration', 'record_declaration'):
            self._handle_class(node, parent)
        elif node.type in ('method_declaration',
                            'constructor_declaration'):
            self._handle_method(node, parent)
        else:
            for child in node.children:
                self._visit(child, parent)

    def _handle_class(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            for child in node.children:
                self._visit(child, parent)
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            for child in node.children:
                self._visit(child, parent)
            return

        typ = ('interface' if 'interface' in node.type
               else 'enum'      if 'enum'      in node.type
               else 'class')

        # Collect base classes and interfaces
        extends_list    = []
        implements_list = []
        if superclass := node.child_by_field_name('superclass'):
            extends_list.append(
                self.get_text(superclass).replace('extends', '').strip()
            )
        if interfaces := node.child_by_field_name('interfaces'):
            itext = self.get_text(interfaces).replace('implements', '').strip()
            implements_list = [i.strip() for i in itext.split(',') if i.strip()]

        # Class-level annotations — check for @RequestMapping base path
        annotations = self._get_annotations(node)
        base_path = ""
        for ann in annotations:
            if ann["name"].lower() == "requestmapping":
                base_path = ann.get("value") or ann["attrs"].get("value", "")
                break
        if name:
            self._class_base_path[name] = base_path

        # Field-level type map — used by method call resolution
        field_type_map = self._extract_field_type_map(node)

        cid = self.make_unique_id(f"{self.file_path}::{typ}::{name}")

        calls_ctx = self._extract_calls_with_context(node, field_type_map)

        self.chunks.append(self.create_chunk(
            node, cid, name, typ, parent,
            calls              = self._extract_calls_flat(node),
            imports            = extends_list,
            signature          = f"{typ} {name}",
            type_map           = field_type_map,
            calls_with_context = calls_ctx,
            implements         = implements_list,
        ))

        if body := node.child_by_field_name('body'):
            for child in body.children:
                self._visit(child, cid)

    def _handle_method(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        params   = self.extract_params(node)
        ret_node = node.child_by_field_name('type')
        ret      = (self.get_text(ret_node)
                    if ret_node else
                    'void' if node.type == 'method_declaration' else '')
        params_str = ', '.join(
            f"{p.get('type', '')} {p['name']}".strip() for p in params
        )

        # Determine enclosing class name for base path lookup
        # parent is the class chunk id: "file::class::ClassName"
        class_name = parent.split("::")[-1] if parent else ""
        base_path  = self._class_base_path.get(class_name, "")

        annotations = self._get_annotations(node)
        ep          = self._classify_annotations(annotations, base_path)

        # Merge field-level type_map with local variable type_map
        # Field map comes from the parent class chunk (already built).
        # We need to re-build it here or pass it down.
        # For simplicity, extract local variables only —
        # field types will be found via parent chunk's type_map in resolver.
        local_type_map = self._extract_local_type_map(node)
        calls_ctx      = self._extract_calls_with_context(
            node, local_type_map
        )

        self.chunks.append(self.create_chunk(
            node,
            self.make_unique_id(f"{self.file_path}::method::{name}"),
            name, 'method', parent,
            calls              = self._extract_calls_flat(node),
            signature          = f"{ret} {name}({params_str})".strip(),
            params             = params,
            returns            = ret if ret else None,
            type_map           = local_type_map,
            calls_with_context = calls_ctx,
            is_entry_point     = ep["is_entry_point"],
            entry_point_type   = ep["entry_point_type"],
            entry_point_route  = ep["entry_point_route"],
            entry_point_method = ep["entry_point_method"],
        ))

    def _extract_calls_flat(self, node) -> List[str]:
        calls = set()
        def walk(n):
            if n.type == 'method_invocation':
                if name_n := n.child_by_field_name('name'):
                    if self.is_valid_name(name := self.get_text(name_n)):
                        calls.add(name)
            for child in n.children:
                walk(child)
        walk(node)
        return [c for c in calls if c not in BUILTIN_JAVA]

    def _extract_single_param(self, node):
        if node.type != 'formal_parameter':
            return []
        if not (name_node := node.child_by_field_name('name')):
            return []
        name = self.get_text(name_node)
        if not self.is_valid_name(name):
            return []
        p = {'name': name}
        if type_node := node.child_by_field_name('type'):
            p['type'] = self.get_text(type_node).strip()
        return [p]


def parse_java(content, file_path):
    if TS_JAVA:
        try:
            return JavaTreeSitterParser(content, file_path).parse()
        except Exception as e:
            print(f"  Tree-sitter error: {e}")
    return []
