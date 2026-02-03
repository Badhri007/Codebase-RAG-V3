


from parsers.utils import TreeSitterBase


SKIP_JAVA = {'if', 'else', 'for', 'while', 'switch', 'case', 'return', 'throw', 'new',
              'public', 'private', 'protected', 'static', 'final', 'void', 'class', 'interface'}
BUILTIN_JAVA = {'println', 'print', 'toString', 'equals', 'hashCode', 'length', 'size', 'get', 'set', 'add'}

try:
    from tree_sitter import Language, Parser
    import tree_sitter_java as ts_java
    TS_JAVA = True
except:
    TS_JAVA = False


class JavaTreeSitterParser(TreeSitterBase):
    def __init__(self, content, file_path):
        super().__init__(content, file_path, 'java', SKIP_JAVA)
        self.parser = Parser(Language(ts_java.language()))

    def parse(self):
        tree = self.parser.parse(bytes(self.content, 'utf8'))
        self._visit(tree.root_node)
        return self.chunks

    def _extract_single_param(self, node):
        if node.type != 'formal_parameter':
            return []
        if (name_node := node.child_by_field_name('name')) and self.is_valid_name(name := self.get_text(name_node)):
            p = {'name': name}
            if type_node := node.child_by_field_name('type'):
                p['type'] = self.get_text(type_node)
            return [p]
        return []

    def _visit(self, node, parent=None):
        if node.type in ('class_declaration', 'interface_declaration', 'enum_declaration'):
            self._handle_class(node, parent)
        elif node.type in ('method_declaration', 'constructor_declaration'):
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

        typ = 'interface' if 'interface' in node.type else 'enum' if 'enum' in node.type else 'class'
        cid = self.make_unique_id(f"{self.file_path}::{typ}::{name}")

        extends = []
        if superclass := node.child_by_field_name('superclass'):
            extends.append(self.get_text(superclass).replace('extends ', ''))
        if interfaces := node.child_by_field_name('interfaces'):
            extends.append(self.get_text(interfaces).replace('implements ', ''))

        self.chunks.append(self.create_chunk(
            node, cid, name, typ, parent,
            calls=self._extract_calls(node),
            imports=extends,
            signature=f"{typ} {name}"
        ))

        if body := node.child_by_field_name('body'):
            for child in body.children:
                self._visit(child, cid)

    def _handle_method(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        params = self.extract_params(node)
        ret_type = node.child_by_field_name('type')
        ret = self.get_text(ret_type) if ret_type else 'void' if node.type == 'method_declaration' else ''

        params_str = ', '.join(f"{p.get('type', '')} {p['name']}".strip() for p in params)

        self.chunks.append(self.create_chunk(
            node, self.make_unique_id(f"{self.file_path}::method::{name}"),
            name, 'method', parent,
            calls=self._extract_calls(node),
            signature=f"{ret} {name}({params_str})".strip(),
            params=params, returns=ret if ret else None
        ))

    def _extract_calls(self, node):
        calls = set()
        def walk(n):
            if n.type == 'method_invocation' and (name_node := n.child_by_field_name('name')):
                if self.is_valid_name(name := self.get_text(name_node)):
                    calls.add(name)
            for child in n.children:
                walk(child)
        walk(node)
        return [c for c in calls if c not in BUILTIN_JAVA]


def parse_java(content, file_path):
    if TS_JAVA:
        try:
            return JavaTreeSitterParser(content, file_path).parse()
        except Exception as e:
            print(f"  Tree-sitter error: {e}")
    return []
