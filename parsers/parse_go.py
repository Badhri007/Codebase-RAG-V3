from typing import List, Dict
from parsers.utils import *

SKIP_GO = {'if', 'else', 'for', 'range', 'switch', 'case', 'return', 'defer', 'go',
            'func', 'type', 'struct', 'interface', 'make', 'new', 'len', 'cap', 'append'}
BUILTIN_GO = {'Printf', 'Println', 'Sprintf', 'Errorf', 'New', 'Get', 'Set', 'Error', 'String'}

try:
    from tree_sitter import Language, Parser
    import tree_sitter_go as ts_go
    TS_GO = True
except:
    TS_GO = False


class GoTreeSitterParser(TreeSitterBase):
    def __init__(self, content, file_path):
        super().__init__(content, file_path, 'go', SKIP_GO)
        self.parser = Parser(Language(ts_go.language()))

    def parse(self):
        tree = self.parser.parse(bytes(self.content, 'utf8'))
        self._visit(tree.root_node)
        return self.chunks

    def _extract_single_param(self, node):
        if node.type != 'parameter_declaration':
            return []
        names, typ = [], ''
        for sub in node.children:
            if sub.type == 'identifier' and self.is_valid_name(name := self.get_text(sub)):
                names.append(name)
            elif sub.type in ('type_identifier', 'pointer_type', 'slice_type', 'array_type',
                            'map_type', 'interface_type', 'struct_type'):
                typ = self.get_text(sub)
        return [{'name': n, 'type': typ} for n in names]

    def _visit(self, node, parent=None):
        handlers = {
            'function_declaration': self._handle_function,
            'method_declaration': self._handle_method,
            'type_declaration': self._handle_type,
        }
        if handler := handlers.get(node.type):
            handler(node, parent)
        else:
            for child in node.children:
                self._visit(child, parent)

    def _handle_function(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        params = self.extract_params(node)
        ret = self.get_text(r) if (r := node.child_by_field_name('result')) else None
        sig = f"func {name}(f"', '.join(f"{p['name']} {p.get('type', '')}" for p in params)
        if ret:
            sig += f" {ret}"

        self.chunks.append(self.create_chunk(
            node, self.make_unique_id(f"{self.file_path}::function::{name}"),
            name, 'function', parent,
            calls=[c for c in self.extract_calls(node, 'call_expression', 'function', 'field') if c not in BUILTIN_GO],
            signature=sig, params=params, returns=ret
        ))

    def _handle_method(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        receiver_type = None
        if receiver := node.child_by_field_name('receiver'):
            for child in receiver.children:
                if child.type == 'parameter_declaration':
                    for sub in child.children:
                        if sub.type in ('type_identifier', 'pointer_type'):
                            receiver_type = self.get_text(sub).lstrip('*')

        self.chunks.append(self.create_chunk(
            node, self.make_unique_id(f"{self.file_path}::method::{name}"),
            name, 'method', f"{self.file_path}::struct::{receiver_type}" if receiver_type else None,
            calls=[c for c in self.extract_calls(node, 'call_expression', 'function', 'field') if c not in BUILTIN_GO],
            signature=f"func ({receiver_type}) {name}()" if receiver_type else f"func {name}()"
        ))

    def _handle_type(self, node, parent):
        for child in node.children:
            if child.type == 'type_spec' and (name_node := child.child_by_field_name('name')):
                if not self.is_valid_name(name := self.get_text(name_node)):
                    continue
                if type_node := child.child_by_field_name('type'):
                    typ = 'struct' if type_node.type == 'struct_type' else 'interface' if type_node.type == 'interface_type' else 'type'
                    self.chunks.append(self.create_chunk(
                        node, self.make_unique_id(f"{self.file_path}::{typ}::{name}"),
                        name, typ, parent, calls=[], signature=f"type {name} {typ}"
                    ))


def parse_go(content, file_path):
    if TS_GO:
        try:
            return GoTreeSitterParser(content, file_path).parse()
        except Exception as e:
            print(f"  Tree-sitter error: {e}")
    return []
