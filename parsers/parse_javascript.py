from parsers.utils import *

SKIP_JS = {'if', 'else', 'for', 'while', 'switch', 'case', 'return', 'throw', 'new',
            'async', 'await', 'const', 'let', 'var', 'function', 'class', 'true', 'false', 'null'}
BUILTIN_JS = {'console', 'log', 'error', 'require', 'parseInt', 'Promise', 'setTimeout', 'fetch'}

try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript as ts_js
    import tree_sitter_typescript as ts_ts
    TS_JS = True
except:
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

    def parse(self):
        tree = self.parser.parse(bytes(self.content, 'utf8'))
        self._visit(tree.root_node)
        return self.chunks

    def _extract_single_param(self, node):
        if node.type not in ('required_parameter', 'optional_parameter', 'identifier', 'rest_parameter'):
            return []
        pattern = node.child_by_field_name('pattern')
        name = self.get_text(pattern) if pattern else self.get_text(node)
        if not self.is_valid_name(name) or name in '(),{}[]':
            return []
        p = {'name': name}
        if type_node := node.child_by_field_name('type'):
            p['type'] = self.get_text(type_node).lstrip(': ')
        return [p]

    def _visit(self, node, parent=None):
        handlers = {
            'class_declaration': self._handle_class,
            'class': self._handle_class,
            'function_declaration': self._handle_function,
            'function': self._handle_function,
            'generator_function_declaration': self._handle_function,
            'lexical_declaration': self._handle_variable,
            'variable_declaration': self._handle_variable,
            'method_definition': self._handle_method,
            'interface_declaration': lambda n, p: self._handle_simple(n, p, 'interface'),
            'type_alias_declaration': lambda n, p: self._handle_simple(n, p, 'type'),
        }

        if handler := handlers.get(node.type):
            handler(node, parent)
        elif node.type in ('export_statement', 'export_default_declaration'):
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
        for child in node.children:
            if child.type in ('heritage_clause', 'class_heritage'):
                if 'extends' in (text := self.get_text(child)):
                    extends.append(text.replace('extends', '').strip())

        cid = self.make_unique_id(f"{self.file_path}::class::{name}")
        self.chunks.append(self.create_chunk(
            node, cid, name, 'class', parent,
            calls=[c for c in self.extract_calls(node, 'call_expression', 'function', 'property') if c not in BUILTIN_JS],
            imports=extends,
            signature=f"class {name}" + (f" extends {', '.join(extends)}" if extends else "")
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
        ret = self.get_text(r).lstrip(': ') if (r := node.child_by_field_name('return_type')) else None
        params_str = ', '.join(f"{p['name']}: {p.get('type', 'any')}" for p in params)
        sig = f"function {name}({params_str})" + (f": {ret}" if ret else "")

        self.chunks.append(self.create_chunk(
            node, self.make_unique_id(f"{self.file_path}::function::{name}"),
            name, 'function', parent,
            calls=[c for c in self.extract_calls(node, 'call_expression', 'function', 'property') if c not in BUILTIN_JS],
            signature=sig, params=params, returns=ret
        ))

    def _handle_variable(self, node, parent):
        for decl in node.children:
            if decl.type == 'variable_declarator':
                if (name_node := decl.child_by_field_name('name')) and (value_node := decl.child_by_field_name('value')):
                    if value_node.type in ('arrow_function', 'function', 'function_expression'):
                        if self.is_valid_name(name := self.get_text(name_node)):
                            params = self.extract_params(value_node)
                            params_str = ', '.join(f"{p['name']}: {p.get('type', 'any')}" for p in params)

                            self.chunks.append(self.create_chunk(
                                node, self.make_unique_id(f"{self.file_path}::function::{name}"),
                                name, 'function', parent,
                                calls=[c for c in self.extract_calls(value_node, 'call_expression', 'function', 'property') if c not in BUILTIN_JS],
                                signature=f"const {name} = ({params_str}) =>",
                                params=params
                            ))

    def _handle_method(self, node, parent):
        if not (name_node := node.child_by_field_name('name')):
            return
        if not self.is_valid_name(name := self.get_text(name_node)):
            return

        params = self.extract_params(node)
        ret = self.get_text(r).lstrip(': ') if (r := node.child_by_field_name('return_type')) else None
        params_str = ', '.join(f"{p['name']}: {p.get('type', 'any')}" for p in params)
        sig = f"{name}({params_str})" + (f": {ret}" if ret else "")

        self.chunks.append(self.create_chunk(
            node, self.make_unique_id(f"{self.file_path}::method::{name}"),
            name, 'method', parent,
            calls=[c for c in self.extract_calls(node, 'call_expression', 'function', 'property') if c not in BUILTIN_JS],
            signature=sig, params=params, returns=ret
        ))

    def _handle_simple(self, node, parent, typ):
        if (name_node := node.child_by_field_name('name')) and self.is_valid_name(name := self.get_text(name_node)):
            self.chunks.append(self.create_chunk(
                node, self.make_unique_id(f"{self.file_path}::{typ}::{name}"),
                name, typ, parent, calls=[], signature=f"{typ} {name}"
            ))


def parse_javascript(content, file_path, language='javascript'):
    if TS_JS:
        try:
            chunks = JSTreeSitterParser(content, file_path, language).parse()

            if not chunks and content.strip():
                from chunk import Chunk
                import re

                imports = []
                imports.extend(re.findall(r'require\(["\']([^"\']+)["\']\)', content))
                imports.extend(re.findall(r'from\s+["\']([^"\']+)["\']', content))
                imports.extend(re.findall(r'import\s+["\']([^"\']+)["\']', content))


                calls = []
                call_matches = re.findall(r'(\w+)\.(\w+)\s*\(', content)
                calls.extend([method for obj, method in call_matches if method not in BUILTIN_JS])

                standalone_calls = re.findall(r'^\s*(\w+)\s*\(', content, re.MULTILINE)
                calls.extend([c for c in standalone_calls if c not in BUILTIN_JS and c not in {'const', 'let', 'var', 'if', 'for', 'while'}])

                chunk_id = f"{file_path}::module::main"
                chunks = [Chunk(
                    id=chunk_id,
                    name=file_path.split('/')[-1],
                    type='module',
                    file=file_path,
                    start=1,
                    end=len(content.split('\n')),
                    language=language,
                    code=content,
                    docstring=f"Entry point/configuration module: {file_path}",
                    calls=list(set(calls)),
                    imports=list(set(imports)),
                )]

            return chunks
        except Exception as e:
            print(f"  Tree-sitter error: {e}")
    return []


def parse_typescript(content, file_path, language='typescript'):
    return parse_javascript(content, file_path, language)
