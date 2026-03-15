import re
from typing import List, Dict, Optional, Set, Callable
from chunk import Chunk


class ImportExtractor:
    """Extract ALL types of imports for any language."""

    PATTERNS = {
        'python': [
            (r'from\s+([\w.]+)\s+import\s+\*',
             lambda m: [(f"{m[1]}.*", {})]),
            (r'from\s+([\w.]+)\s+import\s+((?:\w+(?:\s+as\s+\w+)?(?:\s*,\s*)?)+)',
             'parse_python_from'),
            (r'import\s+((?:\w+(?:\s+as\s+\w+)?(?:\s*,\s*)?)+)',
             'parse_python_import'),
        ],
        'javascript': [
            (r'import\s+\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']',
             lambda m: [(m[2], {m[1]: m[2]})]),
            (r'import\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']',
             'parse_js_named'),
            (r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']',
             lambda m: [(m[2], {m[1]: m[2]})]),
            (r'(?:const|let|var)\s+(\w+)\s*=\s*require\(["\']([^"\']+)["\']\)',
             lambda m: [(m[2], {m[1]: m[2]})]),
        ],
        'typescript': [
            (r'import\s+\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']',
             lambda m: [(m[2], {m[1]: m[2]})]),
            (r'import\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']',
             'parse_js_named'),
            (r'import\s+type\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']',
             'parse_js_named'),
            (r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']',
             lambda m: [(m[2], {m[1]: m[2]})]),
        ],
        'java': [
            (r'import\s+static\s+([\w.]+)\.\*',
             lambda m: [(f"{m[1]}.*", {})]),
            (r'import\s+static\s+([\w.]+)',
             lambda m: [(m[1], {m[1].split('.')[-1]: m[1]})]),
            (r'import\s+([\w.]+)\.\*',
             lambda m: [(f"{m[1]}.*", {})]),
            (r'import\s+([\w.]+)',
             lambda m: [(m[1], {m[1].split('.')[-1]: m[1]})]),
        ],

        # Go patterns cover three forms:
        #   1. single-line aliased:  import db "github.com/lib/pq"
        #   2. single-line bare:     import "github.com/user/models"
        #   3. block:                import ( ... )
        # re.DOTALL is applied in extract() so [\s\S]+? matches newlines
        # inside the block body.
        'go': [
            (r'import\s+(\w+)\s+"([^"]+)"',
             lambda m: [(m[2], {m[1]: m[2]})]),
            (r'import\s+"([^"]+)"',
             lambda m: [(m[1], {m[1].split('/')[-1]: m[1]})]),
            (r'import\s*\(([\s\S]+?)\)',
             'parse_go_block'),
        ],
    }

    @staticmethod
    def extract(content: str, language: str) -> tuple:
        """
        Extract all imports and create import map.

        Returns:
            imports    — flat list of all import path strings
            import_map — {local_alias: full_path}
        """
        imports, import_map = [], {}
        patterns = ImportExtractor.PATTERNS.get(
            language,
            ImportExtractor.PATTERNS.get('javascript', [])
        )

        for pattern, handler in patterns:
            # re.DOTALL needed for Go block pattern so [\s\S]+?
            # matches newlines. Harmless for all other patterns.
            for match in re.finditer(pattern, content,
                                     re.MULTILINE | re.DOTALL):
                try:
                    if callable(handler):
                        results = handler(match.groups())
                    else:
                        results = getattr(ImportExtractor, handler)(
                            match.groups()
                        )
                    for imp, mapping in results:
                        imports.append(imp)
                        import_map.update(mapping)
                except Exception:
                    continue

        return imports, import_map

    # ── Go block import handler ───────────────────────────────────────

    @staticmethod
    def parse_go_block(groups):
        """
        Parse the body of a Go block import statement.

        Handles all four line forms inside import( ... ):
            "github.com/user/models"        bare    → local = last segment
            db "github.com/lib/pq"          aliased → local = alias
            _ "github.com/driver"           blank   → path only, no local name
            . "github.com/util"             dot     → local = last segment

        Returns list of (path, {local_name: path}) tuples.
        """
        block = groups[0]
        results = []

        for raw_line in block.splitlines():
            line = raw_line.strip()

            # skip blank lines and inline comments
            if not line or line.startswith('//'):
                continue

            # aliased / blank / dot:  <token> "path"
            aliased = re.match(r'^(\S+)\s+"([^"]+)"$', line)
            if aliased:
                alias, path = aliased.group(1), aliased.group(2)
                if alias == '_':
                    # blank import — side-effect only, no usable local name
                    results.append((path, {}))
                elif alias == '.':
                    # dot import — merges namespace, use last segment as key
                    results.append((path, {path.split('/')[-1]: path}))
                else:
                    results.append((path, {alias: path}))
                continue

            # bare:  "path"
            bare = re.match(r'^"([^"]+)"$', line)
            if bare:
                path  = bare.group(1)
                local = path.split('/')[-1]
                results.append((path, {local: path}))

        return results

    # ── Python and JS handlers (unchanged) ───────────────────────────

    @staticmethod
    def parse_python_from(groups):
        module, names_str = groups[0], groups[1]
        results = []
        for name in names_str.split(','):
            name = name.strip()
            if ' as ' in name:
                orig, alias = [x.strip() for x in name.split(' as ')]
                results.append((f"{module}.{orig}",
                                 {alias: f"{module}.{orig}"}))
            else:
                results.append((f"{module}.{name}",
                                 {name: f"{module}.{name}"}))
        return results

    @staticmethod
    def parse_python_import(groups):
        results = []
        for mod in groups[0].split(','):
            mod = mod.strip()
            if ' as ' in mod:
                orig, alias = [x.strip() for x in mod.split(' as ')]
                results.append((orig, {alias: orig}))
            else:
                results.append((mod, {mod: mod}))
        return results

    @staticmethod
    def parse_js_named(groups):
        names_str, module = groups[0], groups[1]
        results = []
        for name in names_str.split(','):
            name = name.strip()
            if ' as ' in name:
                orig, alias = [x.strip() for x in name.split(' as ')]
                results.append((f"{module}.{orig}",
                                 {alias: f"{module}.{orig}"}))
            else:
                results.append((f"{module}.{name}",
                                 {name: f"{module}.{name}"}))
        return results


class ParserUtils:
    """Core parsing utilities."""

    @staticmethod
    def make_id_generator():
        seen = set()
        def gen(base):
            if base not in seen:
                seen.add(base)
                return base
            i = 1
            while (uid := f"{base}_{i}") in seen:
                i += 1
            seen.add(uid)
            return uid
        return gen

    @staticmethod
    def extract_comment(lines, start, single='//', mstart='/*', mend='*/'):
        if not lines or start < 0 or start >= len(lines):
            return None
        comments = []
        for i in range(start - 1, max(0, start - 10), -1):
            if i >= len(lines):
                continue
            line = lines[i].strip()
            if line.startswith(single):
                comments.insert(0, line[len(single):].strip())
            elif mend and line.endswith(mend):
                for j in range(i, max(0, i - 20), -1):
                    if j < len(lines) and mstart in lines[j]:
                        block = '\n'.join(lines[j:i+1]).replace(
                            mstart, '').replace(mend, '')
                        return '\n'.join(
                            l.strip().lstrip('* ')
                            for l in block.split('\n')
                        ).strip()
                break
            elif line and not line.startswith((single, '*', '@')):
                break
        return '\n'.join(comments) if comments else None

    @staticmethod
    def is_valid_name(name, skip):
        if not name or not isinstance(name, str):
            return False
        return (name not in skip
                and len(name) >= 2
                and (name[0].isalpha() or name[0] == '_'))

    @staticmethod
    def split_lines(content):
        return content.split('\n') if content else []

    @staticmethod
    def get_code_range(lines, start, end):
        if not lines:
            return ""
        start = max(0, start)
        end   = min(len(lines), end + 1)
        return '\n'.join(lines[start:end])


class TreeSitterBase:
    """Base for all tree-sitter parsers with robust error handling."""

    def __init__(self, content, file_path, language, skip_names):
        self.content    = content or ""
        self.file_path  = file_path
        self.language   = language
        self.skip_names = skip_names
        self.lines      = self.content.split('\n')
        self.chunks     = []
        self.make_unique_id = ParserUtils.make_id_generator()

        try:
            self.all_imports, self.import_map = ImportExtractor.extract(
                content, language
            )
        except Exception as e:
            print(f"  Warning: Import extraction failed for {file_path}: {e}")
            self.all_imports, self.import_map = [], {}

    def get_text(self, node):
        if not node:
            return ""
        try:
            return self.content[node.start_byte:node.end_byte]
        except (AttributeError, IndexError, TypeError):
            return ""

    def get_code(self, start, end):
        if start < 0 or end < 0:
            return ""
        start = max(0, min(start, len(self.lines) - 1))
        end   = max(0, min(end,   len(self.lines) - 1))
        return '\n'.join(self.lines[start:end + 1])

    def is_valid_name(self, name):
        return ParserUtils.is_valid_name(name, self.skip_names)

    def extract_comment(self, line, **kw):
        try:
            return ParserUtils.extract_comment(self.lines, line, **kw)
        except Exception:
            return None

    def extract_calls(self, node, call_type, func_field, prop_field=None):
        calls = set()
        def walk(n):
            if not n:
                return
            try:
                if n.type == call_type:
                    func = n.child_by_field_name(func_field)
                    if func:
                        if func.type == 'identifier':
                            name = self.get_text(func)
                            if self.is_valid_name(name):
                                calls.add(name)
                        elif prop_field and func.type in (
                            'selector_expression',
                            'member_expression',
                            'field_access',
                        ):
                            prop = func.child_by_field_name(prop_field)
                            if prop:
                                calls.add(self.get_text(prop))
                for child in n.children:
                    walk(child)
            except (AttributeError, TypeError):
                pass
        try:
            walk(node)
        except Exception:
            pass
        return list(calls)

    def extract_params(self, node, field='parameters'):
        params = []
        try:
            params_node = node.child_by_field_name(field)
            if params_node:
                for child in params_node.children:
                    try:
                        params.extend(self._extract_single_param(child))
                    except Exception:
                        continue
        except (AttributeError, TypeError):
            pass
        return params

    def get_node_position(self, node):
        try:
            return node.start_point[0] + 1, node.end_point[0] + 1
        except (AttributeError, IndexError, TypeError):
            return 1, min(10, len(self.lines))

    def create_chunk(self, node, cid, name, typ, parent=None, **kw):
        try:
            start_line, end_line = self.get_node_position(node)
            try:
                code = self.get_code(start_line - 1, end_line - 1)
            except Exception:
                code = ""
            try:
                docstring = self.extract_comment(start_line - 1)
            except Exception:
                docstring = None
            return Chunk(
                id=cid, name=name, type=typ,
                file=self.file_path,
                start=start_line, end=end_line,
                language=self.language,
                code=code, parent=parent,
                docstring=docstring,
                imports=self.all_imports,
                **kw
            )
        except Exception as e:
            print(f"  Warning: Failed to create chunk for "
                  f"{name} in {self.file_path}: {e}")
            return Chunk(
                id=cid, name=name, type=typ,
                file=self.file_path,
                start=1, end=1,
                language=self.language,
                code="", parent=parent,
                docstring=None, imports=[],
                **kw
            )

    def _extract_single_param(self, node):
        raise NotImplementedError("Implement in subclass")
