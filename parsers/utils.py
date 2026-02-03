import re
from typing import List, Dict, Optional, Set, Callable
from chunk import Chunk


class ImportExtractor:
    """Extract ALL types of imports for any language."""

    PATTERNS = {
        'python': [
            (r'from\s+([\w.]+)\s+import\s+\*', lambda m: [(f"{m[1]}.*", {})]),
            (r'from\s+([\w.]+)\s+import\s+((?:\w+(?:\s+as\s+\w+)?(?:\s*,\s*)?)+)', 'parse_python_from'),
            (r'import\s+((?:\w+(?:\s+as\s+\w+)?(?:\s*,\s*)?)+)', 'parse_python_import'),
        ],
        'javascript': [
            (r'import\s+\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']', lambda m: [(m[2], {m[1]: m[2]})]),
            (r'import\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']', 'parse_js_named'),
            (r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']', lambda m: [(m[2], {m[1]: m[2]})]),
            (r'(?:const|let|var)\s+(\w+)\s*=\s*require\(["\']([^"\']+)["\']\)', lambda m: [(m[2], {m[1]: m[2]})]),
        ],
        'typescript': [
            (r'import\s+\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']', lambda m: [(m[2], {m[1]: m[2]})]),
            (r'import\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']', 'parse_js_named'),
            (r'import\s+type\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']', 'parse_js_named'),
            (r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']', lambda m: [(m[2], {m[1]: m[2]})]),
        ],
        'java': [
            (r'import\s+static\s+([\w.]+)\.\*', lambda m: [(f"{m[1]}.*", {})]),
            (r'import\s+static\s+([\w.]+)', lambda m: [(m[1], {m[1].split('.')[-1]: m[1]})]),
            (r'import\s+([\w.]+)\.\*', lambda m: [(f"{m[1]}.*", {})]),
            (r'import\s+([\w.]+)', lambda m: [(m[1], {m[1].split('.')[-1]: m[1]})]),
        ],
        'go': [
            (r'import\s+(\w+)\s+"([^"]+)"', lambda m: [(m[2], {m[1]: m[2]})]),
            (r'import\s+"([^"]+)"', lambda m: [(m[1], {m[1].split('/')[-1]: m[1]})]),
        ],
    }

    @staticmethod
    def extract(content: str, language: str) -> tuple:
        """Extract all imports and create import map."""
        imports, import_map = [], {}
        patterns = ImportExtractor.PATTERNS.get(language, ImportExtractor.PATTERNS.get('javascript', []))

        for pattern, handler in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                try:
                    if callable(handler):
                        results = handler(match.groups())
                    else:
                        results = getattr(ImportExtractor, handler)(match.groups())

                    for imp, mapping in results:
                        imports.append(imp)
                        import_map.update(mapping)
                except Exception as e:
                    # Silently skip malformed imports
                    continue

        return imports, import_map

    @staticmethod
    def parse_python_from(groups):
        """Parse: from module import name1, name2 as alias"""
        module, names_str = groups[0], groups[1]
        results = []
        for name in names_str.split(','):
            name = name.strip()
            if ' as ' in name:
                orig, alias = [x.strip() for x in name.split(' as ')]
                results.append((f"{module}.{orig}", {alias: f"{module}.{orig}"}))
            else:
                results.append((f"{module}.{name}", {name: f"{module}.{name}"}))
        return results

    @staticmethod
    def parse_python_import(groups):
        """Parse: import module1, module2 as alias"""
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
        """Parse: import {name1, name2 as alias} from 'module'"""
        names_str, module = groups[0], groups[1]
        results = []
        for name in names_str.split(','):
            name = name.strip()
            if ' as ' in name:
                orig, alias = [x.strip() for x in name.split(' as ')]
                results.append((f"{module}.{orig}", {alias: f"{module}.{orig}"}))
            else:
                results.append((f"{module}.{name}", {name: f"{module}.{name}"}))
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
        """Extract comments before a line with error handling."""
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
                        block = '\n'.join(lines[j:i+1]).replace(mstart, '').replace(mend, '')
                        return '\n'.join(l.strip().lstrip('* ') for l in block.split('\n')).strip()
                break
            elif line and not line.startswith((single, '*', '@')):
                break
        return '\n'.join(comments) if comments else None

    @staticmethod
    def is_valid_name(name, skip):
        """Check if name is valid identifier."""
        if not name or not isinstance(name, str):
            return False
        return name not in skip and len(name) >= 2 and (name[0].isalpha() or name[0] == '_')

    @staticmethod
    def split_lines(content):
        """Split content into lines."""
        return content.split('\n') if content else []

    @staticmethod
    def get_code_range(lines, start, end):
        """Get code from line range with bounds checking."""
        if not lines:
            return ""
        start = max(0, start)
        end = min(len(lines), end + 1)
        return '\n'.join(lines[start:end])


class TreeSitterBase:
    """Base for all tree-sitter parsers with robust error handling."""

    def __init__(self, content, file_path, language, skip_names):
        self.content = content or ""
        self.file_path = file_path
        self.language = language
        self.skip_names = skip_names
        self.lines = self.content.split('\n')
        self.chunks = []
        self.make_unique_id = ParserUtils.make_id_generator()

        # Extract imports with error handling
        try:
            self.all_imports, self.import_map = ImportExtractor.extract(content, language)
        except Exception as e:
            print(f"  Warning: Import extraction failed for {file_path}: {e}")
            self.all_imports, self.import_map = [], {}

    def get_text(self, node):
        """Safely get text from node."""
        if not node:
            return ""
        try:
            return self.content[node.start_byte:node.end_byte]
        except (AttributeError, IndexError, TypeError):
            return ""

    def get_code(self, start, end):
        """Get code from line range with bounds checking."""
        if start < 0 or end < 0:
            return ""
        start = max(0, min(start, len(self.lines) - 1))
        end = max(0, min(end, len(self.lines) - 1))
        return '\n'.join(self.lines[start:end + 1])

    def is_valid_name(self, name):
        """Check if name is valid."""
        return ParserUtils.is_valid_name(name, self.skip_names)

    def extract_comment(self, line, **kw):
        """Extract comment with bounds checking."""
        try:
            return ParserUtils.extract_comment(self.lines, line, **kw)
        except Exception:
            return None

    def extract_calls(self, node, call_type, func_field, prop_field=None):
        """Generic call extraction with error handling."""
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
                        elif prop_field and func.type in ('selector_expression', 'member_expression', 'field_access'):
                            prop = func.child_by_field_name(prop_field)
                            if prop:
                                calls.add(self.get_text(prop))

                for child in n.children:
                    walk(child)
            except (AttributeError, TypeError):
                # Skip nodes that don't have expected structure
                pass

        try:
            walk(node)
        except Exception:
            pass

        return list(calls)

    def extract_params(self, node, field='parameters'):
        """Extract parameters with error handling."""
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
        """Safely get node start/end positions."""
        try:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            return start_line, end_line
        except (AttributeError, IndexError, TypeError):
            # Fallback to reasonable defaults
            return 1, min(10, len(self.lines))

    def create_chunk(self, node, cid, name, typ, parent=None, **kw):
        """Create chunk with robust error handling."""
        try:
            # Get node positions safely
            start_line, end_line = self.get_node_position(node)

            # Get code safely
            try:
                code = self.get_code(start_line - 1, end_line - 1)
            except Exception:
                code = ""

            # Get docstring safely
            try:
                docstring = self.extract_comment(start_line - 1)
            except Exception:
                docstring = None

            return Chunk(
                id=cid,
                name=name,
                type=typ,
                file=self.file_path,
                start=start_line,
                end=end_line,
                language=self.language,
                code=code,
                parent=parent,
                docstring=docstring,
                imports=self.all_imports,
                **kw
            )
        except Exception as e:
            print(f"  Warning: Failed to create chunk for {name} in {self.file_path}: {e}")
            # Return a minimal chunk
            return Chunk(
                id=cid,
                name=name,
                type=typ,
                file=self.file_path,
                start=1,
                end=1,
                language=self.language,
                code="",
                parent=parent,
                docstring=None,
                imports=[],
                **kw
            )

    def _extract_single_param(self, node):
        """Extract single parameter - must be implemented by subclass."""
        raise NotImplementedError("Implement in subclass")
