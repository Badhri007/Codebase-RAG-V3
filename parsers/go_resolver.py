"""
parsers/go_resolver.py

Package-aware resolution for Go.

Key change from original:
  resolve_call() now handles raw qualified receiver types like
  "*models.UserRepository" and "[]*models.User" stored by the updated
  parse_go.py. It strips leading decorators (* [] [N]) then splits on
  "." to get the package alias and bare type name, then resolves through
  pkg_name_map.

  All other structures (GoPackageMap, build(), is_internal_import(),
  import_path_to_dir()) are unchanged.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from chunk import Chunk


def parse_go_mod(repo_path: str) -> Optional[str]:
    """
    Read go.mod and return the module name.
    Returns None if go.mod is not found.
    """
    go_mod_path = os.path.join(repo_path, "go.mod")
    if not os.path.exists(go_mod_path):
        go_mod_path = os.path.join(os.path.dirname(repo_path), "go.mod")
        if not os.path.exists(go_mod_path):
            return None

    with open(go_mod_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("module "):
                module = line.split()[1]
                print(f"  ✓ go.mod: module = {module}")
                return module
    return None


def extract_package_name(content: str) -> Optional[str]:
    """
    Extract the package name from Go source content.
    Skips comments and blank lines before the package declaration.
    """
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("//") or line.startswith("/*") or not line:
            continue
        m = re.match(r'^package\s+(\w+)', line)
        if m:
            return m.group(1)
    return None


def _strip_type_decorators(raw: str) -> str:
    """
    Strip pointer/slice/array decorators from the front of a raw type string.
    Returns the bare qualified or unqualified type name.

    This is the ONLY place in the codebase where decorator stripping happens.
    parse_go.py stores raw strings. This function runs at resolution time
    when the full context (import_map, chunk names) is available.

    Examples:
        "*models.User"       → "models.User"
        "[]*models.User"     → "models.User"
        "[4]*models.User"    → "models.User"
        "[][]string"         → "string"
        "*UserService"       → "UserService"
        "UserService"        → "UserService"
        "map[string]*User"   → "map[string]*User"  (not a decorator — stop)
        "chan *User"          → "chan *User"         (not a decorator — stop)
    """
    t = raw.strip()
    while t:
        if t.startswith('*'):
            t = t[1:]
        elif t.startswith('[]'):
            t = t[2:]
        elif t.startswith('['):
            m = re.match(r'^\[\d+\]', t)
            if m:
                t = t[m.end():]
            else:
                # map[K]V or something unknown — stop, return as-is
                break
        else:
            break
    return t


class GoPackageMap:
    """
    Lookup structures for Go call resolution.

    package_map:      {pkg_name: [Chunk, ...]}
    dir_package_map:  {"internal/auth": "auth"}
    pkg_name_map:     {(pkg_name, symbol_name): Chunk}
    import_alias_map: {file_path: {local_alias: real_pkg_name}}
    """

    def __init__(self, module_name: Optional[str]):
        self.module_name      = module_name
        self.package_map:      Dict[str, List[Chunk]] = {}
        self.dir_package_map:  Dict[str, str]         = {}
        self.pkg_name_map:     Dict[Tuple, Chunk]     = {}
        self.import_alias_map: Dict[str, Dict[str, str]] = {}

    def build(self, chunks: List[Chunk],
              file_package_map: Dict[str, str]):
        """
        Build all lookup structures from chunks.

        Args:
            chunks:           all parsed chunks (Go only)
            file_package_map: {"internal/auth/service.go": "auth", ...}
        """
        # dir → package
        for file_path, pkg_name in file_package_map.items():
            directory = "/".join(
                file_path.replace("\\", "/").split("/")[:-1]
            )
            self.dir_package_map[directory] = pkg_name

        # chunk grouping and fast lookup
        for chunk in chunks:
            if chunk.language != "go":
                continue
            pkg = file_package_map.get(chunk.file)
            if not pkg:
                continue
            self.package_map.setdefault(pkg, []).append(chunk)
            self.pkg_name_map[(pkg, chunk.name)] = chunk

        # import alias map per file
        for chunk in chunks:
            if chunk.language != "go":
                continue
            if chunk.file not in self.import_alias_map:
                aliases = {}
                for local_name, imp_info in chunk.imports_map.items():
                    real_path = imp_info["from"]
                    real_pkg  = real_path.split("/")[-1]
                    aliases[local_name] = real_pkg
                self.import_alias_map[chunk.file] = aliases

        print(f"  ✓ GoPackageMap: {len(self.package_map)} packages, "
              f"{len(self.pkg_name_map)} symbols indexed")

    def is_internal_import(self, import_path: str) -> bool:
        if self.module_name:
            return import_path.startswith(self.module_name)
        EXTERNAL_HOSTS = (
            "github.com/", "golang.org/", "gopkg.in/",
            "google.golang.org/", "k8s.io/", "go.uber.org/",
            "cloud.google.com/", "sigs.k8s.io/",
        )
        return not any(import_path.startswith(h) for h in EXTERNAL_HOSTS)

    def import_path_to_dir(self, import_path: str) -> Optional[str]:
        if not self.is_internal_import(import_path):
            return None
        if self.module_name and import_path.startswith(self.module_name):
            rel = import_path[len(self.module_name):].lstrip("/")
            return rel if rel else "."
        parts = import_path.split("/")
        return "/".join(parts[1:]) if len(parts) > 1 else None

    def resolve_call(self,
                     call_name:      str,
                     receiver:       Optional[str],
                     receiver_type:  Optional[str],
                     caller_file:    str,
                     caller_package: str,
                     resolved:       bool = False) -> Optional[Chunk]:
        """
        Resolve a Go call to its target Chunk.

        Args:
            call_name:      method/function name, e.g. "FindByUsername"
            receiver:       receiver variable or package alias, e.g. "repo"
            receiver_type:  raw type string from type_map, e.g. "*models.UserRepository"
                            May be None if the call was unresolved at parse time.
            caller_file:    file where the call originates
            caller_package: package of the caller
            resolved:       the "resolved" flag from calls_with_context.
                            If False and receiver_type is None, only same-package
                            lookup is attempted.

        Resolution strategy:
            1. receiver_type is set (resolved=True at parse time):
               Strip decorators → get bare type name.
               If qualified ("models.User"): use package alias → pkg_name_map.
               If bare ("UserService"): same-package lookup.

            2. receiver is set but receiver_type is None:
               receiver may be a package alias ("models.NewUser").
               Try direct pkg_name_map lookup via alias resolution.

            3. No receiver (direct call):
               Same-package lookup by name only.
        """

        # ── Case 1: receiver_type is known ───────────────────────────
        if receiver_type:
            # Strip leading decorators — this is the ONLY place we strip
            bare = _strip_type_decorators(receiver_type)

            if "." in bare:
                # Cross-package: "models.UserRepository"
                pkg_alias, type_name = bare.split(".", 1)
                file_aliases  = self.import_alias_map.get(caller_file, {})
                real_pkg_name = file_aliases.get(pkg_alias, pkg_alias)

                # Look up (real_pkg_name, call_name) in pkg_name_map
                # The call is on an instance of type_name, so we need
                # the method named call_name inside that package.
                # First try exact struct methods:
                target = self.pkg_name_map.get((real_pkg_name, call_name))
                if target:
                    return target

                # Also try resolving via import path → directory → pkg
                for local_name, imp_info in self._get_file_imports(
                    caller_file
                ):
                    if local_name == pkg_alias:
                        imp_dir = self.import_path_to_dir(imp_info["from"])
                        if imp_dir:
                            pkg_name = self.dir_package_map.get(imp_dir)
                            if pkg_name:
                                target = self.pkg_name_map.get(
                                    (pkg_name, call_name)
                                )
                                if target:
                                    return target
                return None

            else:
                # Same-package bare type: "UserService", "PostgresUserRepo"
                # Look for call_name in the same package
                same_pkg = self.package_map.get(caller_package, [])
                # Prefer methods whose parent matches the bare type
                for c in same_pkg:
                    if c.name == call_name:
                        parent_suffix = (c.parent or "").split("::")[-1]
                        if parent_suffix == bare:
                            return c
                # Fallback: any chunk with that name in same package
                for c in same_pkg:
                    if c.name == call_name:
                        return c
                return None

        # ── Case 2: No receiver_type but receiver is set ──────────────
        # receiver may be a package alias: models.NewUser()
        if receiver:
            file_aliases  = self.import_alias_map.get(caller_file, {})
            real_pkg_name = file_aliases.get(receiver, receiver)

            target = self.pkg_name_map.get((real_pkg_name, call_name))
            if target:
                return target

            # Try import path resolution
            for local_name, imp_info in self._get_file_imports(caller_file):
                if local_name == receiver:
                    imp_dir = self.import_path_to_dir(imp_info["from"])
                    if imp_dir:
                        pkg_name = self.dir_package_map.get(imp_dir)
                        if pkg_name:
                            target = self.pkg_name_map.get(
                                (pkg_name, call_name)
                            )
                            if target:
                                return target

        # ── Case 3: No receiver — same-package direct call ────────────
        same_pkg = self.package_map.get(caller_package, [])
        matches  = [c for c in same_pkg if c.name == call_name]
        return matches[0] if matches else None

    def _get_file_imports(self, file_path: str):
        """Yield (local_name, imp_info) for a given file."""
        for pkg_chunks in self.package_map.values():
            for chunk in pkg_chunks:
                if chunk.file == file_path and chunk.imports_map:
                    yield from chunk.imports_map.items()
                    return
