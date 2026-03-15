"""Enhanced Chunk dataclass with contextual retrieval support."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Chunk:
    # ── Core identity ─────────────────────────────────────────────────
    id: str
    name: str
    type: str
    # type values:
    #   function | method | class | struct | interface | type |
    #   module   | component (React)

    # ── Location ──────────────────────────────────────────────────────
    file: str
    start: int
    end: int
    language: str

    # ── Code ──────────────────────────────────────────────────────────
    code: str

    # ── Relationships ─────────────────────────────────────────────────
    calls: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    parent: Optional[str] = None

    # ── Resolution data ───────────────────────────────────────────────
    imports_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # {local_alias: {"from": full_path, "name": local_alias}}

    type_map: Dict[str, str] = field(default_factory=dict)
    # {var_name: raw_type_string}
    # Stored exactly as produced by AST — no stripping.
    # All normalization happens at resolution time in go_resolver.py.
    # Examples:
    #   Go:     {"user": "*models.User", "users": "[]*models.User"}
    #   Python: {"user": "User", "repo": "UserRepository"}
    #   Java:   {"repo": "UserRepository"}  (from @Autowired fields)
    #   JS/TS:  {"service": "AuthService"}  (from TypeScript annotations)

    calls_with_context: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry:
    # {
    #   "name":          str,        call name e.g. "FindByUsername"
    #   "receiver":      str | None, receiver variable name e.g. "repo"
    #   "receiver_type": str | None, raw type string from type_map
    #   "resolved":      bool        True if receiver_type found in type_map
    # }

    # ── Context ───────────────────────────────────────────────────────
    docstring: Optional[str] = None
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    params: List[Dict[str, str]] = field(default_factory=list)
    returns: Optional[str] = None

    # ── Contextual retrieval ──────────────────────────────────────────
    situating_context: Optional[str] = None
    contextual_embedding_text: Optional[str] = None
    use_contextual_embedding: bool = True

    # ── Go-specific ───────────────────────────────────────────────────
    package_name: Optional[str] = None
    receiver_type: Optional[str] = None
    # Always bare name e.g. "AuthService" — Go receiver types are
    # always same-package so no package prefix needed.

    # ── Entry point classification (NEW) ─────────────────────────────
    is_entry_point: bool = False
    # True when this chunk is a top-level trigger for a flow.
    # Set by entry point detection in each language parser.

    entry_point_type: Optional[str] = None
    # Classifies what kind of entry point this is.
    # Values:
    #   "http"       — HTTP route handler (any framework)
    #   "kafka"      — Kafka consumer / listener
    #   "grpc"       — gRPC service method implementation
    #   "cron"       — Scheduled / periodic job
    #   "cli"        — Command-line command handler
    #   "task"       — Async task (Celery, etc.)
    #   "event"      — Generic event listener (EventEmitter, socket.on, etc.)
    #   "websocket"  — WebSocket handler
    #   "main"       — Program main entry (main(), __main__, etc.)
    #   "graphql"    — GraphQL resolver
    #   "init"       — Module/package initializer (Go init(), Python module-level)
    #   None         — Not an entry point

    entry_point_route: Optional[str] = None
    # The route, topic, schedule, or event name associated with this entry point.
    # Examples:
    #   HTTP:   "/api/v1/login"
    #   Kafka:  "order.created"
    #   Cron:   "every 1h" or "0 * * * *"
    #   gRPC:   "UserService.Login"
    #   CLI:    "login" (command name)
    #   None    for non-entry-points

    entry_point_method: Optional[str] = None
    # HTTP method when entry_point_type == "http".
    # Values: "GET" | "POST" | "PUT" | "DELETE" | "PATCH" | "ANY" | None
    # None for all non-HTTP entry points.

    # ── Interface implementation tracking (NEW) ───────────────────────
    implements: List[str] = field(default_factory=list)
    # For struct/class chunks: names of interfaces this type implements.
    # Examples:
    #   Go:     ["http.Handler", "UserRepository"]
    #   Java:   ["UserService", "Serializable"]
    #   Python: ["ABC"]   (from class MyClass(ABC):)
    #   JS/TS:  ["IUserService"]  (from class X implements Y)
    # Used during flow detection to find gRPC/interface-based entry points.

    # ── Test flag ─────────────────────────────────────────────────────
    is_test: bool = False

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id':                       self.id,
            'name':                     self.name,
            'type':                     self.type,
            'file':                     self.file,
            'start':                    self.start,
            'end':                      self.end,
            'language':                 self.language,
            'code':                     self.code,
            'calls':                    self.calls,
            'imports':                  self.imports,
            'parent':                   self.parent,
            'imports_map':              self.imports_map,
            'type_map':                 self.type_map,
            'calls_with_context':       self.calls_with_context,
            'docstring':                self.docstring,
            'signature':                self.signature,
            'decorators':               self.decorators,
            'params':                   self.params,
            'returns':                  self.returns,
            'situating_context':        self.situating_context,
            'use_contextual_embedding': self.use_contextual_embedding,
            'package_name':             self.package_name,
            'receiver_type':            self.receiver_type,
            'is_entry_point':           self.is_entry_point,
            'entry_point_type':         self.entry_point_type,
            'entry_point_route':        self.entry_point_route,
            'entry_point_method':       self.entry_point_method,
            'implements':               self.implements,
            'is_test':                  self.is_test,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Chunk':
        data = dict(d)
        # Migration: handle old misspelled key from stored data
        if 'reciever_type' in data and 'receiver_type' not in data:
            data['receiver_type'] = data.pop('reciever_type')
        elif 'reciever_type' in data:
            data.pop('reciever_type')
        valid = {k: v for k, v in data.items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)

    # ── Retrieval helpers ─────────────────────────────────────────────

    def context_string(self) -> str:
        """
        Compact text summary for embedding fallback.
        Used when situating_context is not yet generated.
        Includes entry point information when present so that
        vector search on entry-point-related queries finds
        these chunks reliably.
        """
        parts = [f"{self.type}: {self.name}", f"File: {self.file}"]

        if self.signature:
            parts.append(f"Signature: {self.signature}")
        if self.docstring:
            doc = (self.docstring[:300] + "..."
                   if len(self.docstring) > 300
                   else self.docstring)
            parts.append(f"Description: {doc}")
        if self.decorators:
            parts.append(f"Decorators: {', '.join(self.decorators)}")
        if self.params:
            params_str = ', '.join(
                f"{p['name']}: {p.get('type', 'any')}" for p in self.params
            )
            parts.append(f"Parameters: {params_str}")
        if self.returns:
            parts.append(f"Returns: {self.returns}")
        if self.calls:
            parts.append(f"Calls: {', '.join(self.calls[:10])}")
        if self.parent:
            parts.append(f"In: {self.parent.split('::')[-1]}")

        # Entry point context — included so embeddings reflect
        # the "entry point" semantic even before situating_context exists
        if self.is_entry_point:
            ep_parts = [f"Entry point type: {self.entry_point_type}"]
            if self.entry_point_route:
                ep_parts.append(f"Route/topic: {self.entry_point_route}")
            if self.entry_point_method:
                ep_parts.append(f"HTTP method: {self.entry_point_method}")
            parts.append(' | '.join(ep_parts))

        if self.implements:
            parts.append(f"Implements: {', '.join(self.implements)}")
        if self.package_name:
            parts.append(f"Package: {self.package_name}")

        return '\n'.join(parts)

    def embedding_text(self) -> str:
        if self.use_contextual_embedding and self.contextual_embedding_text:
            return self.contextual_embedding_text
        return f"{self.context_string()}\n\n{self.code}"

    def llm_context(self) -> str:
        parts = []
        if self.situating_context:
            parts.append(
                f"<context>\n{self.situating_context}\n</context>\n"
            )
        header = (f"File: {self.file} "
                  f"(lines {self.start}-{self.end}) [{self.language}]")
        if self.signature:
            header += f"\n{self.type}: {self.signature}"
        if self.docstring:
            header += f"\nDoc: {self.docstring[:200]}"
        if self.is_entry_point:
            header += (f"\nEntry point: {self.entry_point_type}"
                       + (f" {self.entry_point_method}" if self.entry_point_method else "")
                       + (f" {self.entry_point_route}" if self.entry_point_route else ""))
        parts.append(header)
        parts.append(f"```{self.language}\n{self.code}\n```")
        return "\n".join(parts)
