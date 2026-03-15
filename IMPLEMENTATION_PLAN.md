# Multi-Level Summary System - Implementation Plan

## 🎯 Executive Summary

This plan outlines a **dynamic, non-hardcoded** approach to generating hierarchical summaries for codebases, enabling newbies to understand any codebase from high-level architecture to low-level implementation details without missing information.

---

## 🔑 Key Design Principles

### 1. **Dynamic Detection Over Static Hardcoding**
- Use AST (Abstract Syntax Tree) pattern matching
- Leverage language-specific parsers already in place
- Extract patterns from code structure, not predefined lists
- Make detection rules configurable and extensible

### 2. **Language-Agnostic Framework**
- Common abstraction layer for all languages
- Language-specific adapters/detectors
- Shared pattern recognition engine
- Extensible plugin architecture

### 3. **LLM-Assisted Classification**
- Use LLM to identify ambiguous entry points
- Verify detected patterns with semantic analysis
- Generate descriptions from actual code context

---

## 🏗️ Architecture: 5-Level Hierarchy

```
Level 1: CODEBASE SUMMARY
    │
    ├─── Level 2: PACKAGE/MODULE SUMMARIES
    │       │
    │       ├─── Level 3: FILE SUMMARIES
    │       │       │
    │       │       ├─── Level 4: FEATURE/FLOW SUMMARIES
    │       │       │       │
    │       │       │       └─── Level 5: CHUNK SUMMARIES (existing)
    │       │       │
    │       │       └─── Entry Points (dynamically detected)
    │       │
    │       └─── Package APIs & Exports
    │
    └─── Technology Stack (auto-detected)
```

---

## 🔍 Phase 1: Dynamic Entry Point Detection

### 1.1 **AST-Based Pattern Detection** (No Hardcoding!)

#### Strategy: Extract patterns from Abstract Syntax Tree

**For HTTP API Endpoints:**

Instead of hardcoding framework names, detect patterns:

```python
# Python Dynamic Detection
class EntryPointDetector:
    def detect_http_endpoints(self, ast_node, language):
        patterns = {
            'decorator_route': self._detect_decorator_routes,
            'function_annotation': self._detect_annotation_routes,
            'route_registration': self._detect_route_registration,
        }

        for pattern_name, detector in patterns.items():
            endpoints = detector(ast_node)
            if endpoints:
                return self._classify_endpoints(endpoints)

    def _detect_decorator_routes(self, ast_node):
        """
        Dynamically find: @something.route(), @something.get(), etc.
        Works for Flask, FastAPI, Django views, any decorator-based routing
        """
        if ast_node.type == 'decorated_definition':
            decorators = self._extract_decorators(ast_node)
            for dec in decorators:
                # Pattern: @*.{route|get|post|put|delete|patch}(path)
                if self._matches_route_pattern(dec):
                    return {
                        'type': 'http_endpoint',
                        'method': self._extract_http_method(dec),
                        'path': self._extract_route_path(dec),
                        'handler': ast_node.child_by_field_name('definition').name,
                        'framework': self._infer_framework(dec),  # Flask/FastAPI/etc
                        'decorator': dec,
                    }
```

**Pattern Matching Rules (Configurable, Not Hardcoded):**

```yaml
# entry_point_patterns.yaml
http_endpoints:
  decorators:
    - pattern: "@*.route(*)"
      type: "route_decorator"
      extract: ["method", "path"]
    - pattern: "@*.{get|post|put|delete|patch}(*)"
      type: "method_decorator"
      extract: ["path"]

  annotations:
    - pattern: "@{RequestMapping|GetMapping|PostMapping}(*)"
      language: "java"
      extract: ["value", "method"]

  function_calls:
    - pattern: "app.{get|post|put|delete}(*, *)"
      language: "javascript"
      extract: ["path", "handler"]

kafka_listeners:
  decorators:
    - pattern: "@KafkaListener(*)"
      extract: ["topics", "groupId"]

  annotations:
    - pattern: "@KafkaListener"
      extract: ["topics", "groupId", "containerFactory"]

  function_patterns:
    - pattern: "consumer.subscribe(*)"
      type: "subscription"

cli_commands:
  decorators:
    - pattern: "@click.command(*)"
      extract: ["name", "help"]
    - pattern: "@app.command(*)"  # Typer
      extract: ["name", "help"]

  argparse_patterns:
    - pattern: "parser.add_subparsers(*)"
      type: "subcommand_registration"

scheduled_jobs:
  decorators:
    - pattern: "@*.schedule(*)"
      extract: ["interval", "cron"]
    - pattern: "@celery.task(*)"
      extract: ["name", "schedule"]
```

### 1.2 **Tree-Sitter Query Language for Patterns**

Use Tree-Sitter queries (already in your codebase) for dynamic detection:

```scheme
;; Query for HTTP route decorators (Python)
(decorated_definition
  (decorator_list
    (decorator
      (call
        function: (attribute
          object: (_) @framework
          attribute: (_) @route_method
        )
        arguments: (argument_list
          (string) @route_path
        )
      )
    )
  )
  definition: (function_definition
    name: (identifier) @handler_name
  )
) @http_endpoint

;; Query for Kafka listeners (Java)
(method_declaration
  (modifiers
    (annotation
      name: (identifier) @annotation_name
      (#eq? @annotation_name "KafkaListener")
      arguments: (annotation_argument_list) @kafka_config
    )
  )
  name: (identifier) @listener_name
) @kafka_listener
```

### 1.3 **LLM-Assisted Classification**

For ambiguous cases, use LLM to classify:

```python
def classify_entry_point_with_llm(code_chunk, context):
    prompt = f"""
    Analyze this code and determine if it's an entry point:

    Context: {context.file_path}, {context.language}

    Code:
    ```{context.language}
    {code_chunk.code}
    ```

    Decorators: {code_chunk.decorators}
    Function calls: {code_chunk.calls}

    Is this an entry point? If yes, classify as:
    - http_endpoint (REST API, GraphQL endpoint)
    - kafka_listener (message consumer)
    - cli_command (command-line interface)
    - scheduled_job (cron, periodic task)
    - websocket_handler
    - event_handler (AWS Lambda, event-driven)
    - message_queue_consumer (RabbitMQ, SQS)
    - grpc_service
    - other (specify type)

    Return JSON:
    {{
      "is_entry_point": true/false,
      "type": "...",
      "confidence": 0.0-1.0,
      "evidence": "what indicates this is an entry point",
      "metadata": {{...}}  // extracted info like path, method, topic
    }}
    """

    return llm.extract_structured(prompt)
```

### 1.4 **Technology Stack Auto-Detection**

Dynamically identify frameworks and technologies:

```python
class TechStackDetector:
    def detect_from_dependencies(self, repo_path):
        """Extract from package managers"""
        detectors = {
            'requirements.txt': self._detect_python_stack,
            'package.json': self._detect_nodejs_stack,
            'go.mod': self._detect_go_stack,
            'pom.xml': self._detect_java_stack,
            'build.gradle': self._detect_gradle_stack,
        }

        stack = {}
        for file, detector in detectors.items():
            path = os.path.join(repo_path, file)
            if os.path.exists(path):
                stack.update(detector(path))
        return stack

    def detect_from_imports(self, all_chunks):
        """Infer from import statements"""
        imports = defaultdict(int)
        for chunk in all_chunks:
            for imp in chunk.imports:
                root_package = imp.split('.')[0]
                imports[root_package] += 1

        # Map to technologies
        tech_map = {
            'flask': 'Flask (Web Framework)',
            'fastapi': 'FastAPI (Web Framework)',
            'django': 'Django (Web Framework)',
            'kafka': 'Apache Kafka (Message Queue)',
            'celery': 'Celery (Task Queue)',
            'sqlalchemy': 'SQLAlchemy (ORM)',
            'asyncio': 'Async I/O',
            # ... configurable mapping
        }

        return {tech_map.get(pkg, pkg): count
                for pkg, count in imports.most_common(20)}
```

---

## 📊 Phase 2: Multi-Level Summary Generation

### 2.1 **New Data Models**

```python
# summaries.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class SummaryLevel(Enum):
    CODEBASE = 1
    PACKAGE = 2
    FILE = 3
    FEATURE = 4
    CHUNK = 5

@dataclass
class EntryPoint:
    """Dynamically detected entry point"""
    type: str  # http_endpoint, kafka_listener, etc.
    name: str
    file: str
    line: int
    chunk_id: str

    # Type-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For HTTP: {method, path, middlewares}
    # For Kafka: {topics, group_id, consumer_config}
    # For CLI: {command, arguments, help_text}

    confidence: float = 1.0  # If LLM-detected
    detection_method: str = "ast"  # ast, llm, hybrid

    # Tracing
    calls_chain: List[str] = field(default_factory=list)  # Call graph from this entry
    feature_id: Optional[str] = None  # Which feature it implements

@dataclass
class CodebaseSummary:
    """Level 1: Overall codebase understanding"""
    repo_name: str
    purpose: str  # What the codebase does
    domain: str  # E-commerce, ML pipeline, API gateway, etc.

    # Architecture
    architecture_pattern: str  # Microservice, Monolith, Serverless, etc.
    tech_stack: Dict[str, str]  # {category: technology}

    # Entry points
    entry_points: List[EntryPoint] = field(default_factory=list)
    entry_point_summary: str = ""  # Natural language summary

    # Core functionalities
    core_functionalities: List[Dict[str, str]] = field(default_factory=list)
    # [{"name": "User Authentication", "description": "...", "files": [...]}]

    # Statistics
    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    languages: Dict[str, int] = field(default_factory=dict)  # {language: line_count}

    # Generated summary
    summary: str = ""

    # LLM-generated insights
    key_design_decisions: List[str] = field(default_factory=list)
    external_integrations: List[str] = field(default_factory=list)

@dataclass
class PackageSummary:
    """Level 2: Package/module summary"""
    package_name: str
    path: str  # Directory path
    purpose: str  # What this package does

    # Relationships
    parent_package: Optional[str] = None
    sub_packages: List[str] = field(default_factory=list)

    # Contents
    files: List[str] = field(default_factory=list)
    exported_apis: List[str] = field(default_factory=list)  # Public interface

    # Dependencies
    internal_dependencies: List[str] = field(default_factory=list)  # Other packages
    external_dependencies: List[str] = field(default_factory=list)  # Third-party

    # Entry points in this package
    entry_points: List[EntryPoint] = field(default_factory=list)

    # Patterns
    design_patterns: List[str] = field(default_factory=list)

    summary: str = ""

@dataclass
class FileSummary:
    """Level 3: Individual file summary"""
    file_path: str
    language: str
    package: str

    # Purpose
    primary_responsibility: str

    # Contents
    chunks: List[str] = field(default_factory=list)  # chunk IDs
    classes: List[Dict[str, str]] = field(default_factory=list)
    functions: List[Dict[str, str]] = field(default_factory=list)

    # Dependencies
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)

    # Entry points in this file
    entry_points: List[EntryPoint] = field(default_factory=list)

    # Patterns
    design_patterns: List[str] = field(default_factory=list)

    summary: str = ""

@dataclass
class FeatureSummary:
    """Level 4: Cross-cutting feature/flow"""
    feature_id: str
    name: str
    description: str

    # Entry point that triggers this feature
    entry_point: EntryPoint

    # Implementation
    files_involved: List[str] = field(default_factory=list)
    functions_involved: List[str] = field(default_factory=list)  # chunk IDs

    # Flow
    data_flow: str = ""  # Mermaid diagram or text
    call_sequence: List[Dict[str, Any]] = field(default_factory=list)
    # [{"step": 1, "function": "...", "file": "...", "action": "..."}]

    # Integrations
    external_calls: List[str] = field(default_factory=list)  # APIs, DBs, etc.

    summary: str = ""

@dataclass
class ChunkSummary:
    """Level 5: Enhanced individual chunk (extends existing Chunk)"""
    chunk_id: str
    summary: str  # What this function/class does

    # Context
    contributes_to_file: str  # How it serves file purpose
    contributes_to_feature: Optional[str] = None  # Which feature it's part of

    # Technical details
    algorithm_used: Optional[str] = None
    complexity: Optional[str] = None  # O(n), O(log n), etc.
    side_effects: List[str] = field(default_factory=list)
```

### 2.2 **Summary Generation Pipeline**

```python
# summary_generator.py

from typing import List, Dict
from llm.providers import get_llm
from summaries import *
from chunk import Chunk

class SummaryGenerator:
    def __init__(self, llm_provider="claude", model=None):
        self.llm = get_llm(llm_provider, model=model)
        self.cache = {}  # Cache summaries to avoid re-generation

    def generate_all_summaries(self, chunks: List[Chunk],
                               entry_points: List[EntryPoint],
                               repo_name: str,
                               repo_path: str) -> Dict[SummaryLevel, List]:
        """
        Generate summaries at all levels in bottom-up fashion
        """
        print("🔄 Generating multi-level summaries...")

        # Level 5: Chunk summaries
        print("  📝 Level 5: Generating chunk summaries...")
        chunk_summaries = self.generate_chunk_summaries(chunks)

        # Level 3: File summaries (aggregate chunks)
        print("  📝 Level 3: Generating file summaries...")
        file_summaries = self.generate_file_summaries(chunks, chunk_summaries)

        # Level 2: Package summaries (aggregate files)
        print("  📝 Level 2: Generating package summaries...")
        package_summaries = self.generate_package_summaries(
            file_summaries, chunks, entry_points
        )

        # Level 4: Feature summaries (trace from entry points)
        print("  📝 Level 4: Generating feature summaries...")
        feature_summaries = self.generate_feature_summaries(
            entry_points, chunks, file_summaries
        )

        # Level 1: Codebase summary (top-level synthesis)
        print("  📝 Level 1: Generating codebase summary...")
        codebase_summary = self.generate_codebase_summary(
            repo_name, repo_path, package_summaries,
            feature_summaries, entry_points, chunks
        )

        return {
            SummaryLevel.CODEBASE: [codebase_summary],
            SummaryLevel.PACKAGE: package_summaries,
            SummaryLevel.FILE: file_summaries,
            SummaryLevel.FEATURE: feature_summaries,
            SummaryLevel.CHUNK: chunk_summaries,
        }

    def generate_chunk_summaries(self, chunks: List[Chunk]) -> List[ChunkSummary]:
        """Generate summary for each chunk with context"""
        summaries = []

        # Group by file for context
        by_file = defaultdict(list)
        for chunk in chunks:
            by_file[chunk.file].append(chunk)

        for file_path, file_chunks in by_file.items():
            # Get file-level context
            file_context = self._infer_file_purpose(file_chunks)

            for chunk in file_chunks:
                summary = self._generate_chunk_summary(chunk, file_context)
                summaries.append(summary)

        return summaries

    def _generate_chunk_summary(self, chunk: Chunk, file_context: str) -> ChunkSummary:
        """Generate summary for a single chunk"""

        # Build context
        context_parts = [
            f"File: {chunk.file}",
            f"File Purpose: {file_context}",
            f"Type: {chunk.type}",
        ]

        if chunk.parent:
            context_parts.append(f"Parent: {chunk.parent}")

        if chunk.calls:
            context_parts.append(f"Calls: {', '.join(chunk.calls[:5])}")

        context = "\n".join(context_parts)

        prompt = f"""Analyze this code and provide a technical summary.

{context}

Code:
```{chunk.language}
{chunk.signature or ''}
{chunk.code}
```

Provide a JSON response with:
{{
  "summary": "2-3 sentence technical explanation of what this does",
  "contributes_to_file": "How this serves the file's purpose",
  "algorithm_used": "Algorithm/pattern if applicable (e.g., 'binary search', 'factory pattern')",
  "complexity": "Time/space complexity if applicable",
  "side_effects": ["list", "of", "side effects"] or []
}}

Requirements:
- Be technically accurate, reference actual code
- Explain WHAT it does and HOW
- No hallucination - only describe what's in the code
- Include implementation details (algorithms, data structures)
"""

        response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

        try:
            data = json.loads(response)
            return ChunkSummary(
                chunk_id=chunk.id,
                summary=data['summary'],
                contributes_to_file=data['contributes_to_file'],
                algorithm_used=data.get('algorithm_used'),
                complexity=data.get('complexity'),
                side_effects=data.get('side_effects', []),
            )
        except json.JSONDecodeError:
            # Fallback
            return ChunkSummary(
                chunk_id=chunk.id,
                summary=response[:500],
                contributes_to_file="Part of file functionality",
            )

    def generate_file_summaries(self, chunks: List[Chunk],
                                chunk_summaries: List[ChunkSummary]) -> List[FileSummary]:
        """Aggregate chunks into file summaries"""

        # Group chunks by file
        by_file = defaultdict(list)
        for chunk in chunks:
            by_file[chunk.file].append(chunk)

        # Create summary mapping
        summary_map = {s.chunk_id: s for s in chunk_summaries}

        file_summaries = []
        for file_path, file_chunks in by_file.items():
            # Collect chunk summaries for this file
            chunk_texts = []
            classes = []
            functions = []

            for chunk in file_chunks:
                if chunk.id in summary_map:
                    chunk_texts.append(
                        f"- {chunk.name} ({chunk.type}): {summary_map[chunk.id].summary}"
                    )

                if chunk.type == 'class':
                    classes.append({"name": chunk.name, "purpose": summary_map.get(chunk.id, ChunkSummary(chunk.id, "")).summary})
                elif chunk.type in ('function', 'method'):
                    functions.append({"name": chunk.name, "purpose": summary_map.get(chunk.id, ChunkSummary(chunk.id, "")).summary})

            # Generate file summary with LLM
            prompt = f"""Summarize this file's purpose and contents.

File: {file_path}
Language: {file_chunks[0].language}
Package: {getattr(file_chunks[0], 'package_name', 'unknown')}

Components:
{chr(10).join(chunk_texts)}

Imports: {list(set(imp for c in file_chunks for imp in c.imports))[:10]}

Provide JSON:
{{
  "primary_responsibility": "Main purpose of this file in 1 sentence",
  "summary": "Detailed explanation of what this file does (3-4 sentences)",
  "design_patterns": ["patterns", "used"] or []
}}

Be specific and technical. Reference actual classes/functions.
"""

            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

            try:
                data = json.loads(response)
                file_summaries.append(FileSummary(
                    file_path=file_path,
                    language=file_chunks[0].language,
                    package=getattr(file_chunks[0], 'package_name', ''),
                    primary_responsibility=data['primary_responsibility'],
                    summary=data['summary'],
                    chunks=[c.id for c in file_chunks],
                    classes=classes,
                    functions=functions,
                    imports=list(set(imp for c in file_chunks for imp in c.imports)),
                    design_patterns=data.get('design_patterns', []),
                ))
            except:
                # Fallback
                file_summaries.append(FileSummary(
                    file_path=file_path,
                    language=file_chunks[0].language,
                    package=getattr(file_chunks[0], 'package_name', ''),
                    primary_responsibility=f"Implementation in {file_path}",
                    summary=response[:500],
                    chunks=[c.id for c in file_chunks],
                    classes=classes,
                    functions=functions,
                ))

        return file_summaries

    def generate_package_summaries(self, file_summaries: List[FileSummary],
                                   chunks: List[Chunk],
                                   entry_points: List[EntryPoint]) -> List[PackageSummary]:
        """Aggregate files into package summaries"""

        # Group files by package
        by_package = defaultdict(list)
        for fs in file_summaries:
            package = self._extract_package(fs.file_path)
            by_package[package].append(fs)

        package_summaries = []
        for package_path, pkg_files in by_package.items():
            # Collect file purposes
            file_purposes = [f"- {fs.file_path}: {fs.primary_responsibility}"
                            for fs in pkg_files]

            # Find entry points in this package
            pkg_entry_points = [ep for ep in entry_points
                               if any(ep.file == fs.file_path for fs in pkg_files)]

            prompt = f"""Summarize this package's purpose and role.

Package: {package_path}

Files and their purposes:
{chr(10).join(file_purposes)}

Entry points: {[f"{ep.type}: {ep.name}" for ep in pkg_entry_points]}

Provide JSON:
{{
  "purpose": "What this package does (1-2 sentences)",
  "summary": "Detailed description of package responsibilities (3-4 sentences)",
  "exported_apis": ["public", "interfaces"],
  "design_patterns": ["patterns"] or []
}}
"""

            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

            try:
                data = json.loads(response)
                package_summaries.append(PackageSummary(
                    package_name=package_path,
                    path=package_path,
                    purpose=data['purpose'],
                    summary=data['summary'],
                    files=[fs.file_path for fs in pkg_files],
                    entry_points=pkg_entry_points,
                    exported_apis=data.get('exported_apis', []),
                    design_patterns=data.get('design_patterns', []),
                ))
            except:
                package_summaries.append(PackageSummary(
                    package_name=package_path,
                    path=package_path,
                    purpose=f"Package: {package_path}",
                    summary=response[:500],
                    files=[fs.file_path for fs in pkg_files],
                ))

        return package_summaries

    def generate_feature_summaries(self, entry_points: List[EntryPoint],
                                   chunks: List[Chunk],
                                   file_summaries: List[FileSummary]) -> List[FeatureSummary]:
        """Trace features from entry points through call graph"""

        feature_summaries = []

        for ep in entry_points:
            # Trace call graph from this entry point
            call_chain = self._trace_call_graph(ep, chunks)

            # Collect files involved
            files_involved = list(set(chunk.file for chunk in call_chain))

            # Generate feature summary
            prompt = f"""Describe this feature implementation.

Entry Point: {ep.type} - {ep.name}
File: {ep.file}
Metadata: {ep.metadata}

Call sequence ({len(call_chain)} functions):
{self._format_call_chain(call_chain[:20])}

Files involved: {files_involved}

Provide JSON:
{{
  "name": "Feature name (e.g., 'User Authentication API')",
  "description": "What this feature does from user perspective",
  "summary": "Technical flow explanation (4-5 sentences)",
  "data_flow": "Input → Processing → Output description",
  "external_calls": ["External APIs", "Database", "Message queues"] or []
}}
"""

            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

            try:
                data = json.loads(response)
                feature_summaries.append(FeatureSummary(
                    feature_id=f"feature_{ep.name}_{hash(ep.file) % 10000}",
                    name=data['name'],
                    description=data['description'],
                    summary=data['summary'],
                    entry_point=ep,
                    files_involved=files_involved,
                    functions_involved=[c.id for c in call_chain],
                    data_flow=data['data_flow'],
                    external_calls=data.get('external_calls', []),
                ))
            except:
                feature_summaries.append(FeatureSummary(
                    feature_id=f"feature_{ep.name}_{hash(ep.file) % 10000}",
                    name=ep.name,
                    description=f"Feature: {ep.name}",
                    summary=response[:500],
                    entry_point=ep,
                    files_involved=files_involved,
                    functions_involved=[c.id for c in call_chain],
                ))

        return feature_summaries

    def generate_codebase_summary(self, repo_name: str, repo_path: str,
                                 package_summaries: List[PackageSummary],
                                 feature_summaries: List[FeatureSummary],
                                 entry_points: List[EntryPoint],
                                 chunks: List[Chunk]) -> CodebaseSummary:
        """Top-level codebase summary synthesis"""

        # Detect tech stack
        tech_detector = TechStackDetector()
        tech_stack = tech_detector.detect_from_dependencies(repo_path)
        tech_stack.update(tech_detector.detect_from_imports(chunks))

        # Statistics
        languages = defaultdict(int)
        for chunk in chunks:
            languages[chunk.language] += len(chunk.code.split('\n'))

        total_functions = sum(1 for c in chunks if c.type in ('function', 'method'))
        total_classes = sum(1 for c in chunks if c.type == 'class')

        # Group entry points by type
        ep_by_type = defaultdict(list)
        for ep in entry_points:
            ep_by_type[ep.type].append(ep)

        prompt = f"""Create a comprehensive codebase summary for a new developer.

Repository: {repo_name}

Statistics:
- Files: {len(set(c.file for c in chunks))}
- Functions: {total_functions}
- Classes: {total_classes}
- Languages: {dict(languages)}

Entry Points ({len(entry_points)} total):
{self._format_entry_points(ep_by_type)}

Packages ({len(package_summaries)}):
{chr(10).join(f"- {ps.package_name}: {ps.purpose}" for ps in package_summaries[:10])}

Features ({len(feature_summaries)}):
{chr(10).join(f"- {fs.name}: {fs.description}" for fs in feature_summaries[:10])}

Technology Stack:
{tech_stack}

Provide JSON:
{{
  "purpose": "What this codebase does (2-3 sentences)",
  "domain": "Domain/industry (e.g., 'E-commerce Backend', 'ML Pipeline')",
  "architecture_pattern": "Microservice/Monolith/Serverless/Layered/etc",
  "summary": "Comprehensive overview (5-6 sentences)",
  "core_functionalities": [
    {{"name": "Functionality 1", "description": "...", "files": ["..."]}}
  ],
  "key_design_decisions": ["Decision 1", "Decision 2"],
  "external_integrations": ["Database", "API", "Queue"]
}}

Be thorough and technical. This is for onboarding new developers.
"""

        response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)

        try:
            data = json.loads(response)

            # Create entry point summary
            ep_summary = self._create_entry_point_summary(ep_by_type)

            return CodebaseSummary(
                repo_name=repo_name,
                purpose=data['purpose'],
                domain=data['domain'],
                architecture_pattern=data['architecture_pattern'],
                tech_stack=tech_stack,
                entry_points=entry_points,
                entry_point_summary=ep_summary,
                core_functionalities=data['core_functionalities'],
                total_files=len(set(c.file for c in chunks)),
                total_functions=total_functions,
                total_classes=total_classes,
                languages=dict(languages),
                summary=data['summary'],
                key_design_decisions=data.get('key_design_decisions', []),
                external_integrations=data.get('external_integrations', []),
            )
        except:
            # Fallback
            return CodebaseSummary(
                repo_name=repo_name,
                purpose=f"Codebase: {repo_name}",
                domain="Unknown",
                architecture_pattern="Unknown",
                tech_stack=tech_stack,
                entry_points=entry_points,
                entry_point_summary=self._create_entry_point_summary(ep_by_type),
                total_files=len(set(c.file for c in chunks)),
                total_functions=total_functions,
                total_classes=total_classes,
                languages=dict(languages),
                summary=response[:1000],
            )

    # Helper methods...
    def _infer_file_purpose(self, chunks: List[Chunk]) -> str:
        # Quick inference from chunk names and types
        return f"File containing {len(chunks)} components"

    def _extract_package(self, file_path: str) -> str:
        # Extract package from file path
        parts = file_path.split('/')
        return '/'.join(parts[:-1]) if len(parts) > 1 else 'root'

    def _trace_call_graph(self, entry_point: EntryPoint, chunks: List[Chunk]) -> List[Chunk]:
        # BFS traversal from entry point
        chunk_map = {c.id: c for c in chunks}
        visited = set()
        queue = deque([entry_point.chunk_id])
        result = []

        while queue and len(result) < 50:  # Limit depth
            chunk_id = queue.popleft()
            if chunk_id in visited or chunk_id not in chunk_map:
                continue

            visited.add(chunk_id)
            chunk = chunk_map[chunk_id]
            result.append(chunk)

            # Add called functions to queue
            for call in chunk.calls[:5]:  # Limit breadth
                for c in chunks:
                    if c.name == call and c.id not in visited:
                        queue.append(c.id)
                        break

        return result

    def _format_call_chain(self, chunks: List[Chunk]) -> str:
        return "\n".join(
            f"{i+1}. {c.name} ({c.type}) in {c.file}"
            for i, c in enumerate(chunks)
        )

    def _format_entry_points(self, ep_by_type: Dict) -> str:
        lines = []
        for ep_type, eps in ep_by_type.items():
            lines.append(f"\n{ep_type.upper()} ({len(eps)}):")
            for ep in eps[:5]:
                metadata_str = ', '.join(f"{k}={v}" for k, v in list(ep.metadata.items())[:3])
                lines.append(f"  - {ep.name} ({metadata_str})")
        return "\n".join(lines)

    def _create_entry_point_summary(self, ep_by_type: Dict) -> str:
        parts = []
        for ep_type, eps in ep_by_type.items():
            parts.append(f"{len(eps)} {ep_type}(s)")
        return f"Total: {sum(len(eps) for eps in ep_by_type.values())} entry points - " + ", ".join(parts)
```

---

## 💾 Phase 3: Storage Extensions

### 3.1 **Neo4j Graph Schema**

```cypher
// New node types
CREATE CONSTRAINT codebase_name IF NOT EXISTS FOR (c:Codebase) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT package_path IF NOT EXISTS FOR (p:Package) REQUIRE (p.repo, p.path) IS UNIQUE;
CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE;

// Create codebase node
CREATE (cb:Codebase {
  name: $repo_name,
  purpose: $purpose,
  domain: $domain,
  architecture: $architecture,
  tech_stack: $tech_stack_json,
  summary: $summary,
  entry_point_count: $ep_count,
  indexed_at: datetime()
})

// Create package nodes and relationships
MATCH (cb:Codebase {name: $repo_name})
CREATE (p:Package {
  repo: $repo_name,
  path: $package_path,
  name: $package_name,
  purpose: $purpose,
  summary: $summary
})
CREATE (cb)-[:HAS_PACKAGE]->(p)

// Link files to packages
MATCH (p:Package {repo: $repo_name, path: $package_path})
MATCH (f:File {repo: $repo_name, path: $file_path})
CREATE (p)-[:CONTAINS_FILE]->(f)

// Create feature nodes
CREATE (feat:Feature {
  id: $feature_id,
  repo: $repo_name,
  name: $name,
  description: $description,
  summary: $summary,
  data_flow: $data_flow
})

// Link feature to entry point
MATCH (feat:Feature {id: $feature_id})
MATCH (ep:Chunk {id: $entry_point_chunk_id, repo: $repo_name})
CREATE (feat)-[:STARTS_AT]->(ep)

// Link feature to all involved chunks
MATCH (feat:Feature {id: $feature_id})
MATCH (c:Chunk {id: $chunk_id, repo: $repo_name})
CREATE (feat)-[:USES]->(c)

// Query examples:

// Get all entry points
MATCH (cb:Codebase {name: $repo})-[:HAS_PACKAGE]->(p)-[:CONTAINS_FILE]->(f)-[:CONTAINS]->(c:Chunk)
WHERE c.is_entry_point = true
RETURN c, c.entry_point_type, c.entry_point_metadata

// Get feature flow
MATCH (f:Feature {id: $feature_id})-[:STARTS_AT]->(entry)
MATCH (f)-[:USES]->(c:Chunk)
RETURN entry, collect(c) AS flow

// Get package dependencies
MATCH (p1:Package {repo: $repo})-[:CONTAINS_FILE]->(f1)-[:CONTAINS]->(c1)
MATCH (c1)-[:CALLS|IMPORTS]->(c2)<-[:CONTAINS]-(f2)<-[:CONTAINS_FILE]-(p2:Package)
WHERE p1.path <> p2.path
RETURN p1.name, p2.name, count(*) AS dependency_count
```

### 3.2 **ChromaDB Collections**

```python
# vectordb.py - Enhanced with multiple collections

class VectorDB:
    def __init__(self, ...):
        # ... existing code ...
        self.collection_types = {
            'chunks': 'entities',  # existing
            'files': 'file_summaries',
            'packages': 'package_summaries',
            'features': 'feature_summaries',
            'codebase': 'codebase_summary',
        }

    def index_summaries(self, repo_name: str, summaries: Dict[SummaryLevel, List]):
        """Index all summary levels"""

        # Level 5: Chunks (existing)
        # Already indexed

        # Level 3: File summaries
        file_data = []
        for fs in summaries[SummaryLevel.FILE]:
            file_data.append({
                "id": f"{repo_name}::file::{fs.file_path}",
                "text": f"{fs.primary_responsibility}\n\n{fs.summary}",
                "metadata": {
                    "file": fs.file_path,
                    "package": fs.package,
                    "type": "file_summary",
                    "classes_count": len(fs.classes),
                    "functions_count": len(fs.functions),
                }
            })

        self._index_collection(repo_name, 'file_summaries', file_data)

        # Level 2: Package summaries
        pkg_data = []
        for ps in summaries[SummaryLevel.PACKAGE]:
            pkg_data.append({
                "id": f"{repo_name}::package::{ps.package_name}",
                "text": f"{ps.purpose}\n\n{ps.summary}",
                "metadata": {
                    "package": ps.package_name,
                    "type": "package_summary",
                    "files_count": len(ps.files),
                    "entry_points_count": len(ps.entry_points),
                }
            })

        self._index_collection(repo_name, 'package_summaries', pkg_data)

        # Level 4: Feature summaries
        feat_data = []
        for feat in summaries[SummaryLevel.FEATURE]:
            feat_data.append({
                "id": f"{repo_name}::feature::{feat.feature_id}",
                "text": f"{feat.name}\n{feat.description}\n\n{feat.summary}\n\nData Flow: {feat.data_flow}",
                "metadata": {
                    "feature_id": feat.feature_id,
                    "feature_name": feat.name,
                    "type": "feature_summary",
                    "entry_point_type": feat.entry_point.type,
                    "files_count": len(feat.files_involved),
                }
            })

        self._index_collection(repo_name, 'feature_summaries', feat_data)

        # Level 1: Codebase summary
        cb = summaries[SummaryLevel.CODEBASE][0]
        cb_data = [{
            "id": f"{repo_name}::codebase",
            "text": f"{cb.purpose}\n\n{cb.summary}\n\nDomain: {cb.domain}\nArchitecture: {cb.architecture_pattern}\n\n{cb.entry_point_summary}",
            "metadata": {
                "repo": repo_name,
                "type": "codebase_summary",
                "domain": cb.domain,
                "architecture": cb.architecture_pattern,
                "entry_points_count": len(cb.entry_points),
            }
        }]

        self._index_collection(repo_name, 'codebase_summary', cb_data)

    def _index_collection(self, repo_name: str, collection_type: str, data: List[Dict]):
        col_name = self._get_collection_name(repo_name, collection_type)
        # ... same as existing index_batch logic ...
```

---

## 🔍 Phase 4: Enhanced Query System

### 4.1 **Hierarchical Query Router**

```python
# hierarchical_query.py

class HierarchicalQueryEngine:
    def __init__(self, llm_provider="claude", model=None):
        self.llm = get_llm(llm_provider, model=model)
        self.graph = Graph()
        self.vector_db = VectorDB()

    def answer(self, query: str, repo_name: str) -> str:
        """Route query to appropriate summary level"""

        # Classify query intent
        intent = self._classify_query_intent(query)

        # Route to appropriate handler
        handlers = {
            'overview': self._handle_overview_query,
            'entry_points': self._handle_entry_point_query,
            'feature': self._handle_feature_query,
            'implementation': self._handle_implementation_query,
            'architecture': self._handle_architecture_query,
            'package': self._handle_package_query,
        }

        handler = handlers.get(intent, self._handle_implementation_query)
        return handler(query, repo_name)

    def _classify_query_intent(self, query: str) -> str:
        """Determine what level of summary is needed"""

        q_lower = query.lower()

        # Pattern matching for intent
        if any(word in q_lower for word in ['what does this codebase', 'overall purpose', 'what is this project', 'overview']):
            return 'overview'

        if any(word in q_lower for word in ['entry point', 'how to start', 'api endpoints', 'kafka listeners', 'cli commands']):
            return 'entry_points'

        if any(word in q_lower for word in ['how does', 'flow', 'feature', 'work', 'implement']):
            return 'feature'

        if any(word in q_lower for word in ['architecture', 'structure', 'organized', 'design pattern']):
            return 'architecture'

        if any(word in q_lower for word in ['package', 'module', 'component']):
            return 'package'

        return 'implementation'

    def _handle_overview_query(self, query: str, repo_name: str) -> str:
        """Answer with codebase-level summary"""

        # Get codebase summary from ChromaDB
        results = self.vector_db.search(
            repo_name, query, k=1,
            collection_type='codebase_summary'
        )

        if not results:
            return "Codebase summary not found."

        # Get detailed codebase node from Neo4j
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (cb:Codebase {name: $repo})
                OPTIONAL MATCH (cb)-[:HAS_PACKAGE]->(p:Package)
                RETURN cb, collect(p) AS packages
            """, repo=repo_name).single()

            if not result:
                return "Codebase not found in graph."

            cb_data = dict(result['cb'])
            packages = [dict(p) for p in result['packages']]

        # Format response
        prompt = f"""Answer the user's question using this codebase summary.

Question: {query}

Codebase Summary:
Name: {cb_data['name']}
Purpose: {cb_data['purpose']}
Domain: {cb_data['domain']}
Architecture: {cb_data['architecture']}
Summary: {cb_data['summary']}

Entry Points: {cb_data.get('entry_point_count', 0)} total

Packages ({len(packages)}):
{chr(10).join(f"- {p['name']}: {p['purpose']}" for p in packages[:10])}

Technology Stack: {cb_data.get('tech_stack', 'Unknown')}

Provide a comprehensive answer that helps a new developer understand this codebase.
"""

        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)

    def _handle_entry_point_query(self, query: str, repo_name: str) -> str:
        """List and explain all entry points"""

        with self.graph.driver.session() as session:
            # Get all entry points
            results = session.run("""
                MATCH (c:Chunk {repo: $repo})
                WHERE c.is_entry_point = true
                RETURN c.id AS id, c.name AS name, c.file AS file,
                       c.entry_point_type AS type, c.entry_point_metadata AS metadata
                ORDER BY c.entry_point_type, c.name
            """, repo=repo_name)

            entry_points = []
            for record in results:
                entry_points.append({
                    'name': record['name'],
                    'type': record['type'],
                    'file': record['file'],
                    'metadata': record['metadata'],
                })

        if not entry_points:
            return "No entry points detected in this codebase."

        # Group by type
        by_type = defaultdict(list)
        for ep in entry_points:
            by_type[ep['type']].append(ep)

        # Format
        formatted = []
        for ep_type, eps in by_type.items():
            formatted.append(f"\n**{ep_type.upper()}** ({len(eps)} total):")
            for ep in eps:
                metadata_str = ', '.join(f"{k}={v}" for k, v in ep.get('metadata', {}).items())
                formatted.append(f"  - `{ep['name']}` in {ep['file']}")
                if metadata_str:
                    formatted.append(f"    {metadata_str}")

        entry_point_text = "\n".join(formatted)

        prompt = f"""Answer the user's question about entry points.

Question: {query}

Detected Entry Points:
{entry_point_text}

Total: {len(entry_points)} entry points across {len(by_type)} categories.

Explain what these entry points do and how to use them.
"""

        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)

    def _handle_feature_query(self, query: str, repo_name: str) -> str:
        """Explain a feature with flow"""

        # Search feature summaries
        results = self.vector_db.search(
            repo_name, query, k=3,
            collection_type='feature_summaries'
        )

        if not results:
            # Fallback to regular implementation search
            return self._handle_implementation_query(query, repo_name)

        # Get top matching feature
        feature_id = results[0]['metadata']['feature_id']

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (f:Feature {id: $feature_id})
                MATCH (f)-[:STARTS_AT]->(entry:Chunk)
                MATCH (f)-[:USES]->(c:Chunk)
                RETURN f, entry, collect(c) AS chunks
            """, feature_id=feature_id).single()

            if not result:
                return "Feature not found."

            feature_data = dict(result['f'])
            entry_point = dict(result['entry'])
            chunks = [dict(c) for c in result['chunks']]

        # Format call flow
        call_flow = "\n".join(
            f"{i+1}. {c['name']} ({c['type']}) in {c['file']}"
            for i, c in enumerate(chunks[:20])
        )

        prompt = f"""Explain this feature implementation.

Question: {query}

Feature: {feature_data['name']}
Description: {feature_data['description']}
Summary: {feature_data['summary']}

Entry Point: {entry_point['name']} ({entry_point.get('entry_point_type', 'function')})
File: {entry_point['file']}

Call Flow ({len(chunks)} functions):
{call_flow}

Data Flow: {feature_data.get('data_flow', 'N/A')}

Provide a clear explanation of how this feature works, step by step.
"""

        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

    def _handle_architecture_query(self, query: str, repo_name: str) -> str:
        """Explain architecture and structure"""

        with self.graph.driver.session() as session:
            # Get codebase and packages
            result = session.run("""
                MATCH (cb:Codebase {name: $repo})
                MATCH (cb)-[:HAS_PACKAGE]->(p:Package)
                OPTIONAL MATCH (p)-[:CONTAINS_FILE]->(f:File)
                RETURN cb, p, count(f) AS file_count
                ORDER BY p.name
            """, repo=repo_name)

            packages = []
            codebase = None

            for record in result:
                if not codebase:
                    codebase = dict(record['cb'])
                packages.append({
                    'name': record['p']['name'],
                    'purpose': record['p']['purpose'],
                    'file_count': record['file_count'],
                })

        prompt = f"""Explain the codebase architecture.

Question: {query}

Architecture Pattern: {codebase.get('architecture', 'Unknown')}
Domain: {codebase.get('domain', 'Unknown')}

Packages ({len(packages)}):
{chr(10).join(f"- {p['name']} ({p['file_count']} files): {p['purpose']}" for p in packages)}

Provide architectural overview with package responsibilities and relationships.
"""

        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)

    def _handle_package_query(self, query: str, repo_name: str) -> str:
        """Explain package/module"""

        results = self.vector_db.search(
            repo_name, query, k=3,
            collection_type='package_summaries'
        )

        if not results:
            return "Package not found."

        package_name = results[0]['metadata']['package']

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (p:Package {repo: $repo, path: $package})
                MATCH (p)-[:CONTAINS_FILE]->(f:File)
                RETURN p, collect(f) AS files
            """, repo=repo_name, package=package_name).single()

            if not result:
                return "Package not found in graph."

            pkg_data = dict(result['p'])
            files = [dict(f) for f in result['files']]

        prompt = f"""Explain this package.

Question: {query}

Package: {pkg_data['name']}
Purpose: {pkg_data['purpose']}
Summary: {pkg_data['summary']}

Files ({len(files)}):
{chr(10).join(f"- {f['path']}" for f in files[:20])}

Explain what this package does and how it fits into the overall system.
"""

        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)

    def _handle_implementation_query(self, query: str, repo_name: str) -> str:
        """Handle detailed implementation questions (existing logic)"""
        # Use existing query.py logic with graph expansion
        from query import answer_query
        return answer_query(query, repo_name)
```

---

## 🚀 Phase 5: Integration into Indexer

### 5.1 **Modified indexer.py**

```python
# indexer.py - Enhanced with summary generation

from entry_points import EntryPointDetector
from summary_generator import SummaryGenerator
from summaries import SummaryLevel

def index_repository(repo_identifier: str,
                     force_reindex: bool = False,
                     embedding_provider: str = "huggingface",
                     **embedding_kwargs) -> str:

    # ... existing code up to chunk parsing ...

    chunks, file_package_map = parse_all_files(repo_path, module_name)

    if not chunks:
        print("❌ No chunks extracted!")
        return repo_name

    # ═══════════════════════════════════════════════════════
    # NEW: Entry Point Detection
    # ═══════════════════════════════════════════════════════
    print("\n🔍 Detecting entry points...")
    entry_detector = EntryPointDetector(llm_provider=embedding_kwargs.get('llm_provider', 'claude'))
    entry_points = entry_detector.detect_all_entry_points(chunks, repo_path)
    print(f"  ✓ Found {len(entry_points)} entry points")

    # Mark chunks as entry points
    entry_chunk_ids = {ep.chunk_id for ep in entry_points}
    for chunk in chunks:
        if chunk.id in entry_chunk_ids:
            chunk.is_entry_point = True
            ep = next(ep for ep in entry_points if ep.chunk_id == chunk.id)
            chunk.entry_point_type = ep.type
            chunk.entry_point_metadata = ep.metadata

    # ═══════════════════════════════════════════════════════
    # NEW: Multi-Level Summary Generation
    # ═══════════════════════════════════════════════════════
    print("\n📝 Generating multi-level summaries...")
    summary_gen = SummaryGenerator(llm_provider=embedding_kwargs.get('llm_provider', 'claude'))
    all_summaries = summary_gen.generate_all_summaries(
        chunks, entry_points, repo_name, repo_path
    )

    # ═══════════════════════════════════════════════════════
    # Store in Graph (Enhanced)
    # ═══════════════════════════════════════════════════════
    print("\n🔗 Building enhanced graph...")
    graph.store_chunks(chunks, repo_name, go_package_map=go_package_map)
    graph.store_summaries(all_summaries, repo_name)  # NEW method

    # ═══════════════════════════════════════════════════════
    # Store in Vector DB (Enhanced)
    # ═══════════════════════════════════════════════════════
    print("\n🔢 Embedding & indexing (all levels)...")

    # Existing: Chunk embeddings
    vector_data = [{
        "id":   c.id,
        "text": c.embedding_text(),
        "metadata": {
            "name":    c.name,
            "type":    c.type,
            "file":    c.file,
            "is_test": c.is_test,
            "is_entry_point": getattr(c, 'is_entry_point', False),
            "entry_point_type": getattr(c, 'entry_point_type', None),
        }
    } for c in chunks]
    vector_db.index_batch(repo_name, vector_data)

    # NEW: Summary embeddings
    vector_db.index_summaries(repo_name, all_summaries)

    print(f"\n✅ Done — {len(chunks)} chunks + multi-level summaries indexed.")

    # Print summary
    print("\n" + "="*60)
    print("📊 INDEXING SUMMARY")
    print("="*60)
    cb_summary = all_summaries[SummaryLevel.CODEBASE][0]
    print(f"Repository: {repo_name}")
    print(f"Purpose: {cb_summary.purpose}")
    print(f"Domain: {cb_summary.domain}")
    print(f"Architecture: {cb_summary.architecture_pattern}")
    print(f"\nEntry Points: {len(entry_points)}")
    print(cb_summary.entry_point_summary)
    print(f"\nPackages: {len(all_summaries[SummaryLevel.PACKAGE])}")
    print(f"Features: {len(all_summaries[SummaryLevel.FEATURE])}")
    print("="*60)

    graph.close()
    return repo_name
```

### 5.2 **Modified query.py**

```python
# query.py - Use hierarchical query engine

from hierarchical_query import HierarchicalQueryEngine

def answer_query(query, repo_name, llm_provider="claude", llm_model=None,
                 embedding_provider="huggingface", **embedding_kwargs):

    print(f"\n{'='*60}")
    print(f"❓ Question: {query}")
    print(f"{'='*60}\n")

    # Use hierarchical query engine
    query_engine = HierarchicalQueryEngine(llm_provider=llm_provider, model=llm_model)
    answer = query_engine.answer(query, repo_name)

    print("✅ Answer generated\n")
    return answer
```

---

## 📦 New Files to Create

1. **`entry_points.py`** - Dynamic entry point detection
2. **`summaries.py`** - Data models for all summary levels
3. **`summary_generator.py`** - LLM-based summary generation
4. **`hierarchical_query.py`** - Multi-level query routing
5. **`entry_point_patterns.yaml`** - Configurable detection patterns (no hardcoding!)

---

## ✅ Anti-Hallucination Guarantees

### 1. **Grounded Generation**
- Every summary references actual code locations (file:line)
- LLM prompts include actual code, not descriptions
- Verification step: cross-check generated summary against code

### 2. **Fact Extraction First**
```python
def generate_summary_with_verification(chunk):
    # Step 1: Extract facts from code
    facts = {
        'name': chunk.name,
        'type': chunk.type,
        'calls': chunk.calls,
        'parameters': chunk.params,
        'returns': chunk.returns,
        'file': chunk.file,
        'lines': f"{chunk.start}-{chunk.end}",
    }

    # Step 2: Generate summary grounded in facts
    summary = llm.generate(facts, code=chunk.code)

    # Step 3: Verify summary mentions real facts
    assert chunk.name in summary
    assert any(call in summary or 'calls' in summary for call in chunk.calls)

    return summary
```

### 3. **Source Attribution**
Every statement in summaries includes source:
- "In `app.py:45`, the `create_user` function..."
- "This package exposes 3 APIs (see `routes/users.py`, `routes/products.py`...)"

### 4. **Consistency Checks**
- Child summaries must align with parent summaries
- Cross-reference between levels
- Detect contradictions

---

## 🎯 Implementation Checklist

- [ ] **Phase 1: Data Models** (2-3 hours)
  - [ ] Create `summaries.py` with all dataclasses
  - [ ] Add new fields to `chunk.py`
  - [ ] Create `entry_point_patterns.yaml`

- [ ] **Phase 2: Entry Point Detection** (3-4 hours)
  - [ ] Create `entry_points.py`
  - [ ] Implement AST-based detectors for each language
  - [ ] Implement Tree-Sitter query patterns
  - [ ] Add LLM-assisted classification
  - [ ] Test on sample codebases

- [ ] **Phase 3: Summary Generation** (5-6 hours)
  - [ ] Create `summary_generator.py`
  - [ ] Implement chunk summary generation
  - [ ] Implement file summary generation
  - [ ] Implement package summary generation
  - [ ] Implement feature summary generation (with call tracing)
  - [ ] Implement codebase summary generation
  - [ ] Add verification logic

- [ ] **Phase 4: Storage Extensions** (2-3 hours)
  - [ ] Extend Neo4j schema in `graph.py`
  - [ ] Add `store_summaries()` method
  - [ ] Extend ChromaDB collections in `vectordb.py`
  - [ ] Add `index_summaries()` method

- [ ] **Phase 5: Query Enhancement** (3-4 hours)
  - [ ] Create `hierarchical_query.py`
  - [ ] Implement query intent classification
  - [ ] Implement level-specific query handlers
  - [ ] Integrate with existing `query.py`

- [ ] **Phase 6: Integration** (2-3 hours)
  - [ ] Modify `indexer.py` to call summary generation
  - [ ] Update `app.py` UI to show summaries
  - [ ] Add summary display in Streamlit

- [ ] **Phase 7: Testing & Refinement** (3-4 hours)
  - [ ] Test on Python codebase
  - [ ] Test on Go codebase
  - [ ] Test on JavaScript codebase
  - [ ] Verify no hallucinations
  - [ ] Verify all entry points detected
  - [ ] Performance optimization

---

## 🔬 Example Queries & Expected Responses

### Query 1: "What does this codebase do?"

**Response using Level 1 (Codebase Summary):**
```
This is a **Codebase RAG** system that enables intelligent code understanding through multi-language parsing, graph-based relationship modeling, and LLM-powered Q&A.

**Domain:** Developer Tools / Code Intelligence

**Architecture:** Layered Architecture with separate indexing and query pipelines

**Entry Points (7 total):**
- **HTTP APIs (3):**
  - POST /index - Repository indexing endpoint
  - POST /query - Question answering endpoint
  - GET /stats - Statistics endpoint
- **CLI Commands (2):**
  - `python indexer.py <url>` - Index from command line
  - `python query.py <repo> <question>` - Query from CLI
- **Streamlit UI (2):**
  - Web interface for repository indexing
  - Interactive chat interface

**Core Functionalities:**
1. Multi-language code parsing (Python, Go, Java, JavaScript/TypeScript)
2. Graph-based relationship modeling using Neo4j
3. Semantic code search with ChromaDB
4. LLM-powered contextual Q&A
5. Intelligent call graph traversal

**Technology Stack:**
- Languages: Python
- Databases: Neo4j (graph), ChromaDB (vector)
- LLMs: Claude, GPT-4, Gemini, DeepSeek
- Parsers: Tree-sitter
- UI: Streamlit
```

### Query 2: "Show me all entry points"

**Response using Entry Point Detection:**
```
This codebase has **7 entry points** across 3 categories:

**HTTP ENDPOINTS (3):**
1. `index_endpoint` in app.py:150
   - Method: POST
   - Path: /index
   - Purpose: Triggers repository indexing

2. `query_endpoint` in app.py:185
   - Method: POST
   - Path: /query
   - Purpose: Answers code questions

3. `stats_endpoint` in app.py:210
   - Method: GET
   - Path: /stats
   - Purpose: Returns repository statistics

**CLI COMMANDS (2):**
1. `main` in indexer.py:220
   - Command: `python indexer.py <url> [--force]`
   - Purpose: Index repository from command line

2. `main` in query.py:180
   - Command: `python query.py <repo> <question>`
   - Purpose: Query repository from CLI

**STREAMLIT UI (2):**
1. Index UI in app.py:50-120
   - Purpose: Web interface for indexing repositories

2. Chat UI in app.py:200-350
   - Purpose: Interactive question-answering interface
```

### Query 3: "How does the indexing feature work?"

**Response using Level 4 (Feature Summary):**
```
The **Repository Indexing** feature processes codebases in 5 stages:

**Entry Point:** POST /index or `index_repository()` function

**Flow:**
1. **Repository Acquisition** (app.py:150 → indexer.py:45)
   - Clones Git repository or loads local path
   - File: indexer.py, function: `clone_repo()`

2. **Code Parsing** (indexer.py:85 → parsers/*)
   - Detects file languages
   - Parses with Tree-sitter AST
   - Extracts functions, classes, calls, imports
   - Files: parse_python.py, parse_go.py, parse_java.py, parse_javascript.py

3. **Graph Building** (graph.py:120)
   - Creates Neo4j nodes for chunks
   - Builds relationships: CALLS, IMPORTS, HAS_MEMBER, CONTAINS
   - File: graph.py, function: `store_chunks()`

4. **Vector Embedding** (vectordb.py:45)
   - Generates embeddings for each chunk
   - Stores in ChromaDB
   - File: vectordb.py, function: `index_batch()`

5. **Summary Generation** (summary_generator.py:30) [NEW]
   - Detects entry points
   - Generates multi-level summaries
   - Stores in both Neo4j and ChromaDB

**Data Flow:**
GitHub URL → Git Clone → File Reading → AST Parsing → Chunk Extraction → Graph Storage + Vector Indexing + Summary Generation → Ready for Queries

**External Integrations:**
- Git (for cloning)
- Neo4j (graph database)
- ChromaDB (vector database)
- HuggingFace/Jina (embeddings)

**Files Involved:** indexer.py, graph.py, vectordb.py, chunk.py, parsers/*
```

---

## 📈 Success Criteria

✅ **Completeness:** Can list ALL entry points with 100% accuracy
✅ **Zero Missing Info:** Every function/class has contextual summary
✅ **No Hallucination:** Every fact verifiable in code (file:line references)
✅ **Newbie-Friendly:** High-level summaries understandable without coding knowledge
✅ **Technical Depth:** Can drill down to exact implementation details
✅ **Dynamic:** No hardcoded framework names - works on any codebase
✅ **Extensible:** Easy to add new languages and entry point types

---

## 🚀 Ready to Implement?

This plan provides a **complete, non-hardcoded, LLM-assisted** approach to generating hierarchical codebase summaries that eliminates hallucination through grounded generation and fact verification.

The key innovation is using **AST patterns + Tree-Sitter queries + LLM classification** instead of hardcoded framework lists, making it work on any codebase in any language with any framework.
