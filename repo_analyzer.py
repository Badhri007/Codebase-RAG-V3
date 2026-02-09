"""Repository analyzer for building high-level context."""
from typing import List, Dict, Set, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import re
import json

from chunk import Chunk


@dataclass
class EntryPoint:
    """Represents a code entry point."""
    name: str
    file: str
    type: str  # main, cli, api, script
    line: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Component:
    """Represents a logical component of the codebase."""
    name: str
    files: List[str]
    entry_points: List[str]
    purpose: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FileNode:
    """Represents a file or directory in the tree."""
    name: str
    path: str
    is_directory: bool
    children: List['FileNode'] = None
    chunk_count: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary recursively."""
        return {
            'name': self.name,
            'path': self.path,
            'is_directory': self.is_directory,
            'chunk_count': self.chunk_count,
            'children': [child.to_dict() for child in self.children] if self.children else []
        }


@dataclass
class RepoContext:
    """Complete repository context."""
    name: str
    entry_points: List[EntryPoint]
    architecture_pattern: str
    components: List[Component]
    tech_stack: Set[str]
    directory_structure: Dict[str, int]
    file_tree: FileNode  # NEW: Complete file tree
    all_files: List[str]  # NEW: All file paths
    total_files: int
    total_chunks: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'entry_points': [ep.to_dict() for ep in self.entry_points],
            'architecture_pattern': self.architecture_pattern,
            'components': [comp.to_dict() for comp in self.components],
            'tech_stack': list(self.tech_stack),
            'directory_structure': self.directory_structure,
            'file_tree': self.file_tree.to_dict() if self.file_tree else None,
            'all_files': self.all_files,
            'total_files': self.total_files,
            'total_chunks': self.total_chunks
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_file(self, filepath: str):
        """Save repo context to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        print(f"✓ Repository context saved to: {filepath}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepoContext':
        """Create RepoContext from dictionary."""
        return cls(
            name=data['name'],
            entry_points=[EntryPoint(**ep) for ep in data['entry_points']],
            architecture_pattern=data['architecture_pattern'],
            components=[Component(**comp) for comp in data['components']],
            tech_stack=set(data['tech_stack']),
            directory_structure=data['directory_structure'],
            file_tree=cls._dict_to_file_node(data['file_tree']) if data.get('file_tree') else None,
            all_files=data['all_files'],
            total_files=data['total_files'],
            total_chunks=data['total_chunks']
        )

    @staticmethod
    def _dict_to_file_node(data: Dict[str, Any]) -> FileNode:
        """Convert dictionary to FileNode recursively."""
        node = FileNode(
            name=data['name'],
            path=data['path'],
            is_directory=data['is_directory'],
            chunk_count=data.get('chunk_count', 0)
        )
        if data.get('children'):
            node.children = [RepoContext._dict_to_file_node(child) for child in data['children']]
        return node

    @classmethod
    def from_json(cls, json_str: str) -> 'RepoContext':
        """Load RepoContext from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'RepoContext':
        """Load repo context from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


class RepositoryAnalyzer:
    """Analyze repository structure and build high-level context."""

    # Patterns for detecting entry points
    ENTRY_POINT_PATTERNS = {
        'main_function': r'def\s+main\s*\(',
        'python_main': r'if\s+__name__\s*==\s*["\']__main__["\']',
        'flask_app': r'@app\.route\(|app\.run\(',
        'fastapi_app': r'@app\.(get|post|put|delete)\(|uvicorn\.run\(',
        'django_app': r'django\.setup\(\)|INSTALLED_APPS',
        'cli_click': r'@click\.(command|group)\(',
        'cli_argparse': r'argparse\.ArgumentParser\(',
        'express_app': r'app\.listen\(|express\(\)',
        'react_app': r'ReactDOM\.render\(|createRoot\(',
        'spring_boot': r'@SpringBootApplication|SpringApplication\.run',
        'go_main': r'func\s+main\s*\(',
    }

    # Architecture patterns detection
    ARCHITECTURE_KEYWORDS = {
        'mvc': ['models', 'views', 'controllers'],
        'layered': ['presentation', 'business', 'data', 'service', 'repository'],
        'microservices': ['service', 'api', 'gateway'],
        'clean': ['domain', 'application', 'infrastructure', 'presentation'],
        'modular': ['modules', 'packages', 'components'],
    }

    def analyze(self, chunks: List[Chunk], repo_name: str, all_scanned_files: List[str] = None) -> RepoContext:
        """Build comprehensive repository context."""

        print(f"\n📊 Analyzing repository structure...")

        entry_points = self._find_entry_points(chunks)
        architecture = self._detect_architecture_pattern(chunks)
        components = self._identify_components(chunks)
        tech_stack = self._identify_tech_stack(chunks)
        directory_structure = self._analyze_directory_structure(chunks)


        if all_scanned_files:
            all_files = sorted(set(all_scanned_files))
            print(f"  Using all scanned files: {len(all_files)}")
        else:
            all_files = sorted(set(c.file for c in chunks))
            print(f"  Using chunked files only: {len(all_files)}")

        file_tree = self._build_file_tree(chunks, all_files)

        print(f"  Found {len(entry_points)} entry points")
        print(f"  Detected architecture: {architecture}")
        print(f"  Identified {len(components)} components")
        print(f"  Built file tree with {len(all_files)} files")

        return RepoContext(
            name=repo_name,
            entry_points=entry_points,
            architecture_pattern=architecture,
            components=components,
            tech_stack=tech_stack,
            directory_structure=directory_structure,
            file_tree=file_tree,
            all_files=all_files,
            total_files=len(all_files),
            total_chunks=len(chunks)
        )

    def _find_entry_points(self, chunks: List[Chunk]) -> List[EntryPoint]:
        """Identify entry points in the codebase."""
        entry_points = []

        for chunk in chunks:
            # Check for main functions
            if chunk.type == 'function' and chunk.name == 'main':
                entry_points.append(EntryPoint(
                    name=chunk.name,
                    file=chunk.file,
                    type='main',
                    line=chunk.start,
                    description=chunk.docstring or "Main entry point"
                ))

            # Check for Python __main__
            if 'if __name__ == "__main__"' in chunk.code or "if __name__ == '__main__'" in chunk.code:
                entry_points.append(EntryPoint(
                    name=f"Script: {chunk.file}",
                    file=chunk.file,
                    type='script',
                    line=chunk.start,
                    description="Python script entry point"
                ))

            # Check for web framework decorators
            if chunk.decorators:
                for dec in chunk.decorators:
                    if 'app.route' in dec or 'app.get' in dec or 'app.post' in dec:
                        entry_points.append(EntryPoint(
                            name=chunk.name,
                            file=chunk.file,
                            type='api',
                            line=chunk.start,
                            description=f"API endpoint: {chunk.signature or chunk.name}"
                        ))

            # Check for CLI decorators
            if chunk.decorators and any('click.command' in d or 'click.group' in d for d in chunk.decorators):
                entry_points.append(EntryPoint(
                    name=chunk.name,
                    file=chunk.file,
                    type='cli',
                    line=chunk.start,
                    description=f"CLI command: {chunk.name}"
                ))

            # Check for specific file names that are often entry points
            file_lower = chunk.file.lower()
            entry_files = ['main.py', 'app.py', 'index.js', 'index.ts', 'server.js', 'app.js',
                          'cli.py', '__main__.py', 'run.py', 'manage.py']

            if any(ef in file_lower for ef in entry_files):
                if chunk.type in ('function', 'class') and chunk.name not in [ep.name for ep in entry_points]:
                    entry_points.append(EntryPoint(
                        name=chunk.name,
                        file=chunk.file,
                        type='entry_file',
                        line=chunk.start,
                        description=f"Found in entry file: {chunk.file}"
                    ))

        return entry_points

    def _detect_architecture_pattern(self, chunks: List[Chunk]) -> str:
        """Detect the architectural pattern used."""

        # Get all directory paths
        directories = set()
        for chunk in chunks:
            parts = Path(chunk.file).parts
            directories.update(parts)

        # Convert to lowercase for matching
        dir_lower = [d.lower() for d in directories]

        # Check for each pattern
        pattern_scores = {}
        for pattern_name, keywords in self.ARCHITECTURE_KEYWORDS.items():
            score = sum(1 for kw in keywords if any(kw in d for d in dir_lower))
            if score > 0:
                pattern_scores[pattern_name] = score

        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            return best_pattern

        # Default patterns based on file organization
        if len(directories) > 10:
            return "modular"
        elif any('src' in d for d in dir_lower):
            return "layered"
        else:
            return "simple"

    def _identify_components(self, chunks: List[Chunk]) -> List[Component]:
        """Group chunks into logical components."""

        # Group by top-level directory
        components_map = defaultdict(lambda: {'files': set(), 'chunks': []})

        for chunk in chunks:
            parts = Path(chunk.file).parts
            if len(parts) > 1:
                component_name = parts[0]
            else:
                component_name = "root"

            components_map[component_name]['files'].add(chunk.file)
            components_map[component_name]['chunks'].append(chunk)

        # Create Component objects
        components = []
        for name, data in components_map.items():
            # Find entry points in this component
            entry_points = [
                chunk.name for chunk in data['chunks']
                if chunk.type == 'function' and
                (chunk.name == 'main' or 'if __name__' in chunk.code)
            ]

            # Infer purpose from directory name and contents
            purpose = self._infer_component_purpose(name, data['chunks'])

            components.append(Component(
                name=name,
                files=sorted(list(data['files']))[:10],  # Limit to 10 files
                entry_points=entry_points,
                purpose=purpose
            ))

        return sorted(components, key=lambda c: len(c.files), reverse=True)[:10]

    def _infer_component_purpose(self, name: str, chunks: List[Chunk]) -> str:
        """Infer the purpose of a component from its name and contents."""

        name_lower = name.lower()

        # Common component types
        if 'test' in name_lower:
            return "Testing and test utilities"
        elif name_lower in ('models', 'model', 'entities'):
            return "Data models and entities"
        elif name_lower in ('views', 'templates', 'ui'):
            return "User interface and presentation"
        elif name_lower in ('controllers', 'routes', 'api'):
            return "API endpoints and request handling"
        elif name_lower in ('services', 'business', 'logic'):
            return "Business logic and services"
        elif name_lower in ('utils', 'helpers', 'common'):
            return "Utility functions and helpers"
        elif name_lower in ('config', 'settings'):
            return "Configuration and settings"
        elif name_lower in ('db', 'database', 'repository'):
            return "Database access and data persistence"
        elif name_lower in ('auth', 'authentication'):
            return "Authentication and authorization"
        elif name_lower in ('parsers', 'parser'):
            return "Parsing and data transformation"
        else:
            # Infer from chunk types
            class_count = sum(1 for c in chunks if c.type == 'class')
            function_count = sum(1 for c in chunks if c.type == 'function')

            if class_count > function_count:
                return f"Component with {class_count} classes"
            else:
                return f"Component with {function_count} functions"

    def _identify_tech_stack(self, chunks: List[Chunk]) -> Set[str]:
        """Identify technologies used in the codebase."""

        tech_stack = set()

        # Add languages
        languages = set(c.language for c in chunks)
        tech_stack.update(languages)

        # Common framework/library imports
        framework_indicators = {
            'flask': ['Flask', 'flask'],
            'django': ['django', 'Django'],
            'fastapi': ['FastAPI', 'fastapi'],
            'express': ['express'],
            'react': ['React', 'ReactDOM'],
            'vue': ['Vue'],
            'angular': ['@angular'],
            'spring': ['springframework', 'SpringBoot'],
            'pandas': ['pandas'],
            'numpy': ['numpy'],
            'tensorflow': ['tensorflow'],
            'pytorch': ['torch'],
            'sqlalchemy': ['sqlalchemy'],
            'pytest': ['pytest'],
            'jest': ['jest'],
        }

        # Check imports
        all_imports = set()
        for chunk in chunks:
            all_imports.update(chunk.imports)

        for tech, indicators in framework_indicators.items():
            if any(ind in imp for imp in all_imports for ind in indicators):
                tech_stack.add(tech)

        return tech_stack

    def _analyze_directory_structure(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Analyze directory structure and file counts."""

        dir_counts = defaultdict(int)
        file_counts = defaultdict(int)

        for chunk in chunks:
            parts = Path(chunk.file).parts
            file_counts[chunk.file] += 1

            if len(parts) > 1:
                top_dir = parts[0]
                dir_counts[top_dir] += 1
            else:
                # Root level files
                dir_counts['<root>'] = dir_counts.get('<root>', 0) + 1

        return dict(sorted(dir_counts.items(), key=lambda x: x[1], reverse=True))

    def _build_file_tree(self, chunks: List[Chunk], all_files: List[str]) -> FileNode:
        """Build a complete file tree structure."""

        # Count chunks per file
        file_chunk_counts = defaultdict(int)
        for chunk in chunks:
            file_chunk_counts[chunk.file] += 1

        # Create root node
        root = FileNode(name="<root>", path="", is_directory=True)

        # Build tree structure
        for file_path in all_files:
            parts = Path(file_path).parts
            current = root

            # Navigate/create directory structure
            for i, part in enumerate(parts[:-1]):
                # Look for existing child directory
                child = next((c for c in current.children if c.name == part and c.is_directory), None)

                if not child:
                    # Create new directory node
                    child = FileNode(
                        name=part,
                        path='/'.join(parts[:i+1]),
                        is_directory=True
                    )
                    current.children.append(child)

                current = child

            # Add file node
            file_name = parts[-1] if len(parts) > 0 else file_path
            file_node = FileNode(
                name=file_name,
                path=file_path,
                is_directory=False,
                chunk_count=file_chunk_counts.get(file_path, 0)
            )
            current.children.append(file_node)

        # Sort children alphabetically
        self._sort_tree(root)

        return root

    def _sort_tree(self, node: FileNode):
        """Recursively sort tree nodes."""
        if node.children:
            # Sort: directories first, then files, both alphabetically
            node.children.sort(key=lambda x: (not x.is_directory, x.name.lower()))
            for child in node.children:
                if child.is_directory:
                    self._sort_tree(child)

    def create_overview_chunk(self, repo_context: RepoContext) -> Chunk:
        """Create a searchable repository overview chunk."""

        # Format entry points
        entry_points_text = "\n".join([
            f"  - {ep.name} ({ep.type}) in {ep.file}:{ep.line}"
            for ep in repo_context.entry_points[:10]
        ])

        # Format components
        components_text = "\n".join([
            f"  - {comp.name}: {comp.purpose} ({len(comp.files)} files)"
            for comp in repo_context.components[:10]
        ])

        # Format directory structure
        dir_structure = "\n".join([
            f"  - {dir_name}: {count} chunks"
            for dir_name, count in list(repo_context.directory_structure.items())[:10]
        ])

        # NEW: Format file tree
        file_tree_text = self._format_file_tree(repo_context.file_tree, max_depth=3, max_files=50)

        # NEW: Root level files
        root_files = [f for f in repo_context.all_files if '/' not in f and '\\' not in f]
        root_files_text = "\n".join([f"  - {f}" for f in root_files[:10]])

        overview_text = f"""# Repository Overview: {repo_context.name}

## Architecture Pattern
{repo_context.architecture_pattern}

## Entry Points ({len(repo_context.entry_points)})
{entry_points_text or "  No clear entry points detected"}

## Root Level Files ({len(root_files)})
{root_files_text or "  No root files"}

## File Tree Structure (top 3 levels)
{file_tree_text}

## Main Components ({len(repo_context.components)})
{components_text}

## Technology Stack
{', '.join(sorted(repo_context.tech_stack))}

## Directory Structure
{dir_structure}

## Statistics
- Total Files: {repo_context.total_files}
- Total Code Chunks: {repo_context.total_chunks}
- Average Chunks per File: {repo_context.total_chunks // repo_context.total_files if repo_context.total_files > 0 else 0}
- Root Level Files: {len(root_files)}
- Directory Depth: {self._calculate_tree_depth(repo_context.file_tree)}

## All Files
{', '.join(repo_context.all_files[:30])}{'...' if len(repo_context.all_files) > 30 else ''}

## Key Insights
- This is a {repo_context.architecture_pattern} architecture
- Main entry points are in: {', '.join(set(ep.file for ep in repo_context.entry_points[:5]))}
- Primary components: {', '.join([c.name for c in repo_context.components[:3]])}
"""

        return Chunk(
            id=f"{repo_context.name}::repo::overview",
            name="Repository Overview",
            type="repo_overview",
            file="<repository>",
            start=0,
            end=0,
            language="text",
            code=overview_text,
            docstring="High-level repository context and architecture overview",
            calls=[],
            imports=[],
        )

    def _format_file_tree(self, node: FileNode, prefix: str = "", max_depth: int = 3,
                         current_depth: int = 0, max_files: int = 50,
                         file_count: List[int] = None) -> str:
        """Format file tree as text with limited depth and file count."""

        if file_count is None:
            file_count = [0]

        if current_depth >= max_depth or file_count[0] >= max_files:
            return ""

        lines = []

        for i, child in enumerate(node.children):
            if file_count[0] >= max_files:
                lines.append(f"{prefix}└── ... ({len(node.children) - i} more items)")
                break

            is_last = i == len(node.children) - 1
            current_prefix = "└── " if is_last else "├── "

            if child.is_directory:
                lines.append(f"{prefix}{current_prefix}{child.name}/")
                next_prefix = prefix + ("    " if is_last else "│   ")
                subtree = self._format_file_tree(
                    child, next_prefix, max_depth, current_depth + 1, max_files, file_count
                )
                if subtree:
                    lines.append(subtree)
            else:
                file_count[0] += 1
                chunk_info = f" ({child.chunk_count} chunks)" if child.chunk_count > 0 else ""
                lines.append(f"{prefix}{current_prefix}{child.name}{chunk_info}")

        return "\n".join(lines)

    def _calculate_tree_depth(self, node: FileNode, current_depth: int = 0) -> int:
        """Calculate the maximum depth of the file tree."""

        if not node.children:
            return current_depth

        max_child_depth = current_depth
        for child in node.children:
            if child.is_directory:
                child_depth = self._calculate_tree_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth
