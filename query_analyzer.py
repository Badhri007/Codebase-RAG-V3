"""Query analyzer for understanding user intent."""
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

from llm import LLMProvider


class QueryType(Enum):
    """Types of queries users can ask."""
    ARCHITECTURAL = "architectural"      # "explain flow", "architecture", "how does it work"
    ENTRY_POINT = "entry_point"          # "where to start", "main entry", "how to run"
    SPECIFIC_FUNCTION = "specific"        # "find function X", "where is Y"
    IMPLEMENTATION = "implementation"     # "how does X work", "implementation of Y"
    DEBUG = "debug"                       # "why is X failing", "error in Y"
    USAGE = "usage"                       # "how to use X", "example of Y"
    TESTING = "testing"                   # "test for X", "how to test"
    DEPENDENCY = "dependency"             # "what uses X", "dependencies of Y"
    GENERAL = "general"                   # General questions


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    original_query: str
    query_type: QueryType
    entities: List[str]  # Extracted function/class/file names
    keywords: List[str]
    needs_repo_context: bool
    include_tests: bool
    rewritten_queries: List[str]
    confidence: float


class QueryAnalyzer:
    """Analyze and classify user queries to improve retrieval."""

    # Keywords for each query type
    TYPE_KEYWORDS = {
        QueryType.ARCHITECTURAL: [
            'architecture', 'structure', 'design', 'pattern', 'flow',
            'overall', 'explain', 'overview', 'organize', 'layout'
        ],
        QueryType.ENTRY_POINT: [
            'entry point', 'start', 'main', 'begin', 'initialize',
            'startup', 'bootstrap', 'run', 'execute', 'launch', 'begin'
        ],
        QueryType.SPECIFIC_FUNCTION: [
            'find', 'where is', 'locate', 'search for', 'show me'
        ],
        QueryType.IMPLEMENTATION: [
            'how does', 'implementation', 'works', 'logic', 'algorithm',
            'process', 'mechanism', 'internally'
        ],
        QueryType.DEBUG: [
            'error', 'bug', 'issue', 'problem', 'fail', 'wrong',
            'broken', 'debug', 'fix', 'troubleshoot'
        ],
        QueryType.USAGE: [
            'how to use', 'example', 'usage', 'use case', 'tutorial',
            'guide', 'call', 'invoke'
        ],
        QueryType.TESTING: [
            'test', 'testing', 'unit test', 'integration test', 'spec',
            'test coverage', 'test case'
        ],
        QueryType.DEPENDENCY: [
            'depends on', 'dependency', 'uses', 'calls', 'imports',
            'requires', 'relationship', 'connected'
        ]
    }

    def __init__(self, llm: Optional[LLMProvider] = None, use_llm: bool = True):
        """
        Initialize query analyzer.

        Args:
            llm: LLM provider for advanced analysis
            use_llm: Whether to use LLM for classification (more accurate but slower)
        """
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def analyze(self, query: str) -> QueryAnalysis:
        # Classify query type
        if self.use_llm:
            query_type, confidence = self._classify_with_llm(query)
        else:
            query_type, confidence = self._classify_with_keywords(query)

        # Extract entities (function/class/file names)
        entities = self._extract_entities(query)

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Determine if repo context needed
        needs_repo_context = query_type in [
            QueryType.ARCHITECTURAL,
            QueryType.ENTRY_POINT,
            QueryType.GENERAL
        ]

        # Determine if tests should be included
        include_tests = query_type == QueryType.TESTING or 'test' in query.lower()

        # Generate rewritten queries
        rewritten_queries = self._rewrite_query(query, query_type, entities)

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            needs_repo_context=needs_repo_context,
            include_tests=include_tests,
            rewritten_queries=rewritten_queries,
            confidence=confidence
        )

    def _classify_with_keywords(self, query: str) -> tuple[QueryType, float]:
        """Classify query using keyword matching."""

        query_lower = query.lower()
        scores = {}

        # Score each type based on keyword matches
        for query_type, keywords in self.TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[query_type] = score

        if not scores:
            return QueryType.GENERAL, 0.5

        # Get best match
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 3.0, 1.0)  # Normalize confidence

        return best_type, confidence

    def _classify_with_llm(self, query: str) -> tuple[QueryType, float]:
        """Classify query using LLM for better accuracy."""

        prompt = f"""Classify this codebase question into ONE category:

Categories:
- ARCHITECTURAL: Questions about overall structure, flow, design patterns, architecture
- ENTRY_POINT: Questions about where code starts, main functions, how to run
- SPECIFIC_FUNCTION: Looking for a specific function, class, or file
- IMPLEMENTATION: How something is implemented, internal logic, algorithms
- DEBUG: Debugging, errors, issues, problems, troubleshooting
- USAGE: How to use a feature, examples, tutorials
- TESTING: Questions about tests, test coverage, testing strategies
- DEPENDENCY: Questions about dependencies, what uses what, relationships
- GENERAL: General questions that don't fit other categories

Question: "{query}"

Respond ONLY with the category name and confidence (0.0-1.0):
CATEGORY: <name>
CONFIDENCE: <score>"""

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

            # Parse response
            lines = response.strip().split('\n')
            category_line = next((l for l in lines if 'CATEGORY:' in l), '')
            confidence_line = next((l for l in lines if 'CONFIDENCE:' in l), '')

            category = category_line.split(':', 1)[1].strip().upper()
            confidence = float(confidence_line.split(':', 1)[1].strip())

            # Map to enum
            query_type = QueryType[category]
            return query_type, confidence

        except Exception as e:
            print(f"  Warning: LLM classification failed: {e}, using keywords")
            return self._classify_with_keywords(query)

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entity names (functions, classes, files) from query."""

        import re

        entities = []

        # Look for quoted strings (often function/file names)
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        entities.extend(quoted)

        # Look for camelCase or snake_case identifiers
        identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', query.lower())

        # Filter out common words
        common_words = {
            'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'this', 'that', 'what', 'where',
            'how', 'when', 'why', 'which', 'who', 'does', 'do', 'can', 'will',
            'would', 'should', 'could', 'function', 'class', 'method', 'file',
            'code', 'implementation', 'work', 'works', 'used', 'use'
        }

        entities.extend([
            word for word in identifiers
            if word not in common_words and len(word) > 2
        ])

        return list(set(entities))[:10]  # Limit to 10 unique entities

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""

        import re

        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        return keywords[:10]

    def _rewrite_query(self, query: str, query_type: QueryType,
                       entities: List[str]) -> List[str]:
        """Generate alternative query formulations."""

        rewritten = [query]  # Always include original

        if query_type == QueryType.ARCHITECTURAL:
            rewritten.extend([
                f"architecture overview",
                f"main components and structure",
                f"high-level design and flow",
                f"system architecture patterns"
            ])

        elif query_type == QueryType.ENTRY_POINT:
            rewritten.extend([
                "main function entry point",
                "where does execution start",
                "initialization code",
                "startup and bootstrap"
            ])

        elif query_type == QueryType.SPECIFIC_FUNCTION and entities:
            entity = entities[0]
            rewritten.extend([
                f"function {entity}",
                f"class {entity}",
                f"{entity} implementation",
                f"where is {entity} defined"
            ])

        elif query_type == QueryType.IMPLEMENTATION:
            rewritten.extend([
                f"{query.replace('how does', '')} implementation",
                f"{query.replace('how does', '')} logic",
                f"algorithm for {query.replace('how does', '')}"
            ])

        elif query_type == QueryType.USAGE and entities:
            entity = entities[0]
            rewritten.extend([
                f"how to use {entity}",
                f"{entity} usage example",
                f"{entity} API documentation"
            ])

        elif query_type == QueryType.DEPENDENCY and entities:
            entity = entities[0]
            rewritten.extend([
                f"what calls {entity}",
                f"{entity} dependencies",
                f"functions that use {entity}"
            ])

        return rewritten[:5]  # Limit to 5 variations

    def should_use_graph_traversal(self, query_analysis: QueryAnalysis) -> bool:
        """Determine if graph traversal would be beneficial."""

        return query_analysis.query_type in [
            QueryType.DEPENDENCY,
            QueryType.IMPLEMENTATION,
            QueryType.ARCHITECTURAL
        ]

    def get_retrieval_config(self, query_analysis: QueryAnalysis) -> Dict:
        """Get optimal retrieval configuration based on query type."""

        config = {
            'k': 25,  # Default number of chunks
            'include_tests': query_analysis.include_tests,
            'use_graph': self.should_use_graph_traversal(query_analysis),
            'boost_types': [],
            'filter_types': []
        }

        # Adjust based on query type
        if query_analysis.query_type == QueryType.ARCHITECTURAL:
            config['k'] = 30
            config['boost_types'] = ['repo_overview', 'file_summary', 'class_summary']

        elif query_analysis.query_type == QueryType.ENTRY_POINT:
            config['k'] = 20
            config['boost_types'] = ['function', 'file_summary']

        elif query_analysis.query_type == QueryType.SPECIFIC_FUNCTION:
            config['k'] = 15
            config['use_graph'] = True

        elif query_analysis.query_type == QueryType.TESTING:
            config['include_tests'] = True
            config['boost_types'] = ['function', 'method']

        return config
