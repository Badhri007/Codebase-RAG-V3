"""LLM providers"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional



SYSTEM_PROMPT = """You are an expert code analyst. Help developers understand codebases by:
1. Explaining code structure, architecture, and how components interact
2. Describing how functions, methods, and classes work
3. Creating clear Mermaid diagrams when asked (flowchart, sequence, class diagrams)
4. Referencing specific files and line numbers
5. Being thorough but concise

When creating Mermaid diagrams:
- Use ```mermaid code blocks
- Choose appropriate diagram type (flowchart TD/LR, sequenceDiagram, classDiagram)
- Include clear, descriptive labels
- Show key relationships and data flow"""


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Synchronous chat."""
        pass



class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or "claude-sonnet-4-20250514"
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY required")
        import anthropic
        self.client = anthropic.Anthropic(api_key=key)

    @property
    def name(self) -> str:
        return f"claude ({self.model})"

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        return response.content[0].text

    def expand_query(self, query: str) -> List[str]:
        """Generate 3 query variations using Claude."""
        prompt = f'''Given: "{query}"
Generate 3 alternative search queries for code. Return only queries, one per line.'''

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        variations = [q.strip() for q in response.content[0].text.strip().split('\n') if q.strip()]
        return [query] + variations[:3]


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or "gpt-4o"
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required")
        from openai import OpenAI
        self.client = OpenAI(api_key=key)

    @property
    def name(self) -> str:
        return f"openai ({self.model})"

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=4096
        )
        return response.choices[0].message.content

    def expand_query(self, query: str) -> List[str]:
        """Generate 3 query variations using OpenAI."""
        prompt = f'''Given: "{query}"
Generate 3 alternative search queries for code. Return only queries, one per line.'''

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        variations = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
        return [query] + variations[:3]


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or "gemini-2.0-flash-exp"
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY required")
        import google.generativeai as genai
        genai.configure(api_key=key)
        self.client = genai.GenerativeModel(self.model)
        self.genai = genai

    @property
    def name(self) -> str:
        return f"gemini ({self.model})"

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        # Combine system + user message
        user_content = messages[-1]["content"] if messages else ""
        prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"

        response = self.client.generate_content(
            prompt,
            generation_config=self.genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4096
            )
        )
        return response.text

    def expand_query(self, query: str) -> List[str]:
        """Generate 3 query variations using Gemini."""
        prompt = f'''Given: "{query}"
Generate 3 alternative search queries for code that is being used for this Q&A.
Based on overall context of the code, nothing innovative or outside this code context.
Return only queries, one per line.'''

        response = self.client.generate_content(
            prompt,
            generation_config=self.genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=200
            )
        )

        variations = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        return [query] + variations[:3]


class DeepSeekProvider(LLMProvider):
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or "deepseek-chat"
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY required")
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")

    @property
    def name(self) -> str:
        return f"deepseek ({self.model})"

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=4096
        )
        return response.choices[0].message.content


PROVIDERS = {
    'claude': ClaudeProvider,
    'anthropic': ClaudeProvider,
    'openai': OpenAIProvider,
    'gpt': OpenAIProvider,
    'gemini': GeminiProvider,
    'google': GeminiProvider,
    'deepseek': DeepSeekProvider,
}


def get_llm(provider: str, model: str = None, api_key: str = None) -> LLMProvider:
    """Get LLM provider by name."""
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider](model, api_key)
