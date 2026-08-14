"""LLM provider interface with graceful fallback."""

import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

import httpx

from .config import settings


class LLMResponse:
    """Structured LLM response."""

    def __init__(self, content: str, usage: dict = None):
        self.content = content
        self.usage = usage or {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai is not installed. "
                    "Install it with: pip install 'codebrain[openai]'"
                )
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            } if response.usage else {},
        )

    async def stream_chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic is not installed. "
                    "Install it with: pip install 'codebrain[anthropic]'"
                )
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = self._get_client()

        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        return LLMResponse(
            content=response.content[0].text,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            } if response.usage else {},
        )

    async def stream_chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()

        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


class OllamaProvider(LLMProvider):
    """Local Ollama provider."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return LLMResponse(content=data["message"]["content"])

    async def stream_chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature},
                },
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM provider used when the real provider is unavailable.
    Shows a helpful message telling the user what to do.
    """

    def __init__(self, reason: str = "default"):
        self.reason = reason

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # Extract the question from messages
        question = "your question"
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "Question:" in content:
                    question = content.split("Question:")[-1].strip()
                else:
                    question = content[:200]
                break

        if self.reason == "ollama_not_running":
            content = (
                f"**Ollama is not running**\n\n"
                f"CodeBrain is configured to use Ollama (free, local AI), but it could not connect.\n\n"
                f"**To fix this:**\n"
                f"1. Install Ollama from [ollama.com](https://ollama.com)\n"
                f"2. Open a terminal and run: `ollama pull llama3.2`\n"
                f"3. Make sure Ollama is running (it starts automatically after install)\n"
                f"4. Refresh this page and try again\n\n"
                f"**Or switch to a cloud provider:**\n"
                f"- Set `OPENAI_API_KEY` and `LLM_PROVIDER=openai` in your `.env` file\n"
                f"- Set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic` in your `.env` file\n\n"
                f"Your question was: *{question}*\n\n"
                f"The retrieval pipeline found relevant code — the AI just needs to be connected."
            )
        elif self.reason == "ollama_no_model":
            content = (
                f"**Model not found: `{settings.ollama_model}`**\n\n"
                f"Ollama is running, but the model `{settings.ollama_model}` hasn't been downloaded yet.\n\n"
                f"**To fix this:**\n"
                f"1. Open a terminal\n"
                f"2. Run: `ollama pull {settings.ollama_model}`\n"
                f"3. Wait for the download to finish\n"
                f"4. Refresh this page and try again\n\n"
                f"**Or use a different model:**\n"
                f"Set `OLLAMA_MODEL=llama3.1` or another model in your `.env` file\n\n"
                f"Your question was: *{question}*"
            )
        else:
            content = (
                f"**No AI provider connected**\n\n"
                f"CodeBrain found relevant code for your question, but no LLM is configured.\n\n"
                f"**Free option (recommended):**\n"
                f"1. Install Ollama from [ollama.com](https://ollama.com)\n"
                f"2. Run: `ollama pull llama3.2`\n"
                f"3. Make sure Ollama is running\n"
                f"4. Restart CodeBrain (`CTRL+C` then `python src/codebrain/cli.py serve`)\n\n"
                f"**Or use a cloud provider:**\n"
                f"- Set `OPENAI_API_KEY` and `LLM_PROVIDER=openai` in `.env`\n"
                f"- Set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic` in `.env`\n\n"
                f"Your question was: *{question}*"
            )
        return LLMResponse(content=content)

    async def stream_chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        response = await self.chat(messages, temperature, max_tokens)
        words = response.content.split(" ")
        for word in words:
            yield word + " "


def _provider_is_ready() -> bool:
    """Check if the configured provider is actually usable."""
    provider = settings.llm_provider
    if provider == "openai":
        key = settings.openai_api_key
        return bool(key) and not key.startswith("your-") and len(key) > 20
    elif provider == "anthropic":
        key = settings.anthropic_api_key
        return bool(key) and not key.startswith("your-") and len(key) > 20
    elif provider == "ollama":
        # Check if Ollama is running AND has the requested model
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{settings.ollama_base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                # Model name might include :latest tag
                target = settings.ollama_model
                for model in models:
                    if model == target or model.startswith(target + ":"):
                        return True
                return False  # Ollama running but model not found
        except Exception:
            return False
    return False


def get_llm_provider() -> LLMProvider:
    """Factory function to get the configured LLM provider."""
    provider = settings.llm_provider

    if provider == "ollama":
        # Check if Ollama is reachable
        try:
            import urllib.request
            urllib.request.urlopen(
                f"{settings.ollama_base_url}/api/tags",
                timeout=2,
            )
            ollama_running = True
        except Exception:
            ollama_running = False

        if not ollama_running:
            return MockLLMProvider(reason="ollama_not_running")

        # Check if model is available
        if _provider_is_ready():
            return OllamaProvider()
        else:
            return MockLLMProvider(reason="ollama_no_model")

    elif provider == "anthropic":
        if _provider_is_ready():
            try:
                import anthropic  # noqa: F401
                return AnthropicProvider()
            except ImportError:
                pass
        return MockLLMProvider()

    elif provider == "openai":
        if _provider_is_ready():
            try:
                import openai  # noqa: F401
                return OpenAIProvider()
            except ImportError:
                pass
        return MockLLMProvider()

    return MockLLMProvider()
