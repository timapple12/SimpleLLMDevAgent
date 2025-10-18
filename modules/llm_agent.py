from __future__ import annotations

from typing import List, Dict, Any
import requests

SYSTEM_PROMPT = (
    "You are a code editing agent. Produce a unified git diff (patch) that applies to the repository root. "
    "Output ONLY the diff. Do not include explanations. Use correct file paths."
)


class LLMAgent:
    def __init__(
            self,
            provider: str,
            model: str,
            project_root: str,
            openai_api_key: str | None = None,
            openai_model: str | None = None,
            ollama_host: str = "http://localhost:11434",
    ) -> None:
        self.provider = provider
        self.model = model
        self.project_root = project_root
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model or model
        self.ollama_host = ollama_host.rstrip("/")

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)

    def _build_prompt(self, task: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        ctx_snippets = []
        for item in context:
            path = item["path"]
            content = item["content"]
            # snippets SHOULD be short
            snippet = content[:12000]
            ctx_snippets.append(f"FILE: {path}\n" + snippet)

        user_prompt = (
                "TASK:\n" + task.strip() + "\n\n" +
                "Relevant code snippets (read-only):\n\n" + "\n\n".join(ctx_snippets) + "\n\n" +
                "Respond with a single valid unified git diff."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _chat_openai(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.openai_model,
            temperature=0.2,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        # Convert to Ollama chat format
        ollama_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        r = requests.post(
            f"{self.ollama_host}/api/chat",
            json={"model": self.model, "messages": ollama_msgs, "options": {"temperature": 0.2}, "stream": False},
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]
        return ""

    def generate_git_diff(self, task: str, context: List[Dict[str, Any]]) -> str:
        messages = self._build_prompt(task, context)
        if self.provider == "openai":
            return self._chat_openai(messages)
        return self._chat_ollama(messages)
