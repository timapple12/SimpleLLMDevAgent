from __future__ import annotations

from typing import Dict, Any

from .llm_agent import LLMAgent


ANALYZE_SYSTEM = (
    "You are a senior software engineer assessing task sufficiency. "
    "Given a task description and limited project context snippets, output a minimal JSON verdict with fields: "
    "{\"sufficient\": boolean, \"reason\": string, \"missing\": [string]}."
)


class TaskAnalyzer:
    def __init__(self, llm: LLMAgent) -> None:
        self.llm = llm

    def analyze(self, task_text: str, context_snippets: list[dict[str, Any]]) -> dict[str, Any]:
        msgs = [
            {"role": "system", "content": ANALYZE_SYSTEM},
            {"role": "user", "content": self._build_user_prompt(task_text, context_snippets)},
        ]
        # Reuse provider-specific chat but ensure we don't request a diff
        if self.llm.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.llm.openai_api_key)
            resp = client.chat.completions.create(model=self.llm.openai_model, messages=msgs, temperature=0)
            content = resp.choices[0].message.content or "{}"
        else:
            import requests
            r = requests.post(
                f"{self.llm.ollama_host}/api/chat",
                json={"model": self.llm.model, "messages": msgs, "options": {"temperature": 0}},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content") or data.get("response", "{}")
        try:
            import json
            return json.loads(content)
        except Exception:
            return {"sufficient": False, "reason": "Analyzer returned non-JSON", "missing": ["Provide clearer acceptance criteria"]}

    def _build_user_prompt(self, task_text: str, context_snippets: list[dict[str, Any]]) -> str:
        parts = ["TASK:", task_text.strip(), "\nContext snippets:"]
        for item in context_snippets[:5]:
            parts.append(f"FILE: {item['path']}\n{item['content'][:2000]}")
        parts.append("\nRespond with strict JSON only, no prose.")
        return "\n\n".join(parts)


