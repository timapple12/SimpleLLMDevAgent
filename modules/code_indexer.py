import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
import requests


class EmbeddingProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI  # lazy import
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for t in texts:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            vectors.append(data["embedding"])  # type: ignore[index]
        return vectors


class CodeIndexer:
    def __init__(
        self,
        project_root: str,
        vector_store_path: str,
        provider: str = "ollama",
        openai_api_key: str | None = None,
        openai_embeddings_model: str = "text-embedding-3-small",
        ollama_host: str = "http://localhost:11434",
        ollama_embeddings_model: str = "nomic-embed-text:latest",
        include_extensions: List[str] | None = None,
        exclude_dirs: List[str] | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.vector_store_path = Path(vector_store_path)
        self.include_extensions = set(include_extensions or [".java", ".vue", ".js", ".html"])
        self.exclude_dirs = set(exclude_dirs or ["node_modules", "target", "dist", ".git", "build"])

        if provider == "openai":
            if not openai_api_key:
                raise ValueError("openai_api_key is required when provider=openai")
            self.embedding = OpenAIEmbeddingProvider(openai_api_key, openai_embeddings_model)
        else:
            self.embedding = OllamaEmbeddingProvider(ollama_host, ollama_embeddings_model)

        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="code_index",
            metadata={"hnsw:space": "cosine"},
        )

    def _iter_files(self):
        for root, dirs, files in os.walk(self.project_root):
            # prune excluded dirs
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.include_extensions:
                    yield Path(root) / f

    def build_index(self) -> None:
        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for p in self._iter_files():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = str(p.relative_to(self.project_root))
            docs.append(text)
            metadatas.append({"path": rel})
            ids.append(rel)

        if not docs:
            return

        try:
            self.client.delete_collection("code_index")
            self.collection = self.client.get_or_create_collection(
                name="code_index",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass

        embeddings = self.embedding.embed(docs)
        # Batch add to avoid payload limits
        batch = 64
        for i in range(0, len(docs), batch):
            self.collection.add(
                ids=ids[i : i + batch],
                documents=docs[i : i + batch],
                metadatas=metadatas[i : i + batch],
                embeddings=embeddings[i : i + batch],
            )

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q_embed = self.embedding.embed([query])[0]
        res = self.collection.query(query_embeddings=[q_embed], n_results=top_k)
        items: List[Dict[str, Any]] = []
        for i in range(len(res.get("ids", [[]])[0])):
            items.append(
                {
                    "path": res["metadatas"][0][i]["path"],
                    "content": res["documents"][0][i],
                }
            )
        return items


