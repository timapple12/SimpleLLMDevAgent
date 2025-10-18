from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import requests
from chromadb.config import Settings

from modules.progress_bar import print_progress


class EmbeddingProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
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
            vectors.append(data["embedding"])
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
            verbose: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.vector_store_path = Path(vector_store_path)
        self.include_extensions = set(include_extensions or [".java", ".vue", ".js", ".html"])
        self.exclude_dirs = set(exclude_dirs or ["node_modules", "target", "dist", ".git", "build"])
        self.verbose = verbose

        self.index_metadata_path = self.vector_store_path / "index_metadata.json"

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

    def _get_file_hash(self, filepath: Path) -> str:
        return hashlib.md5(filepath.read_bytes()).hexdigest()

    def _load_index_metadata(self) -> Dict[str, Any]:
        if self.index_metadata_path.exists():
            with open(self.index_metadata_path, 'r') as f:
                return json.load(f)
        return {"files": {}, "last_indexed": None}

    def _save_index_metadata(self, metadata: Dict[str, Any]) -> None:
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        with open(self.index_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _iter_files(self):
        for root, dirs, files in os.walk(self.project_root):
            # prune excluded dirs
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.include_extensions:
                    yield Path(root) / f

    def needs_reindex(self) -> bool:
        metadata = self._load_index_metadata()

        if not metadata["last_indexed"]:
            return True

        if self.verbose:
            print("Checking for file changes...")

        current_files = {}
        files_list = list(self._iter_files())

        for i, p in enumerate(files_list):
            if self.verbose and i % 10 == 0:
                print(f"\rChecking files... {i}/{len(files_list)}", end='', flush=True)
            try:
                current_files[str(p)] = self._get_file_hash(p)
            except Exception:
                continue

        if self.verbose:
            print("\rChecking files... Done!                    ")

        old_files = set(metadata["files"].keys())
        new_files = set(current_files.keys())

        if old_files != new_files:
            return True

        for filepath, filehash in current_files.items():
            if metadata["files"].get(filepath) != filehash:
                return True

        return False

    def build_index(self, force: bool = False) -> None:
        start_time = time.time()

        if not force and not self.needs_reindex():
            if self.verbose:
                print("Index is up to date, skipping rebuild.")
            return

        if not force and self.index_metadata_path.exists():
            self.incremental_update()
            return

        if self.verbose:
            print("Building/updating index...")

        if self.verbose:
            print("Collecting files...")

        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        file_hashes: Dict[str, str] = {}

        files_list = list(self._iter_files())
        total_files = len(files_list)

        if self.verbose:
            print(f"Found {total_files} files to index")

        for i, p in enumerate(files_list):
            if self.verbose:
                print_progress(i + 1, total_files, f"Reading {p.name}...", time.time() - start_time)

            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                file_hash = self._get_file_hash(p)
            except Exception as e:
                if self.verbose:
                    print(f"\n  \033[31mError reading {p}: {e}\033[0m")
                continue

            rel = str(p.relative_to(self.project_root))
            docs.append(text)
            metadatas.append({"path": rel})
            ids.append(rel)
            file_hashes[str(p)] = file_hash

        if self.verbose:
            print()

        if not docs:
            if self.verbose:
                print("\033[31mNo files to index!\033[0m")
            return

        if self.verbose:
            print("  Clearing old index...")

        try:
            self.client.delete_collection("code_index")
            self.collection = self.client.get_or_create_collection(
                name="code_index",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass

        if self.verbose:
            print(f"Generating embeddings for {len(docs)} files...")

        embed_start = time.time()
        embeddings = []

        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_embeddings = self.embedding.embed(batch_docs)
            embeddings.extend(batch_embeddings)

            if self.verbose:
               print_progress(
                    min(i + batch_size, len(docs)),
                    len(docs),
                    "Generating embeddings...",
                    time.time() - embed_start
                )

        if self.verbose:
            print()

        if self.verbose:
            print("Saving to vector store...")

        save_start = time.time()
        batch = 64
        for i in range(0, len(docs), batch):
            self.collection.add(
                ids=ids[i: i + batch],
                documents=docs[i: i + batch],
                metadatas=metadatas[i: i + batch],
                embeddings=embeddings[i: i + batch],
            )

            if self.verbose:
                print_progress(
                    min(i + batch, len(docs)),
                    len(docs),
                    "Saving vectors...",
                    time.time() - save_start
                )

        if self.verbose:
            print()

        self._save_index_metadata({
            "files": file_hashes,
            "last_indexed": datetime.now().isoformat(),
            "total_files": len(docs)
        })

        total_time = time.time() - start_time
        if self.verbose:
            print(f" Index built successfully!")
            print(f"   Indexed {len(docs)} files in {total_time:.1f} seconds")
            print(f"   Average: {total_time / len(docs):.2f} seconds per file")

    def incremental_update(self) -> None:
        start_time = time.time()

        if self.verbose:
            print("Checking for changes...")

        metadata = self._load_index_metadata()
        old_files = metadata.get("files", {})

        if not old_files:
            if self.verbose:
                print("No existing index found, building full index...")
            self.build_index(force=True)
            return

        files_to_update = []
        files_to_remove = []
        current_files = {}

        all_files = list(self._iter_files())

        if self.verbose:
            print(f"Scanning {len(all_files)} files for changes...")

        for i, p in enumerate(all_files):
            if self.verbose and i % 50 == 0:
                print(f"\rScanning... {i}/{len(all_files)}", end='', flush=True)

            try:
                filepath = str(p)
                file_hash = self._get_file_hash(p)
                current_files[filepath] = file_hash

                if filepath not in old_files or old_files[filepath] != file_hash:
                    files_to_update.append(p)
            except Exception as e:
                if self.verbose:
                    print(f"\nError checking {p}: {e}")
                continue

        if self.verbose:
            print(f"\rScanning... Done!                    ")

        for old_path in old_files:
            if old_path not in current_files:
                files_to_remove.append(old_path)

        if self.verbose:
            if files_to_update:
                print(f"Found {len(files_to_update)} files to update:")
                for p in files_to_update[:5]:  # Показуємо перші 5
                    status = "new" if str(p) not in old_files else "modified"
                    print(f"   - {status}: {p.relative_to(self.project_root)}")
                if len(files_to_update) > 5:
                    print(f"   ... and {len(files_to_update) - 5} more")

            if files_to_remove:
                print(f"Found {len(files_to_remove)} files to remove")

        if not files_to_update and not files_to_remove:
            if self.verbose:
                print("Index is up to date!")
            return

        if files_to_remove:
            if self.verbose:
                print(f"Removing {len(files_to_remove)} deleted files...")

            # Батчами для швидкості
            batch_size = 100
            for i in range(0, len(files_to_remove), batch_size):
                batch = files_to_remove[i:i + batch_size]
                try:
                    self.collection.delete(ids=batch)
                except Exception as e:
                    if self.verbose:
                        print(f"Error removing batch: {e}")

        if files_to_update:
            if self.verbose:
                print(f"Processing {len(files_to_update)} files...")

            docs = []
            metadatas = []
            ids = []

            for i, p in enumerate(files_to_update):
                if self.verbose:
                    print_progress(i + 1, len(files_to_update), f"Reading {p.name}...", time.time() - start_time)

                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    rel = str(p.relative_to(self.project_root))
                    docs.append(text)
                    metadatas.append({"path": rel})
                    ids.append(str(p))
                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠Error reading {p}: {e}")
                    continue

            if self.verbose:
                print()

            if docs:
                if self.verbose:
                    print("Generating embeddings...")

                embeddings = []
                embed_start = time.time()
                batch_size = 20

                for i in range(0, len(docs), batch_size):
                    batch_docs = docs[i:i + batch_size]
                    batch_embeddings = self.embedding.embed(batch_docs)
                    embeddings.extend(batch_embeddings)

                    if self.verbose:
                        print_progress(
                            min(i + batch_size, len(docs)),
                            len(docs),
                            "Generating embeddings...",
                            time.time() - embed_start
                        )

                if self.verbose:
                    print()

                existing_ids = [id_ for id_ in ids if id_ in old_files]
                if existing_ids:
                    self.collection.delete(ids=existing_ids)

                if self.verbose:
                    print("Saving to vector store...")

                save_start = time.time()
                batch = 64
                for i in range(0, len(docs), batch):
                    self.collection.add(
                        ids=ids[i: i + batch],
                        documents=docs[i: i + batch],
                        metadatas=metadatas[i: i + batch],
                        embeddings=embeddings[i: i + batch],
                    )

                    if self.verbose:
                        print_progress(
                            min(i + batch, len(docs)),
                            len(docs),
                            "Saving vectors...",
                            time.time() - save_start
                        )

                if self.verbose:
                    print()

        self._save_index_metadata({
            "files": current_files,
            "last_indexed": datetime.now().isoformat(),
            "total_files": len(current_files)
        })

        total_time = time.time() - start_time
        if self.verbose:
            print(f"Index updated successfully in {total_time:.1f} seconds!")
            if files_to_update:
                print(f"Updated: {len(files_to_update)} files")
            if files_to_remove:
                print(f"Removed: {len(files_to_remove)} files")

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if self.verbose:
            print(f"Searching for: {query}")

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

        if self.verbose:
            print(f"   Found {len(items)} relevant files")

        return items