from __future__ import annotations

import datetime

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from pathlib import Path
from modules.code_indexer import CodeIndexer
from modules.llm_agent import LLMAgent
from modules.patch_applier import PatchApplier
from modules.config_loader import ConfigLoader
from modules.task_source import ManualTaskSource, TrelloTaskSource
from modules.task_analyzer import TaskAnalyzer


def prompt_user(prompt: str) -> str:
    try:
        return input(prompt)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)


def ensure_paths():
    agent_root = Path(__file__).resolve().parent
    (agent_root / "vector_store").mkdir(parents=True, exist_ok=True)


def main():
    ensure_paths()
    agent_root = Path(__file__).resolve().parent
    config_path = agent_root / "config.yaml"
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        sys.exit(1)

    app = ConfigLoader.load(str(config_path))

    project_path = Path(app.project_path).resolve()
    if not project_path.exists():
        print(f"Project path does not exist: {project_path}")
        sys.exit(1)

    # Task source selection
    if app.agent.task_source == "trello" and app.trello.enabled and app.trello.api_key and app.trello.token:
        task_source = TrelloTaskSource(
            api_key=app.trello.api_key,
            token=app.trello.token,
            board_id=app.trello.board_id,
            list_id=app.trello.list_id,
            card_id=app.trello.card_id,
        )
        task = task_source.get()
        task_description = task.description
        print(f"Using Trello task: {task.title}")
        if task.url:
            print(f"{task.url}")
    else:
        print("Enter your task/feature description (end with Enter):")
        task_description = prompt_user("> ")
        if not task_description.strip():
            print("Task description is required.")
            sys.exit(1)

    # 1) Build or update index
    indexer = CodeIndexer(
        project_root=str(project_path),
        vector_store_path=str(agent_root / "vector_store"),
        provider=app.provider.provider,
        openai_api_key=app.provider.openai_api_key,
        openai_embeddings_model=app.provider.openai_embeddings_model or "text-embedding-3-small",
        ollama_host=app.provider.ollama_host,
        ollama_embeddings_model=app.provider.embeddings_model or "nomic-embed-text:latest",
        include_extensions=app.index.include_extensions,
        exclude_dirs=app.index.exclude_dirs,
    )
    print("Indexing project files...")
    indexer.build_index()

    top_k = int(app.agent.top_k)
    relevant = indexer.search(task_description, top_k=top_k)

    # Analyze sufficiency if task came from Trello
    """if app.agent.task_source == "trello" and app.trello.enabled:
        llm_for_analysis = LLMAgent(
            provider=app.provider.provider,
            model=app.provider.model,
            project_root=str(project_path),
            openai_api_key=app.provider.openai_api_key,
            openai_model=app.provider.openai_model or app.provider.model,
            ollama_host=app.provider.ollama_host,
        )
        analyzer = TaskAnalyzer(llm_for_analysis)
        verdict = analyzer.analyze(task_description, relevant)
        print(f"Task sufficiency: {verdict.get('sufficient')} | {verdict.get('reason')}")
        missing = verdict.get("missing") or []
        if missing:
            print("Missing details:")
            for m in missing:
                print(f"- {m}")"""

    # 2) Generate patch via LLM
    llm = LLMAgent(
        provider=app.provider.provider,
        model=app.provider.model,
        openai_api_key=app.provider.openai_api_key,
        openai_model=app.provider.openai_model or app.provider.model,
        ollama_host=app.provider.ollama_host,
        project_root=str(project_path),
    )

    print("Generating git diff patch...")
    patch_text = llm.generate_git_diff(task_description, relevant)

    output_patch_dir = Path(app.agent.output_patch_path).resolve()
    if not output_patch_dir.is_absolute():
        output_patch_dir = agent_root / output_patch_dir

    output_patch_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #patch_filename = f"patch_{timestamp}.patch"

    task_slug = task_description[:30].replace(" ", "_").replace("/", "_")
    patch_filename = f"patch_{timestamp}_{task_slug}.patch"

    output_patch_path = output_patch_dir / patch_filename

    with open(output_patch_path, "w", encoding="utf-8") as f:
        f.write(patch_text)

    print(f"Patch saved to {output_patch_path}")

    # 3) Ask user to apply
    apply_answer = prompt_user("Apply patch, create branch and open PR? [y/N]: ").strip().lower()
    if apply_answer != "y":
        print("Skipping apply.")
        return

    applier = PatchApplier(
        repo_root=str(project_path),
        github_config={
            "token": app.github.token,
            "repo": app.github.repo,
            "base_branch": app.github.base_branch,
            "pr_title_prefix": app.github.pr_title_prefix,
            "pr_body_footer": app.github.pr_body_footer,
        },
    )

    branch_name = applier.generate_branch_name(task_description)
    print(f"Using branch: {branch_name}")
    applier.apply_and_push(
        patch_file=str(output_patch_path),
        branch_name=branch_name,
        commit_message=f"feat: {task_description.strip()[:72]}",
        base_branch=app.github.base_branch,
        pr_title_prefix=app.github.pr_title_prefix,
        pr_body_footer=app.github.pr_body_footer,
        repo_full_name=app.github.repo,
        github_token=app.github.token,
    )


if __name__ == "__main__":
    main()


