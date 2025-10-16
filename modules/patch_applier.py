import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .github_client import GitHubClient


class PatchApplier:
    def __init__(self, repo_root: str, github_config: dict) -> None:
        self.repo_root = Path(repo_root)
        self.github_config = github_config

    def run(self, *args: str) -> str:
        result = subprocess.run(
            args,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        return result.stdout.strip()

    def generate_branch_name(self, task: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9-_]+", "-", task.strip().lower())
        slug = re.sub(r"-+", "-", slug).strip("-")
        short = slug[:40] or "change"
        return f"feat/{short}"

    def apply_and_push(
        self,
        patch_file: str,
        branch_name: str,
        commit_message: str,
        base_branch: str,
        pr_title_prefix: str,
        pr_body_footer: str,
        repo_full_name: Optional[str],
        github_token: Optional[str],
    ) -> None:
        patch_path = Path(patch_file)
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch not found: {patch_path}")

        # Ensure repo is clean and up to date
        self.run("git", "fetch", "origin", base_branch)
        self.run("git", "checkout", base_branch)
        self.run("git", "pull", "--ff-only", "origin", base_branch)

        # Create new branch
        self.run("git", "checkout", "-b", branch_name)

        # Apply patch
        self.run("git", "apply", "--whitespace=fix", str(patch_path))
        self.run("git", "add", ".")
        self.run("git", "commit", "-m", commit_message)
        self.run("git", "push", "-u", "origin", branch_name)

        # Open PR if configured
        if repo_full_name and github_token:
            gh = GitHubClient(github_token, repo_full_name)
            title = f"{pr_title_prefix} {commit_message}".strip()
            body = f"{commit_message}{pr_body_footer}"
            gh.create_pull_request(title=title, body=body, head=branch_name, base=base_branch)


