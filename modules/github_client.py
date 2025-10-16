from github import Github


class GitHubClient:
    def __init__(self, token: str, repo_full_name: str):
        self.gh = Github(token)
        self.repo = self.gh.get_repo(repo_full_name)

    def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        draft: bool = False,
    ):
        return self.repo.create_pull(title=title, body=body, head=head, base=base, draft=draft)


