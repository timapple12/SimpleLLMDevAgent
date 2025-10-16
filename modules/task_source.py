from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import requests


@dataclass
class Task:
    source: str  # manual | trello
    title: str
    description: str
    url: Optional[str] = None
    id: Optional[str] = None


class ManualTaskSource:
    def get(self) -> Task:
        text = input("> Enter task description: ")
        return Task(source="manual", title=text.strip()[:72] or "Manual Task", description=text.strip())


class TrelloTaskSource:
    def __init__(self, api_key: str, token: str, board_id: Optional[str] = None, list_id: Optional[str] = None, card_id: Optional[str] = None) -> None:
        self.api_key = api_key
        self.token = token
        self.board_id = board_id
        self.list_id = list_id
        self.card_id = card_id
        self.base = "https://api.trello.com/1"

    def _params(self):
        return {"key": self.api_key, "token": self.token}

    def _get(self, path: str, **params):
        r = requests.get(f"{self.base}{path}", params={**self._params(), **params}, timeout=60)
        r.raise_for_status()
        return r.json()

    def get(self) -> Task:
        if self.card_id:
            card = self._get(f"/cards/{self.card_id}")
        else:
            if self.list_id:
                cards = self._get(f"/lists/{self.list_id}/cards")
            elif self.board_id:
                cards = self._get(f"/boards/{self.board_id}/cards")
            else:
                raise ValueError("Trello requires at least card_id, list_id, or board_id")
            if not cards:
                raise RuntimeError("No Trello cards found")
            card = cards[0]
        desc = card.get("desc", "")
        name = card.get("name", "Trello Task")
        url = card.get("url")
        return Task(source="trello", title=name, description=f"{name}\n\n{desc}".strip(), url=url, id=card.get("id"))


