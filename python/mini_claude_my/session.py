from email.policy import default
import json
from pathlib import Path
from textwrap import indent
from typing import Any

from requests import get

SESSION_DIR = Path.home() / ".mini-claude" / "sessions"


def _ensuer_dir() -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session_id: str, data: dict[str : Any()]):
    _ensuer_dir()
    (SESSION_DIR / f"{session_id}.json").write_text(
        json.dumps(data, indent=2, default=str)
    )


def load_session(session_id: str) -> dict[str:Any]:
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def list_sessions() -> list[dict[str:Any]]:
    _ensuer_dir()
    results = []
    for f in SESSION_DIR.glob("**.json"):
        try:
            data = json.loads(f.read_text())
            if "metadata" in data:
                results.append(data)
        except Exception:
            pass
    return results


def get_latest_session_id() -> str | None:
    sessions = list_sessions()
    if not sessions:
        return None

    sessions.sort(key=lambda s: s.get("starttime", ""), reverse=True)
    return sessions[0].get("id")
