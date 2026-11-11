"""Модуль автоматичного оновлення RealDub.
Безпечно перевіряє та застосовує оновлення, якщо налаштовано репозиторій оновлень."""
import io
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

_NO_WIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run_git(args: list[str], cwd: Path = None, timeout: int = 10):
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_NO_WIN,
    )


def _get_repo_url(target_dir: Path) -> str:
    """Отримує URL репозиторію з локального .git або config.json без вшивання особистих даних у код."""
    try:
        cfg_file = target_dir / "config.json"
        if cfg_file.exists():
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if data.get("update_repo_url"):
                return data["update_repo_url"]
    except Exception:
        pass
    try:
        if (target_dir / ".git").exists():
            res = _run_git(["config", "--get", "remote.origin.url"], cwd=target_dir, timeout=4)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
    except Exception:
        pass
    return ""


def parse_version(v_str: str) -> tuple:
    """Перетворює рядок версії на зразок 'v2.2.1' у кортеж чисел для порівняння (2, 2, 1)."""
    clean = v_str.strip().lstrip("vV")
    parts = []
    for x in clean.split("."):
        m = re.match(r"(\d+)", x)
        parts.append(int(m.group(1)) if m else 0)
    return tuple(parts)


def check_for_updates(current_version: str, target_dir: Path = None) -> dict | None:
    """Перевіряє наявність новішої версії у підключеному репозиторії."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent

    repo_url = _get_repo_url(target_dir)
    if not repo_url:
        return None

    # 1. Перевіряємо віддалені теги через git ls-remote
    try:
        res = _run_git(["ls-remote", "--tags", repo_url], timeout=8)
        if res.returncode == 0 and res.stdout:
            tags = re.findall(r"refs/tags/(v\d+\.\d+(?:\.\d+)?)", res.stdout)
            if tags:
                best_tag = max(set(tags), key=parse_version)
                if parse_version(best_tag) > parse_version(current_version):
                    return {
                        "version": best_tag,
                        "name": f"Реліз {best_tag}",
                        "body": "Доступна нова версія з покращеннями стабільності та швидкодії.",
                        "method": "git_remote",
                    }
    except Exception:
        pass

    # 2. Якщо у папці є .git — перевіряємо нові коміти у гілці main
    git_dir = target_dir / ".git"
    if git_dir.exists():
        try:
            _run_git(["fetch", "origin", "main", "--quiet"], cwd=target_dir, timeout=6)
            res = _run_git(["rev-list", "HEAD..origin/main", "--count"], cwd=target_dir, timeout=4)
            behind_count = int(res.stdout.strip() or "0")
            if behind_count > 0:
                return {
                    "version": f"+{behind_count} оновлень",
                    "name": "Нові зміни",
                    "body": f"Доступно {behind_count} нових оновлень у гілці main.",
                    "method": "git",
                }
        except Exception:
            pass

    return None


def apply_update(target_dir: Path = None, on_progress=None) -> tuple[bool, str]:
    """Завантажує та застосовує оновлення, зберігаючи моделі (models/) та налаштування (config.json)."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent

    def report(msg: str):
        if on_progress:
            on_progress(msg)

    repo_url = _get_repo_url(target_dir)
    if not repo_url:
        return False, f"✅ Автономна офлайн-версія (усі моделі активні)"

    git_dir = target_dir / ".git"

    try:
        report("Синхронізую файли з репозиторієм...")
        if not git_dir.exists():
            _run_git(["init"], cwd=target_dir, timeout=5)
            _run_git(["remote", "add", "origin", repo_url], cwd=target_dir, timeout=5)
        else:
            _run_git(["remote", "set-url", "origin", repo_url], cwd=target_dir, timeout=5)

        fetch_res = _run_git(["fetch", "origin", "main", "--quiet"], cwd=target_dir, timeout=25)
        if fetch_res.returncode == 0:
            chk_res = _run_git(
                ["checkout", "origin/main", "--", "realdub", "RealDub.exe", "main.pyw", "download_models.py", "requirements.txt", "start.bat", "README.md"],
                cwd=target_dir,
                timeout=20,
            )
            if chk_res.returncode == 0:
                _run_git(["reset", "--mixed", "origin/main"], cwd=target_dir, timeout=5)
                return True, "✅ Оновлено! Перезапускаю RealDub…"
    except Exception as e:
        return False, f"❌ Помилка оновлення: {e}"

    return False, "❌ Не вдалося з'єднатися з сервером оновлень"
