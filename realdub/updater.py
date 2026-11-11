"""Модуль автоматичного оновлення RealDub через Git та GitHub API.
Підтримує перевірку оновлень та безпечне застосування в 1 клік."""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

GITHUB_REPO = "DimaM13/RealDub"
REPO_URL = f"https://github.com/{GITHUB_REPO}.git"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
TAGS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/tags"
ZIP_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.zip"

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


def parse_version(v_str: str) -> tuple:
    """Перетворює рядок версії на зразок 'v2.1.2' у кортеж чисел для порівняння (2, 1, 2)."""
    clean = v_str.strip().lstrip("vV")
    parts = []
    for x in clean.split("."):
        m = re.match(r"(\d+)", x)
        parts.append(int(m.group(1)) if m else 0)
    return tuple(parts)


def check_for_updates(current_version: str, target_dir: Path = None) -> dict | None:
    """Перевіряє наявність новішої версії на GitHub.
    Працює як для публічних, так і для приватних репозиторіїв (через Git)."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent

    # 1. Перевіряємо віддалені теги через git ls-remote (працює навіть без локального .git)
    try:
        res = _run_git(["ls-remote", "--tags", REPO_URL], timeout=8)
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
                    "name": "Нові зміни на GitHub",
                    "body": f"Доступно {behind_count} нових комітів у гілці main.",
                    "method": "git",
                }
        except Exception:
            pass

    # 3. Через публічний GitHub API
    headers = {"User-Agent": "RealDub-App"}
    try:
        req = urllib.request.Request(API_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                tag = data.get("tag_name", "")
                if parse_version(tag) > parse_version(current_version):
                    return {
                        "version": tag,
                        "name": data.get("name") or tag,
                        "body": data.get("body", "Оновлення коду."),
                        "method": "api",
                    }
    except Exception:
        pass

    return None


def apply_update(target_dir: Path = None, on_progress=None) -> tuple[bool, str]:
    """Завантажує та застосовує оновлення.
    Безпечно оновлює код програми, зберігаючи моделі (models/) та налаштування (config.json)."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent

    def report(msg: str):
        if on_progress:
            on_progress(msg)

    git_dir = target_dir / ".git"

    # 1. Оновлюємо код через Git (працює і для приватного, і для публічного репозиторію)
    try:
        report("Синхронізую файли з GitHub...")
        if not git_dir.exists():
            _run_git(["init"], cwd=target_dir, timeout=5)
            _run_git(["remote", "add", "origin", REPO_URL], cwd=target_dir, timeout=5)
        else:
            _run_git(["remote", "set-url", "origin", REPO_URL], cwd=target_dir, timeout=5)

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
    except Exception:
        pass

    # 2. Резервний спосіб через ZIP (якщо репозиторій відкритий)
    try:
        report("Завантажую архів оновлення з GitHub...")
        headers = {"User-Agent": "RealDub-App"}
        req = urllib.request.Request(ZIP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            zip_bytes = resp.read()

        report("Застосовую файли оновлення...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            root_prefix = names[0].split("/")[0] + "/"

            for member in zf.infolist():
                if member.is_dir():
                    continue
                rel_path = member.filename[len(root_prefix):]
                if not rel_path:
                    continue

                if rel_path == "config.json" or rel_path.startswith("models/") or rel_path.startswith(".venv/"):
                    continue

                dest_file = target_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src_f, open(dest_file, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

        return True, "✅ Оновлено! Перезапускаю RealDub…"
    except Exception as e:
        return False, f"❌ Помилка оновлення: {e}"
