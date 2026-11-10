"""Модуль автоматичного оновлення RealDub через Git та GitHub API.
Підтримує перевірку оновлень та безпечне застосування в 1 клік."""
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

GITHUB_REPO = "DimaM13/RealDub"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
TAGS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/tags"
ZIP_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.zip"


def parse_version(v_str: str) -> tuple:
    """Перетворює рядок версії на зразок 'v2.1.0' у кортеж чисел для порівняння (2, 1, 0)."""
    clean = v_str.strip().lstrip("vV")
    parts = []
    for x in clean.split("."):
        try:
            parts.append(int(x))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def check_for_updates(current_version: str, target_dir: Path = None) -> dict | None:
    """Перевіряє наявність новішої версії або комітів на GitHub.
    Працює як через Git (для приватних та публічних репозиторіїв), так і через GitHub API."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent

    # Спосіб 1: Якщо програма встановлена через Git
    git_dir = target_dir / ".git"
    if git_dir.exists():
        try:
            subprocess.run(["git", "fetch", "origin", "main", "--quiet"], cwd=str(target_dir), timeout=6)
            res = subprocess.run(
                ["git", "rev-list", "HEAD..origin/main", "--count"],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=4,
            )
            behind_count = int(res.stdout.strip() or "0")
            if behind_count > 0:
                return {
                    "version": f"+{behind_count} оновлень",
                    "name": "Нова версія на GitHub",
                    "body": f"Доступно нових оновлень у репозиторії: {behind_count}.",
                    "method": "git",
                }
        except Exception:
            pass

    # Спосіб 2: Через GitHub API (якщо репозиторій публічний або має релізи)
    headers = {"User-Agent": "RealDub-App"}
    try:
        req = urllib.request.Request(API_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                tag = data.get("tag_name", "")
                if parse_version(tag) > parse_version(current_version):
                    return {
                        "version": tag,
                        "name": data.get("name") or tag,
                        "body": data.get("body", "Оновлення коду та покращення стабільності."),
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

    # 1. Якщо репозиторій має .git — оновлюємося через git pull
    git_dir = target_dir / ".git"
    if git_dir.exists():
        try:
            report("Оновлюю файли через Git...")
            res = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if res.returncode == 0:
                return True, "✅ Оновлено через Git! Будь ласка, перезапустіть RealDub."
            else:
                report(f"Git помилка: {res.stderr.strip()[:60]}")
        except Exception as e:
            report(f"Помилка Git pull: {e}")

    # 2. Якщо git недоступний — завантажуємо zip-архів з GitHub
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

                # Ніколи не перезаписуємо персональні налаштування та папки моделей
                if rel_path == "config.json" or rel_path.startswith("models/") or rel_path.startswith(".venv/"):
                    continue

                dest_file = target_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src_f, open(dest_file, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

        return True, "✅ Оновлення успішно встановлено! Перезапустіть RealDub."
    except Exception as e:
        return False, f"❌ Помилка оновлення: {e}"
