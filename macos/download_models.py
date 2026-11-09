"""Одноразове завантаження моделей для macOS (потрібен інтернет лише 1 раз)."""
import io
import sys
import zipfile

import requests

from realdub import config

ARGOS_INDEX = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"


def fetch(url: str) -> bytes:
    print(f"  ↓ {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def get_whisper(name: str):
    from faster_whisper import download_model
    out = config.MODELS / f"whisper-{name}"
    if (out / "model.bin").exists():
        print(f"✓ Whisper {name} вже завантажено")
        return
    print(f"• Завантажую Whisper {name}…")
    download_model(name, output_dir=str(out))


def get_nllb():
    out = config.MODELS / "nllb-600m"
    if (out / "model.bin").exists():
        print("✓ NLLB-200 вже завантажено")
        return
    print("• Завантажую NLLB-200 (≈600 МБ)…")
    from huggingface_hub import snapshot_download
    snapshot_download("osa911/nllb-200-distilled-600M-ct2-int8", local_dir=str(out))


def get_translator(src: str, dst: str, index: list):
    out = config.MODELS / f"mt-{src}-{dst}"
    if out.exists() and any(out.rglob("model.bin")):
        print(f"✓ Переклад {src}→{dst} вже є")
        return
    pkg = next((p for p in index if p.get("from_code") == src and p.get("to_code") == dst), None)
    if not pkg:
        return
    print(f"• Завантажую запасну модель {src}→{dst}…")
    for link in pkg["links"]:
        try:
            data = fetch(link)
            break
        except Exception:
            pass
    else:
        return
    out.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)


def get_voice(lang: str):
    path = config.VOICES[lang]
    name = path.rsplit("/", 1)[-1]
    out = config.MODELS / "voices"
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".onnx", ".onnx.json"):
        f = out / (name + ext)
        if not f.exists():
            f.write_bytes(fetch(config.PIPER_BASE + path + ext))
    print(f"✓ Голос {lang}: {name}")


if __name__ == "__main__":
    cfg = config.load()
    config.MODELS.mkdir(exist_ok=True)
    get_whisper(cfg["whisper_model"])
    get_nllb()
    try:
        index = requests.get(ARGOS_INDEX, timeout=30).json()
        get_translator("en", "uk", index)
        get_translator("uk", "en", index)
    except Exception:
        pass
    get_voice("uk")
    get_voice("en")
    print("\nГотово! Моделі завантажено. Тепер можна запускати офлайн: ./start.sh")
