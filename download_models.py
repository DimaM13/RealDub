"""Перевірка та одноразове завантаження моделей ШІ.
Якщо моделі вже є у папці models/ (як в установленому пакеті), інтернет взагалі не потрібен."""
import sys
from realdub import config


def fetch(url: str) -> bytes:
    import requests
    print(f"  ↓ {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def get_whisper(name: str):
    out = config.MODELS / f"whisper-{name}"
    if (out / "model.bin").exists():
        print(f"✓ Whisper {name} вже є")
        return
    print(f"• Завантажую Whisper {name}…")
    from faster_whisper import download_model
    out.mkdir(parents=True, exist_ok=True)
    download_model(name, output_dir=str(out))


def get_text_translator(src: str, dst: str):
    out = config.MODELS / f"text-{src}-{dst}"
    if (out / "model.bin").exists():
        print(f"✓ Модель перекладу тексту {src}↔{dst} вже є")
        return
    repo = f"mults/hplt-translate-{src}-{dst}-ct2-int8"
    print(f"• Завантажую модель перекладу тексту {src}→{dst} ({repo})…")
    from huggingface_hub import snapshot_download
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=str(out))


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


def get_nllb():
    out = config.MODELS / "nllb-600m"
    if (out / "model.bin").exists():
        print("✓ NLLB-200 вже є")
        return
    print("• Завантажую перекладач NLLB-200 (≈600 МБ)…")
    from huggingface_hub import snapshot_download
    snapshot_download("osa911/nllb-200-distilled-600M-ct2-int8", local_dir=str(out))


if __name__ == "__main__":
    cfg = config.load()
    config.MODELS.mkdir(exist_ok=True)
    get_whisper("medium")
    get_whisper("small")
    get_nllb()
    get_text_translator("en", "uk")
    get_text_translator("uk", "en")
    get_voice("uk")
    get_voice("en")
    print("\nГотово! Усі моделі на місці для 100% автономної офлайн-роботи.")
