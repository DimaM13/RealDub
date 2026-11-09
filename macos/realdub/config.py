"""Налаштування RealDub для macOS."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CONFIG_FILE = ROOT / "config.json"

DEFAULTS = {
    # Модель розпізнавання: tiny / base / small / medium (на Apple Silicon small працює чудово)
    "whisper_model": "small",
    # Для macOS використовується процесор Apple Silicon (ARM NEON)
    "device": "cpu",
    # Частина назви пристрою для захоплення звуку комп'ютера на macOS (зазвичай BlackHole)
    "blackhole_device_name": "BlackHole",
    # Назва мікрофона ("" — мікрофон за замовчуванням)
    "mic_device_name": "",
    # Пристрій виводу для озвучення ("" — типовий динамік/навушники)
    "tts_output_device": "",
    # Поріг гучності (0..1)
    "min_level": 0.008,
    # Пауза (сек) між фразами
    "silence_sec": 0.4,
    # Максимальна довжина фрази при безперервному мовленні (сек)
    "max_phrase_sec": 4.5,
    # Показувати проміжний переклад під час мовлення
    "live_partial": True,
    "font_size": 20,
    "opacity": 0.82,
    "max_lines": 3,
    "line_ttl_sec": 15,
}

CONFIG_VERSION = 2

# Офлайн голоси Piper
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICES = {
    "uk": "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium",
}


def load() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            user = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            cfg.update(user)
        except Exception:
            pass
    cfg["version"] = CONFIG_VERSION
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg
