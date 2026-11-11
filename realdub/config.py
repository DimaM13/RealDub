"""Налаштування RealDub. Можна змінювати у програмі або у файлі config.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CONFIG_FILE = ROOT / "config.json"
ICON_ICO = Path(__file__).resolve().parent / "icon.ico"
ICON_PNG = Path(__file__).resolve().parent / "icon.png"

# Доступні моделі розпізнавання Whisper (код -> (Назва в меню, HuggingFace репозиторій))
WHISPER_MODELS = {
    "small": ("⚡ Small (Легка)", "Systran/faster-whisper-small"),
    "medium": ("🎯 Medium (Точна)", "Systran/faster-whisper-medium"),
    "large-v3-turbo": ("🚀 Large-Turbo (Найкраща)", "Lunaxod/large-v3-turbo-faster-whisper"),
    "large-v3": ("💎 Large-v3 (Максимум)", "Systran/faster-whisper-large-v3"),
}

DEFAULTS = {
    # Модель розпізнавання: small / medium / large-v3-turbo / large-v3
    "whisper_model": "large-v3-turbo",
    # "auto" — відеокарта NVIDIA, якщо є, інакше процесор
    "device": "auto",
    # Пристрій виводу для озвучення (частина назви, напр. "CABLE Input"); "" — типовий
    "tts_output_device": "",
    # Поріг гучності мікрофона/звуку (0..1), нижче — тиша
    "min_level": 0.008,
    # Пауза (сек), після якої фраза вважається завершеною
    "silence_sec": 0.4,
    # Максимальна довжина фрази (сек) — при безперервному мовленні ріжемо частіше
    "max_phrase_sec": 4.5,
    # Показувати проміжний переклад під час мовлення
    "live_partial": True,
    # Основний розмір шрифту перекладу (pt)
    "font_size": 21,
    # Розмір шрифту оригіналу / англійського тексту (відносно основного: 0 = авто fs-2)
    "orig_font_size": 18,
    "opacity": 0.86,
    "max_lines": 4,
    # Через скільки секунд застарілі рядки зникають (останні 2 завжди залишаються)
    "line_ttl_sec": 120,
    # Автоматично вмикати захоплення звуку ПК при запуску
    "auto_listen_system": True,
    # Автоматично перевіряти оновлення на GitHub при старті
    "check_updates_on_start": True,
}

# Версія налаштувань: при зміні старий config.json оновлюється новими типовими значеннями
CONFIG_VERSION = 6

# Голоси Piper (офлайн синтез мовлення)
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
            if user.get("version") == CONFIG_VERSION:
                cfg.update(user)
            else:
                for k in ("device", "tts_output_device"):
                    if k in user:
                        cfg[k] = user[k]
        except Exception:
            pass
    cfg["version"] = CONFIG_VERSION
    save(cfg)
    return cfg


def save(cfg: dict):
    try:
        cfg["version"] = CONFIG_VERSION
        CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
