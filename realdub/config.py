"""Налаштування RealDub. Можна змінювати у файлі config.json поруч з програмою."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CONFIG_FILE = ROOT / "config.json"

DEFAULTS = {
    # Модель розпізнавання: tiny / base / small / medium (small — найкращий баланс)
    "whisper_model": "small",
    # "auto" — відеокарта, якщо є, інакше процесор
    "device": "auto",
    # Пристрій виводу для озвучення (частина назви, напр. "CABLE Input"); "" — типовий
    "tts_output_device": "",
    # Поріг гучності мікрофона (0..1), нижче — тиша
    "min_level": 0.008,
    # Пауза (сек), після якої фраза вважається завершеною
    "silence_sec": 0.4,
    # Максимальна довжина фрази (сек) — при безперервному мовленні ріжемо частіше
    "max_phrase_sec": 4.5,
    # Показувати проміжний переклад під час мовлення
    "live_partial": True,
    "font_size": 20,
    "opacity": 0.82,
    "max_lines": 4,
    # Через скільки секунд застарілі рядки стають прозорими (0 = не зникають)
    "line_ttl_sec": 120,
    # Автоматично вмикати захоплення звуку ПК при запуску
    "auto_listen_system": True,
    # Автоматично перевіряти оновлення на GitHub при старті
    "check_updates_on_start": True,
}

# Версія налаштувань: при зміні старий config.json оновлюється новими типовими значеннями
CONFIG_VERSION = 5

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
            else:  # зберігаємо лише особисті налаштування пристроїв
                for k in ("whisper_model", "device", "tts_output_device"):
                    if k in user:
                        cfg[k] = user[k]
        except Exception:
            pass
    cfg["version"] = CONFIG_VERSION
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg
