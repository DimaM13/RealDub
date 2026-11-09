"""Захоплення звуку для macOS: CoreAudio через sounddevice.
Підтримка системного звуку через BlackHole та вбудованого мікрофона."""
import threading
import time

import numpy as np
import sounddevice as sd

SR = 16000
FRAME = 480  # 30 мс


def to_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return data
    n = int(len(data) * SR / sr)
    return np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)


class Segmenter:
    """Збирає аудіо у фрази; викликає on_phrase(audio, final: bool)."""

    def __init__(self, on_phrase, min_level, silence_sec, max_sec, partial):
        self.on_phrase = on_phrase
        self.min_level = min_level
        self.sil_frames = int(silence_sec * SR / FRAME)
        self.max_len = int(max_sec * SR)
        self.partial = partial
        self.floor = 0.003
        self.rest = np.zeros(0, np.float32)
        self.pre = []
        self.buf = []
        self.levels = []
        self.speech = False
        self.voiced = 0
        self.silent = 0
        self.last_partial = 0.0

    def push(self, x: np.ndarray):
        x = np.concatenate([self.rest, x])
        n = len(x) // FRAME
        self.rest = x[n * FRAME:]
        for i in range(n):
            self._frame(x[i * FRAME:(i + 1) * FRAME])

    def _frame(self, f):
        rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
        loud = rms > max(self.min_level, self.floor * 3.0)
        if not self.speech:
            self.floor = 0.97 * self.floor + 0.03 * min(rms, 0.05)
            self.pre.append(f)
            self.pre = self.pre[-10:]  # 300 мс перед початком фрази
            self.voiced = self.voiced + 1 if loud else 0
            if self.voiced >= 3:
                self.speech, self.buf, self.silent = True, list(self.pre), 0
                self.levels = [0.0] * len(self.buf)
                self.last_partial = time.time()
            return
        self.buf.append(f)
        self.levels.append(rms)
        self.silent = 0 if loud else self.silent + 1
        size = len(self.buf) * FRAME
        if self.silent >= self.sil_frames:
            audio = np.concatenate(self.buf)
            self.speech, self.buf, self.levels, self.pre, self.voiced = False, [], [], [], 0
            if size >= int(SR * 0.3):
                self.on_phrase(audio, True)
        elif size >= self.max_len:
            # Розрізаємо довгу фразу в найбільш тихому моменті
            tail = self.levels[-40:]
            cut = len(self.buf) - len(tail) + int(np.argmin(tail)) + 1
            audio = np.concatenate(self.buf[:cut])
            self.buf, self.levels = self.buf[cut:], self.levels[cut:]
            self.last_partial = time.time()
            self.on_phrase(audio, True)
        elif self.partial and size >= int(SR * 0.6) and time.time() - self.last_partial > 0.5:
            self.last_partial = time.time()
            self.on_phrase(np.concatenate(self.buf), False)


def find_input_device(match: str):
    """Шукає пристрій введення за частиною назви в CoreAudio."""
    devices = sd.query_devices()
    m = match.strip().lower()
    if m:
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and m in dev["name"].lower():
                return idx, dev["name"], int(dev["default_samplerate"])
    # Типовий пристрій введення
    try:
        def_idx = sd.default.device[0]
        if def_idx is not None and def_idx >= 0:
            dev = devices[def_idx]
            return def_idx, dev["name"], int(dev["default_samplerate"])
    except Exception:
        pass
    # Перший доступний пристрій з вхідними каналами
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            return idx, dev["name"], int(dev["default_samplerate"])
    return None, "Невідомий пристрій", 16000


class Capture:
    """Потік захоплення аудіо на macOS через sounddevice.
    source: 'system' (шукає BlackHole) або 'mic' (вбудований мікрофон)."""

    def __init__(self, source: str, segmenter: Segmenter, cfg: dict):
        self.source, self.seg, self.cfg = source, segmenter, cfg
        self.enabled = False
        self.muted = False
        self._stop = False
        self.device_name = ""
        threading.Thread(target=self._run, daemon=True).start()

    def _resolve_device(self):
        if self.source == "system":
            target = self.cfg.get("blackhole_device_name", "BlackHole")
            idx, name, sr = find_input_device(target)
            if "blackhole" not in name.lower():
                print(f"[Увага] Пристрій '{target}' не знайдено на Mac. Захоплюємо з: {name}")
                print("[Порада] Встановіть BlackHole: brew install blackhole-2ch")
            return idx, name, sr
        else:
            target = self.cfg.get("mic_device_name", "")
            return find_input_device(target)

    def _run(self):
        while not self._stop:
            if not self.enabled:
                time.sleep(0.1)
                continue
            try:
                idx, self.device_name, sr = self._resolve_device()
                if idx is None:
                    time.sleep(1)
                    continue

                def callback(indata, frames, time_info, status):
                    if self.muted:
                        data = np.zeros(frames, dtype=np.float32)
                    else:
                        data = indata.mean(axis=1) if indata.ndim > 1 else indata
                    self.seg.push(to_16k(data.astype(np.float32), sr))

                with sd.InputStream(
                    device=idx,
                    channels=1,
                    samplerate=sr,
                    blocksize=int(sr * 0.03),  # блоки по ~30 мс
                    callback=callback,
                    dtype="float32",
                ):
                    while self.enabled and not self._stop:
                        time.sleep(0.05)
            except Exception as e:
                print(f"[macOS аудіо {self.source}] Помилка: {e}")
                time.sleep(1)
