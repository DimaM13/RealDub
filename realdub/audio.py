"""Захоплення звуку: системний звук (WASAPI loopback) та мікрофон.
Розбиття на фрази простим енергетичним детектором мовлення."""
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio

SR = 16000
FRAME = 480  # 30 мс
_PA = None  # один спільний екземпляр PortAudio
_PA_LOCK = threading.Lock()


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
            self.pre = self.pre[-10:]  # 300 мс до початку мовлення
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
            if size > SR * 0.5:
                self.on_phrase(audio, True)
        elif size >= self.max_len:
            # Безперервне мовлення: ріжемо в найтихішому місці останніх ~1.2 с, решта йде далі
            tail = self.levels[-40:]
            cut = len(self.buf) - len(tail) + int(np.argmin(tail)) + 1
            audio = np.concatenate(self.buf[:cut])
            self.buf, self.levels = self.buf[cut:], self.levels[cut:]
            self.last_partial = time.time()
            self.on_phrase(audio, True)
        elif self.partial and size > SR * 0.8 and time.time() - self.last_partial > 0.6:
            self.last_partial = time.time()
            self.on_phrase(np.concatenate(self.buf), False)


class Capture:
    """Потік захоплення з пристрою. source: 'system' або 'mic'."""

    def __init__(self, source: str, segmenter: Segmenter):
        self.source, self.seg = source, segmenter
        self.enabled = False
        self.muted = False  # тимчасово (під час нашого озвучення)
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _device(self, pa):
        if self.source == "system":
            return pa.get_default_wasapi_loopback()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        return pa.get_device_info_by_index(wasapi["defaultInputDevice"])

    def _run(self):
        with _PA_LOCK:
            global _PA
            if _PA is None:
                _PA = pyaudio.PyAudio()
        pa = _PA
        while not self._stop:
            if not self.enabled:
                time.sleep(0.1)
                continue
            try:
                dev = self._device(pa)
                ch = max(1, int(dev["maxInputChannels"]))
                sr = int(dev["defaultSampleRate"])
                st = pa.open(format=pyaudio.paFloat32, channels=ch, rate=sr, input=True,
                             input_device_index=dev["index"], frames_per_buffer=sr // 20)
                while self.enabled and not self._stop:
                    raw = st.read(sr // 20, exception_on_overflow=False)
                    x = np.frombuffer(raw, np.float32).reshape(-1, ch).mean(axis=1)
                    if self.muted:
                        x = np.zeros_like(x)
                    self.seg.push(to_16k(x, sr))
                st.stop_stream()
                st.close()
            except Exception as e:
                print(f"[аудіо {self.source}] {e}")
                time.sleep(1)
