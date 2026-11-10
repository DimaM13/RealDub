"""Захоплення звуку для Windows: системний звук комп'ютера (WASAPI loopback) та мікрофон.
Надійне розбиття на фрази та захист від зависань аудіодрайвера."""
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio

SR = 16000
FRAME = 480  # 30 мс
_PA = None
_PA_LOCK = threading.Lock()


def get_pyaudio_instance():
    """Повертає спільний екземпляр PyAudio з потокобезпечною ініціалізацією."""
    global _PA
    with _PA_LOCK:
        if _PA is None:
            _PA = pyaudio.PyAudio()
        return _PA


def reset_pyaudio_instance():
    """Переініціалізує PortAudio при системних помилках драйвера Windows."""
    global _PA
    with _PA_LOCK:
        if _PA is not None:
            try:
                _PA.terminate()
            except Exception:
                pass
            _PA = None


def to_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return data
    n = int(len(data) * SR / sr)
    return np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)


class Segmenter:
    """Збирає аудіо у фрази; викликає on_phrase(audio, final: bool)."""

    def __init__(self, on_phrase, min_level, silence_sec, max_sec, partial):
        self.on_phrase = on_phrase
        self.min_level = max(0.005, float(min_level))
        self.sil_frames = int(silence_sec * SR / FRAME)
        self.max_len = int(max_sec * SR)
        self.partial = partial
        self.rest = np.zeros(0, np.float32)
        self.pre = []
        self.buf = []
        self.levels = []
        self.speech = False
        self.voiced = 0
        self.silent = 0
        self.last_partial = 0.0
        self.had_partial = False

    def push(self, x: np.ndarray):
        x = np.concatenate([self.rest, x])
        n = len(x) // FRAME
        self.rest = x[n * FRAME:]
        for i in range(n):
            self._frame(x[i * FRAME:(i + 1) * FRAME])

    def _frame(self, f):
        rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
        # Стабільний поріг гучності без задирання: мова завжди чітко детектується
        loud = rms > self.min_level

        if not self.speech:
            self.pre.append(f)
            self.pre = self.pre[-10:]  # 300 мс перед початком мовлення
            self.voiced = self.voiced + 1 if loud else 0
            if self.voiced >= 3:
                self.speech = True
                self.buf = list(self.pre)
                self.levels = [0.0] * len(self.buf)
                self.silent = 0
                self.last_partial = time.time()
                self.had_partial = False
            return

        self.buf.append(f)
        self.levels.append(rms)
        self.silent = 0 if loud else self.silent + 1
        size = len(self.buf) * FRAME

        if self.silent >= self.sil_frames:
            # Завершення фрази паузою
            audio = np.concatenate(self.buf)
            had_p = self.had_partial
            self.speech, self.buf, self.levels, self.pre, self.voiced = False, [], [], [], 0
            self.had_partial = False

            if size >= int(SR * 0.3):
                self.on_phrase(audio, True)
            elif had_p:
                # Очищаємо незавершений чернетковий фрагмент у UI
                self.on_phrase(np.zeros(0, np.float32), True)

        elif size >= self.max_len:
            # Безперервне мовлення: ріжемо в найбільш тихому місці останніх ~1.2 с
            tail = self.levels[-40:]
            cut = len(self.buf) - len(tail) + int(np.argmin(tail)) + 1
            audio = np.concatenate(self.buf[:cut])
            self.buf, self.levels = self.buf[cut:], self.levels[cut:]
            self.last_partial = time.time()
            self.had_partial = False
            self.on_phrase(audio, True)

        elif self.partial and size >= int(SR * 0.6) and time.time() - self.last_partial > 0.45:
            self.last_partial = time.time()
            self.had_partial = True
            self.on_phrase(np.concatenate(self.buf), False)


class Capture:
    """Потік захоплення звуку на Windows: WASAPI loopback для системи та мікрофон.
    Оснащений захистом від збоїв драйвера, автоперезапуском та контролем життя стріму."""

    def __init__(self, source: str, segmenter: Segmenter, cfg: dict = None):
        self.source, self.seg, self.cfg = source, segmenter, cfg or {}
        self.enabled = False
        self.muted = False
        self._stop = False
        self.last_read_time = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _device(self, pa):
        if self.source == "system":
            return pa.get_default_wasapi_loopback()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        return pa.get_device_info_by_index(wasapi["defaultInputDevice"])

    def _run(self):
        consecutive_errors = 0

        while not self._stop:
            if not self.enabled:
                time.sleep(0.1)
                continue

            pa = get_pyaudio_instance()
            st = None
            try:
                dev = self._device(pa)
                ch = max(1, int(dev["maxInputChannels"]))
                sr = int(dev["defaultSampleRate"])

                st = pa.open(
                    format=pyaudio.paFloat32,
                    channels=ch,
                    rate=sr,
                    input=True,
                    input_device_index=dev["index"],
                    frames_per_buffer=sr // 20,
                )
                consecutive_errors = 0

                while self.enabled and not self._stop:
                    try:
                        raw = st.read(sr // 20, exception_on_overflow=False)
                        self.last_read_time = time.time()
                    except Exception as read_err:
                        # Якщо буфер переповнився або тимчасовий збій — не падати намертво
                        time.sleep(0.02)
                        continue

                    x = np.frombuffer(raw, np.float32).reshape(-1, ch).mean(axis=1)
                    if self.muted:
                        x = np.zeros_like(x)
                    self.seg.push(to_16k(x, sr))

            except Exception as e:
                consecutive_errors += 1
                time.sleep(0.5)
                # Якщо кілька помилок поспіль — перезавантажуємо драйвер PortAudio
                if consecutive_errors >= 2:
                    reset_pyaudio_instance()
                    time.sleep(0.5)

            finally:
                if st is not None:
                    try:
                        st.stop_stream()
                    except Exception:
                        pass
                    try:
                        st.close()
                    except Exception:
                        pass
