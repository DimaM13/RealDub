"""Захоплення звуку для Windows: системний звук комп'ютера (WASAPI loopback) та мікрофон.
Неблокуючий режим stream_callback гарантує відсутність зависань у паузах та миттєве вивантаження при виході."""
import queue
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
    """Повністю закриває та звільняє ресурси PortAudio."""
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
    if len(data) == 0:
        return np.zeros(0, np.float32)
    n = max(1, int(len(data) * SR / sr))
    return np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)


class Segmenter:
    """Збирає аудіо у фрази; викликає on_phrase(audio, final: bool)."""

    def __init__(self, on_phrase, min_level, silence_sec, max_sec, partial):
        self.on_phrase = on_phrase
        self.min_level = max(0.004, float(min_level))
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
        if len(x) == 0:
            return
        x = np.concatenate([self.rest, x])
        n = len(x) // FRAME
        self.rest = x[n * FRAME:]
        for i in range(n):
            self._frame(x[i * FRAME:(i + 1) * FRAME])

    def _frame(self, f):
        rms = float(np.sqrt(np.mean(f * f)) + 1e-9)

        if not self.speech:
            loud = rms > self.min_level
            self.pre.append(f)
            self.pre = self.pre[-10:]  # 300 мс перед початком мовлення
            self.voiced = self.voiced + 1 if loud else 0
            if self.voiced >= 3:
                self.speech = True
                self.buf = list(self.pre)
                self.levels = [rms] * len(self.buf)
                self.silent = 0
                self.last_partial = time.time()
                self.had_partial = False
            return

        # Під час фрази враховуємо відносний спад гучності, щоб фоновий шум не заважав фіксувати паузи
        peak = max(self.levels[-50:]) if self.levels else rms
        dyn_thresh = max(self.min_level, peak * 0.22) if peak > 0.04 else self.min_level
        loud = rms > dyn_thresh

        self.buf.append(f)
        self.levels.append(rms)
        self.silent = 0 if loud else self.silent + 1
        size = len(self.buf) * FRAME

        if self.silent >= self.sil_frames:
            audio = np.concatenate(self.buf)
            had_p = self.had_partial
            self.speech, self.buf, self.levels, self.pre, self.voiced = False, [], [], [], 0
            self.had_partial = False

            if size >= int(SR * 0.3):
                self.on_phrase(audio, True)
            elif had_p:
                self.on_phrase(np.zeros(0, np.float32), True)

        elif size >= self.max_len:
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
    """Потік захоплення звуку на Windows через неблокуючий callback WASAPI.
    Не зависає, коли звук у Windows зупиняється, і коректно закриває фрази тишею."""

    def __init__(self, source: str, segmenter: Segmenter, cfg: dict = None):
        self.source, self.seg, self.cfg = source, segmenter, cfg or {}
        self.enabled = False
        self.muted = False
        self._stop = False
        self.last_read_time = time.time()
        self.last_error = ""
        self._q = queue.Queue(maxsize=200)
        threading.Thread(target=self._run, daemon=True).start()

    def _device(self, pa):
        if self.source == "system":
            return pa.get_default_wasapi_loopback()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        return pa.get_device_info_by_index(wasapi["defaultInputDevice"])

    def _run(self):
        consecutive_errors = 0
        silence_chunk = np.zeros(FRAME * 2, dtype=np.float32)

        while not self._stop:
            if not self.enabled:
                time.sleep(0.08)
                continue

            pa = get_pyaudio_instance()
            st = None
            try:
                dev = self._device(pa)
                dev_idx = dev["index"]
                ch = max(1, int(dev["maxInputChannels"]))
                sr = int(dev["defaultSampleRate"])

                # Очищаємо чергу перед стартом
                while not self._q.empty():
                    try:
                        self._q.get_nowait()
                    except Exception:
                        break

                def callback(in_data, frame_count, time_info, status):
                    if self._stop or not self.enabled:
                        return (None, pyaudio.paComplete)
                    self.last_read_time = time.time()
                    try:
                        self._q.put_nowait(in_data)
                    except queue.Full:
                        pass
                    return (None, pyaudio.paContinue)

                with _PA_LOCK:
                    st = pa.open(
                        format=pyaudio.paFloat32,
                        channels=ch,
                        rate=sr,
                        input=True,
                        input_device_index=dev_idx,
                        frames_per_buffer=max(256, sr // 25),
                        stream_callback=callback,
                    )
                    st.start_stream()

                consecutive_errors = 0
                self.last_read_time = time.time()
                last_dev_check = time.time()

                while self.enabled and not self._stop:
                    try:
                        raw = self._q.get(timeout=0.06)
                        x = np.frombuffer(raw, np.float32).reshape(-1, ch).mean(axis=1)
                        if self.muted:
                            x = np.zeros_like(x)
                        self.seg.push(to_16k(x, sr))
                    except queue.Empty:
                        # Коли в Windows нічого не грає, WASAPI loopback не шле пакети —
                        # подаємо тишу, щоб поточна фраза вчасно фіналізувалася!
                        self.seg.push(silence_chunk)

                    # Раз на 3 секунди перевіряємо, чи стрім активний і чи не переключили навушники
                    now = time.time()
                    if now - last_dev_check > 3.0:
                        last_dev_check = now
                        if not st.is_active():
                            break
                        if self.source == "system":
                            try:
                                cur_dev = self._device(pa)
                                if cur_dev["index"] != dev_idx:
                                    break  # користувач переключив пристрій виводу звуку
                            except Exception:
                                break

            except Exception as e:
                consecutive_errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(0.4)
                if consecutive_errors >= 2:
                    reset_pyaudio_instance()
                    time.sleep(0.4)

            finally:
                if st is not None:
                    with _PA_LOCK:
                        try:
                            st.stop_stream()
                        except Exception:
                            pass
                        try:
                            st.close()
                        except Exception:
                            pass
