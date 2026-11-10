"""Головне вікно-оверлей RealDub."""
import gc
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSizeGrip, QSizePolicy, QVBoxLayout, QWidget)

from realdub import __version__, config
from realdub.audio import Capture, Segmenter, reset_pyaudio_instance
from realdub.updater import apply_update, check_for_updates

PID_FILE = config.ROOT / "realdub.pid"
_NO_WIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

STYLE = """
#panel { background: rgba(14,16,24,%d); border-radius: 14px; border: 1px solid rgba(255,255,255,35); }
QLabel { color: #f2f4f8; background: transparent; }
QPushButton { color:#cfd6e4; background: rgba(255,255,255,18); border:none; border-radius:8px;
              padding:5px 10px; font-size:13px; }
QPushButton:hover { background: rgba(255,255,255,40); }
QPushButton:checked { background:#3d7cff; color:white; font-weight:600; }
QLineEdit { color:white; background: rgba(255,255,255,20); border:none; border-radius:8px;
            padding:6px 10px; font-size:14px; }
"""


def _ensure_single_instance():
    """Завершує попередню завислу копію RealDub, щоб у пам'яті ПК ніколи не накопичувалися дублікати."""
    my_pid = os.getpid()
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            if old_pid != my_pid and old_pid > 0:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(old_pid)],
                    capture_output=True,
                    timeout=3,
                    creationflags=_NO_WIN,
                )
                time.sleep(0.2)
        except Exception:
            pass
    try:
        PID_FILE.write_text(str(my_pid), encoding="utf-8")
    except Exception:
        pass


def _remove_pid_file():
    try:
        if PID_FILE.exists() and PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


class Bus(QObject):
    line = Signal(str, str, str, bool)  # напрям, оригінал, переклад, фінальний
    status = Signal(str)
    update_info = Signal(dict)
    ready = Signal()
    restart_app = Signal()


class Overlay(QWidget):
    def __init__(self, cfg, bus: Bus):
        super().__init__()
        self.cfg, self.bus = cfg, bus
        self.lines, self.partial = [], {}
        self.setWindowTitle(f"RealDub v{__version__}")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(STYLE % int(cfg["opacity"] * 255))
        self.resize(840, 310)
        self.setMinimumSize(520, 180)
        scr = QApplication.primaryScreen().availableGeometry()
        self.move(scr.center().x() - 420, scr.bottom() - 350)

        panel = QWidget(objectName="panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 10, 14, 10)

        bar = QHBoxLayout()
        title = QLabel(f"🌐 RealDub v{__version__}")
        title.setStyleSheet("font-weight:600; font-size:14px; color:#9fb4ff;")
        bar.addWidget(title)
        self.st = QLabel("Завантаження моделей…")
        self.st.setStyleSheet("color:#8a93a6; font-size:12px;")
        bar.addWidget(self.st, 1)

        self.b_sys = self._btn("🎧 Звук ПК  EN→UK", True)
        self.b_mic = self._btn("🎤 Мікрофон  UK→EN", True)
        self.b_tts = self._btn("🔊 Озвучувати", True)
        self.b_tts.setToolTip("Озвучувати англійський переклад моєї мови")
        for b in (self.b_sys, self.b_mic, self.b_tts):
            bar.addWidget(b)

        self.b_upd = self._btn("🔄", False)
        self.b_upd.setToolTip("Перевірити та встановити оновлення з GitHub")
        self.b_upd.clicked.connect(self._on_update_click)
        bar.addWidget(self.b_upd)

        clr = self._btn("🧹", False)
        clr.setToolTip("Очистити субтитри")
        clr.clicked.connect(self._clear)
        close = self._btn("✕", False)
        close.setToolTip("Закрити програму та вивантажити з пам'яті")
        close.clicked.connect(self._quit_app)
        bar.addWidget(clr)
        bar.addWidget(close)
        lay.addLayout(bar)

        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        self.text.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text.setMinimumHeight(90)
        self.text.setFont(QFont("Segoe UI", cfg["font_size"]))
        # Дозволяємо перетягувати вікно прямо за область тексту субтитрів
        self.text.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay.addWidget(self.text, 1)

        row = QHBoxLayout()
        self.inp = QLineEdit(placeholderText="Напишіть українською — перекладу англійською й озвучу…")
        self.inp.returnPressed.connect(self._typed)
        row.addWidget(self.inp, 1)
        row.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        lay.addLayout(row)

        bus.line.connect(self._on_line)
        bus.status.connect(self.st.setText)
        bus.update_info.connect(self._on_update_available)
        bus.ready.connect(self._on_engine_ready)
        bus.restart_app.connect(self._restart_now)
        self.latest_update = None
        self._updating = False
        self._drag = None
        self.on_typed = None
        self.on_shutdown = None
        self._timer = QTimer(self, interval=2000, timeout=self._expire)
        self._timer.start()

    def _on_engine_ready(self):
        # Автоматично вмикаємо прослуховування звуку ПК після готовності моделей
        if self.cfg.get("auto_listen_system", True) and not self.b_sys.isChecked() and not self.b_mic.isChecked():
            self.b_sys.setChecked(True)

    def _on_update_available(self, info: dict):
        self.latest_update = info
        self.b_upd.setText(f"🔄 {info['version']}")
        self.b_upd.setStyleSheet("background:#10b981; color:white; font-weight:bold; padding:5px 10px; border-radius:8px;")
        self.b_upd.setToolTip(f"Доступна нова версія {info['version']}! Натисніть для миттєвого оновлення.")
        self.st.setText(f"Доступно оновлення {info['version']} — натисніть зелену кнопку 🔄")

    def _on_update_click(self):
        if self._updating:
            return
        if self.latest_update:
            self._run_update()
        else:
            self._updating = True
            self.st.setText("Перевіряю оновлення на GitHub…")
            threading.Thread(target=self._manual_check_and_update, daemon=True).start()

    def _manual_check_and_update(self):
        try:
            info = check_for_updates(__version__)
            if info:
                self.bus.update_info.emit(info)
                ok, msg = apply_update(on_progress=self.bus.status.emit)
                self.bus.status.emit(msg)
                if ok:
                    time.sleep(1.0)
                    self.bus.restart_app.emit()
            else:
                self.bus.status.emit(f"✅ У вас найновіша версія (v{__version__})")
        finally:
            self._updating = False

    def _run_update(self):
        self._updating = True
        self.b_upd.setEnabled(False)
        self.st.setText("Завантажую та встановлюю оновлення…")

        def task():
            try:
                ok, msg = apply_update(on_progress=self.bus.status.emit)
                self.bus.status.emit(msg)
                if ok:
                    time.sleep(1.0)
                    self.bus.restart_app.emit()
            finally:
                self._updating = False

        threading.Thread(target=task, daemon=True).start()

    def _restart_now(self):
        """Перезапускає програму після оновлення, звільняючи старий процес."""
        if self.on_shutdown:
            self.on_shutdown()
        _remove_pid_file()
        try:
            subprocess.Popen(
                [sys.executable, str(config.ROOT / "main.pyw")],
                cwd=str(config.ROOT),
                creationflags=_NO_WIN,
            )
        except Exception:
            pass
        os._exit(0)

    def _quit_app(self):
        if self.on_shutdown:
            self.on_shutdown()
        _remove_pid_file()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        QApplication.quit()
        os._exit(0)

    def _btn(self, text, checkable):
        b = QPushButton(text)
        b.setCheckable(checkable)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def _clear(self):
        self.lines, self.partial = [], {}
        self._render()

    def _typed(self):
        t = self.inp.text().strip()
        if t and self.on_typed:
            self.on_typed(t)
        self.inp.clear()

    def _on_line(self, direction, orig, trans, final):
        if final:
            self.partial.pop(direction, None)
            if orig and trans:
                self.lines.append((direction, orig, trans, time.time()))
                self.lines = self.lines[-self.cfg.get("max_lines", 4):]
        else:
            if orig and trans:
                self.partial[direction] = (orig, trans)
        self._render()

    def _expire(self):
        # Під час перетягування вікна не чіпаємо рядки, щоб нічого не блимало і не зникало
        if self._drag is not None:
            return
        ttl = self.cfg.get("line_ttl_sec", 120)
        if ttl <= 0 or len(self.lines) <= 1:
            return
        now = time.time()
        keep = [l for l in self.lines if now - l[3] < ttl]
        # Завжди залишаємо щонайменше 2 останні репліки, щоб вікно ніколи не пустіло само по собі
        if len(keep) < 2:
            keep = self.lines[-2:]
        if len(keep) != len(self.lines):
            self.lines = keep
            self._render()

    def _render(self):
        fs = self.cfg["font_size"]

        def block(d, o, t, live=False):
            color = "#ffffff" if d == "in" else "#68ffa8"
            tag = "EN → UK" if d == "in" else "UK → EN"
            op = "0.75" if live else "1.0"
            live_badge = "<span style='color:#f59e0b; font-size:11px; font-weight:600;'> ⚡ мовлення…</span>" if live else ""
            return (
                f"<div style='margin-bottom:8px; opacity:{op}; line-height:1.25;'>"
                f"<span style='color:#94a3b8; font-size:{max(11, fs - 7)}pt;'>{tag} · {_esc(o)}</span>{live_badge}<br>"
                f"<span style='color:{color}; font-size:{fs}pt; font-weight:500;'>{_esc(t)}</span>"
                f"</div>"
            )

        html = "".join(block(*l[:3]) for l in self.lines)
        html += "".join(block(d, o, t, True) for d, (o, t) in self.partial.items() if o and t)
        self.text.setText(
            html or f"<span style='color:#64748b; font-size:{max(12, fs - 5)}pt;'>Слухаю… Увімкніть відео/конференцію або говоріть у мікрофон.</span>"
        )

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag is not None and (e.buttons() & Qt.LeftButton):
            self.move(e.globalPosition().toPoint() - self._drag)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag = None
        e.accept()

    def closeEvent(self, e):
        e.accept()
        self._quit_app()


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Lane:
    """Черга одного напряму: фінальні фрази не губляться, проміжний — лише найсвіжіший."""

    def __init__(self):
        self.cv = threading.Condition()
        self.finals, self.partial = [], None

    def put(self, audio, final):
        with self.cv:
            if final:
                if len(audio) > 0:
                    self.finals.append(audio)
                else:
                    self.finals.append(np.zeros(0, np.float32))
                self.partial = None
            else:
                self.partial = audio
            self.cv.notify()

    def get(self):
        with self.cv:
            while not self.finals and self.partial is None:
                self.cv.wait()
            if self.finals:
                non_empty = [a for a in self.finals[-3:] if len(a) > 0]
                self.finals = []
                if not non_empty:
                    return np.zeros(0, np.float32), True
                return np.concatenate(non_empty), True
            audio, self.partial = self.partial, None
            return audio, False


class Pipeline:
    """Аудіо → текст → переклад → оверлей (+ озвучення). Окремий потік на кожен напрям."""

    def __init__(self, cfg, bus, ui: Overlay):
        self.cfg, self.bus, self.ui = cfg, bus, ui
        self.lanes = {"in": Lane(), "out": Lane()}
        self.typed = queue.Queue()
        self._stopped = False
        mk = lambda d: Segmenter(lambda a, f: self.lanes[d].put(a, f), cfg["min_level"],
                                 cfg["silence_sec"], cfg["max_phrase_sec"], cfg["live_partial"])
        self.cap_sys = Capture("system", mk("in"), cfg)
        self.cap_mic = Capture("mic", mk("out"), cfg)
        ui.b_sys.toggled.connect(lambda v: setattr(self.cap_sys, "enabled", v))
        ui.b_mic.toggled.connect(lambda v: setattr(self.cap_mic, "enabled", v))
        ui.on_typed = self.typed.put
        ui.on_shutdown = self.shutdown
        threading.Thread(target=self._load, daemon=True).start()

    def _check_update_bg(self):
        try:
            time.sleep(2)
            info = check_for_updates(__version__)
            if info:
                self.bus.update_info.emit(info)
        except Exception:
            pass

    def _load(self):
        try:
            from realdub.engine import NLLB, Recognizer, Speaker, Translator
            self.asr = Recognizer(self.cfg)
            gpu = "GPU" in self.asr.device
            try:
                nllb = NLLB(gpu)
                self.en_uk, self.uk_en = nllb.make("en", "uk"), nllb.make("uk", "en")
            except Exception:
                self.en_uk = Translator("en", "uk", gpu)
                self.uk_en = Translator("uk", "en", gpu)
            self.tts = Speaker(self.cfg)
            self.asr(np.zeros(16000, np.float32), "en")  # прогрів
        except StopIteration:
            self.bus.status.emit("❌ Моделі не знайдено — запустіть setup.bat")
            return
        except Exception as e:
            self.bus.status.emit(f"❌ Помилка: {e}")
            return
        self.bus.status.emit(f"✅ Готово · {self.asr.device} · офлайн")
        self.bus.ready.emit()
        if self.cfg.get("check_updates_on_start", True):
            threading.Thread(target=self._check_update_bg, daemon=True).start()
        for d in ("in", "out"):
            threading.Thread(target=self._loop, args=(d,), daemon=True).start()
        threading.Thread(target=self._typed_loop, daemon=True).start()

    def _typed_loop(self):
        while not self._stopped:
            t = self.typed.get()
            if self._stopped:
                break
            try:
                self._emit("out", t, self.uk_en(t), True)
            except Exception as e:
                self.bus.status.emit(f"⚠ {e}")

    def _loop(self, d):
        lane = self.lanes[d]
        mt = self.en_uk if d == "in" else self.uk_en
        last_partial_orig = ""
        while not self._stopped:
            audio, final = lane.get()
            if self._stopped:
                break
            if len(audio) == 0:
                if final:
                    last_partial_orig = ""
                    self.bus.line.emit(d, "", "", True)
                continue
            try:
                t0 = time.time()
                orig = self.asr(audio, "en" if d == "in" else "uk")
                if not final and lane.finals:
                    continue
                if not orig:
                    if final:
                        last_partial_orig = ""
                        self.bus.line.emit(d, "", "", True)
                    continue

                if not final:
                    if orig == last_partial_orig:
                        continue
                    last_partial_orig = orig
                    try:
                        tr = mt(orig, fast=True)
                    except TypeError:
                        tr = mt(orig)
                else:
                    last_partial_orig = ""
                    try:
                        tr = mt(orig, fast=False)
                    except TypeError:
                        tr = mt(orig)

                self._emit(d, orig, tr, final)
                if final:
                    self.bus.status.emit(f"✅ {self.asr.device} · затримка {time.time() - t0:.1f} с")
            except Exception as e:
                self.bus.status.emit(f"⚠ {e}")

    def _emit(self, d, orig, tr, final):
        self.bus.line.emit(d, orig, tr, final)
        if final and d == "out" and self.ui.b_tts.isChecked() and tr:
            threading.Thread(target=self._speak, args=(tr,), daemon=True).start()

    def _speak(self, text):
        self.cap_sys.muted = True
        try:
            self.tts.say(text, "en")
        except Exception as e:
            self.bus.status.emit(f"⚠ Озвучення: {e}")
        finally:
            self.cap_sys.muted = False

    def shutdown(self):
        """Повністю зупиняє потоки, закриває аудіо та звільняє моделі з GPU/ОЗП."""
        if self._stopped:
            return
        self._stopped = True
        for cap in (self.cap_sys, self.cap_mic):
            cap.enabled = False
            cap._stop = True
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        try:
            reset_pyaudio_instance()
        except Exception:
            pass
        for attr in ("asr", "en_uk", "uk_en", "tts"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()


def main():
    _ensure_single_instance()
    cfg = config.load()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    bus = Bus()
    ui = Overlay(cfg, bus)
    ui._render()
    pipe = Pipeline(cfg, bus, ui)
    app.aboutToQuit.connect(pipe.shutdown)
    ui.show()
    code = app.exec()
    pipe.shutdown()
    _remove_pid_file()
    time.sleep(0.1)
    os._exit(code)
