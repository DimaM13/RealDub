"""Головне вікно-оверлей RealDub."""
import queue
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSizeGrip, QVBoxLayout, QWidget)

from realdub import __version__, config
from realdub.audio import Capture, Segmenter

STYLE = """
#panel { background: rgba(14,16,24,%d); border-radius: 14px; border: 1px solid rgba(255,255,255,30); }
QLabel { color: #f2f4f8; }
QPushButton { color:#cfd6e4; background: rgba(255,255,255,18); border:none; border-radius:8px;
              padding:5px 10px; font-size:13px; }
QPushButton:hover { background: rgba(255,255,255,40); }
QPushButton:checked { background:#3d7cff; color:white; }
QLineEdit { color:white; background: rgba(255,255,255,20); border:none; border-radius:8px;
            padding:6px 10px; font-size:14px; }
"""


class Bus(QObject):
    line = Signal(str, str, str, bool)  # напрям, оригінал, переклад, фінальний
    status = Signal(str)


class Overlay(QWidget):
    def __init__(self, cfg, bus: Bus):
        super().__init__()
        self.cfg, self.bus = cfg, bus
        self.lines, self.partial = [], {}
        self.setWindowTitle(f"RealDub v{__version__}")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(STYLE % int(cfg["opacity"] * 255))
        self.resize(820, 300)
        scr = QApplication.primaryScreen().availableGeometry()
        self.move(scr.center().x() - 410, scr.bottom() - 340)

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
        clr = self._btn("🧹", False)
        clr.setToolTip("Очистити")
        clr.clicked.connect(self._clear)
        close = self._btn("✕", False)
        close.setToolTip("Закрити")
        close.clicked.connect(QApplication.quit)
        bar.addWidget(clr)
        bar.addWidget(close)
        lay.addLayout(bar)

        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        self.text.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.text.setFont(QFont("Segoe UI", cfg["font_size"]))
        lay.addWidget(self.text, 1)

        row = QHBoxLayout()
        self.inp = QLineEdit(placeholderText="Напишіть українською — перекладу англійською й озвучу…")
        self.inp.returnPressed.connect(self._typed)
        row.addWidget(self.inp, 1)
        row.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        lay.addLayout(row)

        bus.line.connect(self._on_line)
        bus.status.connect(self.st.setText)
        self._drag = None
        self.on_typed = None
        self._timer = QTimer(self, interval=1000, timeout=self._expire)
        self._timer.start()

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
            self.lines.append((direction, orig, trans, time.time()))
            self.lines = self.lines[-self.cfg["max_lines"]:]
        else:
            self.partial[direction] = (orig, trans)
        self._render()

    def _expire(self):
        ttl = self.cfg["line_ttl_sec"]
        keep = [l for l in self.lines if time.time() - l[3] < ttl]
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
        html += "".join(block(d, o, t, True) for d, (o, t) in self.partial.items())
        self.text.setText(
            html or f"<span style='color:#64748b; font-size:{max(12, fs - 5)}pt;'>Очікую на мовлення… Говоріть у мікрофон або увімкніть звук ПК.</span>"
        )

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None


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
                self.finals.append(audio)
                self.partial = None  # застарів
            else:
                self.partial = audio
            self.cv.notify()

    def get(self):
        with self.cv:
            while not self.finals and self.partial is None:
                self.cv.wait()
            if self.finals:
                # якщо накопичилось кілька фраз — обробляємо разом (наздоганяємо)
                audio = np.concatenate(self.finals[-3:])
                self.finals = []
                return audio, True
            audio, self.partial = self.partial, None
            return audio, False


class Pipeline:
    """Аудіо → текст → переклад → оверлей (+ озвучення). Окремий потік на кожен напрям."""

    def __init__(self, cfg, bus, ui: Overlay):
        self.cfg, self.bus, self.ui = cfg, bus, ui
        self.lanes = {"in": Lane(), "out": Lane()}
        self.typed = queue.Queue()
        mk = lambda d: Segmenter(lambda a, f: self.lanes[d].put(a, f), cfg["min_level"],
                                 cfg["silence_sec"], cfg["max_phrase_sec"], cfg["live_partial"])
        self.cap_sys = Capture("system", mk("in"))
        self.cap_mic = Capture("mic", mk("out"))
        ui.b_sys.toggled.connect(lambda v: setattr(self.cap_sys, "enabled", v))
        ui.b_mic.toggled.connect(lambda v: setattr(self.cap_mic, "enabled", v))
        ui.on_typed = self.typed.put
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from realdub.engine import NLLB, Recognizer, Speaker, Translator
            self.asr = Recognizer(self.cfg)
            gpu = self.asr.device == "відеокарта"
            try:
                nllb = NLLB(gpu)
                self.en_uk, self.uk_en = nllb.make("en", "uk"), nllb.make("uk", "en")
            except Exception:  # запасний варіант — легкі моделі Argos
                self.en_uk = Translator("en", "uk", gpu)
                self.uk_en = Translator("uk", "en", gpu)
            self.tts = Speaker(self.cfg)
            self.asr(np.zeros(16000, np.float32), "en")  # прогрів
        except StopIteration:
            self.bus.status.emit("❌ Моделі не знайдено — запустіть install.bat")
            return
        except Exception as e:
            self.bus.status.emit(f"❌ Помилка: {e}")
            return
        self.bus.status.emit(f"✅ Готово · {self.asr.device} · офлайн")
        for d in ("in", "out"):
            threading.Thread(target=self._loop, args=(d,), daemon=True).start()
        threading.Thread(target=self._typed_loop, daemon=True).start()

    def _typed_loop(self):
        while True:
            t = self.typed.get()
            self._emit("out", t, self.uk_en(t), True)

    def _loop(self, d):
        lane = self.lanes[d]
        mt = self.en_uk if d == "in" else self.uk_en
        last_partial_orig = ""
        while True:
            audio, final = lane.get()
            try:
                t0 = time.time()
                orig = self.asr(audio, "en" if d == "in" else "uk")
                if not final and lane.finals:
                    continue  # поки розпізнавали, фраза вже завершилась, обробляємо фінальну
                if not orig:
                    if final:
                        last_partial_orig = ""
                    continue

                if not final:
                    if orig == last_partial_orig:
                        continue  # текст не змінився, уникаємо зайвого навантаження GPU
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
        self.cap_sys.muted = True  # щоб не перекладати власне озвучення
        try:
            self.tts.say(text, "en")
        except Exception as e:
            self.bus.status.emit(f"⚠ Озвучення: {e}")
        finally:
            self.cap_sys.muted = False


def main():
    cfg = config.load()
    app = QApplication(sys.argv)
    bus = Bus()
    ui = Overlay(cfg, bus)
    ui._render()
    Pipeline(cfg, bus, ui)
    ui.show()
    sys.exit(app.exec())
