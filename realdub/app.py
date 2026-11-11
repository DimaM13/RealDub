"""Прозорий оверлей поверх усіх вікон та конвеєр живого перекладу.
Працює повністю автономно офлайн на Windows: швидке перемикання Whisper (Medium / Small), адаптивна прокрутка тексту за будь-якого розміру вікна та повне очищення ОЗП/VRAM при виході."""
import ctypes
import gc
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer, Signal, QObject
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QScrollArea,
                               QSizeGrip, QVBoxLayout, QWidget)

from realdub import __version__, config
from realdub.audio import Capture, Segmenter, reset_pyaudio_instance
from realdub.updater import apply_update, check_for_updates

STYLE = """
QWidget {
    color: #f1f5f9;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
QPushButton {
    background: rgba(255, 255, 255, 18);
    border: 1px solid rgba(255, 255, 255, 32);
    border-radius: 9px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: rgba(255, 255, 255, 34);
    border-color: rgba(255, 255, 255, 55);
}
QPushButton:checked {
    background: rgba(37, 99, 235, 225);
    border-color: #60a5fa;
    color: #ffffff;
}
QPushButton#mic:checked {
    background: rgba(5, 150, 105, 225);
    border-color: #34d399;
    color: #ffffff;
}
QPushButton#send {
    background: rgba(37, 99, 235, 190);
    border-color: #60a5fa;
    padding: 7px 14px;
    font-size: 13px;
}
QPushButton#send:hover {
    background: rgba(29, 78, 216, 230);
}
QPushButton#close {
    background: rgba(239, 68, 68, 45);
    border-color: rgba(239, 68, 68, 90);
    padding: 5px 10px;
}
QPushButton#close:hover {
    background: rgba(239, 68, 68, 210);
    color: #ffffff;
}
QComboBox {
    background: rgba(30, 41, 59, 210);
    border: 1px solid rgba(148, 163, 184, 70);
    border-radius: 9px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 600;
    color: #e2e8f0;
}
QComboBox:hover {
    border-color: #60a5fa;
    background: rgba(30, 41, 59, 240);
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox QAbstractItemView {
    background-color: #0f172a;
    color: #f1f5f9;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px;
}
QLineEdit {
    background: rgba(15, 23, 42, 195);
    border: 1px solid rgba(148, 163, 184, 65);
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 15px;
    color: #ffffff;
}
QLineEdit:focus {
    border-color: #60a5fa;
    background: rgba(15, 23, 42, 235);
}
QScrollArea {
    background: transparent;
    border: none;
}
"""

PID_FILE = config.ROOT / "realdub.pid"
_NO_WIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _ensure_single_instance():
    """Завершує попередню завислу копію RealDub, щоб не дублювати моделі у відеопам'яті та ОЗП."""
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
                time.sleep(0.25)
        except Exception:
            pass
    try:
        PID_FILE.write_text(str(my_pid), encoding="utf-8")
    except Exception:
        pass


def _remove_pid_file():
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


class Bus(QObject):
    line = Signal(str, str, str, bool)  # напрям ("in"|"out"), оригінал, переклад, чи фінальна
    status = Signal(str)
    update_info = Signal(dict)          # інформація про нову версію з GitHub
    ready = Signal()                    # сигнал готовності моделей
    restart_app = Signal()              # безпечний перезапуск із головного потоку


class Overlay(QWidget):
    def __init__(self, cfg, bus: Bus):
        super().__init__()
        self.cfg, self.bus = cfg, bus
        self.lines = []      # [(напрям, оригінал, переклад, час)]
        self.partial = {}    # напрям -> (оригінал, переклад)

        self.setWindowTitle(f"RealDub v{__version__}")
        if config.ICON_ICO.exists():
            self.setWindowIcon(QIcon(str(config.ICON_ICO)))

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(STYLE)

        scr = QGuiApplication.primaryScreen().availableGeometry()
        w, h = min(980, int(scr.width() * 0.66)), 280
        self.setGeometry(scr.center().x() - w // 2, scr.bottom() - h - 48, w, h)
        self.setMinimumSize(480, 145)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(6)

        # Верхня панель керування
        top = QHBoxLayout()
        top.setSpacing(6)

        if config.ICON_PNG.exists():
            logo = QLabel()
            pix = QPixmap(str(config.ICON_PNG)).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(pix)
            logo.setToolTip(f"RealDub v{__version__} — автономний офлайн перекладач")
            logo.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            top.addWidget(logo)
            top.addSpacing(2)

        self.b_sys = self._btn("🎧 Звук ПК EN→UK", True)
        self.b_sys.setToolTip("Перекладати звук із комп'ютера (відео, Zoom, Meet, ігри) з англійської на українську")

        self.b_mic = self._btn("🎤 Мікрофон UK→EN", True)
        self.b_mic.setObjectName("mic")
        self.b_mic.setToolTip("Перекладати ваш голос із мікрофона з української на англійську")

        self.b_tts = self._btn("🔊 Озвучка", True)
        self.b_tts.setToolTip("Озвучувати перекладений англійський текст голосом")

        # Вибір моделі Whisper (лише Medium та Small)
        self.cmb_model = QComboBox()
        self.cmb_model.setCursor(Qt.PointingHandCursor)
        self.cmb_model.setToolTip("Модель розпізнавання Whisper (обидві працюють повністю офлайн)")
        cur_m = self.cfg.get("whisper_model", "medium")
        if cur_m not in config.WHISPER_MODELS:
            cur_m = "medium"
        for idx, (m_key, (m_label, _)) in enumerate(config.WHISPER_MODELS.items()):
            self.cmb_model.addItem(m_label, userData=m_key)
            if m_key == cur_m:
                self.cmb_model.setCurrentIndex(idx)

        b_font_down = self._btn("A−", False)
        b_font_down.setToolTip("Зменшити розмір шрифту")
        b_font_down.clicked.connect(lambda: self._change_font(-2))

        b_font_up = self._btn("A+", False)
        b_font_up.setToolTip("Збільшити розмір шрифту")
        b_font_up.clicked.connect(lambda: self._change_font(+2))

        self.b_upd = self._btn("🔄", False)
        self.b_upd.setToolTip("Перевірити та встановити оновлення з GitHub")
        self.b_upd.clicked.connect(self._on_update_click)

        b_copy = self._btn("📋", False)
        b_copy.setToolTip("Скопіювати останній переклад у буфер обміну")
        b_copy.clicked.connect(self._copy_last)

        b_clr = self._btn("🧹", False)
        b_clr.setToolTip("Очистити субтитри")
        b_clr.clicked.connect(self._clear)

        b_quit = self._btn("✕", False)
        b_quit.setObjectName("close")
        b_quit.setToolTip("Закрити RealDub і вивантажити з пам'яті")
        b_quit.clicked.connect(self._quit_app)

        self.st = QLabel("⏳ Завантаження моделей…")
        self.st.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        self.st.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        for w_item in (self.b_sys, self.b_mic, self.b_tts, self.cmb_model, b_font_down, b_font_up):
            top.addWidget(w_item)
        top.addSpacing(4)
        top.addWidget(self.st, 1)
        for w_item in (b_copy, self.b_upd, b_clr, b_quit):
            top.addWidget(w_item)
        lay.addLayout(top)

        # Область субтитрів з автопрокруткою вниз (працює ідеально навіть у маленькому вікні)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.scroll.viewport().setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.text.setStyleSheet("background: transparent;")
        self.text.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.scroll.setWidget(self.text)
        lay.addWidget(self.scroll, 1)

        # Нижня панель: поле ручного вводу слова або речення + кнопка перекладу + зміна розміру
        row = QHBoxLayout()
        row.setSpacing(6)
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("✍️ Введіть слово або фразу (укр ↔ англ) і натисніть Enter…")
        self.inp.returnPressed.connect(self._typed)

        b_send = QPushButton("Перекласти ➤")
        b_send.setObjectName("send")
        b_send.setCursor(Qt.PointingHandCursor)
        b_send.clicked.connect(self._typed)

        row.addWidget(self.inp, 1)
        row.addWidget(b_send)
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
        self.on_model_change = None
        self.cmb_model.currentIndexChanged.connect(self._on_model_selected)
        self._timer = QTimer(self, interval=2000, timeout=self._expire)
        self._timer.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(11, 15, 27, int(self.cfg["opacity"] * 255)))
        p.setPen(QColor(148, 163, 184, 48))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16, 16)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def _scroll_to_bottom(self):
        vbar = self.scroll.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def _change_font(self, delta: int):
        new_fs = max(14, min(36, self.cfg.get("font_size", 21) + delta))
        self.cfg["font_size"] = new_fs
        self.cfg["orig_font_size"] = max(13, new_fs - 3)
        config.save(self.cfg)
        self._render()

    def _copy_last(self):
        if self.lines:
            _, orig, trans, _ = self.lines[-1]
            clean_tr = trans.split("  ·  ")[0].strip()
            QGuiApplication.clipboard().setText(clean_tr)
            self.st.setText(f"📋 Скопійовано: {clean_tr[:35]}")

    def _on_model_selected(self, idx: int):
        m_key = self.cmb_model.itemData(idx)
        if m_key and self.on_model_change:
            self.on_model_change(m_key)

    def _on_engine_ready(self):
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
        if self.on_shutdown:
            self.on_shutdown()
        _remove_pid_file()
        try:
            exe_launcher = config.ROOT / "RealDub.exe"
            if exe_launcher.exists():
                subprocess.Popen([str(exe_launcher)], cwd=str(config.ROOT), creationflags=_NO_WIN)
            else:
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
                self.lines = self.lines[-self.cfg.get("max_lines", 6):]
        else:
            if orig and trans:
                self.partial[direction] = (orig, trans)
        self._render()

    def _expire(self):
        if self._drag is not None:
            return
        ttl = self.cfg.get("line_ttl_sec", 120)
        if ttl <= 0 or len(self.lines) <= 2:
            return
        now = time.time()
        keep = [l for l in self.lines if now - l[3] < ttl]
        if len(keep) < 2:
            keep = self.lines[-2:]
        if len(keep) != len(self.lines):
            self.lines = keep
            self._render()

    def _render(self):
        fs = self.cfg.get("font_size", 21)
        orig_fs = self.cfg.get("orig_font_size", max(16, fs - 3))

        # Розраховуємо, скільки блоків гарантовано вміщується у поточну висоту вікна,
        # щоб нові репліки знизу НІКОЛИ не обрізалися навіть при маленькому вікні
        avail_h = max(60, self.height() - 102)
        est_block_h = max(44, int((fs + orig_fs) * 1.65))
        max_blocks = max(1, avail_h // est_block_h)
        active_partials = [(d, o, t) for d, (o, t) in self.partial.items() if o and t]
        keep_finals = max(1, max_blocks - len(active_partials))
        visible_finals = self.lines[-keep_finals:] if self.lines else []

        def block(d, o, t, live=False):
            op = "0.80" if live else "1.0"
            live_badge = "<span style='color:#fbbf24; font-size:12px; font-weight:600;'> ⚡ мовлення…</span>" if live else ""
            if d == "in":
                tag = "<span style='color:#38bdf8; font-weight:700;'>EN → UK</span>"
                return (
                    f"<div style='margin-bottom:8px; opacity:{op}; line-height:1.25;'>"
                    f"<span style='font-size:{orig_fs}pt; color:#cbd5e1; font-weight:500;'>{tag} · {_esc(o)}</span>{live_badge}<br>"
                    f"<span style='color:#ffffff; font-size:{fs}pt; font-weight:600;'>{_esc(t)}</span>"
                    f"</div>"
                )
            else:
                tag = "<span style='color:#34d399; font-weight:700;'>UK → EN</span>"
                en_fs = fs + 1
                return (
                    f"<div style='margin-bottom:8px; opacity:{op}; line-height:1.25;'>"
                    f"<span style='font-size:{max(15, orig_fs - 1)}pt; color:#94a3b8; font-weight:500;'>{tag} · {_esc(o)}</span>{live_badge}<br>"
                    f"<span style='color:#68ffa8; font-size:{en_fs}pt; font-weight:600;'>{_esc(t)}</span>"
                    f"</div>"
                )

        html = "".join(block(*l[:3]) for l in visible_finals)
        html += "".join(block(d, o, t, True) for d, o, t in active_partials)
        self.text.setText(
            html or (
                f"<span style='color:#94a3b8; font-size:{max(13, fs - 5)}pt;'>"
                f"🎧 Слухаю звук ПК… Увімкніть відео/конференцію, говоріть у мікрофон або введіть слово знизу."
                f"</span>"
            )
        )
        self._scroll_to_bottom()
        QTimer.singleShot(0, self._scroll_to_bottom)

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
        self._switching_model = False
        self._switch_lock = threading.Lock()
        mk = lambda d: Segmenter(lambda a, f: self.lanes[d].put(a, f), cfg["min_level"],
                                 cfg["silence_sec"], cfg["max_phrase_sec"], cfg["live_partial"])
        self.cap_sys = Capture("system", mk("in"), cfg)
        self.cap_mic = Capture("mic", mk("out"), cfg)
        ui.b_sys.toggled.connect(lambda v: setattr(self.cap_sys, "enabled", v))
        ui.b_mic.toggled.connect(lambda v: setattr(self.cap_mic, "enabled", v))
        ui.on_typed = self.typed.put
        ui.on_shutdown = self.shutdown
        ui.on_model_change = self.change_whisper_model
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
            from realdub.engine import NLLB, Recognizer, Speaker, Translator, WordAndTextTranslator
            self.asr = Recognizer(self.cfg, on_status=self.bus.status.emit)
            gpu = "GPU" in self.asr.device
            nllb = None
            opus_uk_en = None
            opus_en_uk = None
            try:
                nllb = NLLB(gpu)
                self.en_uk, self.uk_en = nllb.make("en", "uk"), nllb.make("uk", "en")
            except Exception:
                pass
            try:
                opus_en_uk = Translator("en", "uk", gpu)
                opus_uk_en = Translator("uk", "en", gpu)
                if not hasattr(self, "en_uk"):
                    self.en_uk, self.uk_en = opus_en_uk, opus_uk_en
            except Exception:
                pass

            self.word_tr = WordAndTextTranslator(
                nllb=nllb,
                opus_uk_en=opus_uk_en,
                opus_en_uk=opus_en_uk,
                fallback_uk_en=self.uk_en,
                fallback_en_uk=self.en_uk,
            )
            self.tts = Speaker(self.cfg)
            self.asr(np.zeros(16000, np.float32), "en", fast=True)  # прогрів
        except StopIteration:
            self.bus.status.emit("❌ Моделі не знайдено — запустіть установник")
            return
        except Exception as e:
            self.bus.status.emit(f"❌ Помилка: {e}")
            return
        self.bus.status.emit(f"✅ Whisper {self.asr.display_name} · {self.asr.device}")
        self.bus.ready.emit()
        if self.cfg.get("check_updates_on_start", True):
            threading.Thread(target=self._check_update_bg, daemon=True).start()
        for d in ("in", "out"):
            threading.Thread(target=self._loop, args=(d,), daemon=True).start()
        threading.Thread(target=self._typed_loop, daemon=True).start()

    def change_whisper_model(self, new_model: str):
        """Динамічна заміна моделі Whisper (Medium <-> Small) без перезапуску програми."""
        cur_loaded = getattr(getattr(self, "asr", None), "model_name", None)
        if self._switching_model or new_model == cur_loaded:
            return

        def worker():
            with self._switch_lock:
                self._switching_model = True
                try:
                    from realdub.engine import Recognizer
                    self.cfg["whisper_model"] = new_model
                    config.save(self.cfg)
                    disp = "Medium" if new_model == "medium" else "Small"
                    self.bus.status.emit(f"⏳ Перемикаю на Whisper {disp}…")
                    old_asr = getattr(self, "asr", None)
                    if old_asr is not None:
                        with old_asr.lock:
                            self.asr = None
                        del old_asr
                        gc.collect()
                    new_asr = Recognizer(self.cfg, on_status=self.bus.status.emit)
                    self.asr = new_asr
                    self.bus.status.emit(f"✅ Whisper {self.asr.display_name} · {self.asr.device}")
                except Exception as e:
                    self.bus.status.emit(f"⚠ Помилка зміни моделі: {e}")
                finally:
                    self._switching_model = False

        threading.Thread(target=worker, daemon=True).start()

    def _typed_loop(self):
        while not self._stopped:
            t = self.typed.get()
            if self._stopped:
                break
            try:
                d, disp, speak_txt = self.word_tr.translate(t)
                if disp:
                    self.bus.line.emit(d, t, disp, True)
                    if d == "out" and self.ui.b_tts.isChecked() and speak_txt:
                        threading.Thread(target=self._speak, args=(speak_txt,), daemon=True).start()
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
                if self._switching_model or getattr(self, "asr", None) is None:
                    time.sleep(0.1)
                    continue
                t0 = time.time()
                cur_asr = self.asr
                orig = cur_asr(audio, "en" if d == "in" else "uk", fast=not final)
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
                    if d == "out" and len(orig.split()) <= 2 and hasattr(self, "word_tr"):
                        _, tr, speak_txt = self.word_tr.translate(orig)
                        self.bus.line.emit(d, orig, tr, True)
                        if self.ui.b_tts.isChecked() and speak_txt:
                            threading.Thread(target=self._speak, args=(speak_txt,), daemon=True).start()
                        if not self._switching_model:
                            self.bus.status.emit(f"✅ Whisper {self.asr.display_name} · {time.time() - t0:.1f} с")
                        continue
                    try:
                        tr = mt(orig, fast=False)
                    except TypeError:
                        tr = mt(orig)

                self._emit(d, orig, tr, final)
                if final and not self._switching_model:
                    self.bus.status.emit(f"✅ Whisper {self.asr.display_name} · {time.time() - t0:.1f} с")
            except Exception as e:
                if not self._switching_model:
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
        for attr in ("asr", "en_uk", "uk_en", "word_tr", "tts"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RealDub.Translator.2.2")
        except Exception:
            pass
    _ensure_single_instance()
    cfg = config.load()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    if config.ICON_ICO.exists():
        app.setWindowIcon(QIcon(str(config.ICON_ICO)))
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
