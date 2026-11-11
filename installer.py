"""Графічний установник RealDub для Windows.
Розгортає програму, налаштовує оточення, завантажує моделі та створює ярлики."""
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent

STYLE = """
QWidget {
    background-color: #0e1018;
    color: #f1f5f9;
    font-family: 'Segoe UI', sans-serif;
}
QFrame#card {
    background-color: #171a26;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 18);
    padding: 16px;
}
QLabel#title {
    font-size: 20px;
    font-weight: bold;
    color: #93c5fd;
}
QLabel#subtitle {
    font-size: 13px;
    color: #94a3b8;
}
QLineEdit {
    background-color: #1e2333;
    color: #ffffff;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
}
QLineEdit:focus {
    border: 1px solid #3b82f6;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton#secondary {
    background-color: #334155;
    color: #cbd5e1;
}
QPushButton#secondary:hover {
    background-color: #475569;
}
QCheckBox {
    font-size: 13px;
    color: #cbd5e1;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #475569;
    background-color: #1e2333;
}
QCheckBox::indicator:checked {
    background-color: #3b82f6;
    border: 1px solid #3b82f6;
}
QProgressBar {
    background-color: #1e2333;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-size: 12px;
    font-weight: 600;
    height: 18px;
    border: 1px solid #334155;
}
QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 5px;
}
"""


def get_desktop_dirs() -> list[Path]:
    """Повертає реальні шляхи до Робочого столу Windows (включно з OneDrive)."""
    dirs = []
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        # CSIDL_DESKTOPDIRECTORY = 0x0010
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buf) == 0 and buf.value:
            p = Path(buf.value)
            if p.exists():
                dirs.append(p)
    except Exception:
        pass
    fallback = Path(os.environ.get("USERPROFILE", "")) / "Desktop"
    if fallback.exists() and fallback not in dirs:
        dirs.append(fallback)
    return dirs


def create_windows_shortcut(target_exe: str, args: str, shortcut_path: str, working_dir: str, description: str, icon_path: str = ""):
    """Створює Windows .lnk ярлик через PowerShell (-EncodedCommand для підтримки кирилиці)."""
    import base64
    icon_stmt = f"$s.IconLocation = '{icon_path},0'; " if icon_path else ""
    ps_cmd = (
        f"$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{shortcut_path}'); "
        f"$s.TargetPath = '{target_exe}'; "
        f"$s.Arguments = '{args}'; "
        f"$s.WorkingDirectory = '{working_dir}'; "
        f"$s.Description = '{description}'; "
        f"{icon_stmt}"
        f"$s.Save()"
    )
    encoded = base64.b64encode(ps_cmd.encode("utf-16-le")).decode("ascii")
    subprocess.run(
        ["powershell", "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )


class InstallWorker(threading.Thread):
    def __init__(self, target_dir: Path, desktop_shortcut: bool, startmenu_shortcut: bool, on_step, on_done, on_error):
        super().__init__(daemon=True)
        self.target = target_dir
        self.desktop_sc = desktop_shortcut
        self.startmenu_sc = startmenu_shortcut
        self.on_step = on_step
        self.on_done = on_done
        self.on_error = on_error

    def run(self):
        try:
            self.on_step("Створюю структуру папок…", 10)
            self.target.mkdir(parents=True, exist_ok=True)

            self.on_step("Копіюю файли програми…", 25)
            # Копіюємо каталог realdub/
            dst_realdub = self.target / "realdub"
            if dst_realdub.exists():
                shutil.rmtree(dst_realdub, ignore_errors=True)
            shutil.copytree(SOURCE_DIR / "realdub", dst_realdub)

            # Копіюємо основні файли
            for fname in ["RealDub.exe", "main.pyw", "download_models.py", "requirements.txt", "README.md", "start.bat"]:
                src_f = SOURCE_DIR / fname
                if src_f.exists():
                    shutil.copy2(src_f, self.target / fname)

            # Якщо в інсталяторі вже є завантажені моделі — копіюємо їх без повторного скачування
            src_models = SOURCE_DIR / "models"
            dst_models = self.target / "models"
            if src_models.exists() and any(src_models.iterdir()):
                self.on_step("Копіюю моделі ШІ з установника…", 40)
                shutil.copytree(src_models, dst_models, dirs_exist_ok=True)

            # Створення віртуального середовища Python
            venv_dir = self.target / ".venv"
            python_exe = sys.executable
            if not (venv_dir / "Scripts" / "python.exe").exists():
                self.on_step("Налаштовую віртуальне середовище Python…", 55)
                subprocess.run([python_exe, "-m", "venv", str(venv_dir)], check=True)

            venv_python = str(venv_dir / "Scripts" / "python.exe")
            venv_pythonw = str(venv_dir / "Scripts" / "pythonw.exe")

            self.on_step("Встановлюю бібліотеки та оптимізацію GPU…", 70)
            pip_cmd = [
                venv_python, "-m", "pip", "install", "-q",
                "-r", str(self.target / "requirements.txt"),
                "nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*"
            ]
            subprocess.run(pip_cmd, check=True)

            # Перевіряємо та дозавантажуємо моделі, якщо чогось не вистачає
            self.on_step("Перевіряю та налаштовую моделі ШІ…", 85)
            subprocess.run([venv_python, str(self.target / "download_models.py")], cwd=str(self.target), check=True)

            # Ініціалізуємо git-зв'язок для надійних наступних автооновлень
            try:
                subprocess.run(["git", "init"], cwd=str(self.target), capture_output=True)
                subprocess.run(
                    ["git", "remote", "add", "origin", "https://github.com/DimaM13/RealDub.git"],
                    cwd=str(self.target),
                    capture_output=True,
                )
            except Exception:
                pass

            # Створення деінсталятора
            uninst_file = self.target / "uninstall.bat"
            uninst_content = (
                "@echo off\n"
                "echo Uninstall RealDub? [Y/N]\n"
                "set /p choice=\n"
                'if /i "%choice%"=="Y" (\n'
                '  del /q "%USERPROFILE%\\Desktop\\RealDub.lnk" 2>nul\n'
                '  del /q "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\RealDub.lnk" 2>nul\n'
                '  cd /d "%TEMP%"\n'
                f'  rmdir /s /q "{self.target}"\n'
                '  echo RealDub uninstalled.\n'
                ")\n"
                "pause\n"
            )
            uninst_file.write_text(uninst_content, encoding="ascii", errors="ignore")

            # Створення ярликів
            self.on_step("Створюю ярлики на комп'ютері…", 95)
            icon_ico = str(self.target / "realdub" / "icon.ico")
            exe_launcher = self.target / "RealDub.exe"
            sc_target = str(exe_launcher) if exe_launcher.exists() else venv_pythonw
            sc_args = "" if exe_launcher.exists() else "main.pyw"

            if self.desktop_sc:
                for desktop in get_desktop_dirs():
                    create_windows_shortcut(
                        target_exe=sc_target,
                        args=sc_args,
                        shortcut_path=str(desktop / "RealDub.lnk"),
                        working_dir=str(self.target),
                        description="RealDub — офлайн перекладач мовлення в реальному часі",
                        icon_path=icon_ico,
                    )

            if self.startmenu_sc:
                programs = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
                if programs.exists():
                    create_windows_shortcut(
                        target_exe=sc_target,
                        args=sc_args,
                        shortcut_path=str(programs / "RealDub.lnk"),
                        working_dir=str(self.target),
                        description="RealDub — офлайн перекладач мовлення в реальному часі",
                        icon_path=icon_ico,
                    )

            self.on_step("Встановлення завершено!", 100)
            self.on_done(str(self.target))
        except Exception as e:
            self.on_error(str(e))


def run_gui():
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QFrame,
                                   QHBoxLayout, QLabel, QLineEdit,
                                   QProgressBar, QPushButton, QStackedWidget,
                                   QVBoxLayout, QWidget)

    class EventBridge(QObject):
        step = Signal(str, int)
        done = Signal(str)
        error = Signal(str)

    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Установка RealDub")
    window.resize(600, 430)
    window.setStyleSheet(STYLE)

    bridge = EventBridge()

    main_layout = QVBoxLayout(window)
    main_layout.setContentsMargins(24, 24, 24, 24)

    # Верхній заголовок
    header = QVBoxLayout()
    lbl_title = QLabel("🌐 RealDub — Офлайн перекладач", objectName="title")
    lbl_subtitle = QLabel("Майстер швидкого встановлення на комп'ютер", objectName="subtitle")
    header.addWidget(lbl_title)
    header.addWidget(lbl_subtitle)
    main_layout.addLayout(header)
    main_layout.addSpacing(16)

    stack = QStackedWidget()
    main_layout.addWidget(stack, 1)

    # --- СТОРІНКА 1: Налаштування шляху та опцій ---
    page1 = QWidget()
    p1_layout = QVBoxLayout(page1)
    p1_layout.setContentsMargins(0, 0, 0, 0)

    card = QFrame(objectName="card")
    card_lay = QVBoxLayout(card)

    card_lay.addWidget(QLabel("Папка для встановлення:"))
    row_path = QHBoxLayout()
    default_install_dir = str(Path(os.environ.get("LOCALAPPDATA", "C:")) / "RealDub")
    inp_path = QLineEdit(default_install_dir)
    btn_browse = QPushButton("Огляд…", objectName="secondary")

    def browse():
        f = QFileDialog.getExistingDirectory(window, "Виберіть папку для встановлення", inp_path.text())
        if f:
            inp_path.setText(os.path.join(f, "RealDub"))

    btn_browse.clicked.connect(browse)
    row_path.addWidget(inp_path, 1)
    row_path.addWidget(btn_browse)
    card_lay.addLayout(row_path)

    card_lay.addSpacing(12)
    card_lay.addWidget(QLabel("Додаткові опції:"))
    cb_desktop = QCheckBox("Створити ярлик на Робочому столі")
    cb_desktop.setChecked(True)
    cb_startmenu = QCheckBox("Створити ярлик у меню «Пуск»")
    cb_startmenu.setChecked(True)
    card_lay.addWidget(cb_desktop)
    card_lay.addWidget(cb_startmenu)

    p1_layout.addWidget(card)
    p1_layout.addStretch(1)

    bar1 = QHBoxLayout()
    bar1.addStretch(1)
    btn_install = QPushButton("Встановити")
    bar1.addWidget(btn_install)
    p1_layout.addLayout(bar1)

    stack.addWidget(page1)

    # --- СТОРІНКА 2: Прогрес інсталяції ---
    page2 = QWidget()
    p2_layout = QVBoxLayout(page2)
    p2_layout.setContentsMargins(0, 0, 0, 0)

    card2 = QFrame(objectName="card")
    card2_lay = QVBoxLayout(card2)
    card2_lay.addSpacing(20)
    lbl_progress = QLabel("Розпочинаю процес встановлення…")
    lbl_progress.setStyleSheet("font-size: 14px; color: #93c5fd;")
    card2_lay.addWidget(lbl_progress)
    card2_lay.addSpacing(10)
    prog_bar = QProgressBar()
    prog_bar.setRange(0, 100)
    prog_bar.setValue(0)
    card2_lay.addWidget(prog_bar)
    card2_lay.addSpacing(20)

    p2_layout.addWidget(card2)
    p2_layout.addStretch(1)
    stack.addWidget(page2)

    # --- СТОРІНКА 3: Фініш ---
    page3 = QWidget()
    p3_layout = QVBoxLayout(page3)
    p3_layout.setContentsMargins(0, 0, 0, 0)

    card3 = QFrame(objectName="card")
    card3_lay = QVBoxLayout(card3)
    lbl_done = QLabel("🎉 RealDub успішно встановлено!")
    lbl_done.setStyleSheet("font-size: 16px; font-weight: bold; color: #4ade80;")
    lbl_done_desc = QLabel("Програма готова до роботи повністю офлайн.\nВи можете запускати її через створені ярлики або кнопку нижче.")
    lbl_done_desc.setStyleSheet("color: #cbd5e1; font-size: 13px; line-height: 1.4;")
    card3_lay.addWidget(lbl_done)
    card3_lay.addSpacing(8)
    card3_lay.addWidget(lbl_done_desc)
    card3_lay.addSpacing(12)
    cb_launch = QCheckBox("Запустити RealDub зараз")
    cb_launch.setChecked(True)
    card3_lay.addWidget(cb_launch)

    p3_layout.addWidget(card3)
    p3_layout.addStretch(1)

    bar3 = QHBoxLayout()
    bar3.addStretch(1)
    btn_finish = QPushButton("Завершити")
    bar3.addWidget(btn_finish)
    p3_layout.addLayout(bar3)

    stack.addWidget(page3)

    final_start_bat = [""]

    def start_install():
        target_path = Path(inp_path.text().strip())
        stack.setCurrentIndex(1)
        worker = InstallWorker(
            target_dir=target_path,
            desktop_shortcut=cb_desktop.isChecked(),
            startmenu_shortcut=cb_startmenu.isChecked(),
            on_step=bridge.step.emit,
            on_done=bridge.done.emit,
            on_error=bridge.error.emit,
        )
        worker.start()

    def on_step(msg, val):
        lbl_progress.setText(msg)
        prog_bar.setValue(val)

    def on_done(start_bat_path):
        final_start_bat[0] = start_bat_path
        stack.setCurrentIndex(2)

    def on_error(err_msg):
        lbl_progress.setText(f"❌ Помилка встановлення: {err_msg}")
        lbl_progress.setStyleSheet("color: #ef4444; font-size: 13px;")

    def on_finish():
        if cb_launch.isChecked() and final_start_bat[0]:
            try:
                t_dir = Path(final_start_bat[0])
                pw = t_dir / ".venv" / "Scripts" / "pythonw.exe"
                if pw.exists():
                    subprocess.Popen([str(pw), str(t_dir / "main.pyw")], cwd=str(t_dir))
                else:
                    os.startfile(str(t_dir / "start.bat"))
            except Exception:
                pass
        window.close()

    btn_install.clicked.connect(start_install)
    bridge.step.connect(on_step)
    bridge.done.connect(on_done)
    bridge.error.connect(on_error)
    btn_finish.clicked.connect(on_finish)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
