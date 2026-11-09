"""Розпізнавання (faster-whisper), переклад (CTranslate2 + SentencePiece), озвучення (Piper)."""
import re
import threading
import wave
import io

import numpy as np

from realdub import config

# Бібліотеки CUDA, встановлені через pip (nvidia-cublas / nvidia-cudnn)
import os, site, glob
for _sp in site.getsitepackages():
    for _d in glob.glob(os.path.join(_sp, "nvidia", "*", "bin")):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]

JUNK = {"", "you", "thank you.", "thanks for watching!", "дякую.", "дякую за перегляд!",
        "субтитри зроблені спільнотою amara.org", "thank you", "дякую", "bye.", "bye", "бувайте."}

QUESTION_STARTERS = {
    "en": ("what", "who", "where", "when", "why", "how", "can", "could", "would", "should", "is", "are", "do", "does", "did", "will", "have", "has"),
    "uk": ("чи", "як", "де", "коли", "чому", "хто", "що", "який", "яка", "яке", "які", "куди", "звідки", "навіщо"),
}


def prepare_audio(audio: np.ndarray) -> np.ndarray:
    """Видаляє постійне зміщення без штучного посилення фонового шуму."""
    if len(audio) == 0:
        return audio
    return (audio - float(np.mean(audio))).astype(np.float32)


def remove_repetitions(text: str) -> str:
    """Виявляє та видаляє циклічні галюцинації Whisper (коли одна фраза повторюється поспіль)."""
    if not text:
        return ""
    words = text.split()
    if len(words) < 6:
        return text
    # Перевіряємо повторення n-грам від 2 до 10 слів
    for n in range(2, min(12, len(words) // 2 + 1)):
        for i in range(len(words) - 2 * n + 1):
            pattern = words[i:i + n]
            if words[i + n:i + 2 * n] == pattern:
                # Знайдено точний цикл — повертаємо лише очищену частину без зациклення
                return " ".join(words[:i + n])
    return text


def clean_text(text: str) -> str:
    """Виправляє пунктуацію, апострофи, прибирає галюцинації та повтори."""
    if not text:
        return ""
    text = remove_repetitions(text)
    # Прибираємо пробіли перед знаками пунктуації
    text = re.sub(r"\s+([,.:;!?…])", r"\1", text)
    # З'єднуємо розірвані апострофи в українських словах
    text = re.sub(r"(\w)\s*[''`’]\s*(\w)", r"\1'\2", text)
    # Прибираємо зайві пробіли всередині лапок та дужок
    text = re.sub(r'([«"“(\[])\s+', r"\1", text)
    text = re.sub(r'\s+([»"”)\]])', r"\1", text)
    # Нормалізуємо пробіли
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Recognizer:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel
        path = config.MODELS / f"whisper-{cfg['whisper_model']}"
        gpu = cfg["device"] == "cuda" or (cfg["device"] == "auto" and _cuda())
        try:
            self.model = WhisperModel(str(path), device="cuda" if gpu else "cpu",
                                      compute_type="int8_float16" if gpu else "int8")
        except Exception:
            self.model = WhisperModel(str(path), device="cpu", compute_type="int8")
            gpu = False
        self.device = "відеокарта" if gpu else "процесор"
        self.lock = threading.Lock()

    def __call__(self, audio: np.ndarray, lang: str) -> str:
        audio = prepare_audio(audio)
        with self.lock:
            segs, _ = self.model.transcribe(
                audio,
                language=lang,
                beam_size=1,
                temperature=0.0,
                initial_prompt=None,  # вимкнено для запобігання галюцинаціям
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=80),
                compression_ratio_threshold=2.2,
                no_speech_threshold=0.6,
            )
            text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()
        text = clean_text(text)
        return "" if text.lower() in JUNK else text


NLLB_CODES = {"en": "eng_Latn", "uk": "ukr_Cyrl"}


class NLLB:
    """Одна модель NLLB-200 для обох напрямів (якісніший переклад)."""

    def __init__(self, gpu: bool):
        import ctranslate2
        import sentencepiece as spm
        root = config.MODELS / "nllb-600m"
        self.sp = spm.SentencePieceProcessor(model_proto=(root / "sentencepiece.bpe.model").read_bytes())
        self.tr = ctranslate2.Translator(str(root), device="cuda" if gpu else "cpu",
                                         compute_type="int8_float16" if gpu else "int8")
        self.lock = threading.Lock()

    def make(self, src: str, dst: str):
        s, t = NLLB_CODES[src], NLLB_CODES[dst]
        starters = QUESTION_STARTERS.get(src, ())

        def run(text: str, fast: bool = False) -> str:
            text = text.strip()
            if not text:
                return ""
            sents = [x for x in re.split(r"(?<=[.!?…])\s+", text) if x]
            if not sents:
                return ""

            # Якщо речення не має розділового знаку, додаємо його для кращої синтаксичної побудови
            prepared = []
            for s_item in sents:
                clean_s = s_item.strip()
                if clean_s and clean_s[-1] not in ".!?…":
                    first_w = clean_s.split()[0].lower().strip(".,:;!?")
                    clean_s += "?" if first_w in starters else "."
                prepared.append(clean_s)

            toks = [[s] + self.sp.encode(x, out_type=str) + ["</s>"] for x in prepared]
            beam = 1 if fast else 2
            with self.lock:
                res = self.tr.translate_batch(
                    toks,
                    target_prefix=[[t]] * len(toks),
                    beam_size=beam,
                    max_decoding_length=180,
                    repetition_penalty=1.15,
                )
            decoded = " ".join(self.sp.decode([p for p in r.hypotheses[0] if p != t]) for r in res)
            return clean_text(decoded)

        return run


class Translator:
    def __init__(self, src: str, dst: str, gpu: bool):
        import ctranslate2
        root = config.MODELS / f"mt-{src}-{dst}"
        model_dir = next(root.rglob("model.bin")).parent
        sp = next(root.rglob("sentencepiece.model"), None)
        if sp:  # токенізатор SentencePiece
            import sentencepiece as spm
            proc = spm.SentencePieceProcessor(model_proto=sp.read_bytes())
            self.enc = lambda s: proc.encode(s, out_type=str)
            self.dec = lambda t: proc.decode(t).replace("▁", " ")
        else:  # Moses + BPE (subword-nmt)
            from sacremoses import MosesDetokenizer, MosesTokenizer
            from subword_nmt.apply_bpe import BPE
            mt, md = MosesTokenizer(lang=src), MosesDetokenizer(lang=dst)
            with open(next(root.rglob("bpe.model")), encoding="utf-8") as f:
                bpe = BPE(f)
            self.enc = lambda s: bpe.process_line(
                mt.tokenize(s, aggressive_dash_splits=True, return_str=True)).split()
            self.dec = lambda t: md.detokenize(" ".join(t).replace("@@ ", "").split())
        self.tr = ctranslate2.Translator(str(model_dir), device="cuda" if gpu else "cpu",
                                         compute_type="int8_float16" if gpu else "int8")
        self.lock = threading.Lock()

    def __call__(self, text: str) -> str:
        sents = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
        if not sents:
            return ""
        toks = [self.enc(s) for s in sents]
        with self.lock:
            res = self.tr.translate_batch(toks, beam_size=1, max_decoding_length=200)
        return " ".join(self.dec(r.hypotheses[0]) for r in res).strip()


def _espeak_kwargs() -> dict:
    """espeak-ng не відкриває шляхи з не-латинськими символами — копіюємо дані в латинський шлях."""
    try:
        import shutil
        from pathlib import Path
        from piper.phonemize_espeak import ESPEAK_DATA_DIR
        src = Path(ESPEAK_DATA_DIR)
        if str(src).isascii():
            return {}
        base = Path(os.environ.get("LOCALAPPDATA", "C:/ProgramData"))
        if not str(base).isascii():
            base = Path("C:/ProgramData")
        dst = base / "RealDub" / "espeak-ng-data"
        if not (dst / "phontab").exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return {"espeak_data_dir": str(dst)}
    except ImportError:
        return {}


class Speaker:
    """Офлайн озвучення голосами Piper у вибраний пристрій виводу."""

    def __init__(self, cfg):
        self.cfg, self.voices, self.lock = cfg, {}, threading.Lock()

    def _voice(self, lang):
        if lang not in self.voices:
            from piper import PiperVoice
            name = config.VOICES[lang].rsplit("/", 1)[-1]
            self.voices[lang] = PiperVoice.load(str(config.MODELS / "voices" / f"{name}.onnx"),
                                                **_espeak_kwargs())
        return self.voices[lang]

    def _synth(self, v, text):
        try:  # piper-tts >= 1.3
            chunks = list(v.synthesize(text))
            return np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate
        except (TypeError, AttributeError):  # старіші версії
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                v.synthesize(text, w)
            buf.seek(0)
            with wave.open(buf) as w:
                return np.frombuffer(w.readframes(w.getnframes()), np.int16), w.getframerate()

    def _device(self):
        import sounddevice as sd
        want = self.cfg.get("tts_output_device", "").lower()
        if want:
            for i, d in enumerate(sd.query_devices()):
                if want in d["name"].lower() and d["max_output_channels"] > 0:
                    return i
        return None

    def say(self, text: str, lang: str):
        import sounddevice as sd
        with self.lock:
            audio, sr = self._synth(self._voice(lang), text)
            sd.play(audio, sr, device=self._device())
            sd.wait()
