"""ШІ-рушій: Faster-Whisper (Medium / Small), NLLB-200, Opus-MT (для слів), Piper TTS.
Працює повністю автономно офлайн на Windows (NVIDIA CUDA або процесор)."""
import io
import os
import re
import sys
import threading
import wave
from pathlib import Path

import numpy as np

from realdub import config

# Безпечне підключення бібліотек CUDA на Windows
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    try:
        import glob
        import site
        for _sp in site.getsitepackages():
            for _d in glob.glob(os.path.join(_sp, "nvidia", "*", "bin")):
                try:
                    os.add_dll_directory(_d)
                    os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]
                except Exception:
                    pass
    except Exception:
        pass

JUNK = {
    "", "you", "thank you", "thanks for watching", "дякую", "дякую за перегляд",
    "субтитри зроблені спільнотою amara.org", "bye", "бувайте",
    "ah", "oh", "uh", "um", "hmm", "mm", "eh", "ha", "ooh", "aah", "huh", "whoa", "yo",
    "а", "о", "е", "у", "м", "угу", "ага", "ем", "хм",
    "продовження буде", "далі буде", "підписуйтесь на канал",
}

QUESTION_STARTERS = {
    "en": ("what", "who", "where", "when", "why", "how", "can", "could", "would", "should", "is", "are", "do", "does", "did", "will", "have", "has"),
    "uk": ("чи", "як", "де", "коли", "чому", "хто", "що", "який", "яка", "яке", "які", "куди", "звідки", "навіщо"),
}

_RUS_FIXES = {
    r"\bнастройки\b": "налаштування",
    r"\bнастройка\b": "налаштування",
    r"\bсообщение\b": "повідомлення",
    r"\bсообщения\b": "повідомлення",
    r"\bприложение\b": "застосунок",
    r"\bсейчас\b": "зараз",
    r"\bспасибо\b": "дякую",
    r"\bпожалуйста\b": "будь ласка",
    r"\bвообще\b": "взагалі",
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
    for n in range(2, min(12, len(words) // 2 + 1)):
        for i in range(len(words) - 2 * n + 1):
            pattern = words[i:i + n]
            if words[i + n:i + 2 * n] == pattern:
                return " ".join(words[:i + n])
    return text


def clean_text(text: str) -> str:
    """Виправляє пунктуацію, апострофи, прибирає галюцинації та повтори."""
    if not text:
        return ""
    text = remove_repetitions(text)
    for pat, repl in _RUS_FIXES.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.:;!?…])", r"\1", text)
    text = re.sub(r"(\w)\s*[''`’]\s*(\w)", r"\1'\2", text)
    text = re.sub(r'([«"“(\[])\s+', r"\1", text)
    text = re.sub(r'\s+([»"”)\]])', r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _has_cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def ensure_whisper_model(model_name: str, on_status=None) -> Path:
    """Перевіряє локальну модель Whisper (medium або small)."""
    if model_name not in ("medium", "small"):
        model_name = "medium"
    out = config.MODELS / f"whisper-{model_name}"
    if (out / "model.bin").exists():
        return out

    # Якщо раптом моделі ще немає на диску і є інтернет — довантажуємо
    try:
        if on_status:
            on_status(f"📥 Завантажую модель Whisper {model_name.capitalize()}…")
        from faster_whisper import download_model
        out.mkdir(parents=True, exist_ok=True)
        download_model(model_name, output_dir=str(out))
        if (out / "model.bin").exists():
            return out
    except Exception:
        pass

    # Автономний офлайн-резерв: беремо наявну локальну модель
    for fallback in ("medium", "small"):
        fb_dir = config.MODELS / f"whisper-{fallback}"
        if (fb_dir / "model.bin").exists():
            return fb_dir
    return out


class Recognizer:
    """Офлайн розпізнавання мовлення через faster-whisper (NVIDIA GPU або процесор)."""

    def __init__(self, cfg, on_status=None):
        from faster_whisper import WhisperModel
        req_name = cfg.get("whisper_model", "medium")
        path = ensure_whisper_model(req_name, on_status=on_status)
        self.model_name = path.name.replace("whisper-", "")
        self.display_name = "Medium" if self.model_name == "medium" else "Small"
        want_cuda = cfg["device"] == "cuda" or (cfg["device"] == "auto" and _has_cuda())

        if want_cuda:
            try:
                self.model = WhisperModel(str(path), device="cuda", compute_type="int8_float16", local_files_only=True)
                self.device = "NVIDIA GPU"
            except Exception:
                self.model = WhisperModel(str(path), device="cpu", compute_type="int8", local_files_only=True)
                self.device = "процесор"
        else:
            self.model = WhisperModel(str(path), device="cpu", compute_type="int8", local_files_only=True)
            self.device = "процесор"

        self.lock = threading.Lock()

    def __call__(self, audio: np.ndarray, lang: str, fast: bool = False) -> str:
        audio = prepare_audio(audio)
        beam = 1 if fast else 3
        with self.lock:
            segs, _ = self.model.transcribe(
                audio,
                language=lang,
                beam_size=beam,
                temperature=0.0,
                initial_prompt=None,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=240, speech_pad_ms=90),
                compression_ratio_threshold=2.2,
                no_speech_threshold=0.6,
            )
            text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()
        text = clean_text(text)
        norm = text.lower().strip(" .,!?:;…\"'«»()-")
        if norm in JUNK or len(re.sub(r"[^a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]", "", norm)) < 2:
            return ""
        return text


NLLB_CODES = {"en": "eng_Latn", "uk": "ukr_Cyrl"}


class NLLB:
    """Офлайн модель NLLB-200 для живого перекладу речень (CUDA або CPU)."""

    def __init__(self, use_gpu: bool = False):
        import ctranslate2
        import sentencepiece as spm
        root = config.MODELS / "nllb-600m"
        self.sp = spm.SentencePieceProcessor(model_proto=(root / "sentencepiece.bpe.model").read_bytes())

        dev = "cuda" if use_gpu else "cpu"
        comp = "int8_float16" if use_gpu else "int8"
        try:
            self.tr = ctranslate2.Translator(str(root), device=dev, compute_type=comp)
        except Exception:
            self.tr = ctranslate2.Translator(str(root), device="cpu", compute_type="int8")

        self.lock = threading.Lock()

    def translate_raw(self, text: str, src: str, dst: str, max_len: int = 24) -> str:
        """Прямий переклад без додавання крапки (ідеально для окремих слів та коротких виразів)."""
        s, t = NLLB_CODES[src], NLLB_CODES[dst]
        toks = [[s] + self.sp.encode(text.strip(), out_type=str) + ["</s>"]]
        with self.lock:
            res = self.tr.translate_batch(
                toks,
                target_prefix=[[t]],
                beam_size=2,
                max_decoding_length=max_len,
                repetition_penalty=1.15,
            )
        decoded = self.sp.decode([p for p in res[0].hypotheses[0] if p != t])
        return clean_text(decoded)

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

            prepared = []
            max_words = 1
            for s_item in sents:
                clean_s = s_item.strip()
                w_list = clean_s.split()
                max_words = max(max_words, len(w_list))
                if len(w_list) >= 4 and clean_s[-1] not in ".!?…":
                    first_w = w_list[0].lower().strip(".,:;!?")
                    clean_s += "?" if first_w in starters else "."
                prepared.append(clean_s)

            toks = [[s] + self.sp.encode(x, out_type=str) + ["</s>"] for x in prepared]
            beam = 1 if fast else 2
            max_dec = min(180, max(16, max_words * 6))
            with self.lock:
                res = self.tr.translate_batch(
                    toks,
                    target_prefix=[[t]] * len(toks),
                    beam_size=beam,
                    max_decoding_length=max_dec,
                    repetition_penalty=1.15,
                )
            decoded = " ".join(self.sp.decode([p for p in r.hypotheses[0] if p != t]) for r in res)
            out = clean_text(decoded)
            if max_words <= 2:
                out = out.rstrip(".")
            return out

        return run


class TextTranslator:
    """Високоточний офлайн перекладач для ручного вводу тексту (HPLT EN↔UK).
    Бездоганно перекладає як окремі слова, так і повні речення повністю локально без інтернету."""

    def __init__(self, use_gpu: bool = False):
        import ctranslate2
        import sentencepiece as spm

        root_en_uk = config.MODELS / "text-en-uk"
        root_uk_en = config.MODELS / "text-uk-en"
        dev = "cuda" if use_gpu else "cpu"
        comp = "int8_float16" if use_gpu else "int8"

        spm_en = root_en_uk / "model.en-uk.spm"
        self.sp_en_uk = spm.SentencePieceProcessor(model_proto=spm_en.read_bytes())
        try:
            self.tr_en_uk = ctranslate2.Translator(str(root_en_uk), device=dev, compute_type=comp)
        except Exception:
            self.tr_en_uk = ctranslate2.Translator(str(root_en_uk), device="cpu", compute_type="int8")

        spm_uk = root_uk_en / "model.uk-en.spm"
        self.sp_uk_en = spm.SentencePieceProcessor(model_proto=spm_uk.read_bytes())
        try:
            self.tr_uk_en = ctranslate2.Translator(str(root_uk_en), device=dev, compute_type=comp)
        except Exception:
            self.tr_uk_en = ctranslate2.Translator(str(root_uk_en), device="cpu", compute_type="int8")

        self.lock = threading.Lock()

    def translate(self, text: str) -> tuple[str, str, str]:
        """Повертає (direction: 'out'|'text_in', display_text: str, speak_text: str)."""
        text = text.strip()
        if not text:
            return "out", "", ""

        has_cyr = any("а" <= c.lower() <= "я" or c.lower() in "іїєґ" for c in text)
        has_lat = any("a" <= c.lower() <= "z" for c in text)
        is_en_input = has_lat and not has_cyr
        words = text.split()

        with self.lock:
            if is_en_input:
                toks = self.sp_en_uk.encode(text, out_type=str)
                res = self.tr_en_uk.translate_batch([toks], beam_size=3)
                out = self.sp_en_uk.decode(res[0].hypotheses[0]).strip()
                out = clean_text(out)
                if len(words) == 1 and text[0].islower() and out:
                    out = out[0].lower() + out[1:]
                return "text_in", out, out
            else:
                clean_w = text.strip(" .,!?:;\"'«»()").lower()
                if len(words) == 1 and clean_w == "налаштування":
                    out = "Settings" if text[0].isupper() else "settings"
                    return "out", out, out
                toks = self.sp_uk_en.encode(text, out_type=str)
                res = self.tr_uk_en.translate_batch([toks], beam_size=3)
                out = self.sp_uk_en.decode(res[0].hypotheses[0]).strip()
                if len(words) == 1:
                    out = re.sub(r"^(the|a|an)\s+", "", out, flags=re.I).strip()
                    if text[0].islower() and out:
                        out = out[0].lower() + out[1:]
                    elif text[0].isupper() and out:
                        out = out[0].upper() + out[1:]
                else:
                    out = clean_text(out)
                return "out", out, out


# Сумісність для попередніх викликів
WordAndTextTranslator = TextTranslator
Translator = TextTranslator


def _espeak_kwargs() -> dict:
    """Налаштування espeak-ng для захисту від не-ASCII шляхів на Windows."""
    try:
        from piper.phonemize_espeak import ESPEAK_DATA_DIR
        src = Path(ESPEAK_DATA_DIR)
        if sys.platform != "win32" or str(src).isascii():
            if src.exists():
                return {"espeak_data_dir": str(src)}
            return {}

        import shutil
        base = Path(os.environ.get("LOCALAPPDATA", "C:/ProgramData"))
        if not str(base).isascii():
            base = Path("C:/ProgramData")
        dst = base / "RealDub" / "espeak-ng-data"
        if not (dst / "phontab").exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return {"espeak_data_dir": str(dst)}
    except Exception:
        return {}


class Speaker:
    """Офлайн синтез мовлення через Piper."""

    def __init__(self, cfg):
        self.cfg, self.voices, self.lock = cfg, {}, threading.Lock()

    def _voice(self, lang):
        if lang not in self.voices:
            from piper import PiperVoice
            name = config.VOICES[lang].rsplit("/", 1)[-1]
            self.voices[lang] = PiperVoice.load(
                str(config.MODELS / "voices" / f"{name}.onnx"),
                **_espeak_kwargs()
            )
        return self.voices[lang]

    def _synth(self, v, text):
        try:
            chunks = list(v.synthesize(text))
            return np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate
        except Exception:
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
