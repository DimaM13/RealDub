"""ШІ-рушій для macOS: Faster-Whisper, NLLB-200, Piper TTS.
Оптимізовано для процесорів Apple Silicon (M1/M2/M3/M4) та Intel Mac."""
import io
import os
import re
import threading
import wave
from pathlib import Path

import numpy as np

from realdub import config

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
    """Виявляє та видаляє циклічні повторення."""
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
    text = re.sub(r"\s+([,.:;!?…])", r"\1", text)
    text = re.sub(r"(\w)\s*[''`’]\s*(\w)", r"\1'\2", text)
    text = re.sub(r'([«"“(\[])\s+', r"\1", text)
    text = re.sub(r'\s+([»"”)\]])', r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class Recognizer:
    """Розпізнавання мовлення через faster-whisper (CPU int8 на Apple Silicon)."""

    def __init__(self, cfg):
        from faster_whisper import WhisperModel
        path = config.MODELS / f"whisper-{cfg['whisper_model']}"
        # На macOS Apple Silicon 'cpu' з compute_type='int8' працює надзвичайно швидко
        self.model = WhisperModel(str(path), device="cpu", compute_type="int8")
        self.device = "Apple Silicon CPU"
        self.lock = threading.Lock()

    def __call__(self, audio: np.ndarray, lang: str) -> str:
        audio = prepare_audio(audio)
        with self.lock:
            segs, _ = self.model.transcribe(
                audio,
                language=lang,
                beam_size=1,
                temperature=0.0,
                initial_prompt=None,
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
    """Одна модель NLLB-200 для обох напрямків (Apple Silicon CPU int8)."""

    def __init__(self, _unused_gpu: bool = False):
        import ctranslate2
        import sentencepiece as spm
        root = config.MODELS / "nllb-600m"
        self.sp = spm.SentencePieceProcessor(model_proto=(root / "sentencepiece.bpe.model").read_bytes())
        self.tr = ctranslate2.Translator(str(root), device="cpu", compute_type="int8")
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
    """Запасний легкий перекладач Opus-MT / Argos."""

    def __init__(self, src: str, dst: str, _unused_gpu: bool = False):
        import ctranslate2
        root = config.MODELS / f"mt-{src}-{dst}"
        model_dir = next(root.rglob("model.bin")).parent
        sp = next(root.rglob("sentencepiece.model"), None)
        if sp:
            import sentencepiece as spm
            proc = spm.SentencePieceProcessor(model_proto=sp.read_bytes())
            self.enc = lambda s: proc.encode(s, out_type=str)
            self.dec = lambda t: proc.decode(t).replace("▁", " ")
        else:
            from sacremoses import MosesDetokenizer, MosesTokenizer
            from subword_nmt.apply_bpe import BPE
            mt, md = MosesTokenizer(lang=src), MosesDetokenizer(lang=dst)
            with open(next(root.rglob("bpe.model")), encoding="utf-8") as f:
                bpe = BPE(f)
            self.enc = lambda s: bpe.process_line(
                mt.tokenize(s, aggressive_dash_splits=True, return_str=True)).split()
            self.dec = lambda t: md.detokenize(" ".join(t).replace("@@ ", "").split())
        self.tr = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8")
        self.lock = threading.Lock()

    def __call__(self, text: str, fast: bool = False) -> str:
        sents = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
        if not sents:
            return ""
        toks = [self.enc(s) for s in sents]
        with self.lock:
            res = self.tr.translate_batch(toks, beam_size=1 if fast else 2, max_decoding_length=180)
        return clean_text(" ".join(self.dec(r.hypotheses[0]) for r in res))


def _espeak_kwargs() -> dict:
    """Налаштування даних espeak для macOS."""
    try:
        from piper.phonemize_espeak import ESPEAK_DATA_DIR
        src = Path(ESPEAK_DATA_DIR)
        if src.exists() and (src / "phontab").exists():
            return {"espeak_data_dir": str(src)}
    except Exception:
        pass
    return {}


class Speaker:
    """Офлайн синтез мовлення через Piper для macOS."""

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

    def say(self, text: str, lang: str):
        import sounddevice as sd
        with self.lock:
            audio, sr = self._synth(self._voice(lang), text)
            # Вивід звуку через CoreAudio
            sd.play(audio, sr)
            sd.wait()
