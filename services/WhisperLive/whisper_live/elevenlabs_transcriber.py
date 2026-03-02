"""
ElevenLabs Scribe v2 transcriber wrapper for WhisperLive.

Drop-in replacement for RemoteTranscriber that uses ElevenLabs Scribe v2 API
instead of OpenAI-compatible endpoints (Groq, Fireworks, etc.).
Implements the same transcribe() interface returning (List[Segment], TranscriptionInfo).
"""

import os
import io
import wave
import logging
import time
from pathlib import Path
from typing import BinaryIO, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import httpx

from .transcriber import Segment, TranscriptionInfo, TranscriptionOptions, VadOptions
from .remote_transcriber import RemoteTranscriberOverloaded, normalize_language_code

logger = logging.getLogger(__name__)

# ISO-639-1 → ISO-639-3 mapping (ElevenLabs uses ISO-639-3)
ISO639_1_TO_3 = {
    "sk": "slk", "cs": "ces", "en": "eng", "de": "deu", "fr": "fra",
    "es": "spa", "it": "ita", "pt": "por", "pl": "pol", "hu": "hun",
    "nl": "nld", "ru": "rus", "uk": "ukr", "ro": "ron", "bg": "bul",
    "hr": "hrv", "sr": "srp", "sl": "slv", "da": "dan", "sv": "swe",
    "no": "nor", "fi": "fin", "el": "ell", "tr": "tur", "ar": "ara",
    "he": "heb", "ja": "jpn", "ko": "kor", "zh": "zho", "hi": "hin",
    "th": "tha", "vi": "vie", "id": "ind", "ms": "msa", "tl": "tgl",
}


def _to_iso639_3(code: Optional[str]) -> Optional[str]:
    """Convert ISO-639-1 code to ISO-639-3 for ElevenLabs API."""
    if not code:
        return None
    code = code.lower().strip()
    # Already 3 letters? Pass through.
    if len(code) == 3:
        return code
    return ISO639_1_TO_3.get(code, code)


class ElevenLabsTranscriber:
    """
    ElevenLabs Scribe v2 transcriber matching the WhisperModel.transcribe() interface.

    Uses direct httpx calls to the ElevenLabs Speech-to-Text API.
    No additional dependencies beyond httpx (already used by WhisperLive).
    """

    API_URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        sampling_rate: int = 16000,
        **kwargs,  # Accept and ignore extra kwargs for compatibility
    ):
        self.api_key = (api_key or os.getenv("ELEVENLABS_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable."
            )

        self.model = model or os.getenv("ELEVENLABS_MODEL", "scribe_v2")
        self.sampling_rate = sampling_rate
        self.default_prompt = os.getenv("REMOTE_TRANSCRIBER_PROMPT")

        # Retry configuration
        self.max_retries = 3
        self.initial_retry_delay = 1.0
        self.max_retry_delay = 10.0

        # HTTP client with connection pooling
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(120.0),  # Scribe can be slower on long audio
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            http2=False,
        )

        # Load hallucination patterns (reuse same files as RemoteTranscriber)
        self._hallucinations: Set[str] = self._load_hallucination_patterns()

        api_key_masked = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "***"
        logger.info(f"ElevenLabsTranscriber initialized: model={self.model}, key={api_key_masked}")

    @staticmethod
    def _load_hallucination_patterns() -> Set[str]:
        """Load hallucination strings from hallucinations/ directory."""
        patterns: Set[str] = set()
        search_dirs = [
            Path("/app/hallucinations"),
            Path(__file__).parent.parent / "hallucinations",
        ]
        for d in search_dirs:
            if not d.is_dir():
                continue
            for txt_file in d.glob("*.txt"):
                try:
                    for line in txt_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.add(line.lower())
                except Exception:
                    pass
        if patterns:
            logger.info(f"ElevenLabsTranscriber: loaded {len(patterns)} hallucination patterns")
        return patterns

    def _is_hallucination(self, text: str) -> bool:
        if not text or not self._hallucinations:
            return False
        return text.strip().lower() in self._hallucinations

    def _numpy_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV file bytes in memory."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sampling_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return wav_buffer.getvalue()

    def _call_api(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> dict:
        """Call ElevenLabs Scribe API with retry logic."""
        retry_count = 0
        last_exception = None

        headers = {"xi-api-key": self.api_key}

        data = {"model_id": self.model}
        if language:
            iso3 = _to_iso639_3(language)
            if iso3:
                data["language_code"] = iso3

        while retry_count <= self.max_retries:
            try:
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}

                response = self.http_client.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                    data=data,
                )

                if response.status_code in (429, 503):
                    retry_after_raw = response.headers.get("Retry-After", "1")
                    try:
                        retry_after = float(retry_after_raw)
                    except Exception:
                        retry_after = 1.0
                    raise RemoteTranscriberOverloaded(
                        status_code=response.status_code,
                        retry_after_s=retry_after,
                        detail=response.text[:500] if response.text else "",
                    )

                response.raise_for_status()
                return response.json()

            except RemoteTranscriberOverloaded:
                raise
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code in (429, 503):
                    retry_after_raw = e.response.headers.get("Retry-After", "1")
                    try:
                        retry_after = float(retry_after_raw)
                    except Exception:
                        retry_after = 1.0
                    raise RemoteTranscriberOverloaded(
                        status_code=e.response.status_code,
                        retry_after_s=retry_after,
                        detail=e.response.text[:500] if e.response.text else "",
                    )
                last_exception = e
                retry_count += 1
                if retry_count <= self.max_retries:
                    delay = min(self.initial_retry_delay * (2 ** (retry_count - 1)), self.max_retry_delay)
                    logger.warning(f"ElevenLabs API call failed (attempt {retry_count}/{self.max_retries}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"ElevenLabs API call failed after {self.max_retries} retries: {e}")
                    raise
            except Exception as e:
                last_exception = e
                retry_count += 1
                if retry_count <= self.max_retries:
                    delay = min(self.initial_retry_delay * (2 ** (retry_count - 1)), self.max_retry_delay)
                    logger.warning(f"ElevenLabs API call failed (attempt {retry_count}/{self.max_retries}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"ElevenLabs API call failed after {self.max_retries} retries: {e}")
                    raise

        raise last_exception or RuntimeError("ElevenLabs API call failed")

    def _response_to_segments(self, api_response: dict, segment_id_start: int = 0) -> List[Segment]:
        """Convert ElevenLabs word-level response to Segment objects.

        Groups words into sentence-like segments split on punctuation or time gaps.
        """
        words = [w for w in api_response.get("words", []) if w.get("type") == "word"]
        if not words:
            # Fallback: use plain text if no words
            text = api_response.get("text", "").strip()
            if text and not self._is_hallucination(text):
                return [Segment(
                    id=segment_id_start, seek=0, start=0.0,
                    end=len(text) * 0.05,
                    text=text, tokens=[], avg_logprob=-0.3,
                    compression_ratio=1.0, no_speech_prob=0.0,
                    words=None, temperature=0.0,
                )]
            return []

        segments = []
        current_words = []
        current_start = None

        for w in words:
            if current_start is None:
                current_start = w.get("start", 0.0)
            current_words.append(w)

            text_so_far = " ".join(cw.get("text", "") for cw in current_words)
            # Split on sentence-ending punctuation (with min 3 words) or after 25 words
            is_sentence_end = text_so_far.rstrip().endswith((".", "?", "!", "...")) and len(current_words) >= 3
            is_long = len(current_words) >= 25
            # Time gap > 2s between consecutive words
            has_gap = (len(current_words) > 1 and
                       w.get("start", 0) - current_words[-2].get("end", 0) > 2.0)

            if is_sentence_end or is_long or has_gap:
                seg_text = " ".join(cw.get("text", "") for cw in current_words)
                if not self._is_hallucination(seg_text):
                    seg_end = current_words[-1].get("end", current_start + 0.5)
                    segments.append(Segment(
                        id=segment_id_start + len(segments),
                        seek=0,
                        start=float(current_start),
                        end=float(seg_end),
                        text=seg_text,
                        tokens=[],
                        avg_logprob=-0.3,
                        compression_ratio=1.0,
                        no_speech_prob=0.0,
                        words=None,
                        temperature=0.0,
                    ))
                current_words = []
                current_start = None

        # Flush remaining words
        if current_words:
            seg_text = " ".join(cw.get("text", "") for cw in current_words)
            if not self._is_hallucination(seg_text):
                seg_end = current_words[-1].get("end", (current_start or 0) + 0.5)
                segments.append(Segment(
                    id=segment_id_start + len(segments),
                    seek=0,
                    start=float(current_start or 0),
                    end=float(seg_end),
                    text=seg_text,
                    tokens=[],
                    avg_logprob=-0.3,
                    compression_ratio=1.0,
                    no_speech_prob=0.0,
                    words=None,
                    temperature=0.0,
                ))

        return segments

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        **kwargs,  # Accept and ignore all other WhisperModel.transcribe() params
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """
        Transcribe audio using ElevenLabs Scribe v2 API.

        Matches WhisperModel.transcribe() signature for drop-in compatibility.
        Extra parameters are accepted but ignored.
        """
        # Convert audio to numpy array
        if isinstance(audio, np.ndarray):
            audio_array = audio
        elif isinstance(audio, str):
            try:
                import soundfile as sf
                audio_array, sr = sf.read(audio)
                if sr != self.sampling_rate:
                    try:
                        from scipy import signal
                        audio_array = signal.resample(audio_array, int(len(audio_array) * self.sampling_rate / sr))
                    except ImportError:
                        logger.warning("scipy not available for resampling.")
            except ImportError:
                logger.error("soundfile not available. Cannot read audio file.")
                raise
        else:
            try:
                import soundfile as sf
                audio_array, sr = sf.read(audio)
                if sr != self.sampling_rate:
                    try:
                        from scipy import signal
                        audio_array = signal.resample(audio_array, int(len(audio_array) * self.sampling_rate / sr))
                    except ImportError:
                        logger.warning("scipy not available for resampling.")
            except ImportError:
                logger.error("soundfile not available. Cannot read audio file.")
                raise

        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Normalize language code (ISO-639-1 input from WhisperLive)
        normalized_language = normalize_language_code(language)

        # Convert to WAV bytes
        audio_wav_bytes = self._numpy_to_wav_bytes(audio_array)

        # Call ElevenLabs API
        api_response = self._call_api(
            audio_bytes=audio_wav_bytes,
            language=normalized_language,
        )

        # Convert to segments
        segments = self._response_to_segments(api_response)

        # Extract language info
        api_lang = api_response.get("language_code", "")
        language_probability = api_response.get("language_probability", 1.0)
        detected_language = normalized_language or api_lang or "sk"

        # Duration
        duration = len(audio_array) / self.sampling_rate

        info = TranscriptionInfo(
            language=detected_language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration,
            all_language_probs=None,
            transcription_options=TranscriptionOptions(
                beam_size=1, best_of=5, patience=1, length_penalty=1,
                repetition_penalty=1, no_repeat_ngram_size=0,
                log_prob_threshold=-1.0, no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,
                prompt_reset_on_temperature=0.5,
                temperatures=[0.0],
                initial_prompt=initial_prompt,
                prefix=None, suppress_blank=True, suppress_tokens=[-1],
                without_timestamps=False, max_initial_timestamp=1.0,
                word_timestamps=False,
                prepend_punctuations="\"'\"¿([{-",
                append_punctuations="\"'.。,，!！?？:：\")]}、",
                multilingual=False, max_new_tokens=None,
                clip_timestamps="0",
                hallucination_silence_threshold=None, hotwords=None,
            ),
            vad_options=VadOptions(),
        )

        return segments, info

    def __del__(self):
        if hasattr(self, 'http_client'):
            try:
                self.http_client.close()
            except Exception:
                pass
