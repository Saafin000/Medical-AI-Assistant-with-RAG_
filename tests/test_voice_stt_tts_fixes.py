"""
Targeted unit tests for Clinical Pre-Consultation Speech-to-Text (/transcribe)
and Text-to-Speech (/generate-speech) endpoints and multilingual support.
Uses httpx.ASGITransport so tests run cleanly without local socket dependencies.
"""

import asyncio
import io
import os
import sys
import struct
import wave
from unittest.mock import MagicMock, patch
import httpx

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = "gsk_test_key_voice_verification_12345"
os.environ["TESTING"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, config, SPEECH_DIR
from config import is_elevenlabs_key_id


def _make_wav_bytes(duration_ms: int = 300, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid PCM WAV buffer so pydub/ffmpeg decodes it cleanly."""
    num_frames = int(sample_rate * (duration_ms / 1000.0))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = b"".join(struct.pack("<h", 0) for _ in range(num_frames))
        wf.writeframes(frames)
    return buf.getvalue()


def _asgi_request(method: str, path: str, **kwargs) -> httpx.Response:
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


class TestTranscribeEndpoint:
    def test_transcribe_english_omits_hardcoded_language(self):
        wav_bytes = _make_wav_bytes()
        mock_result = MagicMock()
        mock_result.text = "I have a fever since yesterday evening"

        with patch("app.client.speech_to_text.convert", return_value=mock_result) as mock_convert:
            res = _asgi_request(
                "POST",
                "/transcribe",
                files={"audio": ("clinical_voice.wav", wav_bytes, "audio/wav")},
            )

        assert res.status_code == 200
        assert res.json() == {"transcript": "I have a fever since yesterday evening"}
        mock_convert.assert_called_once()
        _, kwargs = mock_convert.call_args
        assert kwargs.get("model_id") == "scribe_v1"
        assert "language_code" not in kwargs

    def test_transcribe_hindi_transcript(self):
        wav_bytes = _make_wav_bytes()
        mock_result = MagicMock()
        mock_result.text = "मुझे कल शाम से तेज बुखार और सिर दर्द है"

        with patch("app.client.speech_to_text.convert", return_value=mock_result) as mock_convert:
            res = _asgi_request(
                "POST",
                "/transcribe",
                files={"audio": ("clinical_voice.wav", wav_bytes, "audio/wav")},
            )

        assert res.status_code == 200
        assert res.json()["transcript"] == "मुझे कल शाम से तेज बुखार और सिर दर्द है"
        _, kwargs = mock_convert.call_args
        assert "language_code" not in kwargs

    def test_transcribe_hinglish_transcript(self):
        wav_bytes = _make_wav_bytes()
        mock_result = MagicMock()
        mock_result.text = "Mujhe kal se high fever aur throat pain ho raha hai"

        with patch("app.client.speech_to_text.convert", return_value=mock_result) as mock_convert:
            res = _asgi_request(
                "POST",
                "/transcribe",
                files={"audio": ("clinical_voice.wav", wav_bytes, "audio/wav")},
            )

        assert res.status_code == 200
        assert res.json()["transcript"] == "Mujhe kal se high fever aur throat pain ho raha hai"
        _, kwargs = mock_convert.call_args
        assert "language_code" not in kwargs

    def test_transcribe_empty_audio_returns_400(self):
        before = set(os.listdir(SPEECH_DIR)) if os.path.exists(SPEECH_DIR) else set()
        res = _asgi_request(
            "POST",
            "/transcribe",
            files={"audio": ("empty.webm", b"", "audio/webm")},
        )
        after = set(os.listdir(SPEECH_DIR)) if os.path.exists(SPEECH_DIR) else set()

        assert res.status_code == 400
        assert "empty" in res.json()["error"].lower()
        assert before == after

    def test_transcribe_corrupted_format_returns_400_and_cleans_temp(self):
        before = set(os.listdir(SPEECH_DIR)) if os.path.exists(SPEECH_DIR) else set()
        res = _asgi_request(
            "POST",
            "/transcribe",
            files={"audio": ("bad.webm", b"not-valid-audio-bytes", "audio/webm")},
        )
        after = set(os.listdir(SPEECH_DIR)) if os.path.exists(SPEECH_DIR) else set()

        assert res.status_code == 400
        assert "unsupported" in res.json()["error"].lower() or "corrupted" in res.json()["error"].lower()
        assert before == after

    def test_transcribe_blank_provider_transcript_returns_422(self):
        wav_bytes = _make_wav_bytes()
        mock_result = MagicMock()
        mock_result.text = "   "

        with patch("app.client.speech_to_text.convert", return_value=mock_result):
            res = _asgi_request(
                "POST",
                "/transcribe",
                files={"audio": ("silent.wav", wav_bytes, "audio/wav")},
            )

        assert res.status_code == 422
        assert "no speech detected" in res.json()["error"].lower()


class TestGenerateSpeechEndpointContract:
    def test_generate_speech_success_and_error_cases(self):
        orig_key = getattr(config.speech, "eleven_labs_api_key", None)
        config.speech.eleven_labs_api_key = "test_valid_elevenlabs_key_12345"
        try:
            # Valid MP3 response
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"ID3\x04\x00\x00\x00\x00\x00#" + (b"\xaa" * 200)
            with patch("requests.post", return_value=mock_resp):
                res = _asgi_request(
                    "POST",
                    "/generate-speech",
                    json={"text": "How long have you had the fever?", "voice_id": "EXAVITQu4vr4xnSDxMaL"},
                )
            assert res.status_code == 200
            assert "audio/mpeg" in res.headers.get("content-type", "")
            assert len(res.content) > 100

            # Empty audio response from upstream -> 502
            empty_resp = MagicMock()
            empty_resp.status_code = 200
            empty_resp.content = b""
            with patch("requests.post", return_value=empty_resp):
                res_empty = _asgi_request(
                    "POST",
                    "/generate-speech",
                    json={"text": "How long have you had the fever?", "voice_id": "EXAVITQu4vr4xnSDxMaL"},
                )
            assert res_empty.status_code == 502
            assert res_empty.json().get("error_code") == "TTS_EMPTY_AUDIO"

            # Empty input text -> 400
            res_blank = _asgi_request(
                "POST",
                "/generate-speech",
                json={"text": "   ", "voice_id": "EXAVITQu4vr4xnSDxMaL"},
            )
            assert res_blank.status_code == 400
            assert res_blank.json().get("error_code") == "EMPTY_TEXT"
        finally:
            config.speech.eleven_labs_api_key = orig_key

    def test_key_id_detection_and_http_400_auth_handling(self):
        fake_key_id = "74aef0c6" + ("0" * 56)
        assert is_elevenlabs_key_id(fake_key_id) is True
        assert is_elevenlabs_key_id("sk_1234567890abcdef") is False
        assert is_elevenlabs_key_id("test_valid_elevenlabs_key_12345") is False

        orig_key = getattr(config.speech, "eleven_labs_api_key", None)
        try:
            # 1. TTS catches 64-hex Key ID before calling upstream
            config.speech.eleven_labs_api_key = fake_key_id
            res_tts_pre = _asgi_request(
                "POST",
                "/generate-speech",
                json={"text": "Testing Key ID guard.", "voice_id": "EXAVITQu4vr4xnSDxMaL"},
            )
            assert res_tts_pre.status_code == 401
            assert res_tts_pre.json().get("error_code") == "ELEVENLABS_KEY_ID_USED"

            # 2. TTS parses upstream HTTP 400 api_key_id_used_as_api_key cleanly
            config.speech.eleven_labs_api_key = "test_valid_elevenlabs_key_12345"
            mock_400 = MagicMock()
            mock_400.status_code = 400
            mock_400.json.return_value = {
                "detail": {
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                    "status": "api_key_id_used_as_api_key",
                    "message": "API key ID used as API key",
                }
            }
            mock_400.text = '{"detail":{"status":"api_key_id_used_as_api_key"}}'
            with patch("requests.post", return_value=mock_400):
                res_tts_400 = _asgi_request(
                    "POST",
                    "/generate-speech",
                    json={"text": "Testing upstream 400 auth.", "voice_id": "EXAVITQu4vr4xnSDxMaL"},
                )
            assert res_tts_400.status_code == 401
            assert res_tts_400.json().get("error_code") == "ELEVENLABS_KEY_ID_USED"

            # 3. STT handles upstream api_key_id_used_as_api_key cleanly as 401
            wav_bytes = _make_wav_bytes()
            with patch(
                "app.client.speech_to_text.convert",
                side_effect=Exception("status_code: 400, body: {'detail': {'status': 'api_key_id_used_as_api_key'}}"),
            ):
                res_stt_400 = _asgi_request(
                    "POST",
                    "/transcribe",
                    files={"audio": ("voice.wav", wav_bytes, "audio/wav")},
                )
            assert res_stt_400.status_code == 401
            assert res_stt_400.json().get("error_code") == "ELEVENLABS_KEY_ID_USED"
        finally:
            config.speech.eleven_labs_api_key = orig_key
