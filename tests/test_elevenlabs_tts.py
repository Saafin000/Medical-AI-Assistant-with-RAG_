"""
Focused Unit Test Suite for ElevenLabs Text-to-Speech (TTS) Integration

Verifies all 10 requirements:
1. Successful TTS generation using a mocked ElevenLabs response
2. Correct model ID (eleven_multilingual_v2) in the outgoing request
3. Missing API key handling (503 Service Unavailable)
4. Invalid API credentials handling (401 Unauthorized)
5. Rate-limit response handling (429 Too Many Requests)
6. Upstream service failure handling (502 Bad Gateway)
7. Network timeout handling (504 Gateway Timeout)
8. Empty text input handling (400 Bad Request)
9. Invalid or empty audio response handling (502 Bad Gateway)
10. Existing text-based /chat functionality remains unaffected
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import requests

# Ensure offline environment for HuggingFace / Transformers to prevent network timeouts during test runs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = "gsk_test_key_tts_verification_12345"
os.environ["TESTING"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from app import app, config


class TestElevenLabsTTS(unittest.TestCase):
    """Focused automated tests for /generate-speech endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Default mock API key
        self.orig_key = getattr(config.speech, "eleven_labs_api_key", None)
        config.speech.eleven_labs_api_key = "test_valid_elevenlabs_key_12345"

    def tearDown(self):
        config.speech.eleven_labs_api_key = self.orig_key

    # -------------------------------------------------------------------------
    # Test 1: Successful TTS generation using a mocked ElevenLabs response
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_01_successful_tts_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Valid MP3 payload (ID3 header + dummy bytes)
        mock_resp.content = b"ID3\x04\x00\x00\x00\x00\x00#" + (b"\xaa" * 200)
        mock_post.return_value = mock_resp

        payload = {
            "text": "Your prescription has been sent to the pharmacy.",
            "voice_id": "EXAVITQu4vr4xnSDxMaL"
        }
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("audio/mpeg", res.headers.get("content-type", ""))
        self.assertGreater(len(res.content), 100)

    # -------------------------------------------------------------------------
    # Test 2: Correct model ID (eleven_multilingual_v2) in the outgoing request
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_02_correct_model_id_multilingual_v2(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"ID3\x04\x00\x00\x00\x00\x00#" + (b"\xbb" * 200)
        mock_post.return_value = mock_resp

        payload = {
            "text": "Testing model ID payload verification.",
            "voice_id": "EXAVITQu4vr4xnSDxMaL"
        }
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 200)

        # Inspect the arguments passed to requests.post
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        sent_payload = kwargs.get("json", {})
        self.assertEqual(
            sent_payload.get("model_id"),
            "eleven_multilingual_v2",
            "Model ID must be updated to 'eleven_multilingual_v2'"
        )
        url = args[0] if args else kwargs.get("url", "")
        self.assertIn("EXAVITQu4vr4xnSDxMaL", url)

    # -------------------------------------------------------------------------
    # Test 3: Missing API key handling (503 Service Unavailable)
    # -------------------------------------------------------------------------
    def test_03_missing_api_key(self):
        config.speech.eleven_labs_api_key = ""
        payload = {"text": "Hello without API key", "voice_id": "EXAVITQu4vr4xnSDxMaL"}
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 503)
        data = res.json()
        self.assertEqual(data.get("error_code"), "TTS_KEY_MISSING")
        self.assertIn("not configured", data.get("error", "").lower())

    # -------------------------------------------------------------------------
    # Test 4: Invalid API credentials handling (401 Unauthorized)
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_04_invalid_credentials_401(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"detail":{"status":"invalid_api_key","message":"Invalid API key"}}'
        mock_post.return_value = mock_resp

        payload = {"text": "Testing bad credentials.", "voice_id": "EXAVITQu4vr4xnSDxMaL"}
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 401)
        data = res.json()
        self.assertEqual(data.get("error_code"), "TTS_AUTH_FAILED")

    # -------------------------------------------------------------------------
    # Test 5: Rate-limit response handling (429 Too Many Requests)
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_05_rate_limit_429(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = '{"detail":{"status":"quota_exceeded","message":"Rate limit exceeded"}}'
        mock_post.return_value = mock_resp

        payload = {"text": "Testing rate limit handling.", "voice_id": "EXAVITQu4vr4xnSDxMaL"}
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 429)
        data = res.json()
        self.assertEqual(data.get("error_code"), "TTS_RATE_LIMITED")

    # -------------------------------------------------------------------------
    # Test 6: Upstream service failure handling (502 Bad Gateway)
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_06_upstream_service_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = '{"detail":{"message":"Internal ElevenLabs Server Error"}}'
        mock_post.return_value = mock_resp

        payload = {"text": "Testing upstream service error.", "voice_id": "EXAVITQu4vr4xnSDxMaL"}
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 502)
        data = res.json()
        self.assertEqual(data.get("error_code"), "TTS_UPSTREAM_ERROR")

    # -------------------------------------------------------------------------
    # Test 7: Network timeout handling (504 Gateway Timeout)
    # -------------------------------------------------------------------------
    @patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out"))
    def test_07_network_timeout(self, mock_post):
        payload = {"text": "Testing timeout handling.", "voice_id": "EXAVITQu4vr4xnSDxMaL"}
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 504)
        data = res.json()
        self.assertEqual(data.get("error_code"), "TTS_TIMEOUT")

    # -------------------------------------------------------------------------
    # Test 8: Empty or invalid text input handling (400 Bad Request)
    # -------------------------------------------------------------------------
    def test_08_empty_text_input(self):
        # Empty string
        res = self.client.post("/generate-speech", json={"text": "", "voice_id": "EXAVITQu4vr4xnSDxMaL"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json().get("error_code"), "EMPTY_TEXT")

        # Whitespace only
        res2 = self.client.post("/generate-speech", json={"text": "   ", "voice_id": "EXAVITQu4vr4xnSDxMaL"})
        self.assertEqual(res2.status_code, 400)
        self.assertEqual(res2.json().get("error_code"), "EMPTY_TEXT")

    # -------------------------------------------------------------------------
    # Test 9: Invalid or empty audio response handling (502 Bad Gateway)
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_09_invalid_or_empty_audio_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b""  # Empty audio body
        mock_post.return_value = mock_resp

        payload = {"text": "Testing empty audio response.", "voice_id": "EXAVITQu4vr4xnSDxMaL"}
        res = self.client.post("/generate-speech", json=payload)
        self.assertEqual(res.status_code, 502)
        data = res.json()
        self.assertEqual(data.get("error_code"), "TTS_EMPTY_AUDIO")

    # -------------------------------------------------------------------------
    # Test 10: Existing text-based /chat functionality remains unaffected
    # -------------------------------------------------------------------------
    def test_10_chat_endpoint_unaffected_by_tts(self):
        # Even with ElevenLabs completely unconfigured or failing, text /chat must succeed
        config.speech.eleven_labs_api_key = ""
        mock_ai_resp = {
            "messages": [MagicMock(content="Medical advice: Stay well hydrated and rest.")],
            "agent_name": "CONVERSATION_AGENT"
        }
        with patch("app.process_query", return_value=mock_ai_resp):
            res = self.client.post(
                "/chat",
                json={"query": "What should I do for a mild headache?", "conversation_history": []}
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data.get("status"), "success")
            self.assertIn("hydrated", data.get("response", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
