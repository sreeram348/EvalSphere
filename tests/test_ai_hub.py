import json
import unittest
from unittest.mock import Mock, patch

from ai_hub import (
    AIServiceError,
    MINI_MODELS,
    ProviderConfig,
    build_optimized_prompt,
    complete_chat,
    transcribe_audio,
    _endpoint,
)


class PromptBuilderTests(unittest.TestCase):
    def test_prompt_includes_request_context_and_evidence_rules(self) -> None:
        prompt = build_optimized_prompt(
            task="XPath locator",
            request="Find the checkout button",
            context='<button data-testid="checkout">Pay</button>',
            framework="Browser-neutral",
            output_format="XPath + CSS fallback",
        )

        self.assertIn("Find the checkout button", prompt)
        self.assertIn('data-testid="checkout"', prompt)
        self.assertIn("XPath + CSS fallback", prompt)
        self.assertIn("Use only the supplied DOM as evidence", prompt)

    def test_empty_context_is_explicit(self) -> None:
        prompt = build_optimized_prompt(
            task="Test cases",
            request="Check sign-in",
            context="  ",
            framework="Framework-neutral",
            output_format="Gherkin scenarios",
        )

        self.assertIn("No additional context supplied.", prompt)
        self.assertIn("positive, negative, boundary, and regression", prompt)


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProviderConfig(
            api_key="test-key",
            base_url="https://api.example.test/v1",
            model=MINI_MODELS[0],
        )

    def test_chat_sends_request_and_returns_text(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Generated result"}}]}
        ).encode()

        with patch("ai_hub.urlopen", return_value=response) as urlopen:
            result = complete_chat(config=self.config, prompt="Write a test")

        self.assertEqual(result, "Generated result")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], MINI_MODELS[0])
        self.assertIn("Write a test", payload["messages"][1]["content"])

    def test_chat_rejects_non_mini_model(self) -> None:
        config = ProviderConfig("test-key", self.config.base_url, "premium-model")

        with self.assertRaisesRegex(AIServiceError, "supported mini models"):
            complete_chat(config=config, prompt="Write a test")

    def test_chat_rejects_unexpected_provider_response_shape(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b"[]"

        with patch("ai_hub.urlopen", return_value=response):
            with self.assertRaisesRegex(AIServiceError, "chat result"):
                complete_chat(config=self.config, prompt="Write a test")

    def test_api_endpoint_rejects_nonlocal_http(self) -> None:
        config = ProviderConfig("test-key", "http://localhost.example/v1", MINI_MODELS[0])

        with self.assertRaisesRegex(AIServiceError, "must use HTTPS"):
            _endpoint(config, "chat/completions")

    def test_api_endpoint_allows_local_http(self) -> None:
        config = ProviderConfig("test-key", "http://localhost:11434/v1", MINI_MODELS[0])

        self.assertEqual(
            _endpoint(config, "chat/completions"),
            "http://localhost:11434/v1/chat/completions",
        )

    def test_image_is_sent_as_a_data_url(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Image reviewed"}}]}
        ).encode()
        image = {"content": b"image-bytes", "mime_type": "image/png"}

        with patch("ai_hub.urlopen", return_value=response) as urlopen:
            result = complete_chat(config=self.config, prompt="Inspect", image=image)

        self.assertEqual(result, "Image reviewed")
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertIn(
            "data:image/png;base64,",
            payload["messages"][1]["content"][1]["image_url"]["url"],
        )

    def test_audio_transcription_posts_multipart_audio(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({"text": "The checkout failed."}).encode()

        with patch("ai_hub.urlopen", return_value=response) as urlopen:
            transcript = transcribe_audio(
                config=self.config,
                audio=b"sample-audio",
                filename="run.wav",
            )

        self.assertEqual(transcript, "The checkout failed.")
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url, "https://api.example.test/v1/audio/transcriptions"
        )
        self.assertIn(b"sample-audio", request.data)
        self.assertIn(b"whisper-1", request.data)


if __name__ == "__main__":
    unittest.main()
