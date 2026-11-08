import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.engine import URL

from ai_hub import AIServiceError, MINI_MODELS
from evaluation_api import create_app


class EvaluationAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app("sqlite://")
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": ""})
        self.environment.__enter__()

    def tearDown(self) -> None:
        self.environment.__exit__(None, None, None)
        self.client_context.__exit__(None, None, None)

    def test_health_and_empty_history(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/api/evaluations").json(), [])

    def test_evaluation_requires_a_provider_key(self) -> None:
        response = self.client.post(
            "/api/evaluations",
            json={"model": MINI_MODELS[0], "prompt": "Explain a test strategy"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json()["detail"],
            "Configure a provider API key before running an evaluation.",
        )

    def test_evaluation_saves_result_without_storing_provider_key(self) -> None:
        with patch(
            "evaluation_api.complete_chat",
            return_value="Generated response",
        ) as complete:
            response = self.client.post(
                "/api/evaluations",
                headers={"X-Provider-API-Key": "session-secret"},
                json={
                    "model": MINI_MODELS[0],
                    "prompt": "  Explain a test strategy  ",
                },
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["model"], MINI_MODELS[0])
        self.assertEqual(result["prompt"], "Explain a test strategy")
        self.assertEqual(result["response"], "Generated response")
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertTrue(result["created_at"].endswith("+00:00"))
        self.assertEqual(complete.call_args.kwargs["config"].api_key, "session-secret")

        history = self.client.get("/api/evaluations").json()
        self.assertEqual(len(history), 1)
        self.assertNotIn("api_key", history[0])
        self.assertNotIn("session-secret", repr(history[0]))

    def test_unsupported_model_and_empty_prompt_are_rejected(self) -> None:
        unsupported = self.client.post(
            "/api/evaluations",
            headers={"X-Provider-API-Key": "session-secret"},
            json={"model": "premium-model", "prompt": "Hello"},
        )
        empty_prompt = self.client.post(
            "/api/evaluations",
            headers={"X-Provider-API-Key": "session-secret"},
            json={"model": MINI_MODELS[0], "prompt": "  "},
        )

        self.assertEqual(unsupported.status_code, 422)
        self.assertEqual(empty_prompt.status_code, 422)

    def test_provider_errors_are_reported_and_not_saved(self) -> None:
        with patch(
            "evaluation_api.complete_chat",
            side_effect=AIServiceError("Provider unavailable"),
        ):
            response = self.client.post(
                "/api/evaluations",
                headers={"X-Provider-API-Key": "session-secret"},
                json={"model": MINI_MODELS[0], "prompt": "Hello"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Provider unavailable")
        self.assertEqual(self.client.get("/api/evaluations").json(), [])

    def test_evaluations_persist_across_api_instances(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_url = URL.create(
                "sqlite",
                database=str(Path(temporary_directory) / "evaluations.db"),
            ).render_as_string(hide_password=False)
            first_app = create_app(database_url)
            with TestClient(first_app) as first_client:
                with patch(
                    "evaluation_api.complete_chat",
                    return_value="Persisted response",
                ):
                    created = first_client.post(
                        "/api/evaluations",
                        headers={"X-Provider-API-Key": "session-secret"},
                        json={"model": MINI_MODELS[0], "prompt": "Persist this run"},
                    )
                self.assertEqual(created.status_code, 200)

            second_app = create_app(database_url)
            with TestClient(second_app) as second_client:
                history = second_client.get("/api/evaluations").json()

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["response"], "Persisted response")

    def test_history_limit_is_bounded(self) -> None:
        response = self.client.get("/api/evaluations?limit=51")

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
