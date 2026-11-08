"""Small standard-library client for the local EvalSphere evaluation API."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REQUEST_TIMEOUT_SECONDS = 10


class EvaluationAPIError(RuntimeError):
    """The local evaluation API could not complete a request."""


def _api_url(base_url: str, path: str) -> str:
    normalized = base_url.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
    except ValueError as error:
        raise EvaluationAPIError("The evaluation API URL is invalid.") from error
    local_http = parsed.scheme == "http" and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if (
        not (parsed.scheme == "https" or local_http)
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EvaluationAPIError(
            "The evaluation API URL must use HTTPS. HTTP is allowed only for a local endpoint."
        )
    return f"{normalized}/{path.lstrip('/')}"


def _request(
    *,
    url: str,
    method: str,
    payload: dict[str, Any] | None = None,
    provider_api_key: str | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if provider_api_key:
        headers["X-Provider-API-Key"] = provider_api_key
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_data = json.loads(response.read())
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        try:
            error_body = json.loads(detail)
        except json.JSONDecodeError:
            error_body = {}
        if isinstance(error_body, dict) and isinstance(error_body.get("detail"), str):
            detail = error_body["detail"]
        raise EvaluationAPIError(
            f"Evaluation API returned HTTP {error.code}: {detail or error.reason}"
        ) from error
    except (TimeoutError, URLError, OSError) as error:
        raise EvaluationAPIError(
            f"Could not reach the evaluation API: {error}"
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvaluationAPIError("The evaluation API returned invalid JSON.") from error
    return response_data


def submit_evaluation(
    *,
    base_url: str,
    prompt: str,
    model: str,
    provider_api_key: str,
) -> dict[str, Any]:
    result = _request(
        url=_api_url(base_url, "api/evaluations"),
        method="POST",
        payload={"prompt": prompt, "model": model},
        provider_api_key=provider_api_key,
    )
    if not isinstance(result, dict):
        raise EvaluationAPIError("The evaluation API returned an invalid evaluation.")
    required_fields = {"id", "model", "prompt", "response", "latency_ms", "created_at"}
    if not required_fields.issubset(result):
        raise EvaluationAPIError("The evaluation API returned an incomplete evaluation.")
    return result


def fetch_recent_evaluations(*, base_url: str, limit: int = 20) -> list[dict[str, Any]]:
    result = _request(
        url=_api_url(base_url, f"api/evaluations?limit={limit}"),
        method="GET",
    )
    if not isinstance(result, list) or not all(
        isinstance(item, dict) for item in result
    ):
        raise EvaluationAPIError("The evaluation API returned invalid history.")
    return result
