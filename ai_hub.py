"""Provider helpers and deterministic prompt construction for EvalSphere."""

from __future__ import annotations

import json
import mimetypes
import secrets
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MINI_MODELS = ("gpt-4o-mini", "gpt-4.1-mini")
TRANSCRIPTION_MODEL = "whisper-1"
REQUEST_TIMEOUT_SECONDS = 120


class AIServiceError(RuntimeError):
    """An API request failed or returned an unusable response."""


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    base_url: str
    model: str


def build_optimized_prompt(
    *,
    task: str,
    request: str,
    context: str,
    framework: str,
    output_format: str,
) -> str:
    """Create a precise, evidence-aware prompt without making an API call."""
    task_guidance = {
        "XPath locator": (
            "Use only the supplied DOM as evidence. Prefer stable, semantic attributes "
            "over brittle positional selectors. Return a verified-looking locator only "
            "when the markup supports it; otherwise state what additional DOM is needed."
        ),
        "Robot Framework code": (
            "Produce maintainable Robot Framework syntax for the selected library. "
            "Do not invent URLs, selectors, credentials, or application behavior; mark "
            "unknowns as explicit variables or assumptions."
        ),
        "Test cases": (
            "Cover positive, negative, boundary, and regression behavior where relevant. "
            "Make each case independently executable and tie expected results to the "
            "provided acceptance criteria."
        ),
        "Image analysis": (
            "Report only issues visible in the supplied image. Separate observations "
            "from hypotheses, describe the evidence, and give reproducible verification "
            "steps. Do not claim to verify hidden behavior."
        ),
        "Audio analysis": (
            "Ground findings in the recording transcript. Separate direct evidence "
            "from interpretation, do not identify speakers beyond supplied context, "
            "and mark unclear wording as uncertain."
        ),
        "Custom QA prompt": (
            "Follow the user's goal and constraints. Ask for missing critical inputs "
            "instead of silently inventing facts."
        ),
    }
    guidance = task_guidance.get(task, task_guidance["Custom QA prompt"])
    context_block = context.strip() or "No additional context supplied."
    return (
        "Role: You are a careful senior software quality and test-automation engineer.\n\n"
        f"Task: {request.strip()}\n\n"
        f"Workflow: {task}\n"
        f"Framework or evidence type: {framework}\n\n"
        f"Context and source evidence:\n{context_block}\n\n"
        f"Required output format: {output_format}\n\n"
        f"Quality requirements: {guidance} Identify assumptions, avoid unsupported claims, "
        "and include concise validation checks for the result."
    )


def _endpoint(config: ProviderConfig, path: str) -> str:
    base_url = config.base_url.strip().rstrip("/")
    try:
        parsed_url = urlsplit(base_url)
    except ValueError as error:
        raise AIServiceError("The API base URL is invalid.") from error
    secure_remote = parsed_url.scheme == "https"
    local_http = parsed_url.scheme == "http" and parsed_url.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if (
        not (secure_remote or local_http)
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise AIServiceError(
            "The API base URL must use HTTPS. HTTP is allowed only for a local endpoint."
        )
    return f"{base_url}/{path.lstrip('/')}"


def _request(request: Request, *, purpose: str) -> bytes:
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.read()
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise AIServiceError(
            f"{purpose} failed with HTTP {error.code}: {detail or error.reason}"
        ) from error
    except (TimeoutError, URLError, OSError) as error:
        raise AIServiceError(f"{purpose} could not reach the provider: {error}") from error


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


def complete_chat(
    *,
    config: ProviderConfig,
    prompt: str,
    image: dict[str, Any] | None = None,
) -> str:
    """Send a prompt and optional image to an OpenAI-compatible chat endpoint."""
    if not config.api_key:
        raise AIServiceError("An API key is required to generate a response.")
    if config.model not in MINI_MODELS:
        raise AIServiceError("Select one of the supported mini models.")

    if image is None:
        content: str | list[dict[str, Any]] = prompt
    else:
        import base64

        encoded_image = base64.b64encode(image["content"]).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image['mime_type']};base64,{encoded_image}",
                    "detail": "high",
                },
            },
        ]
    payload = {
        "model": config.model,
        "temperature": 0.1,
        "max_tokens": 2500,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Produce accurate, concise software quality engineering output. "
                    "Use supplied evidence only, state uncertainty, and follow the "
                    "requested output format."
                ),
            },
            {"role": "user", "content": content},
        ],
    }
    request = Request(
        _endpoint(config, "chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        response_data = json.loads(_request(request, purpose="Model generation"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AIServiceError("The provider returned an invalid JSON response.") from error
    if not isinstance(response_data, dict):
        raise AIServiceError("The provider response did not contain a chat result.")
    try:
        result = _message_text(response_data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise AIServiceError("The provider response did not contain a chat result.") from error
    if not result:
        raise AIServiceError("The provider returned an empty chat result.")
    return result


def transcribe_audio(
    *,
    config: ProviderConfig,
    audio: bytes,
    filename: str,
) -> str:
    """Transcribe audio through an OpenAI-compatible audio-transcriptions endpoint."""
    if not config.api_key:
        raise AIServiceError("An API key is required to transcribe audio.")
    boundary = f"----EvalSphere{secrets.token_hex(16)}"
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body = bytearray()
    for name, value in (
        ("model", TRANSCRIPTION_MODEL),
        ("response_format", "json"),
    ):
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(
                "ascii"
            )
        )
    safe_filename = PurePath(filename.replace("\\", "/")).name
    safe_filename = "".join(
        character for character in safe_filename if character not in '"\r\n'
    )
    body.extend(f"--{boundary}\r\n".encode("ascii"))
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'.encode(
            "utf-8"
        )
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("ascii"))
    body.extend(audio)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    request = Request(
        _endpoint(config, "audio/transcriptions"),
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        response_data = json.loads(_request(request, purpose="Audio transcription"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AIServiceError("The transcription service returned invalid JSON.") from error
    if not isinstance(response_data, dict):
        raise AIServiceError("The transcription service returned no transcript.")
    transcript = response_data.get("text")
    if not isinstance(transcript, str) or not transcript.strip():
        raise AIServiceError("The transcription service returned no transcript.")
    return transcript.strip()
