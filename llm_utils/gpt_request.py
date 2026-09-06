"""OpenAI-compatible VLM transport used by the PixelNav planner.

The navigation pipeline still owns the original PixelNav prompt and parses the
same Reason/Angle/Flag response. This module only replaces the remote Vertex
AI transport with the local chat-completions endpoint used by the S2E runtime.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


DEFAULT_API_URL = "http://server-02.cgv:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3.5-9b-instruct"

_PIXELNAV_DIRECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "Reason": {"type": "string"},
        "Angle": {
            "type": "integer",
            "enum": list(range(0, 360, 30)),
        },
        "Flag": {"type": "boolean"},
    },
    "required": ["Reason", "Angle", "Flag"],
}


class VlmApiError(RuntimeError):
    """Raised when the local chat-completions endpoint violates its contract."""


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise VlmApiError(f"{name} must be an integer, got {raw!r}") from error
    if value <= 0:
        raise VlmApiError(f"{name} must be positive, got {value}")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise VlmApiError(f"{name} must be numeric, got {raw!r}") from error
    if value <= 0:
        raise VlmApiError(f"{name} must be positive, got {value}")
    return value


def local_image_to_data_url(image: str | np.ndarray) -> str:
    """Encode a path or OpenCV image as an inline image data URL."""
    if isinstance(image, str):
        path = Path(image)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise VlmApiError(f"image is unreadable: {path}") from error
        mime_type = guess_type(path.name)[0] or "image/jpeg"
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise VlmApiError("image array must be uint8 HxWx3")
        encoded, buffer = cv2.imencode(".jpg", image)
        if not encoded:
            raise VlmApiError("image JPEG encoding failed")
        payload = buffer.tobytes()
        mime_type = "image/jpeg"
    else:
        raise TypeError("image must be a file path or numpy.ndarray")

    content = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{content}"


def _structured_output(payload: dict[str, Any]) -> dict[str, Any]:
    mode = os.getenv("VLM_STRUCTURED_OUTPUT_MODE", "openai_json_schema").strip()
    schema = {
        "name": "pixelnav_direction_v1",
        "strict": True,
        "schema": _PIXELNAV_DIRECTION_SCHEMA,
    }
    if mode in {"", "openai_json_schema"}:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": schema,
        }
        return payload
    if mode == "vllm_no_whitespace":
        payload["structured_outputs"] = {
            "json": _PIXELNAV_DIRECTION_SCHEMA,
            "disable_any_whitespace": True,
        }
        return payload
    raise VlmApiError(f"unsupported VLM_STRUCTURED_OUTPUT_MODE: {mode!r}")


def _extract_content(response_payload: Mapping[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VlmApiError("chat-completions response has no choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, Mapping):
        content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(content, str) or not content.strip():
        raise VlmApiError("chat-completions response has no message content")
    return content.strip()


def _normalize_direction_content(content: str) -> str:
    """Validate JSON output and adapt booleans for PixelNav's literal parser."""
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as error:
        raise VlmApiError("VLM direction response is not valid JSON") from error

    if not isinstance(answer, Mapping):
        raise VlmApiError("VLM direction response must be a JSON object")
    if set(answer) != {"Reason", "Angle", "Flag"}:
        raise VlmApiError(
            "VLM direction response must contain only Reason, Angle, and Flag"
        )

    reason = answer["Reason"]
    angle = answer["Angle"]
    flag = answer["Flag"]
    if not isinstance(reason, str):
        raise VlmApiError("VLM direction Reason must be a string")
    if (
        isinstance(angle, bool)
        or not isinstance(angle, int)
        or angle not in range(0, 360, 30)
    ):
        raise VlmApiError("VLM direction Angle must be one of 0, 30, ..., 330")
    if not isinstance(flag, bool):
        raise VlmApiError("VLM direction Flag must be a boolean")

    # gpt4v_planner.py uses ast.literal_eval, whose boolean spellings are
    # True/False rather than JSON's true/false. repr keeps that contract.
    return repr({"Reason": reason, "Angle": angle, "Flag": flag})


def _chat_completions(payload: Mapping[str, Any]) -> str:
    api_url = os.getenv("VLM_API_URL", DEFAULT_API_URL).strip()
    if not api_url:
        raise VlmApiError("VLM_API_URL is empty")

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("VLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout_s = _positive_float_env("VLM_API_TIMEOUT_S", 120.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise VlmApiError(
            f"VLM API returned HTTP {error.code} for {api_url}"
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise VlmApiError(f"VLM API request failed for {api_url}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VlmApiError("VLM API returned invalid JSON") from error

    if not isinstance(response_payload, Mapping):
        raise VlmApiError("VLM API response must be a JSON object")
    return _extract_content(response_payload)


def _messages(text_prompt: str, system_prompt: str, *, image_url: str | None = None):
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image_url is None:
        user_content: Any = text_prompt
    else:
        user_content = [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    messages.append({"role": "user", "content": user_content})
    return messages


def gptv_response(text_prompt, image_prompt, system_prompt=""):
    """Return one structured PixelNav direction decision from the local VLM."""
    model = os.getenv("VLM_API_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise VlmApiError("VLM_API_MODEL is empty")
    payload = {
        "model": model,
        "messages": _messages(
            str(text_prompt),
            str(system_prompt),
            image_url=local_image_to_data_url(image_prompt),
        ),
        "temperature": 0.0,
        "max_tokens": _positive_int_env("VLM_API_MAX_TOKENS", 1024),
    }
    content = _chat_completions(_structured_output(payload))
    return _normalize_direction_content(content)


def gpt_response(text_prompt, system_prompt=""):
    """Return an unconstrained text completion from the same local model."""
    model = os.getenv("VLM_API_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise VlmApiError("VLM_API_MODEL is empty")
    return _chat_completions(
        {
            "model": model,
            "messages": _messages(str(text_prompt), str(system_prompt)),
            "temperature": 0.0,
            "max_tokens": _positive_int_env("VLM_API_MAX_TOKENS", 1024),
        }
    )
