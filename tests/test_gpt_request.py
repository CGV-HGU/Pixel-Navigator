import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_utils import gpt_request  # noqa: E402


class _Response:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._payload


class GptRequestTest(unittest.TestCase):
    def test_multimodal_request_uses_s2e_endpoint_contract(self):
        answer = '{"Reason":"doorway ahead","Angle":90,"Flag":false}'
        response = _Response(
            {"choices": [{"message": {"role": "assistant", "content": answer}}]}
        )
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return response

        environment = {
            "VLM_API_URL": "http://vlm.test/v1/chat/completions",
            "VLM_API_MODEL": "qwen-test",
            "VLM_API_TIMEOUT_S": "7.5",
            "VLM_API_MAX_TOKENS": "321",
            "VLM_API_KEY": "secret",
            "VLM_STRUCTURED_OUTPUT_MODE": "openai_json_schema",
        }
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ):
            result = gpt_request.gptv_response("<Target Object>:chair", image, "system")

        self.assertEqual(
            result,
            "{'Reason': 'doorway ahead', 'Angle': 90, 'Flag': False}",
        )
        self.assertEqual(captured["timeout"], 7.5)
        request = captured["request"]
        self.assertEqual(request.full_url, environment["VLM_API_URL"])
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen-test")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(
            payload["messages"][0],
            {"role": "system", "content": "system"},
        )
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["text"], "<Target Object>:chair")
        self.assertTrue(
            user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["Angle"]["enum"],
            list(range(0, 360, 30)),
        )
        self.assertEqual(schema["required"], ["Reason", "Angle", "Flag"])

    def test_text_request_uses_same_model_without_direction_schema(self):
        response = _Response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return response

        with mock.patch.dict(
            os.environ,
            {"VLM_API_MODEL": "qwen-test", "VLM_API_TIMEOUT_S": "3"},
            clear=False,
        ), mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = gpt_request.gpt_response("hello", "")

        self.assertEqual(result, "ok")
        self.assertNotIn("response_format", captured["payload"])
        self.assertEqual(
            captured["payload"]["messages"],
            [{"role": "user", "content": "hello"}],
        )

    def test_invalid_response_is_rejected(self):
        with mock.patch(
            "urllib.request.urlopen", return_value=_Response({"choices": []})
        ):
            with self.assertRaisesRegex(gpt_request.VlmApiError, "no choices"):
                gpt_request.gpt_response("hello")

    def test_invalid_direction_contract_is_rejected(self):
        response = _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"Reason":"ahead","Angle":45,"Flag":false}'
                        }
                    }
                ]
            }
        )
        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(gpt_request.VlmApiError, "Angle"):
                gpt_request.gptv_response(
                    "<Target Object>:chair",
                    np.zeros((8, 12, 3), dtype=np.uint8),
                )


if __name__ == "__main__":
    unittest.main()
