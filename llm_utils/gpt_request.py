import os
import cv2
import numpy as np

import vertexai
from vertexai.generative_models import GenerativeModel, Image


_ADC_CANDIDATE_PATHS = [
    os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
    os.path.expanduser("~/gcloud/application_default_credentials.json"),
    os.path.abspath("gcloud/application_default_credentials.json"),
]
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    for _adc_path in _ADC_CANDIDATE_PATHS:
        if os.path.isfile(_adc_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _adc_path
            break


def _read_project_from_gcloud_config():
    active_cfg = "default"
    active_cfg_path = os.path.expanduser("~/.config/gcloud/active_config")
    if os.path.isfile(active_cfg_path):
        try:
            with open(active_cfg_path, "r", encoding="utf-8") as f:
                name = f.read().strip()
                if name:
                    active_cfg = name
        except Exception:
            pass

    cfg_path = os.path.expanduser(f"~/.config/gcloud/configurations/config_{active_cfg}")
    if not os.path.isfile(cfg_path):
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("project"):
                    _, value = s.split("=", 1)
                    project_id = value.strip()
                    if project_id:
                        return project_id
    except Exception:
        return None
    return None


def _resolve_vertex_project():
    for key in ("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        value = os.getenv(key)
        if value:
            return value

    try:
        import google.auth

        _, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if project_id:
            return project_id
    except Exception:
        pass

    project_id = _read_project_from_gcloud_config()
    if project_id:
        return project_id

    raise RuntimeError(
        "Vertex project id를 찾을 수 없습니다. "
        "GOOGLE_CLOUD_PROJECT를 설정하거나 gcloud config set project <PROJECT_ID>를 실행하세요."
    )


_PROJECT_ID = _resolve_vertex_project()
_LOCATION = (
    os.environ.get("VERTEX_LOCATION")
    or os.environ.get("GOOGLE_CLOUD_LOCATION")
    or "us-central1"
)

vertexai.init(project=_PROJECT_ID, location=_LOCATION)

TEXT_MODEL = os.getenv("VERTEX_TEXT_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash"))
VISION_MODEL = os.getenv("VERTEX_VISION_MODEL", os.getenv("GEMINI_VISION_MODEL", TEXT_MODEL))
MAX_OUTPUT_TOKENS = int(os.getenv("VERTEX_MAX_OUTPUT_TOKENS", "1000"))


def local_image_to_vertex_image(image):
    if isinstance(image, str):
        return Image.load_from_file(image)

    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("이미지 인코딩 실패")
        return Image.from_bytes(buf.tobytes())

    raise TypeError("image must be a file path (str) or numpy.ndarray")


def _system_instruction(system_prompt):
    return [system_prompt] if system_prompt else None


def _generation_config():
    return {"max_output_tokens": MAX_OUTPUT_TOKENS}


def gptv_response(text_prompt, image_prompt, system_prompt=""):
    model = GenerativeModel(
        model_name=VISION_MODEL,
        system_instruction=_system_instruction(system_prompt),
    )
    response = model.generate_content(
        [text_prompt, local_image_to_vertex_image(image_prompt)],
        generation_config=_generation_config(),
    )
    return getattr(response, "text", "").strip()


def gpt_response(text_prompt, system_prompt=""):
    model = GenerativeModel(
        model_name=TEXT_MODEL,
        system_instruction=_system_instruction(system_prompt),
    )
    response = model.generate_content(
        text_prompt,
        generation_config=_generation_config(),
    )
    return getattr(response, "text", "").strip()
