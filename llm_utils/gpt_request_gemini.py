import os
import base64
import cv2
import numpy as np
from mimetypes import guess_type

# Google Gemini SDK
import google.generativeai as genai
from google.generativeai import types

# -----------------------------
# Gemini 초기화
# -----------------------------
_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not _API_KEY:
    raise RuntimeError("GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수를 설정하세요.")
genai.configure(api_key=_API_KEY)

# 모델 이름(원하면 환경변수로 덮어쓰기 가능)
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", TEXT_MODEL)  # 1.5 계열은 멀티모달 지원

# -----------------------------
# 이미지 → Gemini 파트 변환 (data URL 대신, 바이트 전달)
# -----------------------------
def _image_to_gemini_part(image):
    """
    image: 파일 경로(str) 또는 numpy.ndarray(BGR, OpenCV)
    return: {"mime_type": "...", "data": <bytes>}
    """
    if isinstance(image, str):
        mime_type, _ = guess_type(image)
        mime_type = mime_type or "image/jpeg"
        with open(image, "rb") as f:
            return {"mime_type": mime_type, "data": f.read()}

    elif isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("이미지 인코딩 실패")
        return {"mime_type": "image/jpeg", "data": buf.tobytes()}

    else:
        raise TypeError("image must be a file path (str) or numpy.ndarray")

# -----------------------------
# 텍스트 전용 응답
# -----------------------------
def gpt_response(text_prompt, system_prompt=""):
    """
    Gemini로 텍스트 응답 생성. 기존 함수명 유지.
    """
    model = genai.GenerativeModel(
        model_name=TEXT_MODEL,
        # system_prompt 개념: Gemini는 system_instruction에 넣어 사용
        system_instruction=system_prompt or None,
    )
    resp = model.generate_content(
        text_prompt,
        generation_config=types.GenerationConfig(
            max_output_tokens=1000,
        ),
    )
    # 실패/빈 응답 대비
    return getattr(resp, "text", "").strip()

# -----------------------------
# 텍스트 + 이미지(멀티모달) 응답
# -----------------------------
def gptv_response(text_prompt, image_prompt, system_prompt=""):
    """
    Gemini로 멀티모달 응답 생성. 기존 함수명 유지.
    image_prompt: 이미지 경로(str) 또는 OpenCV의 np.ndarray
    """
    img_part = _image_to_gemini_part(image_prompt)

    model = genai.GenerativeModel(
        model_name=VISION_MODEL,
        system_instruction=system_prompt or None,
    )
    # 멀티모달은 리스트로 텍스트와 이미지를 함께 전달
    resp = model.generate_content(
        [text_prompt, img_part],
        generation_config=types.GenerationConfig(
            max_output_tokens=1000,
        ),
    )
    return getattr(resp, "text", "").strip()
