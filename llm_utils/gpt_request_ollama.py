import os
import base64
import cv2
import numpy as np
import requests
from mimetypes import guess_type

# ===== Ollama 설정 =====
# 포트 9999로 서비스한다면: export OLLAMA_BASE_URL="http://localhost:9999"
# (미설정 시 기본값은 http://localhost:9999)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:12345")

# 요청 모델: 사용자 지정
DEFAULT_MODEL = "llama4:17b-scout-16e-instruct-q8_0"

# 환경변수로 바꾸고 싶다면 OLLAMA_TEXT_MODEL / OLLAMA_VISION_MODEL 지정
TEXT_MODEL   = os.getenv("OLLAMA_TEXT_MODEL", DEFAULT_MODEL)
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", DEFAULT_MODEL)

# ===== 공통: Ollama Chat 호출 =====
def _ollama_chat(messages, model, num_predict=1000):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # /api/chat 비스트리밍 응답: {"message": {"role":"assistant","content":"..."}}
        content = (data.get("message") or {}).get("content")
        if content is None:
            # 일부 구현/버전에 따라 "response" 필드로 올 수도 있음
            content = data.get("response", "")
        return content
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama chat request failed: {e}\nURL={url}\nModel={model}")

# ===== 기존 함수들 (이름 유지) =====

# Function to encode a local image into data URL 
def local_image_to_data_url(image):
    if isinstance(image, str):
        mime_type, _ = guess_type(image)
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image, np.ndarray):
        base64_encoded_data = base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_encoded_data}"
    else:
        raise ValueError("image must be a file path (str) or a numpy.ndarray")

def gptv_response(text_prompt, image_prompt, system_prompt=""):
    # Ollama 멀티모달(예: llama3.2-vision, llava 등)은 /api/chat에서 images=[base64] 형태 지원
    data_url = local_image_to_data_url(image_prompt)
    # data URL에서 base64 본문만 추출
    base64_img = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text_prompt, "images": [base64_img]})

    return _ollama_chat(messages, VISION_MODEL, num_predict=1000)

def gpt_response(text_prompt, system_prompt=""):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text_prompt})
    return _ollama_chat(messages, TEXT_MODEL, num_predict=1000)
