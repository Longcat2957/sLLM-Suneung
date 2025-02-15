import os
import re
import json
import textwrap

def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON not found: {file_path}")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {file_path}") from e
    
def load_txt(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TXT not found: {file_path}")
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Invalid TXT: {file_path}") from e

def format_text(text: str, width: int = 80) -> str:
    """
    기존 줄바꿈은 유지하면서 각 문단마다 textwrap을 적용하여 적당히 줄바꿈합니다.
    """
    paragraphs = text.splitlines()
    formatted_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph, width=width)
            formatted_paragraphs.append(wrapped)
        else:
            formatted_paragraphs.append("")
    return "\n".join(formatted_paragraphs)


def natural_key(string_):
    """
    파일명에 포함된 숫자를 기준으로 자연스러운 정렬을 위한 key 함수.
    예: "2.txt", "10.txt"를 올바른 순서(2, 10)로 정렬합니다.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string_)]
