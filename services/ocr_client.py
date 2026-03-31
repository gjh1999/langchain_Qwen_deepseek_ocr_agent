import requests

OCR_API_URL = "http://127.0.0.1:8001/ocr_pdf"


def call_ocr_api(file_path: str):
    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(OCR_API_URL, files=files)

    resp.raise_for_status()
    return resp.json()