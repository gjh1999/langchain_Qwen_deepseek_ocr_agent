from langchain.tools import tool
from services.ocr_client import call_ocr_api


@tool
def ocr_pdf_tool(file_path: str) -> str:
    """
    输入PDF路径，返回OCR识别后的markdown文本
    """
    result = call_ocr_api(file_path)
    return result["markdown"]