from agent.build_agent import build_agent
from tools.ocr_tool import ocr_pdf_tool

agent = build_agent()

def _extract_text_from_response(response) -> str:
    if isinstance(response, dict) and "messages" in response:
        msgs = response["messages"]
        if msgs:
            last_msg = msgs[-1]
            if hasattr(last_msg, "content"):
                return last_msg.content
            if isinstance(last_msg, dict):
                return last_msg.get("content", str(last_msg))
    return str(response)

def run_agent(message: str):
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": message}
        ]
    })
    return _extract_text_from_response(response)

def run_agent_with_pdf(file_path: str, question: str = ""):
    # 1. 强制先做 OCR
    ocr_result = ocr_pdf_tool.invoke({"file_path": file_path}) if hasattr(ocr_pdf_tool, "invoke") else ocr_pdf_tool(file_path)

    # 2. 拼给 Qwen 的内容
    prompt = f"""
下面是PDF经过OCR后的内容，请你帮我处理。

用户问题：
{question or "请总结这份PDF的主要内容"}

OCR内容：
{ocr_result}
"""

    response = agent.invoke({
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })
    return _extract_text_from_response(response)