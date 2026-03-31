from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from tools.ocr_tool import ocr_pdf_tool

def build_agent():
    model = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0
    )

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="""
你是一个文档总结助手：
- 我会直接给你OCR后的文本内容
- 你负责提炼、总结、回答问题
- 不需要自己决定是否调用OCR工具
"""
    )

    return agent