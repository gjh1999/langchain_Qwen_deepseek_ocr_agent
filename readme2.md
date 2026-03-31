# 🧠 PDF → Markdown 智能解析系统（Agent + OCR微服务）

基于 **LangChain + Qwen + DeepSeek-OCR（vLLM）** 构建的 PDF 智能解析系统，实现：

👉 **PDF → Markdown → 问答 / 总结**

---

## 🚀 项目简介

本项目采用 **“Agent + OCR微服务”解耦架构**：

* 📄 PDF解析由独立 OCR 服务完成（WSL）
* 🧠 Agent（Windows）负责理解与问答
* 🔗 通过 HTTP 实现跨系统调用

---

## 🏗️ 系统架构（真实流程）

```text
        前端（HTML）
              ↓
        FastAPI（Windows）
              ↓
         runner.py
        /          \
  无PDF            有PDF
   ↓                ↓
Qwen直接回答     调用 OCR 服务
                    ↓
            DeepSeek-OCR（WSL）
                    ↓
             Markdown 文本
                    ↓
             Qwen 总结/问答
```

---

## 📦 技术栈

### 🧠 Agent层（Windows）

* LangChain
* Qwen（通过 Ollama / API）
* FastAPI

### 🔍 OCR层（WSL）

* DeepSeek-OCR
* vLLM 推理引擎
* PyMuPDF（PDF → 图像）
* Pillow（图像处理）

### ⚙️ 通信

* HTTP（跨系统调用）

---

## 📁 项目结构

```text
langchain_Qwen_deepseek_ocr_agent/

├── app.py                     # FastAPI入口（Windows）
├── runner.py                  # 主流程（是否调用OCR）
├── build_agent.py             # LangChain Agent构建

├── tools/
│   └── ocr_tool.py            # OCR工具（封装HTTP调用）

├── services/
│   └── ocr_client.py          # OCR API客户端

├── api.py                     # OCR服务入口（WSL）
├── ocr_pdf_service.py         # OCR核心逻辑（vLLM推理）

├── config.py
├── temp/                      # 临时文件
├── simple_pdf_chat_frontend.html
└── readme.md
```

---

## ⚙️ 核心流程（重点）

### 📄 带 PDF 的请求

```text
1. 前端上传 PDF + 问题
2. FastAPI 接收请求
3. runner.py 调用 OCR 工具
4. HTTP 请求 WSL OCR 服务
5. OCR 返回 markdown
6. 拼接 prompt → Qwen
7. 返回最终答案
```

---

### 💬 普通对话

```text
用户问题 → Qwen → 返回结果
```

---

## 🔌 API 使用方式

### 📄 Chat接口（支持PDF）

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -F "message=帮我总结这个PDF" \
  -F "file=@test.pdf"
```

---

### 🔍 OCR接口（WSL）

```bash
curl -X POST "http://127.0.0.1:8001/ocr_pdf" \
  -F "file=@test.pdf"
```

返回：

```json
{
  "markdown": "...",
  "markdown_det": "...",
  "mmd_path": "...",
  "layout_pdf_path": "..."
}
```

---

## ⚡ 启动方式

### 1️⃣ 启动 OCR 服务（WSL）

```bash
uvicorn api:app --port 8001
```

👉 首次启动会加载 DeepSeek-OCR（vLLM）

---

### 2️⃣ 启动 Agent 服务（Windows）

```bash
uvicorn app:app --port 8000 --reload
```

---

### 3️⃣ 打开前端

```text
http://127.0.0.1:8000/
```

---

## 🧠 项目特点（重点）

### ⭐ 架构设计

* OCR 与 Agent 解耦（微服务化）
* Windows ↔ WSL 跨系统调用

### ⭐ 推理能力

* 基于 vLLM 的多模态 OCR 推理
* 支持 PDF → 图像 → 文本 → Markdown

### ⭐ 工程能力

* FastAPI API封装
* Tool / Service 分层设计
* 模块清晰（agent / tools / services）

### ⭐ OCR增强能力

* 自动图片裁剪
* layout 解析（bbox）
* Markdown结构还原

---

## ⚠️ 注意事项

### ❗ OCR服务必须先启动

否则会报错：

```text
Connection refused
```

---

### ❗ 仅支持 PDF

非 PDF 会直接报错

---

### ❗ GPU要求

建议：

* CUDA 环境
* vLLM 支持 GPU

---

## 🧪 TODO（未来优化）

* [ ] Agent 自动调用 OCR（真正 tool-calling）
* [ ] Gateway 架构（仿 OpenClaw）
* [ ] 多PDF批处理
* [ ] UI优化
* [ ] 推理加速（TensorRT）

---

## 👨‍💻 项目亮点（简历用）

* 构建 **Agent + OCR 微服务架构**
* 实现 **PDF → Markdown 自动解析 pipeline**
* 基于 vLLM 部署 DeepSeek-OCR
* 实现 **跨系统推理调用（Windows + WSL）**
* 模块化设计（Agent / Tool / Service）

---

## 📌 一句话总结（面试用）

👉 将 OCR 推理服务拆分为独立微服务，通过 Agent 调度，实现 PDF 到结构化文本的自动转换与理解。

---
