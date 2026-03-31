

## 总体架构

```text
前端 / 上传页面
    ↓
FastAPI API
    ↓
任务服务（Job Manager）
    ↓
LangChain 1.0 Agent
    ├─ pdf_meta_tool
    ├─ pdf_render_tool
    ├─ ocr_page_tool
    ├─ markdown_merge_tool
    ├─ markdown_cleanup_tool
    └─ save_result_tool
    ↓
模型层
    ├─ Qwen（主模型，API）
    └─ DeepSeek-OCR-1（vLLM 服务）
    ↓
存储层
    ├─ 原始 PDF
    ├─ 页面图片
    ├─ 中间 Markdown
    └─ 最终 .md
```

## 核心分工

**LangChain 1.0**

* 负责 agent 编排
* 用 `create_agent`
* 用 middleware 做日志、重试、限流、输出校验 ([LangChain 文档][1])

**Qwen**

* 负责主决策
* 负责 Markdown 清洗、纠错、格式统一
* 可直接走阿里云百炼的 OpenAI 兼容接口 ([阿里云帮助中心][2])

**DeepSeek-OCR-2**

* 负责识别页面内容
* 负责把单页内容转成 Markdown 初稿
* 适合用 vLLM 跑推理服务；官方仓库明确提供了 vLLM 推理方案 ([GitHub][3])

**vLLM**

* 负责把模型做成 OpenAI 兼容 HTTP 服务
* 方便 LangChain / Python 统一调用 ([vLLM][4])

## 处理流程

```text
1. 上传 PDF
2. 读取 PDF 信息
3. 把 PDF 按页渲染成图片
4. 每页图片送给 DeepSeek-OCR-2
5. 得到每页 Markdown
6. 合并整份 Markdown
7. 用 Qwen 做清洗和规范化
8. 保存并返回 .md
```

## 关键原则

### 1. 不要直接把 PDF 丢给 OCR 模型

DeepSeek-OCR-2 适合处理图像页，所以你的流程应该是：

```text
PDF → 渲染成图片 → OCR → 合并
```

### 2. OCR 和 LLM 分开

* OCR 只负责“看”
* Qwen 只负责“改文”

这样最稳，也方便以后替换模型。

### 3. 模型服务化

* Qwen 一个服务
* DeepSeek-OCR-2 一个 vLLM 服务
* 业务层不要直接把模型塞进 FastAPI 主进程

## 最小可用模块

你第一版只需要这 6 个模块：

* `api/`：上传、查进度、下载
* `agent/`：LangChain 1.0 agent
* `tools/`：PDF、OCR、Markdown 工具
* `services/`：Qwen 和 OCR 客户端
* `storage/`：文件落盘
* `jobs/`：任务状态管理

## 一句话版本

> **FastAPI 接任务，LangChain 1.0 做编排，DeepSeek-OCR-2 + vLLM 做逐页 OCR，Qwen 做 Markdown 清洗，最后输出 .md。**

如果你要，我下一步可以直接给你一版**项目目录结构 + 每个 tool 的输入输出定义**。

[1]: https://docs.langchain.com/oss/python/releases/langchain-v1?utm_source=chatgpt.com "What's new in LangChain v1"
[2]: https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope?utm_source=chatgpt.com "OpenAI Chat接口兼容 - 阿里云文档"
[3]: https://github.com/deepseek-ai/DeepSeek-OCR-2?utm_source=chatgpt.com "DeepSeek-OCR 2: Visual Causal Flow"
[4]: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/?utm_source=chatgpt.com "OpenAI-Compatible Server - vLLM"



# 文件说明

1. test_deepseek_ocr_vllm.py   这个是识别一张图的 脚本   用vllm服务
2. "C:\Users\gjh\PycharmProjects\langchain_Qwen_deepseek_ocr_agent\run_dpsk_ocr_pdf.py"  这个是官方仓库中的识别一个pdf的脚本   不用vllm脚本  是直接调用模型
3. "C:\Users\gjh\PycharmProjects\langchain_Qwen_deepseek_ocr_agent\test_vllm_run_dpsk_ocr_pdf.py" 这个是用vllm服务的 识别pdf的脚本
4. "C:\Users\gjh\PycharmProjects\langchain_Qwen_deepseek_ocr_agent\config.py" 这个是test_vllm_run_dpsk_ocr_pdf脚本的参数
5. "C:\Users\gjh\PycharmProjects\langchain_Qwen_deepseek_ocr_agent\ocr_pdf_service.py"和"C:\Users\gjh\PycharmProjects\langchain_Qwen_deepseek_ocr_agent\api.py"这个是test_vllm_run_dpsk_ocr_pdf包装成api  
6. 这些脚本在C:\Users\gjh\.cache\modelscope\hub\models\deepseek-ai\DeepSeek-OCR这个里面也有一份  是从这个里面复制过来的   这个目录是 modelscope下载的deepseek——ocr模型文件


7. 现在把 run_dpsk_ocr_pdf.py 和test_deepseek_ocr_vllm.py放进other里面了
8. api.py 这个是把ocr包装成api服务
9. config  这个是ocr的配置文件 用于本地测试api
10. ocr_pdf_service.py 这个也是包装ocr为api的文件    
11. agent 是 在windows里的基于langchain的agent   主要是初始化agent
12. services文件夹 里面是 agent 调用wsl里面运行的ocr api  
13. tools 是把这个包装成一个ocr工具
14. app 是fastapi的入口文件

# 整体结构
1. ocr服务包装成api 通过vllm运行在wsl 上
2. windows上基于langchain的agent  主模型是qwen3-8b    然后  如果上传pdf就 调用ocr 这个api来 识别pdf   输出md   到agent  然后 返回qwen3-8b
3. 启动方式  先在wsl上启动   ocr api  入口是api.py这个  然后再启动app.py 这个agent服务   即可  注意 api.py这个需要python -m uvicorn api:app --host 0.0.0.0 --port 8001  注意端口不要冲突