这段代码本质上是一个 **“把 PDF 拆成图片 → 用 DeepSeek-OCR 跑每页 → 生成 markdown 和带框的可视化 PDF”** 的批处理脚本。核心入口是 `run_ocr(INPUT_PATH, OUTPUT_PATH)`。

我按模块给你拆。

## 1. 这段代码整体在干什么

输入：

* 一个 PDF 文件
* 一个输出目录

输出：

* `xxx.mmd`：清洗后的 Markdown
* `xxx_det.mmd`：带检测标记原始结果
* `xxx_layouts.pdf`：画了框的版面可视化 PDF
* `images/`：从页面里裁出来的图片块

也就是说，它不是单纯 OCR 文字，而是顺便把模型返回的版面标注信息也利用起来了。

---

## 2. 全局初始化部分

前面这几块主要是在做 **环境准备和模型单例缓存**：

* `_global_llm`、`_global_sampling_params`
* `get_model()`
* `init_model()`

作用是：

* 第一次调用时真正加载 vLLM 模型
* 后面再次调用 `run_ocr()` 时，不重复加载模型

这很重要，因为 OCR 大模型加载很慢、很占显存。这里用了“全局缓存”的思路。

### `get_model()`

逻辑很简单：

* 如果 `_global_llm is None`

  * 就调用 `init_model()`
* 否则直接返回已经加载好的模型

这就是一个典型的懒加载单例。

---

## 3. 环境变量部分

这里有几行：

* `TRITON_PTXAS_PATH`
* `VLLM_USE_V1 = 0`
* `CUDA_VISIBLE_DEVICES = 0`

意思大概是：

### `TRITON_PTXAS_PATH`

如果 CUDA 版本是 11.8，就手动告诉 Triton 去哪里找 `ptxas`。
这个通常是为了避免 Triton 编译 CUDA kernel 时找不到工具链。

### `VLLM_USE_V1 = 0`

强制 vLLM 不用 V1 路径，可能是为了兼容你当前这个 DeepSeek-OCR 模型实现。

### `CUDA_VISIBLE_DEVICES = 0`

只让程序看见第 0 块 GPU。

注意一点：
这行会**强制锁定 GPU 0**，如果以后你想多卡、或者想换卡，这里要改。

---

## 4. 配置导入

```python
from config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE
```

说明这个脚本依赖外部 `config.py`。
关键配置应该包括：

* `MODEL_PATH`：模型路径
* `PROMPT`：给 OCR 模型的提示词
* `SKIP_REPEAT`：遇到异常输出时是否跳过
* `MAX_CONCURRENCY`：vLLM 同时处理的请求数
* `NUM_WORKERS`：预处理线程数
* `CROP_MODE`：这里虽然导入了，但我看这份代码里**没用上**

所以：
`CROP_MODE` 目前像是遗留配置。

---

## 5. 模型初始化 `init_model()`

这里真正创建了 vLLM 的 `LLM` 和 `SamplingParams`。

### `LLM(...)`

重点参数：

* `model=MODEL_PATH`
  指向你的 DeepSeek-OCR 模型目录

* `enable_prefix_caching=False`
  不开 prefix cache

* `mm_processor_cache_gb=0`
  多模态处理缓存设为 0

* `logits_processors=[NGramPerReqLogitsProcessor]`
  给输出加了 n-gram 重复抑制处理器

* `max_num_seqs=MAX_CONCURRENCY`
  最大并发序列数

### `SamplingParams(...)`

这里虽然温度是 0，但仍然配了重复控制参数：

* `temperature=0.0`
* `max_tokens=8192`
* `extra_args`

  * `ngram_size=30`
  * `window_size=90`
  * `whitelist_token_ids={128821, 128822}`
* `skip_special_tokens=False`

这里的设计意图很明显：
**尽量让 OCR 输出稳定、少重复、保留特殊标记。**

因为后面它还要靠这些特殊标记来解析版面信息。
如果 `skip_special_tokens=True`，有些结构标记可能直接没了。

---

## 6. `process_single_image(image)`

这个函数很简单，就是把一张 PIL 图像包装成 vLLM 多模态输入格式：

```python
{
    "prompt": PROMPT,
    "multi_modal_data": {"image": image},
}
```

同时确保图像是 RGB。
也就是说，每一页 PDF 最终都会变成一个这样的推理请求。

---

## 7. PDF 转图片 `pdf_to_images_high_quality()`

这个函数是 PDF 前处理的核心。

它做了这些事：

1. 用 `fitz.open(pdf_path)` 打开 PDF
2. 按 `dpi=144` 渲染每一页
3. `zoom = dpi / 72.0`
4. `page.get_pixmap(matrix=matrix, alpha=False)`
5. 转成 PIL Image
6. 把所有页放进 `images` 列表返回

### 你可以这么理解

它把 PDF 每一页渲染成图片，再交给 OCR 模型。

### 为什么是 144 DPI

PDF 默认逻辑分辨率一般按 72 DPI 算。
这里乘 2，等于渲染得更清晰，适合 OCR。

### 一个细节

函数名里写了 `high_quality`，本质上就是更高 DPI 渲染，不是什么超分。

---

## 8. `pil_to_pdf_img2pdf()`

这个函数把一组 PIL 图片重新合成 PDF。

用途是：

* 后面每页会画上检测框
* 再把这些“画过框的页面图”拼成一个新的 PDF
* 保存成 `_layouts.pdf`

流程：

1. 把每张 PIL 图转成 JPEG bytes
2. 用 `img2pdf.convert(...)` 合并
3. 写入输出文件

所以 `xxx_layouts.pdf` 不是原始 PDF，而是**渲染图层版的可视化 PDF**。

---

## 9. 正则解析 `re_match(text)`

这个函数非常关键。
它负责从模型输出文本里提取类似这种结构：

```text
<|ref|> ... <|/ref|><|det|> ... <|/det|>
```

它用的正则是：

```python
pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
```

然后把匹配结果分成两类：

* `mathes_image`：`<|ref|>image<|/ref|>` 的块
* `mathes_other`：其他块

也就是说，模型输出里不只是文字，还有“版面区域描述”。
这个函数把它们拆出来，供后续：

* 裁图
* 画框
* 清洗 markdown

使用。

---

## 10. 坐标解析 `extract_coordinates_and_label()`

这里把正则匹配出来的内容继续解析成：

* `label_type`
* `cor_list`

其中：

* `label_type = ref_text[1]`
* `cor_list = eval(ref_text[2])`

### 重点提醒

这里用了 `eval()`。
这很方便，但也有风险：

如果模型输出内容不受信任，`eval()` 可能执行恶意代码。

不过在你这个场景里，通常模型输出应该是类似：

```python
[[x1,y1,x2,y2], ...]
```

所以作者图省事直接 `eval()` 了。

更安全的做法是：

* 用 `ast.literal_eval()` 替代 `eval()`

这会更稳。

---

## 11. 画框和裁图 `draw_bounding_boxes()`

这个函数是“版面可视化”的核心。

它做了几件事：

### 第一件：准备画图对象

* 复制原图
* 创建 `ImageDraw`
* 创建透明 overlay
* 准备字体

### 第二件：确保 `images/` 目录存在

```python
images_dir = os.path.join(output_path, "images")
os.makedirs(images_dir, exist_ok=True)
```

### 第三件：遍历每个检测块

对每个 `ref`：

1. 解析出 `label_type` 和坐标列表
2. 随机生成颜色
3. 把 0~999 的归一化坐标映射到真实图像宽高
4. 如果 `label_type == 'image'`

   * 把这个区域裁出来
   * 保存成 `images/{页号}_{图号}.jpg`
5. 再把这个框画到页面图上

### 这里坐标换算很重要

```python
x1 = int(x1 / 999 * image_width)
```

说明模型给的坐标不是像素值，而是 **0~999 范围的归一化值**。
所以这里恢复成实际像素坐标。

### 画框规则

* `title` 用更粗的框：`width=4`
* 其他类型用 `width=2`

这说明作者想让标题区域更显眼。

---

## 12. `process_image_with_refs()`

这个函数只是个薄封装：

```python
return draw_bounding_boxes(image, ref_texts, jdx, output_path)
```

本质没加新逻辑，只是为了让主流程更清晰。

---

## 13. 主流程 `run_ocr(INPUT_PATH, OUTPUT_PATH)`

这是整个脚本真正的工作入口。

我给你顺一下执行顺序。

### 第一步：拿模型

```python
llm, sampling_params = get_model()
```

### 第二步：创建输出目录

```python
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
```

### 第三步：PDF 转图片

```python
images = pdf_to_images_high_quality(INPUT_PATH)
```

### 第四步：并行做图片预处理

```python
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    batch_inputs = list(tqdm(
        executor.map(process_single_image, images),
        total=len(images),
        desc="Pre-processed images"
    ))
```

这里并没有并行推理，只是并行把 PIL 图像包装成多模态请求。

### 第五步：统一送入 vLLM 批量推理

```python
outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
```

说明它是一次性把整份 PDF 的所有页都喂给模型。
这比一页页单独调更高效。

---

## 14. 输出文件命名

根据输入 PDF 名字，生成这些文件：

* `_det.mmd`
* `.mmd`
* `_layouts.pdf`

例如输入 `test.pdf`，输出会是：

* `test_det.mmd`
* `test.mmd`
* `test_layouts.pdf`

这套命名很清楚。

---

## 15. 每页后处理逻辑

这一段是最关键的“后处理”。

对每一页的模型输出：

### 1）先拿到文本

```python
content = output.outputs[0].text
```

### 2）处理结束标记

如果包含：

```python
'<｜end▁of▁sentence｜>'
```

就删掉它。

如果没有这个标记：

* 且 `SKIP_REPEAT` 为真
* 就 `continue`，直接跳过这一页

#### 这说明什么？

作者怀疑某些输出会出现“重复生成/异常未结束”的情况，
于是用 EOS 标记做一个简陋的质量判断。

这个判断比较硬，但确实实用。

---

## 16. det 版和 clean 版 markdown 的区别

### `contents_det`

直接保留原始内容，加分页标记：

```python
contents_det += content + f'\n{page_num}\n'
```

所以 `_det.mmd` 更像“原始识别结果+检测信息备份”。

### `contents`

会做清洗：

* 图片块替换成 markdown 图片链接
* 其他检测标记删掉
* 一些 LaTeX 符号替换
* 多余空行压缩

所以 `.mmd` 才是真正适合后续使用的 markdown。

---

## 17. 图片块替换逻辑

对 `matches_images`：

```python
content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')
```

意思是：

* 模型标记出来的 `image` 区块
* 被替换成 markdown 里的图片引用

例如：

```markdown
![](images/0_0.jpg)
```

这样最终 `.mmd` 文件就能把裁下来的图嵌进文档结构里。
这是这段代码很有价值的地方。

---

## 18. 非图片检测块清洗逻辑

对 `mathes_other`：

```python
content = content.replace(a_match_other, '')
```

直接删掉所有非图片检测标记。

然后还做了几个符号替换：

* `\coloneqq` → `:=`
* `\eqqcolon` → `=:`

以及把多余空行压缩。

说明作者想得到一个更干净、更适合阅读或继续处理的 markdown。

---

## 19. 最终写文件

最后三件事：

1. 写 `_det.mmd`
2. 写 `.mmd`
3. 把所有画了框的页面拼成 `_layouts.pdf`

整个 OCR 流程就结束了。

---

## 20. `if __name__ == "__main__"` 入口

这部分说明这个文件是可以直接命令行运行的：

```bash
python test_vllm_run_dpsk_ocr_pdf.py --input xxx.pdf --output out_dir
```

参数只有两个：

* `--input`
* `--output`

所以它目前是一个**命令行批处理脚本**，还不是 API。

---

## 21. 这份代码的优点

我觉得主要有这几个优点：

### 1）流程完整

不是只做 OCR，而是：

* PDF 渲染
* 批量推理
* 检测块解析
* 图片裁剪
* markdown 重建
* 可视化 PDF 输出

一条链跑通了。

### 2）模型只初始化一次

`get_model()` 的单例缓存很实用。
这对你后面改成 FastAPI 非常重要。

### 3）输出结果分层

* 原始 `_det.mmd`
* 清洗后 `.mmd`
* 带框 `_layouts.pdf`

方便调试和验收。

### 4）批量推理

`llm.generate(batch_inputs, ...)` 一次处理多页，比每页一次调用更合理。

---

## 22. 这份代码的问题和隐患

### 1）`eval()` 不安全

这里最明显：

```python
cor_list = eval(ref_text[2])
```

建议改成：

```python
import ast
cor_list = ast.literal_eval(ref_text[2])
```

### 2）异常处理比较粗

很多地方只是 `print(...)`，没有抛错。
结果可能是：

* 程序看起来跑完了
* 但中间丢了图、丢了框、丢了页

排查会比较麻烦。

### 3）`SKIP_REPEAT` 的逻辑比较硬

如果某页输出里没有 EOS 标记，就可能被跳过。
这样会导致：

* 页数缺失
* 最终 markdown 少内容

### 4）`CROP_MODE` 未使用

说明代码可能有历史遗留，结构还没完全收敛。

### 5）随机颜色不利于复现

每次画框颜色都不同：

```python
np.random.randint(...)
```

调试时不够稳定。

### 6）显存和内存压力可能偏大

它会：

* 一次把整份 PDF 全部渲染成图片
* 再一次性喂给模型

如果 PDF 很长，容易炸内存或显存。

---

## 23. 你可以把它理解成哪种架构

这段代码其实就是四层：

### 第 1 层：输入层

* 接 PDF 路径
* 接输出目录

### 第 2 层：预处理层

* PDF → PIL images
* image → vLLM 输入格式

### 第 3 层：推理层

* vLLM 批量调用 DeepSeek-OCR

### 第 4 层：后处理层

* 提取 ref/det 标记
* 裁图
* 画框
* 转 markdown
* 保存文件

这个分层是比较清楚的。

---

## 24. 对你当前项目的意义

如果你后面要把它包装成 FastAPI 接口，那最关键的点就是：

* `get_model()` 保留，别每次请求都重新加载模型
* `run_ocr()` 作为核心业务函数保留
* FastAPI 只负责：

  * 接收上传的 PDF
  * 保存临时文件
  * 调用 `run_ocr()`
  * 返回 `.mmd` 内容或输出路径

也就是说，这个脚本已经非常接近你要的“OCR API 核心逻辑”了。
它缺的主要不是算法，而是服务化外壳。

---

## 25. 一句话总结

这段代码是一个 **基于 vLLM + DeepSeek-OCR 的 PDF 转 Markdown 脚本**，它不只是识别文字，还会解析版面标注、裁剪图片、生成带检测框的可视化 PDF，并把最终结果整理成 markdown；整体思路是对的，最值得改的地方是 `eval()`、异常处理，以及长 PDF 下的资源控制。

你要的话，我下一条可以直接继续给你做两件事里的一个：
**1）按函数关系给你画执行流程图**，或者 **2）告诉你怎么最小改动把这份脚本包成 FastAPI 接口**。
