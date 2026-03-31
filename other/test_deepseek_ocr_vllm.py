from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image


def main():
    print("🚀 starting...")

    llm = LLM(
        model="/mnt/c/Users/gjh/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    print("✅ model loaded")

    image_path = "/mnt/c/Users/gjh/Desktop/test.png"
    image = Image.open(image_path).convert("RGB")

    prompt = "<image>\n Free OCR."

    model_input = [{
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }]

    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )

    outputs = llm.generate(model_input, sampling_param)

    for out in outputs:
        print("\n===== OCR RESULT =====\n")
        print(out.outputs[0].text)


if __name__ == "__main__":
    main()
