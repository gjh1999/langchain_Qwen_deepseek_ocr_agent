# ocr_pdf_service.py
import os
import fitz
import img2pdf
import io
import re
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

from config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS

if torch.version.cuda == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llm = None
sampling_params = None


def init_model():
    global llm, sampling_params
    if llm is not None:
        return

    llm = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        max_num_seqs=MAX_CONCURRENCY,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )


def pdf_to_images_high_quality(pdf_path, dpi=300, image_format="PNG"):
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        if image_format.upper() != "PNG" and img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background

        images.append(img)

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path):
    if not pil_images:
        return

    image_bytes_list = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        image_bytes_list.append(img_buffer.getvalue())

    pdf_bytes = img2pdf.convert(image_bytes_list)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for m in matches:
        if "<|ref|>image<|/ref|>" in m[0]:
            matches_image.append(m[0])
        else:
            matches_other.append(m[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception:
        return None
    return label_type, cor_list


def draw_bounding_boxes(image, refs, jdx, output_dir):#cropped.save(f"{output_dir}/images/{jdx}_{img_idx}.jpg")
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()
    img_idx = 0

    for ref in refs:
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if not result:
                continue

            label_type, points_list = result
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(0, 255),
            )
            color_a = color + (20,)

            for points in points_list:
                x1, y1, x2, y2 = points
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)

                if label_type == "image":
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        #cropped.save(f"{output_dir}/images/{jdx}_{img_idx}.jpg")
                        images_dir = os.path.join(output_dir, "images")
                        os.makedirs(images_dir, exist_ok=True)

                        save_path = os.path.join(images_dir, f"{jdx}_{img_idx}.jpg")
                        cropped.save(save_path)
                    except Exception:
                        pass
                    img_idx += 1

                try:
                    if label_type == "title":
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                        draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                    else:
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                    text_x = x1
                    text_y = max(0, y1 - 15)

                    text_bbox = draw.textbbox((0, 0), label_type, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    draw.rectangle(
                        [text_x, text_y, text_x + text_width, text_y + text_height],
                        fill=(255, 255, 255, 30),
                    )
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
                except Exception:
                    pass
        except Exception:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, output_dir):
    return draw_bounding_boxes(image, ref_texts, jdx, output_dir)


def process_single_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return {
        "prompt": PROMPT,
        "multi_modal_data": {"image": image},
    }


def ocr_pdf(pdf_path: str, output_dir: str):
    #init_model()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    images = pdf_to_images_high_quality(pdf_path, dpi=300)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(
            tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc="Pre-processed images",
            )
        )

    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params,
    )

    pdf_name = Path(pdf_path).stem
    mmd_det_path = os.path.join(output_dir, f"{pdf_name}_det.mmd")
    mmd_path = os.path.join(output_dir, f"{pdf_name}.mmd")
    pdf_out_path = os.path.join(output_dir, f"{pdf_name}_layouts.pdf")

    contents_det = ""
    contents = ""
    draw_images = []

    for jdx, (output, img) in enumerate(zip(outputs_list, images)):
        content = output.outputs[0].text

        if "<｜end▁of▁sentence｜>" in content:
            content = content.replace("<｜end▁of▁sentence｜>", "")
        elif SKIP_REPEAT:
            continue

        page_num = "\n<--- Page Split --->"
        contents_det += content + f"\n{page_num}\n"

        image_draw = img.copy()
        matches_ref, matches_images, matches_other = re_match(content)
        result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_dir)
        draw_images.append(result_image)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f"![](images/{jdx}_{idx}.jpg)\n")

        for a_match_other in matches_other:
            content = (
                content.replace(a_match_other, "")
                .replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
                .replace("\n\n\n\n", "\n\n")
                .replace("\n\n\n", "\n\n")
            )

        contents += content + f"\n{page_num}\n"

    with open(mmd_det_path, "w", encoding="utf-8") as f:
        f.write(contents_det)

    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(contents)

    if draw_images:
        try:
            pil_to_pdf_img2pdf(draw_images, pdf_out_path)
        except Exception:
            pass

    return {
        "mmd_path": mmd_path,
        "mmd_det_path": mmd_det_path,
        "layout_pdf_path": pdf_out_path if os.path.exists(pdf_out_path) else None,
        "markdown": contents,
        "markdown_det": contents_det,
    }