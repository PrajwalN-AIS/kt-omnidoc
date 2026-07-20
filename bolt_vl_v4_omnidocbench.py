import os
import json
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline
from tqdm import tqdm

os.environ["HF_HOME"] = "/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/.cache/huggingface"

MODEL_NAME  = "aisquared/bolt-vl-v4"
CACHE_PATH  = "/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/.cache/huggingface"
GT_JSON     = "/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/OmniDocBench/data/OmniDocBench.json"
IMG_DIR     = "/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/OmniDocBench/data/images"
OUTPUT_DIR  = "/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/OmniDocBench/predictions_bolt_vl_v4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = r"""Convert this document page image to Markdown format. /no_think

Rules:
1. Text: Reproduce all text accurately. Use Markdown headings (#, ##, ###) for titles and section headers.
2. Math formulas: Use $...$ for inline formulas and $$...$$ for display/block formulas. Write formulas in LaTeX.
3. Tables: Convert tables to HTML format wrapped in <table>...</table>.
4. Figures/images: Skip figure content entirely. Do not describe images.
5. Reading order: Output content in natural reading order (top to bottom, left to right).
6. Output only the Markdown content. No explanations, no preamble, no commentary.
"""

# Load model
print("Loading model:", MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16,
    cache_dir=CACHE_PATH,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_PATH, trust_remote_code=True)
print("Device map:", getattr(model, "hf_device_map", "N/A"))

pipe = pipeline(
    task="image-text-to-text",
    model=model,
    processor=processor
)
print("Pipeline ready.")

# Load GT JSON to get image list
with open(GT_JSON) as f:
    pages = json.load(f)
print(f"Total pages: {len(pages)}")

def strip_thinking(text):
    """Remove thinking block (chain-of-thought).
    Take everything after </think> if present, otherwise return as-is.
    """
    if '</think>' in text:
        text = text.split('</think>', 1)[-1]
    return text.strip()

def run_one(image_path):
    image = Image.open(image_path).convert("RGB")
    out = pipe(
        text=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": PROMPT}
                ]
            }
        ],
        max_new_tokens=4096,
        do_sample=False,
    )
    result = out[0]["generated_text"][-1]["content"]
    return strip_thinking(result)


done = 0
skipped = 0
failed = 0

for page in tqdm(pages, desc="Inference"):
    img_name = page["page_info"]["image_path"]
    img_path = os.path.join(IMG_DIR, img_name)
    basename = os.path.splitext(img_name)[0]
    md_path = os.path.join(OUTPUT_DIR, f"{basename}.md")

    if os.path.exists(md_path):
        skipped += 1
        continue

    if not os.path.exists(img_path):
        print(f"[WARN] Image not found: {img_path}", flush=True)
        failed += 1
        continue

    try:
        result = run_one(img_path)
        if not result or result.strip() == "":
            raise ValueError("Empty output")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result)
        done += 1
    except Exception as e:
        print(f"[ERROR] {img_name}: {e}", flush=True)
        failed += 1

    if (done + failed) % 50 == 0 and (done + failed) > 0:
        torch.cuda.empty_cache()
        print(f"Progress: {done} done, {skipped} skipped, {failed} failed", flush=True)

print(f"\nFinal: {done} done, {skipped} skipped, {failed} failed")
print(f"Predictions saved to: {OUTPUT_DIR}")
