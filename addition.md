

import os, io, json, time
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ---------------------------
# Helpers (safe at import)
# ---------------------------
def pil_to_rgb_bytes(image_path):
    img = Image.open(image_path).convert("RGB")
    max_side = 2048
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()

def ask_qwen_bbox(llm, image_path, categories):
    cats = ", ".join(categories)
    prompt = (
        "The image contains a visible coordinate grid with labeled rows and columns. "
        "Use it only for reasoning — report all bounding boxes in JSON format "
        f"for categories: {cats}. Each box_2d = [ymin, xmin, ymax, xmax]."
    )
    image_bytes = pil_to_rgb_bytes(image_path)
    max_tokens = min(256 + 32 * len(categories), 1024)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    request = {"prompt": prompt, "multi_modal_data": {"image": image_bytes}}
    outputs = llm.generate([request], sampling_params)
    return prompt, outputs[0].outputs[0].text.strip()

def run_bbox_for_all(llm, metadata_json_path, output_json_dir):
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    os.makedirs(output_json_dir, exist_ok=True)
    for item in tqdm(metadata, desc="Running Qwen3-VL-MoE via vLLM"):
        image_id_str = str(item.get("image_id", "unknown"))
        image_path = item.get("image_path", "")
        if not os.path.exists(image_path):
            print(f"[!] Missing image: {image_path}")
            continue
        categories = item.get("key_words", [])
        if not categories:
            print(f"[!] No key_words for {image_id_str}")
            continue
        prompt, qwen_response = ask_qwen_bbox(llm, image_path, categories)
        item_out = OrderedDict(item)
        item_out["prompt"] = prompt
        item_out["qwen_response"] = qwen_response
        out_path = os.path.join(output_json_dir, f"{image_id_str}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(item_out, f, indent=2, ensure_ascii=False)
        print(f"[✓] Saved: {out_path}")
        time.sleep(1.5)

# ---------------------------
# Main (everything heavy here)
# ---------------------------
if __name__ == "__main__":
    import torch.multiprocessing as mp

    # Use spawn BEFORE creating any multiprocessing objects
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    # ENV
    os.environ["HF_HOME"] = "/aiau010_scratch/hungnh/hf_cache"
    os.environ["VLLM_CACHE_DIR"] = "/aiau010_scratch/hungnh/hf_cache"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    LOCAL_MODEL_PATH = (
        "/aiau010_scratch/hungnh/hf_cache/hub/"
        "models--Qwen--Qwen3-VL-235B-A22B-Instruct/"
        "snapshots/a85c31584af63f6e55c91d93bda2ae78600e1a77"
    )

    print(f"[INFO] Using local model from: {LOCAL_MODEL_PATH}")
    print(f"[INFO] Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Processor (loads tokenizer + preprocessor from your snapshot)
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

    # vLLM engine (no deprecated args)
    llm = LLM(
        model=LOCAL_MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=4,
        max_model_len=1024,
        # trust_remote_code is ignored by vLLM but harmless
        trust_remote_code=True,
    )

    # Paths
    metadata_file = "datasets/coco_grid/metadata.json"
    output_dir = "eval_results/qwen3_vl_moe_vllm/coco_grid"

    run_bbox_for_all(llm, metadata_file, output_dir)