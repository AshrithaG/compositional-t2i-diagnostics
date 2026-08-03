"""
BEAM paper experiment: does a modern instruction-tuned VLM judge share
BLIP-vqa-base's layout blindness?

Scores condition folders with Qwen2.5-VL-Instruct using the SAME question
format as the BLIP diagnostic suite (object presence x2 + relation question),
so the three judges (BLIP, Qwen, OWLv2-geometric) are directly comparable on
the conditions where ground truth is known by construction (box_swap flipped
~83% of layouts; chain_ablate broke object presence).

Requires: pip install qwen-vl-utils   (transformers>=4.49 already supports Qwen2.5-VL)

Usage:
  python exp_vlm_judge.py --dirs r3_outputs/perturb_none_spatial \
      r3_outputs/perturb_box_swap_spatial r3_outputs/oracle2_minimal_alien_spatial_val \
      r3_outputs/perturb_chain_ablate_spatial
Outputs: vlm_judge/<dirname>.json + vlm_judge/summary.md
"""

import argparse, json, os
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", nargs="+", required=True)
parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--split", default="val")
parser.add_argument("--out_dir", default="vlm_judge")
parser.add_argument("--max_prompts", type=int, default=-1)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.bfloat16).to(device).eval()
processor = AutoProcessor.from_pretrained(args.model)
print(f"{args.model} loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# same prompt parsing as the BLIP suite (vm_blip_eval.py) for comparability
SPATIAL_PATTERNS = [
    "on the left of", "on the right of", "on top of", "on the bottom of",
    "on the side of", "above", "below", "beneath", "under", "near", "next to",
]

def parse_spatial_prompt(prompt):
    pl = prompt.lower()
    for pattern in SPATIAL_PATTERNS:
        if pattern in pl:
            parts = pl.split(pattern)
            if len(parts) == 2:
                return (parts[0].strip().lstrip("a ").strip(), pattern,
                        parts[1].strip().lstrip("a ").strip())
    return None, None, None


@torch.inference_mode()
def ask(image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",
         "text": question + " Answer with exactly one word: yes or no."}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    ans = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0].strip().lower()
    if ans.startswith("yes"):
        return 1.0, ans
    if ans.startswith("no"):
        return 0.0, ans
    return 0.5, ans


with open(os.path.join(args.prompt_dir, f"spatial_{args.split}.txt")) as f:
    prompts = [l.strip() for l in f if l.strip()]
if args.max_prompts > 0:
    prompts = prompts[:args.max_prompts]

rows = []
for d in args.dirs:
    name = os.path.basename(d.rstrip("/"))
    out_file = os.path.join(args.out_dir, f"{name}.json")
    if os.path.exists(out_file):
        print(f"skip {name} (done)")
        rows.append((name, json.load(open(out_file))["summary"]))
        continue
    per_image = []
    for idx, prompt in enumerate(tqdm(prompts, desc=name)):
        img_path = os.path.join(d, f"{idx}.png")
        o1, rel, o2 = parse_spatial_prompt(prompt)
        if not os.path.exists(img_path) or o1 is None:
            continue
        img = Image.open(img_path).convert("RGB")
        s1, a1 = ask(img, f"Is there a {o1} in the image?")
        s2, a2 = ask(img, f"Is there a {o2} in the image?")
        s3, a3 = ask(img, f"Is the {o1} {rel} the {o2}?")
        per_image.append({"index": idx, "prompt": prompt, "relation": rel,
                          "obj1_present": s1, "obj2_present": s2,
                          "spatial_correct": s3, "answers": [a1, a2, a3]})
    import numpy as np
    summary = {
        "n": len(per_image),
        "obj1_presence": float(np.mean([r["obj1_present"] for r in per_image])),
        "obj2_presence": float(np.mean([r["obj2_present"] for r in per_image])),
        "spatial_accuracy": float(np.mean([r["spatial_correct"] for r in per_image])),
    }
    json.dump({"dir": d, "judge": args.model, "summary": summary,
               "per_image": per_image}, open(out_file, "w"), indent=2)
    rows.append((name, summary))
    print(name, summary)

lines = ["| condition | obj1 | obj2 | spatial (Qwen) |", "|---|---|---|---|"]
for name, s in rows:
    lines.append(f"| {name} | {s['obj1_presence']:.4f} | {s['obj2_presence']:.4f} "
                 f"| {s['spatial_accuracy']:.4f} |")
open(os.path.join(args.out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
