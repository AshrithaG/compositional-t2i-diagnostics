"""
Shared utilities for the submission-strengthening experiments.
Extracted from r3_got_unique_mods.py so all exp_*.py scripts share one
implementation of parsing, verification, and the two-phase generation loop.

Run every exp_*.py from the GoT-R1 repo root (imports src.processor).
"""

import torch
import numpy as np
import os, json, random, re
from PIL import Image

# ─────────────────────────────────────────────
# Parsing & verification (identical to r3_got_unique_mods.py)
# ─────────────────────────────────────────────

SPATIAL_RELATIONS = {
    "on the left of": ("left", lambda c1, c2: c1[0] < c2[0]),
    "on the right of": ("right", lambda c1, c2: c1[0] > c2[0]),
    "on the top of": ("top", lambda c1, c2: c1[1] < c2[1]),
    "on the bottom of": ("bottom", lambda c1, c2: c1[1] > c2[1]),
    "next to": ("next_to", lambda c1, c2: True),
    "near": ("near", lambda c1, c2: True),
    "on side of": ("side", lambda c1, c2: True),
}

ASYMMETRIC_RELATIONS = ["on the left of", "on the right of", "on the top of", "on the bottom of"]

# semantically-equivalent inverse used by the mention-order control
INVERSE_RELATION = {
    "on the left of": "on the right of",
    "on the right of": "on the left of",
    "on the top of": "on the bottom of",
    "on the bottom of": "on the top of",
}

COLORS = {
    "red", "blue", "green", "brown", "black", "white", "yellow", "gold",
    "gray", "grey", "pink", "purple", "silver", "orange",
}

BOX_RE = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
OBJ_RE = r'<\|obj_start\|>(.*?)<\|obj_end\|>'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_boxes_from_chain(chain_text):
    objects = re.findall(OBJ_RE, chain_text)
    boxes = re.findall(BOX_RE, chain_text)
    parsed = []
    for obj, box in zip(objects, boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        parsed.append({
            "object": obj.strip(),
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
        })
    return parsed


def parse_spatial_prompt(prompt):
    relations = sorted(SPATIAL_RELATIONS.keys(), key=len, reverse=True)
    for rel in relations:
        m = re.match(rf"^(.+?)\s+{re.escape(rel)}\s+(.+?)$", prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip(), rel, m.group(2).strip()
    return None, None, None


def parse_color_prompt(prompt):
    m = re.match(r"^a (\w+) (.+?) and a (\w+) (.+?)$", prompt, re.IGNORECASE)
    if m and m.group(1).lower() in COLORS and m.group(3).lower() in COLORS:
        return [(m.group(1).lower(), m.group(2)), (m.group(3).lower(), m.group(4))]
    return None


def verify_chain(chain_text, prompt, subset):
    parsed = parse_boxes_from_chain(chain_text)
    if len(parsed) < 2:
        return False, {"reason": "fewer_than_2_objects", "parsed": len(parsed)}

    if subset == "spatial":
        obj1_str, rel, obj2_str = parse_spatial_prompt(prompt)
        if rel is None:
            return True, {"reason": "unparseable_prompt"}
        _, check_fn = SPATIAL_RELATIONS[rel]
        c1, c2 = parsed[0]["center"], parsed[1]["center"]
        ok = check_fn(c1, c2)
        return ok, {"reason": "spatial_check", "relation": rel,
                    "centers": [c1, c2],
                    "objects_in_chain": [p["object"] for p in parsed],
                    "spatial_correct": ok}

    if subset == "color":
        expected = parse_color_prompt(prompt)
        if expected is None:
            return True, {"reason": "unparseable_prompt"}
        chain_lower = chain_text.lower()
        details, all_bound = [], True
        for color, obj in expected:
            bound = f"{color} {obj}" in chain_lower or f"{color}" in chain_lower
            details.append({"color": color, "object": obj, "found": bound})
            all_bound = all_bound and bound
        return all_bound, {"reason": "color_check", "bindings": details}

    return True, {"reason": "no_check"}


# ─────────────────────────────────────────────
# Model bundle + two-phase generation
# ─────────────────────────────────────────────

class GotR1:
    """Wraps model + processor with the same generation code as r3 scripts."""

    def __init__(self, ckpt_path="ckpts/GoT-R1-1B", cfg_weight=5.0, image_temperature=1.0):
        from src.processor.processor import get_processor
        from transformers import AutoModelForCausalLM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = get_processor(ckpt_path)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.float16).to(self.device).eval()
        self.image_start_token = self.tokenizer.encode("<begin_of_image>")[-1]
        self.cfg_weight = cfg_weight
        self.image_temperature = image_temperature
        print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    def format_prompt(self, raw_prompt):
        full = f"Follow the caption to generate an image through a chain of thought process: {raw_prompt}"
        conversation = [{"role": "User", "content": full}, {"role": "Assistant", "content": ""}]
        sft = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation, sft_format=self.processor.sft_format, system_prompt="")
        return torch.LongTensor(self.tokenizer.encode(sft)).to(self.device)

    @torch.inference_mode()
    def generate_reasoning_chain(self, prompt_ids, text_temperature=1.0):
        model, device = self.model, self.device
        inputs_embeds = model.language_model.get_input_embeddings()(prompt_ids.unsqueeze(0)).to(device)
        generation_tokens, past_key_values = [], None
        for _ in range(600):
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
            logits = model.language_model.lm_head(outputs.last_hidden_state[:, -1, :])
            probs = torch.softmax(logits / text_temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            inputs_embeds = model.language_model.get_input_embeddings()(next_token).to(device)
            generation_tokens.append(next_token)
            past_key_values = outputs.past_key_values
            if next_token.item() == self.image_start_token:
                break
        if generation_tokens[-1].item() != self.image_start_token:
            generation_tokens.append(torch.tensor(self.image_start_token).to(device))
        chain_ids = torch.tensor(generation_tokens).to(torch.int).to(device)
        return chain_ids, self.tokenizer.decode(chain_ids)

    @torch.inference_mode()
    def generate_image_from_chain(self, prompt_ids, chain_ids):
        model, device = self.model, self.device
        cond_ids = torch.cat([prompt_ids, chain_ids.to(device)], dim=-1)
        use_cfg = self.cfg_weight > 1
        seq_len = len(cond_ids)

        if use_cfg:
            uncond_ids = torch.full((seq_len,), self.processor.pad_id, dtype=torch.int, device=device)
            uncond_ids[0] = cond_ids[0]
            uncond_ids[-1] = cond_ids[-1]
            tokens = torch.stack([cond_ids, uncond_ids], dim=0)
        else:
            tokens = cond_ids.unsqueeze(0)

        n_seqs = tokens.shape[0]
        image_token_num, img_size, patch_size = 576, 384, 16
        generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int, device=device)
        past_key_values = None
        inputs_embeds = model.language_model.get_input_embeddings()(tokens).to(device)

        for step in range(image_token_num):
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
            logits = model.gen_head(outputs.last_hidden_state[:, -1, :])
            if use_cfg:
                logits = logits[1:2, :] + self.cfg_weight * (logits[0:1, :] - logits[1:2, :])
            probs = torch.softmax(logits / self.image_temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[0, step] = next_token.squeeze()
            next_token_expanded = next_token.repeat(n_seqs, 1) if use_cfg else next_token
            inputs_embeds = model.prepare_gen_img_embeds(next_token_expanded)
            past_key_values = outputs.past_key_values

        dec = model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[1, 8, img_size // patch_size, img_size // patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        return Image.fromarray(dec[0].astype(np.uint8))

    def chain_text_to_ids(self, chain_text):
        ids = self.tokenizer.encode(chain_text)
        if ids and ids[-1] != self.image_start_token:
            ids.append(self.image_start_token)
        return torch.LongTensor(ids).to(self.device)

    @torch.inference_mode()
    def chain_nll(self, prompt_ids, chain_text):
        """Teacher-forced mean NLL (nats/token) of a chain given the prompt.

        Measures how in-distribution an injected chain is under the model's
        own planner. Higher = more out-of-distribution.

        Encodes WITHOUT special tokens: tokenizer.encode() would prepend a BOS
        mid-sequence, whose huge NLL inflates short chains' per-token mean
        (condition-dependent bias). Generation paths keep the BOS for
        consistency with the original r3 perturb pipeline; only NLL drops it.
        """
        ids = self.tokenizer.encode(chain_text, add_special_tokens=False)
        if ids and ids[-1] != self.image_start_token:
            ids.append(self.image_start_token)
        chain_ids = torch.LongTensor(ids).to(self.device)
        full = torch.cat([prompt_ids, chain_ids], dim=-1).unsqueeze(0)
        embeds = self.model.language_model.get_input_embeddings()(full).to(self.device)
        outputs = self.model.language_model.model(inputs_embeds=embeds, use_cache=False)
        logits = self.model.language_model.lm_head(outputs.last_hidden_state)
        # positions len(prompt)-1 .. len(full)-2 predict the chain tokens
        p = len(prompt_ids)
        pred_logits = logits[0, p - 1:-1, :].float()
        targets = full[0, p:]
        logprobs = torch.log_softmax(pred_logits, dim=-1)
        token_ll = logprobs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        return {
            "mean_nll": float(-token_ll.mean()),
            "sum_nll": float(-token_ll.sum()),
            "n_tokens": int(targets.numel()),
        }


# ─────────────────────────────────────────────
# Checkpointed run loop shared by image-producing experiments
# ─────────────────────────────────────────────

def load_prompts(prompt_dir, subset, split="val", max_prompts=-1):
    path = os.path.join(prompt_dir, f"{subset}_{split}.txt")
    with open(path) as f:
        prompts = [l.strip() for l in f if l.strip()]
    return prompts[:max_prompts] if max_prompts > 0 else prompts


def run_checkpointed(output_dir, prompts, work_fn, save_every=10):
    """work_fn(idx, prompt) -> (PIL image or None, result_dict)."""
    from pathlib import Path
    from tqdm import tqdm
    import time, gc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_file, results_file = out / "checkpoint.json", out / "results.json"

    completed, results = set(), []
    if ckpt_file.exists():
        completed = set(json.load(open(ckpt_file)).get("completed_indices", []))
    if results_file.exists():
        results = json.load(open(results_file))

    remaining = [i for i in range(len(prompts)) if i not in completed]
    print(f"Output: {out} | Done: {len(completed)} | Remaining: {len(remaining)}")

    def save():
        json.dump(results, open(results_file, "w"), indent=2)
        json.dump({"completed_indices": list(completed)}, open(ckpt_file, "w"))

    errors, t0 = [], time.time()
    for count, idx in enumerate(tqdm(remaining)):
        try:
            img, rec = work_fn(idx, prompts[idx])
            if img is not None:
                img.save(str(out / f"{idx}.png"))
            results.append({"index": idx, "prompt": prompts[idx], **rec})
            completed.add(idx)
        except torch.cuda.OutOfMemoryError:
            errors.append({"index": idx, "error": "OOM"})
            torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
        if (count + 1) % save_every == 0:
            save()
            torch.cuda.empty_cache()
    save()
    if errors:
        json.dump(errors, open(out / "errors.json", "w"), indent=2)
    print(f"DONE {len(completed)}/{len(prompts)} | {len(errors)} errors | {(time.time()-t0)/60:.1f} min")
    return results
