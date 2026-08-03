"""
Phase 9: Mechanism — do image tokens attend to the plan's box coordinates?

During image-token generation, measure cross-attention mass from each
generated image token back to (a) box-coordinate token positions, (b) object
name token positions, (c) the rest of the chain. Aggregated per prompt, this
tests two predictions of the co-adaptation account:
  P1: attention-to-box mass correlates with execution success (per prompt)
  P2: attention-to-box mass is LOWER for injected oracle chains than for the
      model's own chains (the decoder "reads" alien plans less)

Inference only. Attention tensors are large — this script keeps a running
per-source-group sum, not the full maps.

Usage:
  python exp_attention_analysis.py --subset spatial \
      --chains reasoning_chains_spatial.json \
               r3_outputs/oracle_minimal_alien_spatial_val/results.json \
      --max_prompts 150
Output: attention_analysis_{subset}.json — per (source, index):
  frac_attn_box, frac_attn_obj, frac_attn_chain_other, frac_attn_prompt
Join with per-image metrics for the correlation (P1) and compare sources (P2).
"""

import argparse, json, os, re
import torch
from tqdm import tqdm
from exp_common import GotR1, set_seed, BOX_RE, OBJ_RE

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--subset", choices=["spatial", "color"], required=True)
parser.add_argument("--chains", nargs="+", required=True)
parser.add_argument("--max_prompts", type=int, default=150)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--layers", default="last8",
                    help="'all', 'lastN', or comma-separated indices")
args = parser.parse_args()

bot = GotR1(args.ckpt_path)
tok = bot.tokenizer


def token_group_spans(chain_text, chain_ids):
    """Map each chain token position to a group: box / obj / other.

    Robust approach: char-span groups from regex, then align by re-encoding
    the chain prefix. Uses offset via incremental decode (works for BPE
    tokenizers without fast-offset support).
    """
    groups = ["other"] * len(chain_ids)
    spans = []
    for m in re.finditer(BOX_RE.replace("(\\d+)", "\\d+") if False else
                         r'<\|box_start\|>\(\d+,\d+\),\(\d+,\d+\)<\|box_end\|>',
                         chain_text):
        spans.append(("box", m.start(), m.end()))
    for m in re.finditer(OBJ_RE, chain_text):
        spans.append(("obj", m.start(), m.end()))

    # incremental char offsets per token
    offsets, pos = [], 0
    for i in range(len(chain_ids)):
        piece = tok.decode(chain_ids[: i + 1])
        offsets.append((pos, len(piece)))
        pos = len(piece)
    for i, (s, e) in enumerate(offsets):
        for g, gs, ge in spans:
            if s < ge and e > gs:
                groups[i] = g
                break
    return groups


@torch.inference_mode()
def attention_profile(prompt_text, chain_text):
    prompt_ids = bot.format_prompt(prompt_text)
    chain_ids = bot.chain_text_to_ids(chain_text)
    cond_ids = torch.cat([prompt_ids, chain_ids], dim=-1)
    n_prompt, n_chain = len(prompt_ids), len(chain_ids)

    groups = token_group_spans(chain_text, chain_ids.tolist())
    idx_box = [n_prompt + i for i, g in enumerate(groups) if g == "box"]
    idx_obj = [n_prompt + i for i, g in enumerate(groups) if g == "obj"]
    idx_oth = [n_prompt + i for i, g in enumerate(groups) if g == "other"]
    idx_prm = list(range(n_prompt))

    # greedy-ish sampled generation WITHOUT cfg (single stream) for attention
    set_seed(args.seed)
    embeds = bot.model.language_model.get_input_embeddings()(cond_ids.unsqueeze(0))
    past, sums = None, {"box": 0.0, "obj": 0.0, "chain_other": 0.0, "prompt": 0.0}
    n_steps = 576

    n_layers = None
    for step in range(n_steps):
        out = bot.model.language_model.model(
            inputs_embeds=embeds, use_cache=True, past_key_values=past,
            output_attentions=True)
        past = out.past_key_values
        att = out.attentions  # tuple(layers) of [1, heads, q, kv_len]
        if n_layers is None:
            n_layers = len(att)
            if args.layers == "all":
                layer_idx = list(range(n_layers))
            elif args.layers.startswith("last"):
                k = int(args.layers[4:])
                layer_idx = list(range(n_layers - k, n_layers))
            else:
                layer_idx = [int(x) for x in args.layers.split(",")]
        # last query position, mean over selected layers+heads, over cond span
        a = torch.stack([att[l][0, :, -1, :] for l in layer_idx]).mean(dim=(0, 1))
        a = a[: len(cond_ids)]           # attention into the conditioning only
        denom = float(a.sum()) + 1e-9
        sums["box"] += float(a[idx_box].sum()) / denom if idx_box else 0.0
        sums["obj"] += float(a[idx_obj].sum()) / denom if idx_obj else 0.0
        sums["chain_other"] += float(a[idx_oth].sum()) / denom if idx_oth else 0.0
        sums["prompt"] += float(a[idx_prm].sum()) / denom

        logits = bot.model.gen_head(out.last_hidden_state[:, -1, :])
        nxt = torch.multinomial(torch.softmax(logits / 1.0, dim=-1), 1)
        embeds = bot.model.prepare_gen_img_embeds(nxt)

    return {f"frac_attn_{k}": v / n_steps for k, v in sums.items()} | {
        "n_box_tokens": len(idx_box), "n_chain_tokens": n_chain}


records = []
for path in args.chains:
    entries = json.load(open(path))[: args.max_prompts]
    tag = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    print(f"{tag}: {len(entries)} prompts")
    for e in tqdm(entries):
        if not e.get("reasoning"):
            continue
        prof = attention_profile(e["prompt"], e["reasoning"])
        records.append({"source": tag, "index": e["index"],
                        "prompt": e["prompt"], **prof})

out_path = f"attention_analysis_{args.subset}.json"
json.dump(records, open(out_path, "w"), indent=2)

from collections import defaultdict
by = defaultdict(list)
for r in records:
    by[r["source"]].append(r["frac_attn_box"])
print(f"\n{'source':45s} {'n':>4s} {'mean attn→box':>14s}")
for s, v in by.items():
    print(f"{s:45s} {len(v):4d} {sum(v)/len(v):14.4f}")
print(f"Wrote {out_path}")
