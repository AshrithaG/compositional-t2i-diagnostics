# The Plan, Not the Decoder

**Diagnosing and repairing compositional failure in reasoning-augmented text-to-image generation.**

Accepted as an **oral** at [ECCV 2026](https://eccv.ecva.net/) · [arXiv](https://arxiv.org/abs/2608.21713)

Tools for diagnosing why reasoning-augmented text-to-image models fail compositional prompts, and for fixing them at inference time.

Reasoning-augmented generators like [GoT-R1](https://arxiv.org/abs/2505.17022) emit an explicit plan — object names and bounding boxes — before generating image tokens. That plan is machine-readable, so it can be checked, edited, or replaced before the image is decoded. This repo contains the experiments that exploit that.

Built on GoT-R1 and [T2I-CompBench++](https://arxiv.org/abs/2307.06350).

## What's here

**Plan interventions** — modify the plan, keep everything else fixed, measure the effect on the image.

| Script | What it does |
|---|---|
| `exp_minimal_repair.py` | Detects invalid plans and applies the smallest edit that fixes them (box swap for spatial, color insertion for attributes) |
| `exp_margin_repair.py` | Rewrites only box geometry to a target separation, leaving objects and prose untouched |
| `exp_oracle_dial.py` | Injects constructed plans across text styles (terse / verbose / planner-mimicking) and box distributions |
| `exp_none_control.py` | Unperturbed control through the identical conditioning path |
| `exp_gen_chains.py` | Chains-only generation for a given split (no image decoding) |

**Analysis**

| Script | What it does |
|---|---|
| `exp_detector_eval.py` | Open-vocabulary detection (OWLv2) → relation correctness + planned-vs-detected box IoU |
| `exp_detector_all.py` | Stores all raw detections and box sizes for downstream count/size analysis |
| `exp_eval_suite.py` | BLIP-VQA diagnostic suite (object presence, color binding, relation) |
| `exp_vlm_judge.py` | Same questions scored by an instruction-tuned VLM judge |
| `exp_chain_nll.py` | Planner likelihood of a chain, for measuring distribution shift |
| `exp_order_swap.py` | Re-plans prompts in semantically equivalent inverted phrasing |
| `exp_attention_analysis.py` | Cross-attention from image tokens to plan components |
| `exp_human_eval_pack.py` | Builds a blinded rating pack and computes inter-rater agreement |
| `analysis_paired_stats.py` | Paired permutation tests over per-image scores |

`exp_common.py` holds the shared parsing, verification, and two-phase generation code.

## Setup

Requires a CUDA GPU (~15 GB for 1B inference) and the GoT-R1 repo layout with the checkpoint in `ckpts/`.

```bash
python -m venv env && source env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.49.0 peft==0.13.2 accelerate==1.3.0 \
  sentencepiece opencv-python tqdm einops timm attrdict "numpy<2" \
  huggingface_hub scipy

huggingface-cli download gogoduan/GoT-R1-1B --local-dir ckpts/GoT-R1-1B
```

## Usage

```bash
# generate the model's own plans for a split
python exp_gen_chains.py --subset spatial --split val

# control condition
python exp_none_control.py --subset spatial --chains_file <chains.json>

# repair invalid plans in place
python exp_minimal_repair.py --subset spatial

# widen box separation only
python exp_margin_repair.py --target_margin 0.2

# score geometrically
python exp_detector_eval.py --chains_file <chains.json> --dirs r3_outputs/<condition>
```

Every script checkpoints and resumes, so runs can be interrupted.

## Notes

- All conditions share seeds, decoding settings, and a single conditioning path, so differences are attributable to the intervention.
- Spatial results use detector-based geometric scoring rather than VQA relation questions.
- `analysis_paired_stats.py` runs paired tests over per-image scores; per-condition means alone are not enough at these sample sizes.

## License

MIT

## Citation

```bibtex
@inproceedings{gonuguntla2026plan,
  title     = {The Plan, Not the Decoder: Diagnosing and Repairing Compositional
               Failure in Reasoning-Augmented Text-to-Image Generation},
  author    = {Gonuguntla, Ashritha},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  eprint    = {2608.21713},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2608.21713}
}
```
