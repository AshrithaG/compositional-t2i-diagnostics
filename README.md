# The Plan, Not the Decoder

**Diagnosing and repairing compositional failure in reasoning-augmented text-to-image generation.**

Accepted at the **MUCG workshop at [ECCV 2026](https://eccv.ecva.net/)** (non-archival) and selected for an **oral** presentation · [arXiv](https://arxiv.org/abs/2608.21713)

Tools for diagnosing why reasoning-augmented text-to-image models fail compositional prompts, and for fixing them at inference time.

Reasoning-augmented generators like [GoT-R1](https://arxiv.org/abs/2505.17022) emit an explicit plan (object names and bounding boxes) before generating image tokens. That plan is machine-readable, so it can be checked, edited, or replaced before the image is decoded. This repo contains the experiments that exploit that.

Built on GoT-R1 and [T2I-CompBench++](https://arxiv.org/abs/2307.06350).

## Findings

GoT-R1-1B on T2I-CompBench++ `spatial_val`, 300 prompts per condition, spatial
relations scored with the OWLv2 open-vocabulary detector, paired sign-flip
permutation tests. Numbers are from the arXiv version of the paper.

- **A widely used VQA spatial metric is blind to layout.** Swapping the two boxes
  inside the model's own plan flips the generated layout (detector accuracy 0.75
  to 0.48, p < 0.001), yet the BLIP-VQA spatial score goes up.
- **The decoder is a faithful executor.** 94% of generated layouts realize the
  planned relation, with a planned-versus-detected box IoU of 0.75.
- **The planner is the bottleneck.** Its planning accuracy is 98% on one phrasing
  and 54% on a semantically identical one.
- **Geometry decides, not style or familiarity.** Injected plans with clean,
  well-separated boxes beat the planner's own by 13.3 points whether their text
  is terse or verbose, despite a 5x gap in planner negative log-likelihood. Plans
  that copy the planner's own box statistics gain nothing significant.

| condition | compared with | relation accuracy | change [95% CI] | p |
|---|---|---|---|---|
| unperturbed control (own plans) | | 0.750 | | |
| native-path baseline (fresh plans) | | 0.773 | | |
| verify-then-generate (K=5) | native baseline | **0.823** | +5.0 [+2.3, +8.0] | .0007 |
| minimal in-place repair | control | **0.810** | +6.0 [+1.0, +11.0] | .021 |
| geometric plan repair (margin 0.2) | control | **0.857** | +10.7 [+6.7, +14.7] | <.0001 |
| oracle: minimal text, clean boxes | control | **0.883** | +13.3 [+8.3, +18.3] | .0001 |
| oracle: verbose text, clean boxes | control | **0.870** | +12.0 [+6.7, +17.3] | .0001 |
| oracle: minimal text, planner-statistics boxes | control | 0.780 | +3.0 [-3.0, +8.7] | .37 |
| oracle: verbose text, planner-statistics boxes | control | 0.803 | +5.3 [0.0, +10.7] | .062 |
| oracle: planner-mimic (donor plans) | control | 0.740 | -1.0 [-7.0, +5.0] | .83 |

The ladder's ordering holds in each of two further generation seeds. On GoT-R1-7B
every intervention still improves over the control, and box swap collapses
detector accuracy from 0.863 to 0.433.

## What's here

**Plan interventions.** Modify the plan, keep everything else fixed, and measure the effect on the image.

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
  booktitle = {MUCG Workshop at the European Conference on Computer Vision (ECCV)},
  note      = {Non-archival; oral presentation},
  year      = {2026},
  eprint    = {2608.21713},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2608.21713}
}
```
