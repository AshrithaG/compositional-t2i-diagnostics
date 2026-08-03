"""
Human-evaluation pack generator (BEAM paper centerpiece + ICLR Phase E).

Samples N images per condition (stratified, shuffled, condition HIDDEN from
raters), copies them into a self-contained folder with a rating page
(rate.html) that works by double-clicking — no server, no install. Each rater
answers three yes/no questions per image and clicks "Download ratings CSV"
at the end. The answer key (image -> condition/prompt/index) is written
separately to key.json — do NOT show raters that file.

Usage (run where r3_outputs lives; VM or Mac backup both fine):
  python exp_human_eval_pack.py --per_condition 60 --dirs \
      r3_outputs/perturb_none_spatial r3_outputs/perturb_box_swap_spatial \
      r3_outputs/oracle2_minimal_alien_spatial_val r3_outputs/perturb_chain_ablate_spatial
Then zip human_eval_pack/ and send to each rater; collect their CSVs.
Analysis: exp_human_eval_pack.py --analyze ratings1.csv ratings2.csv ...
"""

import argparse, json, os, random, shutil, csv, sys

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", nargs="*", default=[])
parser.add_argument("--per_condition", type=int, default=60)
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--split", default="val")
parser.add_argument("--out", default="human_eval_pack")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--analyze", nargs="*", default=None,
                    help="rater CSV files; computes agreement vs key.json + judges")
args = parser.parse_args()

# ---------------- analysis mode ----------------
if args.analyze is not None:
    key = {r["file"]: r for r in json.load(open(os.path.join(args.out, "key.json")))}
    raters = []
    for f in args.analyze:
        rows = {r["file"]: r for r in csv.DictReader(open(f))}
        raters.append((os.path.basename(f), rows))
        print(f"{f}: {len(rows)} ratings")
    # inter-rater agreement on the relation question
    if len(raters) >= 2:
        common = set(raters[0][1]) & set(raters[1][1])
        a = [raters[0][1][k]["relation_correct"] for k in common]
        b = [raters[1][1][k]["relation_correct"] for k in common]
        agree = sum(1 for x, y in zip(a, b) if x == y) / len(common)
        # Cohen's kappa
        pa = agree
        py = (a.count("yes")/len(a)) * (b.count("yes")/len(b)) + \
             (a.count("no")/len(a)) * (b.count("no")/len(b))
        kappa = (pa - py) / (1 - py) if py < 1 else float("nan")
        print(f"inter-rater (relation): agreement={agree:.3f} kappa={kappa:.3f} n={len(common)}")
    # per-condition human relation accuracy (majority vote)
    from collections import defaultdict
    votes = defaultdict(list)
    for fname, k in key.items():
        vs = [r[1][fname]["relation_correct"] for r in raters if fname in r[1]]
        if vs:
            maj = 1.0 if vs.count("yes") >= (len(vs)+1)//2 else 0.0
            votes[k["condition"]].append((k["index"], maj))
    print(f"\n{'condition':45s} {'n':>4s} {'human relation acc':>18s}")
    for c, v in votes.items():
        print(f"{c:45s} {len(v):4d} {sum(m for _,m in v)/len(v):18.3f}")
    json.dump({c: v for c, v in votes.items()},
              open(os.path.join(args.out, "human_majority.json"), "w"))
    print(f"\nWrote {args.out}/human_majority.json — send to Claude with the "
          "judge JSONs (eval_out/, vlm_judge/, detector_eval/) for the "
          "human-vs-judge agreement table.")
    sys.exit(0)

# ---------------- pack-building mode ----------------
if not args.dirs:
    sys.exit("--dirs required in pack-building mode")
random.seed(args.seed)
with open(os.path.join(args.prompt_dir, f"spatial_{args.split}.txt")) as f:
    prompts = [l.strip() for l in f if l.strip()]

os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
key, items = [], []
for d in args.dirs:
    cond = os.path.basename(d.rstrip("/"))
    idxs = [int(f[:-4]) for f in os.listdir(d) if f.endswith(".png")]
    for idx in random.sample(sorted(idxs), min(args.per_condition, len(idxs))):
        fname = f"{len(key):04d}.png"
        shutil.copy(os.path.join(d, f"{idx}.png"),
                    os.path.join(args.out, "img", fname))
        key.append({"file": fname, "condition": cond, "index": idx,
                    "prompt": prompts[idx]})
        items.append({"file": fname, "prompt": prompts[idx]})
random.shuffle(items)
json.dump(key, open(os.path.join(args.out, "key.json"), "w"), indent=1)

html = """<!doctype html><meta charset="utf-8">
<title>Image rating</title>
<style>body{font-family:sans-serif;max-width:760px;margin:2em auto}
.card{border:1px solid #ccc;padding:1em;margin:1em 0;border-radius:8px}
img{max-width:380px;display:block;margin:.5em 0}
.q{margin:.35em 0}</style>
<h2>Image rating</h2>
<p>Your name: <input id="rater"> &nbsp; For each image, read the caption and
answer the three questions. Go with your first impression; there are no trick
questions. At the end click <b>Download ratings CSV</b>.</p>
<div id="cards"></div>
<button onclick="dl()" style="font-size:1.2em;padding:.5em 1em">Download ratings CSV</button>
<script>
const items = ITEMS_JSON;
const qs = [["obj1_present","Is the FIRST object mentioned in the caption present?"],
            ["obj2_present","Is the SECOND object mentioned present?"],
            ["relation_correct","Is the spatial relation in the caption correct in the image?"]];
const root = document.getElementById("cards");
items.forEach((it,i)=>{
  const div=document.createElement("div"); div.className="card";
  let h=`<b>${i+1}/${items.length}</b> &nbsp; Caption: <i>${it.prompt}</i>
         <img src="img/${it.file}">`;
  qs.forEach(([k,q])=>{h+=`<div class="q">${q}
    <label><input type="radio" name="${it.file}_${k}" value="yes">yes</label>
    <label><input type="radio" name="${it.file}_${k}" value="no">no</label></div>`;});
  div.innerHTML=h; root.appendChild(div);});
function dl(){
  let rows=[["rater","file","obj1_present","obj2_present","relation_correct"]];
  const who=document.getElementById("rater").value||"anon";
  let missing=0;
  items.forEach(it=>{
    const vals=qs.map(([k])=>{const el=document.querySelector(`input[name="${it.file}_${k}"]:checked`);
      if(!el){missing++;return ""}return el.value;});
    rows.push([who,it.file,...vals]);});
  if(missing>0 && !confirm(missing+" answers missing. Download anyway?"))return;
  const csv=rows.map(r=>r.join(",")).join("\\n");
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="ratings_"+who+".csv"; a.click();}
</script>"""
html = html.replace("ITEMS_JSON", json.dumps(items))
open(os.path.join(args.out, "rate.html"), "w").write(html)
print(f"Pack built: {args.out}/ ({len(key)} images, {len(args.dirs)} conditions)")
print("Send raters the WHOLE folder minus key.json (or just don't mention it).")
print("They open rate.html in any browser, rate, and send you the CSV.")
