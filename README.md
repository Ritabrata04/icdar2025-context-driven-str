# A Lightweight Context-Driven Training-Free Network for Scene Text Segmentation and Recognition

[![Conference](https://img.shields.io/badge/ICDAR-2025%20Oral-blueviolet)](https://icdar2025.org/)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-B31B1B.svg)](https://arxiv.org/abs/2503.15639)
[![Project Page](https://img.shields.io/badge/Project%20Page-TBD-1f6feb)](https://ritabrata04.github.io/Context-driven-STR/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

> **ICDAR 2025 Oral (Top ~40 accepted)** — A training-free, **context-driven** pipeline for **scene text segmentation & recognition** that avoids heavy end-to-end text spotters unless necessary. We combine a lightweight **AG-UNet** (attention-gated U-Net) for text foreground masks, **block-level localization**, **BLIP‑2 scene captions** for semantic context, efficient **recognizers** (e.g., TrOCR/PARSeq), and a simple **semantic+lexical fusion** to decide whether a **heavy fallback** (DeepSolo) is needed.

---

## 🔥 TL;DR

- **Training-free recognition**: combines full-image and block-crop recognitions with **contextual captions**.
- **Fast & light**: **bypass** heavy spotters if the semantic+lexical confidence exceeds a threshold (τ).
- **Modular & reproducible**: deterministic scoring, CSV logging, and minimal dependencies.
- **Clean codebase**: small, focused modules; easy to extend with your own recognizers or captioners.

---

## 📖 Paper

**A Lightweight Context-Driven Training-Free Network for Scene Text Segmentation and Recognition**  
Ritabrata Chakraborty, Palaiahnakote Shivakumara, Umapada Pal, Cheng-Lin Liu  
*ICDAR 2025 (Oral)*

- **PDF**: see `overleaf/main.pdf` (provided by authors).  
- **arXiv**: https://arxiv.org/abs/TBD (update once available).  
- **Project Page**: TBD (update once live).

If you use this work, please **cite us** (BibTeX below).

---

## 🏗️ Method Overview

<p align="center">
  <i>Architecture: AG‑UNet → Mask → Blocks → (T1, T3, T2) → Scoring → Decision (Bypass vs. DeepSolo)</i>
</p>

- **Segmentation (AG‑UNet)**: ResNet‑50 encoder + attention‑gated skips + BN bottleneck → foreground **probability map** → **binary mask**.
- **Localization**: Connected components on the mask → padded bboxes → top‑K **blocks**.
- **Context (T2)**: BLIP‑2 caption with a “text‑aware” prompt → **medium‑length** description (~40–80 tokens).
- **Recognition**: T1 on the full image; T3 on block crops (choose best crop by scoring).
- **Scoring**: MPNet cosine **S1/S3** + Levenshtein **L1/L3** → **C = αS + βL**; if `max(C1,C3) ≥ τ`, return the **semantically stronger** (higher S) between T1/T3; else **fallback**.
- **Fallback**: Optional **DeepSolo/MMOCR** call only when needed.

> The code in this repo exactly mirrors the paper’s pipeline and equations (α=0.6, β=0.4, τ=0.8 by default).

---

## 🗂️ Repository Layout

```
- caption
  - caption/blip2_captioner.py
  - caption/init.py
- cli
  - cli/main.py
- localize
  - localize/init.py
  - localize/mask_to_blocks.py
- models
  - models/fallback_deepsolo.py
  - models/init.py
  - models/recognizers.py
  - models/segmenter.py
  - models/unet_adv.py
- pipeline
  - pipeline/init.py
  - pipeline/io_utils.py
  - pipeline/runner.py
- requirements.txt
- scoring
  - scoring/context_score.py
  - scoring/init.py
```

- `models/` — AG‑UNet (`unet_adv.py`), segmenter wrapper, recognizers (TrOCR/Tesseract/PARSeq stub), optional DeepSolo hook.
- `caption/` — BLIP‑2 captioner (medium‑length).
- `localize/` — mask → blocks (connected components).
- `scoring/` — MPNet + Levenshtein fusion.
- `pipeline/` — runner + I/O helpers.
- `cli/` — entrypoint for folder/CSV runs.
- `requirements.txt` — pinned-ish dependencies.

> **Note:** This repository ships core code. For configs and figures, see the sections below.

---

## 📦 Installation

**Python 3.10+** recommended.

```bash
git clone https://github.com/Ritabrata04/icdar2025-context-driven-str.git
cd icdar2025-context-driven-str
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- **HuggingFace** weights (BLIP‑2, MPNet, TrOCR) will download on first use to your HF cache: `~/.cache/huggingface/`.
- **GPU** is recommended (set `device: cuda` in configs). CPU works but is slow for BLIP‑2.

---

## ⚙️ Minimal Configs (create these files)

This repo’s CLI expects a base config and a run preset. Create the following:

**`configs/default.yaml`**
```yaml
seed: 1337
device: "cuda"
num_workers: 4
deterministic: true

segmenter:
  ckpt: ""              # path to AdvancedUNet checkpoint (optional but recommended)
  input_size: [256, 256]
  binarize_thresh: 0.5

localize:
  min_area: 150         # Amin
  pad: 6                # p
  nmax: 10              # max blocks per image

recognition:
  device: "cuda"
  models: ["trocr"]     # choose from: trocr, tesseract, parseq, dummy
  trocr:
    repo_id: "microsoft/trocr-base-printed"
    revision: null
  tesseract:
    lang: "eng"

caption:
  model: "Salesforce/blip2-flan-t5-xl"
  max_len: 80
  min_len: 40

scoring:
  embedder: "sentence-transformers/all-mpnet-base-v2"
  alpha: 0.6
  beta: 0.4
  tau: 0.8

fallback:
  use_deepsolo: false   # set true only if you integrate MMOCR/DeepSolo
  only_if_below_tau: true
```

**`configs/run_presets/demo_folder.yaml`**
```yaml
# Example override for a quick folder demo
device: "cuda"

segmenter:
  ckpt: ""             # optional; if left empty, mask quality will be poor

localize:
  min_area: 150
  pad: 6
  nmax: 5

recognition:
  models: ["trocr"]
  device: "cuda"

caption:
  model: "Salesforce/blip2-flan-t5-xl"
  max_len: 80
  min_len: 40

scoring:
  embedder: "sentence-transformers/all-mpnet-base-v2"
  alpha: 0.6
  beta: 0.4
  tau: 0.8

fallback:
  use_deepsolo: false
```

---

## 🚀 Quick Start

### A) Folder Mode (demo on a directory of images)
```bash
python -m cli.main \
  --config configs/run_presets/demo_folder.yaml \
  --images path/to/images \
  --out_dir outputs/demo \
  --save_artifacts
```

Writes:
- `outputs/demo/results.csv` — per‑image T1/T3/T2 and scores (S/L/C), decision, latency.
- `outputs/demo/masks/*_mask.png` — segmentation masks.
- `outputs/demo/crops/*_crop{i}.png` — top‑K block crops.
- `outputs/demo/debug/*_boxes.png` — overlays of block boxes.

### B) CSV Mode (optionally with ground truth)
Prepare a CSV with rows:
```
/abs/path/image_0001.jpg, hello world
/abs/path/image_0002.jpg, example
/abs/path/image_0003.jpg
```

Run:
```bash
python -m cli.main \
  --config configs/run_presets/demo_folder.yaml \
  --dataset_csv path/to/list.csv \
  --out_dir outputs/bench \
  --save_artifacts
```

---

## 📊 Datasets (per paper)

**Segmentation:** `COCO‑TS`, `ICDAR13 FST`, `TotalText`  
**Recognition:** `ICDAR13`, `ICDAR15`, `TotalText`

Example (from paper):
| Dataset        | Train | Test  |
|----------------|------:|------:|
| COCO‑TS        | 43686 | 10000 |
| ICDAR13 FST    |   229 |   233 |
| TotalText      |  1255 |   300 |

> See the paper for more detailed splits, metrics (IoU, F1), and recognition results.

---

## 🧪 Reproducibility Checklist

- **Determinism**: we set seeds and disable cudnn benchmarking in `pipeline/__init__.py`.
- **Config immutability**: we recommend saving your resolved config to `outputs/*/config.resolved.yaml`.
- **Pinned deps**: `requirements.txt` is provided; consider `pip freeze > requirements.lock` for archival.
- **HF revisions**: optionally pin exact HF commits for BLIP‑2, MPNet, and recognizers via `revision:` keys.
- **Hardware**: the paper used a single RTX GPU; BLIP‑2 can consume several GB — consider `flan‑t5‑xl` vs `xxl` variants.

---

## 🔌 Optional: Heavy Fallback (DeepSolo/MMOCR)

The fallback hook (`models/fallback_deepsolo.py`) is a stub; if you have MMOCR/DeepSolo installed, wire your inference call in `run_deepsolo_once(...)` and set:
```yaml
fallback:
  use_deepsolo: true
```

The pipeline only calls the fallback when `max(C1, C3) < τ`, preserving speed on easy cases.

---

## 📈 Expected Behavior (Paper)

- Strong **mask IoU/F1** on segmentation benchmarks (AG‑UNet).
- Competitive end‑to‑end recognition with **lower FLOPs** by **bypassing** heavy spotters whenever context confidence is high.
- Robustness on cluttered scenes due to semantic guidance from BLIP‑2 captions.

> Detailed tables and qualitative visualizations are available in the Overleaf `figures/` folder (e.g., architecture, dataset samples, failure cases). You can copy selected figures into a `docs/assets/` folder and embed them here.

---

## 📜 Citation

```bibtex
@inproceedings{Chakraborty2025ContextDrivenSTR,
  title     = {A Lightweight Context-Driven Training-Free Network for Scene Text Segmentation and Recognition},
  author    = {Chakraborty, Ritabrata and Shivakumara, Palaiahnakote and Pal, Umapada and Liu, Cheng-Lin},
  booktitle = {Proceedings of the International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2025},
  note      = {Oral},
  eprint    = {arXiv:TBD}  % update when available
}
```

---

## 📎 Add Figures (optional)

Copy any of these from your Overleaf zip into this repo (e.g., `docs/assets/`) and reference in the README:

- `figures/ICDAR_ARCHITECHTURE.pdf` (architecture)
- `figures/TEASER_ICDAR.pdf` (teaser)
- `figures/dataset_qualitative.pdf` (dataset samples)
- `figures/failure_cases.png` (failure modes)

Example embed (after copying & converting PDFs to PNGs):

```markdown
<p align="center">
  <img src="docs/assets/architecture.png" width="800" alt="Architecture">
</p>
```

