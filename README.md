# Reassessing Global Gradient-Norm Imbalance in BLIP Fine-Tuning Across Physical Domains: Reproducibility Package

> **Status:** Paper under review at IEEE Transactions on Multimedia (TMM).
> This repository contains the code, training logs, and analysis scripts
> needed to reproduce every table and figure in the paper, and to re-run
> training from scratch.

Code, logs, and analysis scripts for:

> Kiran Naseer, Samreen Azhar, Dwarikanath Mahapatra. "Reassessing Global Gradient-Norm Imbalance in BLIP Fine-Tuning Across Physical Domains." IEEE Transactions on Multimedia (under review).

---

## Two software environments were used

Training for this paper spans two distinct environments, because a Colab
platform update mid-project changed the default Python version and broke
compatibility with the original pin. **This is disclosed in the paper
(Section IV-E / Limitations) and matters for anyone trying to reproduce
results exactly.**

| Experiments | Python | transformers | peft |
|---|---|---|---|
| Naive FT, Low-LR FT, Isolated Visual, Frozen Vision, Text-Only LoRA Audit, Staged FT, Joint-Schedule FT | 3.12.12 | 4.41.2 | 0.11.1 |
| RMS-Balanced FT, Multimodal LoRA | 3.13.15 | 4.49.0 | 0.11.1 |

Use `requirements-env1.txt` for the first group, `requirements-env2.txt`
for the second. **Do not mix them** — `transformers==4.41.2` will fail to
install cleanly on Python 3.13 (a `tokenizers` wheel is unavailable; see
`notebooks/colab/06_rms_balanced_ft.ipynb` cell 2 for the exact failure
and the version-range workaround used).

PyTorch build, CUDA version, and the exact `pycocoevalcap` version used
during either set of training runs were not captured and are not
recoverable; this is stated explicitly in the paper.

---

## Repository structure

```
.
├── README.md                          (this file)
├── requirements-env1.txt              Python 3.12 / transformers 4.41.2 pin
├── requirements-env2.txt              Python 3.13 / transformers 4.49.0 pin
├── LICENSE
├── CITATION.cff
│
├── notebooks/
│   ├── kaggle/
│   │   └── staged_ft_original_training.ipynb
│   │       Original 7-method x 3-dataset x 3-seed training sweep
│   │       (Naive FT, Low-LR FT, Isolated Visual, Frozen Vision,
│   │       Text-Only LoRA Audit, Staged FT) plus Joint-Schedule FT
│   │       for UICD. Run on Kaggle, environment 1 (see table above).
│   │
│   └── colab/
│       ├── 03_joint_schedule_ft_uicd.ipynb
│       ├── 04_joint_schedule_ft_rocov2.ipynb
│       ├── 05_joint_schedule_ft_rsicd.ipynb
│       │       Joint-Schedule FT control (Section III-D of the
│       │       paper): identical two-phase LR schedule to Staged
│       │       FT, no freezing. Environment 1.
│       │
│       ├── 06_rms_balanced_ft.ipynb
│       │       RMS-Balanced FT (Eq. 4 in the paper): per-step
│       │       gradient rescaling to equal RMS magnitude across
│       │       the visual/language groups. All three datasets in
│       │       one notebook (Parts A/B/C). Environment 2.
│       │
│       ├── 07_multimodal_lora.ipynb
│       │       Architecture-aware LoRA (target_modules extended to
│       │       cover BLIP's vision encoder). Includes the module-
│       │       inspection diagnostic (Cells 6-7) that must be run
│       │       and verified before training — do not skip this;
│       │       it is what catches the silent zero-visual-parameter
│       │       failure documented in the paper (Section VII). All
│       │       three datasets. Environment 2.
│       │
│       └── 08_normalized_rt.ipynb
│               Post-hoc analysis notebook. Computes the parameter-
│               normalised R_t^norm and B_t (Section III-B, Eq. 3)
│               from existing rt_log data — no retraining required.
│               Also runs the measurement-protocol audit (scaled vs
│               unscaled gradients, group membership, optimiser
│               reset at stage boundaries) reported in Section III-B3.
│
├── scripts/
│   ├── rt_normalization.py
│   │       Standalone .py version of 08_normalized_rt.ipynb's core
│   │       computation. Reads existing logs, no training needed.
│   │
│   ├── bertscore_per_seed.py
│   │       Computes per-seed BERTScore F1 (Appendix B) from stored
│   │       example predictions. Requires bert-score package.
│   │
│   └── regenerate_figures.py
│           Regenerates every R_t and stability figure used in the
│           paper from the JSON logs in logs/. Writes to session
│           storage, validates each figure is non-empty before
│           offering it for download (see script docstring for why
│           this matters — an earlier version of this project lost
│           data to silent empty-figure generation more than once).
│
├── logs/
│   └── [experiment]_[dataset]_seed[N].json
│       All training logs referenced by the paper's tables and
│       figures. See "Log file schema" below.
│
├── figures/
│   └── fig*.pdf
│       Final figure PDFs as they appear in the compiled paper.
│
└── paper/
    ├── TMM_manuscript.tex
    ├── references.bib
    └── figures/            (symlink or copy of ../figures/)
```

---

## Log file schema

Each `logs/[method]_[dataset]_seed[N].json` contains:

```json
{
  "experiment": "staged_ft_rsicd_seed42",
  "dataset": "RSICD",
  "seed": 42,
  "method": "staged_ft",
  "bleu4_per_epoch": [0.41, 0.44, ...],
  "cider_per_epoch": [1.8, 1.9, ...],
  "meteor_per_epoch": [0.37, 0.38, ...],
  "train_loss_per_epoch": [2.5, 0.6, ...],
  "val_loss_per_epoch": [0.98, 0.87, ...],
  "rt_log": [[step, ratio], [step, ratio], ...],
  "best_bleu4": 0.4707,
  "best_epoch": 4,
  "best_predictions": ["...", "...", ...],
  "best_references": [["...", "..."], ...]
}
```

Notes on known irregularities, documented here so they don't look like
bugs when you encounter them:
- **`rms_balanced_ft_*.json`'s `rt_log` field uses a different
  convention** than every other method: `rms(g_language)/rms(g_visual)`,
  not the raw total-norm ratio. See `scripts/rt_normalization.py`'s
  docstring and the paper's Section III-B3 before using this field
  directly.
- **Three RSICD `isolated`/`frozen_vis`/`lora` seeds have only 20 stored
  `best_predictions`**, not all seeds present, for the BERTScore
  appendix table specifically — see Appendix B's table footnote in the
  paper.
- **Model checkpoints are not included** (each ~990MB;
  `delete_checkpoint()` ran at the end of every training run to manage
  storage). This means predictions beyond the stored top-20 per run, and
  any UICD CIDEr re-scoring beyond what's reported in Appendix A, cannot
  be regenerated without retraining. This limitation is disclosed in the
  paper.

---

## Reproducing the paper's tables and figures from existing logs

No GPU needed for this path — everything below reads `logs/`.

```bash
pip install -r requirements-env2.txt   # either env works for analysis-only scripts
python scripts/rt_normalization.py
python scripts/bertscore_per_seed.py
python scripts/regenerate_figures.py
```

## Reproducing training from scratch

1. Read `notebooks/kaggle/staged_ft_original_training.ipynb` first — run
   on Kaggle under `requirements-env1.txt`. Produces the original
   7-method sweep.
2. Run `notebooks/colab/03/04/05_joint_schedule_ft_*.ipynb` under
   `requirements-env1.txt`.
3. Run `notebooks/colab/06_rms_balanced_ft.ipynb` and
   `07_multimodal_lora.ipynb` under `requirements-env2.txt`. **Run Cells
   6-7 of Notebook 7 (the module-inspection diagnostic) and confirm
   non-zero visual-parameter counts before proceeding to training** —
   this is not optional, it is what the paper's central LoRA finding
   depends on getting right.
4. Run `08_normalized_rt.ipynb` for the post-hoc analysis.

Total compute: approximately 150–200 A100-hours across all experiments.
ROCOv2 is the dominant cost (largest dataset).

---

## Citation

Repository: **[https://github.com/KiranNaseer-AI/multi-modal-shift-vlm]**

See `CITATION.cff`. BibTeX:

```bibtex
@article{naseer2026gradientimbalance,
  title   = {Reassessing Global Gradient-Norm Imbalance in
             BLIP Fine-Tuning Across Physical Domains},
  author  = {Naseer, Kiran and Azhar, Samreen and Mahapatra,
             Dwarikanath},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Kiran Naseer, University of Gujrat, Pakistan —
25016119-003@uog.edu.pk (corresponding author)
