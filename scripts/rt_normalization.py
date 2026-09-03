"""
Parameter-normalised gradient ratio (R_t^norm, B_t) analysis.

Companion script to 08_normalized_rt.ipynb; standalone .py version
for repository use. Computes:
  - r_t: raw total-norm ratio ||g_language|| / ||g_visual||, as
    logged during training (this IS the balance point at r_t=1
    misconception the paper corrects -- see Section III-B).
  - R_t^norm, B_t: parameter-count-normalised ratio and its log,
    using K = sqrt(N_language / N_visual) for BLIP
    (Salesforce/blip-image-captioning-base).

IMPORTANT: RMS-Balanced FT logs a DIFFERENT quantity in its rt_log
field (rms(g_language)/rms(g_visual), not the raw total-norm ratio
used by every other condition). This script does NOT auto-correct
for that -- see Section III-B3 and Appendix Table IV of the paper
for the correction factor and its derivation. Applying the same
+ln(K) B_t formula to RMS-Balanced FT's logged values without this
correction reproduces the bug documented and fixed during this
project's review process; do not do this.

Usage (Colab, with Drive mounted):
    !python rt_normalization.py
"""

import json
import math
from pathlib import Path
import numpy as np
from transformers import BlipForConditionalGeneration

LOGS = Path('/content/drive/MyDrive/DAMF/logs')

METHODS_STANDARD_CONVENTION = [
    'naive_ft', 'lowlr_ft', 'damf', 'joint_schedule_ft'
]  # rt_log = raw total-norm ratio, standard GradientTracker hooks

METHOD_RMS_CONVENTION = 'rms_balanced_ft'  # rt_log = rms ratio, needs correction

DATASETS = ['uicd', 'rsicd', 'rocov2']
SEEDS = (42, 0, 123)


def compute_balance_constant():
    """K = sqrt(N_language / N_visual) for BLIP, using the same
    vision_model / text_decoder substring grouping as GradientTracker."""
    model = BlipForConditionalGeneration.from_pretrained(
        'Salesforce/blip-image-captioning-base')
    vis_elems = sum(p.numel() for n, p in model.named_parameters()
                    if 'vision_model' in n)
    lang_elems = sum(p.numel() for n, p in model.named_parameters()
                     if 'text_decoder' in n)
    K = math.sqrt(lang_elems / vis_elems)
    print(f"N_visual={vis_elems:,}  N_language={lang_elems:,}  K={K:.4f}")
    return K


def window_stats(vals, n=10):
    v = np.asarray(vals, dtype=float)
    k = min(n, len(v))
    return float(v[:k].mean()), float(v[-k:].mean())


def rt_log_to_Bt(rt_log, K, is_rms_convention=False):
    """Convert a logged rt_log array to B_t (log parameter-normalised
    ratio). is_rms_convention=True applies the correction needed for
    RMS-Balanced FT's differently-defined logged quantity."""
    r = np.array([x[1] for x in rt_log], dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return None
    if is_rms_convention:
        # r already equals raw_ratio / K; B_t = -ln(r), no extra +ln(K)
        b = -np.log(r)
    else:
        b = -np.log(r) + math.log(K)
    return b


if __name__ == '__main__':
    K = compute_balance_constant()

    for ds in DATASETS:
        print(f"\n=== {ds.upper()} ===")
        for method in METHODS_STANDARD_CONVENTION + [METHOD_RMS_CONVENTION]:
            is_rms = (method == METHOD_RMS_CONVENTION)
            b_early, b_late = [], []
            for s in SEEDS:
                f = LOGS / f'{method}_{ds}_seed{s}.json'
                if not f.exists():
                    continue
                d = json.load(open(f))
                rt_log = d.get('rt_log', [])
                if not rt_log:
                    continue
                b = rt_log_to_Bt(rt_log, K, is_rms_convention=is_rms)
                if b is None:
                    continue
                e, l = window_stats(b)
                b_early.append(e)
                b_late.append(l)
            if not b_early:
                continue
            print(f"  {method:20s} B_early={np.mean(b_early):+.3f}  "
                  f"B_late={np.mean(b_late):+.3f}  n={len(b_early)}")
