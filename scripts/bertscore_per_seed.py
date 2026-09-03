"""
Per-seed BERTScore evaluation.

Computes BERTScore F1 (rescaled against a random-pairing baseline)
over each method's stored top-20 example predictions, per seed, for
all three datasets. Used to build the paper's Appendix B BERTScore
table (mean +/- std ACROSS SEEDS, matching the convention used by
every other results table in the paper -- NOT the within-run std
across the 20 pooled images, which is a different, incompatible
quantity).

Requires: bert-score (pip install bert-score)
Input: JSON logs in DAMF/logs/, each containing 'best_predictions'
       and 'best_references' (top-20 stored examples per run).
Output: per-seed BERTScore-F1 printed to console; aggregate into
        mean+/-std across seeds by hand or extend this script.

Usage (Colab, with Drive mounted):
    !python bertscore_per_seed.py
"""

import json
from pathlib import Path
from bert_score import score as bert_score

LOGS = Path('/content/drive/MyDrive/DAMF/logs')

METHODS = ['naive_ft', 'lowlr_ft', 'isolated', 'frozen_vis', 'lora',
           'damf', 'joint_schedule_ft', 'rms_balanced_ft', 'multimodal_lora']
DATASETS = ['uicd', 'rsicd', 'rocov2']
SEEDS = (42, 0, 123)


def per_seed(ds, methods):
    """Print per-seed BERTScore-F1 for each method on one dataset.
    Missing files or empty prediction lists are reported explicitly
    rather than silently skipped, so gaps (e.g. the RSICD n=40 case)
    are visible directly in the output."""
    for m in methods:
        print(f"\n{m} ({ds.upper()}):")
        for s in SEEDS:
            f = LOGS / f'{m}_{ds}_seed{s}.json'
            if not f.exists():
                print(f"  seed {s}: file missing")
                continue
            d = json.load(open(f))
            preds = d.get('best_predictions', [])
            refs = d.get('best_references', [])
            if not preds:
                print(f"  seed {s}: no predictions stored")
                continue
            P, R, F1 = bert_score(preds, refs, lang='en',
                                  rescale_with_baseline=True, verbose=False)
            print(f"  seed {s}: BERTScore-F1={F1.mean().item():.4f}  n={len(preds)}")


if __name__ == '__main__':
    for ds in DATASETS:
        print("=" * 60)
        print(f"{ds.upper()}")
        print("=" * 60)
        per_seed(ds, METHODS)
