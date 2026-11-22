"""The headline claims, each tested with a paired Wilcoxon signed-rank test over categories.

The claims and the record sets were fixed before any p-value was computed, and claims 5 and 6
before any linear-probe result existed:

1. the negative bank beats kNN at k = 5 (mask negatives, selected lambda, raw units);
2. fixed lambda = 1 is not worse than kNN on unseen defect types;
3. fixed lambda = 1 against the selected lambda on unseen defect types;
4. mask negatives beat the 10 % of patches farthest from normal (k = 5, each at its selected
   lambda);
5. the negative bank with the farthest 10 % of patches as negatives (image-level labels only,
   at its selected lambda) against the class-weighted linear probe, at k = 1 and at k = 5;
6. fixed lambda = 1 (mask negatives, normalised units) against the class-weighted linear probe
   on unseen defect types: whether a classifier, which has no notion of the defect types it was
   not trained on, loses more on them than the negative bank does.

A large p-value means no consistent difference, never that two methods are equally good.

Unit = category: seeds are averaged within a category first, and for the unseen-type claims the
held-out types of a category before that. The test is two-sided with exact zeros dropped. The
p-value is reported beside the mean, never as a pass/fail gate, with the number of categories
left after the zeros and how many are positive and negative. With five or fewer left, no
two-sided p-value can fall below 0.05, and the table says so instead of giving one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from cmad.report.few_abnormal import paired_runs
from cmad.report.lodo import _paired, lodo_curves, lodo_runs
from cmad.report.probes import claim_5_units, claim_6_units, has_probes
from cmad.report.runs import BACKBONE_ORDER
from cmad.report.tables import write_table

MIN_UNITS = 6  # with 5 or fewer non-zero units the smallest two-sided p is 0.0625
CLAIMS = {
    "1": "negative bank minus kNN, k = 5",
    "2": "unseen types: λ = 1 minus λ = 0",
    "3": "unseen types: λ = 1 minus selected λ",
    "4": "masks minus farthest 10 %, k = 5",
    "5a": "farthest 10 % minus wCE probe, k = 1",
    "5b": "farthest 10 % minus wCE probe, k = 5",
    "6": "unseen types: λ = 1 minus wCE probe",
}


def signed_rank(differences: pd.Series) -> dict:
    """Two-sided paired Wilcoxon signed-rank on per-category differences (zeros dropped)."""
    nonzero = differences[differences != 0]
    out = {
        "n_units": len(differences),
        "n_left": len(nonzero),
        "n_positive": int((nonzero > 0).sum()),
        "n_negative": int((nonzero < 0).sum()),
        "mean": float(differences.mean()),
        "p": None,
    }
    if len(nonzero) >= MIN_UNITS:
        out["p"] = float(
            wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue
        )
    return out


def _per_category(frame: pd.DataFrame, value: str, levels: list[str]) -> pd.Series:
    """Average `value` over `levels` in order (innermost first), ending with one per category."""
    keys = ["category", *levels]
    series = frame.groupby(keys)[value].mean()
    for _ in levels:
        series = series.groupby(level=list(range(series.index.nlevels - 1))).mean()
    return series


def claim_units(study: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """Per claim and backbone: the paired difference per category, in AUROC points."""
    units: dict[tuple[int, str], pd.Series] = {}
    five = study[(study["family"] != "few_abn") | (study["k"] == 5)]
    masks = paired_runs(five)
    masks = masks[masks["k"] == 5]
    top = paired_runs(five, source="top_q", top_frac=0.10)
    top = top[top["k"] == 5]
    vs = masks.merge(top, on=["bb", "k", "category", "seed"], suffixes=("", "_top"))
    vs["masks_minus_top"] = vs["neg_image_auroc"] - vs["neg_image_auroc_top"]
    curves = lodo_curves(lodo_runs(study))
    for bb in BACKBONE_ORDER:
        units[("1", bb)] = _per_category(
            masks[masks["bb"] == bb], "gain_image_auroc", ["seed"]
        )
        units[("4", bb)] = _per_category(
            vs[vs["bb"] == bb], "masks_minus_top", ["seed"]
        )
        c = curves[curves["bb"] == bb]
        fixed = _paired(c, "unseen", "fixed")
        selected = _paired(c, "unseen", "selected")
        keys = ["category", "held_out", "seed"]
        both = fixed.merge(selected, on=keys, suffixes=("", "_sel"))
        both = both[~both["marked"]]
        both["fixed_minus_selected"] = both["auroc"] - both["auroc_sel"]
        units[("2", bb)] = _per_category(both, "diff", ["seed", "held_out"])
        units[("3", bb)] = _per_category(
            both, "fixed_minus_selected", ["seed", "held_out"]
        )
    if has_probes(study):
        for (k, bb), differences in claim_5_units(study).items():
            units[("5a" if k == 1 else "5b", bb)] = differences
        for bb, differences in claim_6_units(study).items():
            units[("6", bb)] = differences
    return units


def claims_table(study: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"claim": claim, "bb": bb, **signed_rank(differences)}
        for (claim, bb), differences in claim_units(study).items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["claim", "bb"],
            key=lambda c: (
                c.map({b: i for i, b in enumerate(BACKBONE_ORDER)})
                if c.name == "bb"
                else c
            ),
        )
        .reset_index(drop=True)
    )


def _p_text(row: dict) -> str:
    if row["p"] is None or pd.isna(row["p"]):
        return f"not computed: {row['n_left']} categories left"
    return f"{row['p']:.3f}" if row["p"] >= 0.001 else "< 0.001"


def write_claims(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    table = claims_table(study)
    header = [
        "claim",
        "backbone",
        "mean difference (pt)",
        "categories left",
        "positive / negative",
        "p (two-sided)",
    ]
    rows = [
        [
            f"{r['claim']}: {CLAIMS[r['claim']]}"
            if r["bb"] == BACKBONE_ORDER[0]
            else "",
            r["bb"],
            f"{r['mean']:+.2f}".replace("-", "−"),
            f"{r['n_left']} of {r['n_units']}",
            f"{r['n_positive']} / {r['n_negative']}",
            _p_text(r),
        ]
        for r in table.to_dict("records")
    ]
    note = (
        "Image AUROC differences in points; unit = category (seeds averaged first; for the "
        "unseen-type claims, the held-out types of a category before that). Paired Wilcoxon "
        "signed-rank, two-sided, categories with a difference of exactly zero dropped; exact null "
        "distribution (approximate where non-zero differences tie). Claims 1–4 were declared "
        "before testing, and claims 5–6 before any linear-probe result existed; no correction "
        "for multiple tests. With five or fewer categories left a two-sided p-value cannot fall "
        "below 0.05, so none is given. A large p-value means no consistent difference, never "
        "that the two are equally good."
    )
    return write_table(out_dir, "claims", table, header, rows, "llrrrr", note)
