"""Faults of a defect type the negative bank has never seen (one defect type held out per run).

Each run is the few-abnormal setting at 5 labelled faults per defect type, with one type given
no support. The unit of every readout is the held-out type: its test images are the unseen
subset of its own runs, and the faults whose type has support are the seen subset. The normal
test images belong to both. Types that get no support because they are too small (and so are
flagged unseen in every run of their category) are not part of any unit; they get a separate
descriptive line. Folds whose held-out type is such a type are marked and left out of the rule,
since their support is the same as without a held-out type.

The rule decides whether a fixed lambda = 1 (normalised units) can be recommended without
condition: on the unseen subset, take the paired image AUROC difference lambda = 1 minus
lambda = 0 per run, average it over the seeds of each held-out type, and over types; the
recommendation becomes conditional for a backbone where that mean is negative by more than two
standard errors over types. Lambda = 0 is plain kNN on the same images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cmad.data.regimes import GUARD_MARGIN, LODO_SHOTS
from cmad.report.figures import (
    BACKBONE_COLOR,
    BACKBONE_TITLE,
    panel_title,
    panels,
    reference_line,
    save,
)
from cmad.report.runs import SEEDS, order_backbones
from cmad.report.tables import write_table

N_FOLDS = (
    68  # one per defect type of the 14 categories with more than one, mixed types aside
)
FIXED = 1.0
HARM_PT = 0.1
FLOOR_PT = -3.0  # lower end of the figure's y-axis
RULE_SE = 2.0
SUBSETS = ("unseen", "seen")


def lodo_runs(study: pd.DataFrame, scale: str = "val_median") -> pd.DataFrame:
    """The negative-bank runs with a held-out type, checked: every fold has all seeds once, and
    there are `N_FOLDS` folds per backbone."""
    runs = study[
        (study["family"] == "lodo")
        & (study["scorer"] == "negbank")
        & (study["scale"] == scale)
    ]
    if runs.empty:
        raise ValueError("no leave-one-defect-type-out negative-bank runs")
    problems = []
    for bb, group in runs.groupby("bb", observed=True):
        seeds = group.groupby(["category", "held_out"])["seed"].apply(sorted)
        bad = [fold for fold, found in seeds.items() if found != sorted(SEEDS)]
        if len(seeds) != N_FOLDS or bad:
            problems.append(f"{bb}: {len(seeds)} folds, wrong seeds in {bad}")
    if problems:
        raise ValueError("held-out-type runs incomplete:\n  " + "\n  ".join(problems))
    return runs


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return 100.0 * roc_auc_score(labels, scores)


def score_run(record: dict) -> list[dict]:
    """Image AUROC and recall at that lambda's threshold, per lambda and subset, of one run.
    Subsets: the held-out type ("unseen"), the types with support ("seen") and, if any, the
    types that are too small for support ("too_small"); each with all normal test images. The
    false-positive rate is over the normal images and is the same for every subset."""
    test, val = record["image_scores"]["test"], record["image_scores"]["val"]
    held_out = record["split"]["held_out_type"]
    labels = np.array(test["label"])
    types = np.array(test["defect_type"])
    seen = np.array([s is True for s in test["seen"]])
    normal = labels == 0
    members = {
        "unseen": types == held_out,
        "seen": seen,
        # flagged unseen but not held out: too small to give support in any run
        "too_small": (labels == 1)
        & ~seen
        & (types != held_out)
        & np.array([s is False for s in test["seen"]]),
    }
    bank = record["neg_bank"]
    selected = float(bank["selected_lambda"] or 0.0)
    rows = []
    for lam in sorted(bank["lambdas"]):
        key = f"{lam:g}"
        scores = np.array(test["curve"][key])
        flagged = scores > np.max(val["curve"][key])
        for subset, faults in members.items():
            if not faults.any():
                continue
            keep = normal | faults
            rows.append(
                {
                    "lam": float(lam),
                    "selected": float(lam) == selected,
                    "subset": subset,
                    "n_faults": int(faults.sum()),
                    "image_auroc": _auroc(labels[keep], scores[keep]),
                    "recall": 100.0 * float(flagged[faults].mean()),
                    "fpr": 100.0 * float(flagged[normal].mean()),
                }
            )
    return rows


def lodo_curves(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run, lambda and subset. `marked` is a fold whose held-out type is too small
    to give support anyway, so its support is the same as without a held-out type."""
    rows = []
    for run in runs.itertuples():
        record = json.loads(Path(run.path).read_text())
        n_held = sum(
            t == run.held_out for t in record["image_scores"]["test"]["defect_type"]
        )
        base = {
            "bb": str(run.bb),
            "category": run.category,
            "held_out": run.held_out,
            "seed": int(run.seed),
            "marked": n_held < LODO_SHOTS + GUARD_MARGIN,
        }
        rows += [{**base, **row} for row in score_run(record)]
    return order_backbones(pd.DataFrame(rows))


def _paired(curves: pd.DataFrame, subset: str, compare: str) -> pd.DataFrame:
    """Per run: AUROC and recall at the compared lambda (`fixed` = 1, or the run's `selected`)
    minus the same at lambda = 0, on one subset."""
    sub = curves[curves["subset"] == subset]
    keys = ["bb", "category", "held_out", "seed"]
    base = sub[sub["lam"] == 0.0].set_index(keys)
    other = sub[sub["lam"] == FIXED] if compare == "fixed" else sub[sub["selected"]]
    other = other.set_index(keys)
    out = pd.DataFrame(
        {
            "marked": base["marked"],
            "auroc_0": base["image_auroc"],
            "auroc": other["image_auroc"],
            "diff": other["image_auroc"] - base["image_auroc"],
            "recall_0": base["recall"],
            "recall": other["recall"],
            "fpr_0": base["fpr"],
            "fpr": other["fpr"],
        }
    )
    return out.reset_index()


def _summary(paired: pd.DataFrame) -> dict:
    """Mean over units (held-out types; seeds averaged first) with the standard error over
    types, and the standard error over categories (types averaged within a category first)."""
    units = paired.groupby(["category", "held_out"]).mean(numeric_only=True)
    per_category = units.groupby("category")["diff"].mean()
    n = len(units)
    mean = float(units["diff"].mean())
    se_types = float(units["diff"].std(ddof=1) / np.sqrt(n))
    se_categories = float(per_category.std(ddof=1) / np.sqrt(len(per_category)))
    worst = units["diff"].idxmin()
    return {
        "n_units": n,
        "n_categories": len(per_category),
        "mean": mean,
        "se_types": se_types,
        "se_categories": se_categories,
        "n_below": int((units["diff"] < -HARM_PT).sum()),
        "worst": float(units["diff"].min()),
        "worst_unit": f"{worst[0]} / {worst[1]}",
        "auroc_0": float(units["auroc_0"].mean()),
        "auroc": float(units["auroc"].mean()),
        "recall_0": float(units["recall_0"].mean()),
        "recall": float(units["recall"].mean()),
        "fpr_0": float(units["fpr_0"].mean()),
        "fpr": float(units["fpr"].mean()),
    }


def lodo_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Per backbone, subset and comparison (fixed lambda = 1 or the selected lambda, each
    against lambda = 0): the summary over held-out types, marked folds left out. The rule
    verdict is set on the unseen subset at the fixed lambda."""
    rows = []
    for bb in curves["bb"].cat.categories:
        if bb not in set(curves["bb"]):
            continue
        for subset in SUBSETS:
            for compare in ("fixed", "selected"):
                paired = _paired(curves[curves["bb"] == bb], subset, compare)
                summary = _summary(paired[~paired["marked"]])
                verdict = ""
                if subset == "unseen" and compare == "fixed":
                    below = summary["mean"] < -RULE_SE * summary["se_types"]
                    verdict = "below kNN: conditional" if below else "not below kNN"
                rows.append(
                    {
                        "bb": bb,
                        "subset": subset,
                        "compare": compare,
                        **summary,
                        "verdict": verdict,
                    }
                )
    return order_backbones(pd.DataFrame(rows))


def too_small_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Descriptive only: the types too small for support (pill_type, wood color on MVTec AD),
    over the unmarked folds of their categories, as the paired difference at lambda = 1."""
    paired = _paired(curves, "too_small", "fixed")
    paired = paired[~paired["marked"]]
    out = paired.groupby(["bb", "category"], observed=True).agg(
        n_runs=("diff", "size"), mean=("diff", "mean"), worst=("diff", "min")
    )
    return out.reset_index()


def lodo_figure(curves: pd.DataFrame):
    """Mean image AUROC difference to lambda = 0 against lambda, on the unseen (dashed) and the
    seen (solid) subset, one panel per backbone; marked folds left out. Each panel has its own
    y-axis, cut at `FLOOR_PT`: on WRN-50 the curves fall by tens of points past lambda = 1.5,
    which would flatten the DINOv2 panels."""
    curves = curves[~curves["marked"] & curves["subset"].isin(SUBSETS)]
    base = curves[curves["lam"] == 0.0].set_index(
        ["bb", "category", "held_out", "seed", "subset"]
    )
    curves = curves.join(
        base["image_auroc"].rename("auroc_0"),
        on=["bb", "category", "held_out", "seed", "subset"],
    )
    curves["diff"] = curves["image_auroc"] - curves["auroc_0"]
    units = curves.groupby(
        ["bb", "subset", "lam", "category", "held_out"], observed=True
    )["diff"].mean()
    mean = units.groupby(["bb", "subset", "lam"], observed=True).mean().reset_index()
    backbones = [str(b) for b in mean["bb"].unique()]
    grid = sorted(mean["lam"].unique())
    position = {lam: i for i, lam in enumerate(grid)}
    figure, axes = panels(len(backbones))
    for axis, bb in zip(axes, backbones, strict=True):
        for subset, style in (("seen", "-"), ("unseen", "--")):
            line = mean[(mean["bb"] == bb) & (mean["subset"] == subset)].sort_values(
                "lam"
            )
            axis.plot(
                [position[x] for x in line["lam"]],
                line["diff"],
                style,
                color=BACKBONE_COLOR[bb],
                marker="o",
                markersize=2.5,
                label=subset,
            )
        reference_line(axis, 0.0)
        reference_line(axis, position[FIXED], vertical=True)
        last = min(position.get(4.0, len(grid) - 1), len(grid) - 1)
        shown = mean[(mean["bb"] == bb) & (mean["lam"] <= grid[last])]["diff"]
        axis.set_ylim(max(shown.min() - 0.2, FLOOR_PT), shown.max() + 0.2)
        axis.set_xlim(-0.4, last + 0.2)
        axis.set_xticks(
            range(last + 1), [f"{x:g}" for x in grid[: last + 1]], fontsize=6.5
        )
        panel_title(axis, "abc"[backbones.index(bb)], BACKBONE_TITLE[bb])
        axis.set_xlabel(r"$\lambda$ (normalised units)")
        axis.legend(loc="lower left")
    axes[0].set_ylabel(r"AUROC change from $\lambda = 0$ (pt)")
    return figure


def _pm(mean: float, se: float) -> str:
    return f"{mean:+.2f} ± {se:.2f}".replace("-", "−")


def write_lodo(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    curves = lodo_curves(lodo_runs(study))
    table = lodo_table(curves)
    header = [
        "backbone",
        "faults",
        "λ",
        "difference to λ = 0",
        "SE over categories",
        "types below kNN",
        "worst type",
        "recall (λ = 0 → λ)",
        "FPR (λ = 0 → λ)",
        "rule",
    ]
    rows = []
    for r in table.to_dict("records"):
        rows.append(
            [
                str(r["bb"])
                if r["subset"] == "unseen" and r["compare"] == "fixed"
                else "",
                r["subset"] if r["compare"] == "fixed" else "",
                "1" if r["compare"] == "fixed" else "selected",
                _pm(r["mean"], r["se_types"]),
                f"{r['se_categories']:.2f}",
                f"{r['n_below']} of {r['n_units']}",
                f"{r['worst']:+.2f} ({r['worst_unit']})".replace("-", "−", 1),
                f"{r['recall_0']:.1f} → {r['recall']:.1f}",
                f"{r['fpr_0']:.1f} → {r['fpr']:.1f}",
                r["verdict"],
            ]
        )
    note = (
        "Image AUROC in points, paired within each run against λ = 0 (plain kNN on the same "
        "images). Unit: the held-out defect type, seeds averaged first; mean ± standard error "
        "over the held-out types, and the standard error over categories beside it. 'Unseen' "
        "is the held-out type's test images, 'seen' the types with support; both with all "
        "normal test images. Folds whose held-out type is too small for support anyway are "
        "left out. Rule: λ = 1 is recommended without condition unless the unseen mean is "
        "negative by more than two standard errors over types."
    )
    paths = write_table(out_dir, "lodo_rule", table, header, rows, "lllrrrlrrl", note)
    small = too_small_table(curves)
    small.to_csv(out_dir / "lodo_too_small.csv", index=False)
    curves.to_csv(out_dir / "lodo_curves.csv", index=False)
    return [
        *paths,
        out_dir / "lodo_too_small.csv",
        out_dir / "lodo_curves.csv",
        *save(lodo_figure(curves), out_dir, "lodo_lambda"),
    ]
