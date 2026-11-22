"""Long-tailed labelled faults: the same number of faults, spread evenly or unevenly over the types.

Every category spends five labelled faults per defect type in all, and the head-to-tail ratio
rho (1, 3 or 10) only changes how they are spread: the types are put in one seeded order per
category and seed, shared by the three ratios, and type i of T gets a share proportional to
rho^(-i/(T-1)), capped so that it keeps 5 test images. Toothbrush has one defect type and is
left out. The run's split records the type order and the counts actually used.

Fixed population: a type's support at each ratio is a prefix of one shuffle of its images, so
the three runs of a category and seed are scored on their common test set, the images that are
support at none of the three ratios. Every readout below uses it, so the ratios differ only in
the support, never in what is scored.

Every reading is paired within a run against lambda = 0, which is plain kNN on the same images:
lambda = 1 in normalised units (the fixed recommendation) and the lambda the run selected.

Readouts:
- per ratio: realised total and ratio of the counts, image AUROC of kNN and the two differences,
  and how the gain at lambda = 1 changes from rho = 1 (paired by category and seed);
- per defect type, through the randomised order: a type's AUROC (its faults against all normal
  images) and its recall at the count it received, against the same type in the same seed at
  rho = 1 (5 faults). kNN does not use the support and scores the common test set, so its
  per-type AUROC is the same at every ratio and the change is the negative bank's alone;
- per defect type with headroom (kNN type AUROC below 99): the gain over kNN at each count;
- the selected lambda against the head type's share of the support: there is one selection
  fold per support image, so the head dominates the selection as rho grows. Shown as the change
  from lambda = 1 to the selected lambda on the head type and on the tail types (1 or 2 faults).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cmad.data.regimes import LONGTAIL_PREFIX, LONGTAIL_RATIOS, LONGTAIL_SHOTS
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

N_CATEGORIES = 14  # every MVTec AD category with more than one defect type
FIXED = 1.0
TAIL_MAX = 2  # a type with at most this many labelled faults counts as tail
# support-count bins of the per-type readout (5 is the count at rho = 1)
COUNT_BINS = [(1, 1), (2, 2), (3, 4), (5, 5), (6, 8), (9, 99)]
Y_MIN_PT = 0.5  # the type figure shows at least +-0.5 pt, the same range in every panel
_RUN = ["bb", "category", "seed", "rho"]
HEADROOM = 99.0  # a type with kNN AUROC below this, in points, is not already separated
FIRST_ROW = "5 (ρ = 1)"
SEED_SD = "standard deviation over the 5 seeds of the mean over categories"


def _rho(regime: str) -> int:
    return int(regime[len(LONGTAIL_PREFIX) :])


def longtail_runs(study: pd.DataFrame, scale: str = "val_median") -> pd.DataFrame:
    """The long-tail negative-bank runs, checked: per backbone, every category has every ratio
    and seed exactly once."""
    runs = study[
        study["regime"].astype(str).str.startswith(LONGTAIL_PREFIX)
        & (study["scorer"] == "negbank")
        & (study["scale"] == scale)
    ].copy()
    if runs.empty:
        raise ValueError("no long-tail negative-bank runs")
    runs["rho"] = runs["regime"].map(_rho)
    expected = sorted((r, s) for r in LONGTAIL_RATIOS for s in SEEDS)
    problems = []
    for bb, group in runs.groupby("bb", observed=True):
        cells = group.groupby("category")[["rho", "seed"]].apply(
            lambda g: sorted(zip(g["rho"], g["seed"], strict=True))
        )
        bad = [c for c, found in cells.items() if found != expected]
        if len(cells) != N_CATEGORIES or bad:
            problems.append(f"{bb}: {len(cells)} categories, wrong cells in {bad}")
    if problems:
        raise ValueError("long-tail runs incomplete:\n  " + "\n  ".join(problems))
    return runs


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return 100.0 * roc_auc_score(labels, scores)


def score_run(record: dict, common: set[str]) -> list[dict]:
    """One run on the common test set, at lambda = 0, lambda = 1 and the selected lambda: image
    AUROC, recall and false-positive rate at that lambda's threshold over all faults
    (`defect_type` "all"), and AUROC and recall per defect type (that type's faults against
    every normal image), with the type's support count."""
    test, val = record["image_scores"]["test"], record["image_scores"]["val"]
    keep = np.array([i in common for i in test["image_id"]])
    labels = np.array(test["label"])[keep]
    types = np.array(test["defect_type"])[keep]
    normal = labels == 0
    plan = record["split"]["longtail"]
    selected = float(record["neg_bank"]["selected_lambda"] or 0.0)
    head = plan["order"][0]
    rows = []
    for name, lam in (("knn", 0.0), ("fixed", FIXED), ("selected", selected)):
        key = f"{lam:g}"
        scores = np.array(test["curve"][key])[keep]
        flagged = scores > np.max(val["curve"][key])
        fpr = 100.0 * float(flagged[normal].mean())
        groups = {"all": labels == 1} | {t: types == t for t in plan["order"]}
        for defect_type, faults in groups.items():
            if not faults.any():
                raise ValueError(f"no common test image of type {defect_type!r}")
            mask = normal | faults
            rows.append(
                {
                    "at": name,
                    "lam": lam,
                    "defect_type": defect_type,
                    "count": plan["counts"].get(defect_type, plan["total"]),
                    "head": defect_type == head,
                    "n_faults": int(faults.sum()),
                    "image_auroc": _auroc(labels[mask], scores[mask]),
                    "recall": 100.0 * float(flagged[faults].mean()),
                    "fpr": fpr,
                }
            )
    return rows


def longtail_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run, operating point (`at`: knn, fixed, selected) and defect type ("all" for
    every fault), scored on the common test set of the run's category and seed; with the run's
    realised total and ratio, selected lambda and the head type's share of the support."""
    records = {
        run.Index: json.loads(Path(run.path).read_text()) for run in runs.itertuples()
    }
    rows = []
    for (bb, category, seed), group in runs.groupby(
        ["bb", "category", "seed"], observed=True
    ):
        tests = [
            set(records[i]["image_scores"]["test"]["image_id"]) for i in group.index
        ]
        common = set.intersection(*tests)
        for i, run in group.iterrows():
            record = records[i]
            plan = record["split"]["longtail"]
            base = {
                "bb": str(bb),
                "category": category,
                "seed": int(seed),
                "rho": int(run["rho"]),
                "total": plan["total"],
                "budget": LONGTAIL_SHOTS * len(plan["order"]),
                "realised_ratio": plan["realised_ratio"],
                "selected_lambda": float(record["neg_bank"]["selected_lambda"] or 0.0),
                "head_share": plan["counts"][plan["order"][0]] / plan["total"],
            }
            rows += [{**base, **row} for row in score_run(record, common)]
    return order_backbones(pd.DataFrame(rows))


def _mean_sd(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    """Mean over categories, and the standard deviation over seeds of that mean: the
    convention of every result table. Rows are averaged within a category and seed first,
    then over the categories present in each seed."""
    per_seed = frame.groupby(["seed", "category"])[column].mean().groupby("seed").mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1))


def _mean_se_categories(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    """Mean and standard error over categories, rows averaged within a category first. For a
    subset whose categories differ from seed to seed, where a mean over categories per seed
    would weight a category by the number of seeds it appears in."""
    units = frame.groupby("category")[column].mean()
    return float(units.mean()), float(units.std(ddof=1) / np.sqrt(len(units)))


def _paired(scores: pd.DataFrame, at: str, value: str = "image_auroc") -> pd.Series:
    """Per run and defect type: `value` at the operating point `at` minus at lambda = 0."""
    keys = [*_RUN, "defect_type"]
    base = scores[scores["at"] == "knn"].set_index(keys)[value]
    other = scores[scores["at"] == at].set_index(keys)[value]
    return other - base


def at_budget(scores: pd.DataFrame) -> set[str]:
    """The categories whose realised total at rho = 10 is not below the budget in any seed."""
    top = scores[(scores["rho"] == max(LONGTAIL_RATIOS)) & (scores["at"] == "knn")]
    short = top.groupby("category")[["total", "budget"]].apply(
        lambda g: bool((g["total"] < g["budget"]).any())
    )
    return set(short[~short].index)


def ratio_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per backbone and ratio, over all faults: unit = category, seeds averaged first. Adds a
    row for rho = 10 on the categories at full budget only."""
    every = scores[scores["defect_type"] == "all"]
    gain = {at: _paired(every, at) for at in ("fixed", "selected")}
    recall = {at: _paired(every, at, "recall") for at in ("fixed", "selected")}
    knn = every[every["at"] == "knn"].set_index([*_RUN, "defect_type"])
    frame = knn[["image_auroc", "recall", "fpr", "total", "realised_ratio"]].copy()
    frame["fixed"], frame["selected"] = gain["fixed"], gain["selected"]
    frame["recall_fixed"] = frame["recall"] + recall["fixed"]
    fixed = every[every["at"] == "fixed"].set_index([*_RUN, "defect_type"])
    frame["fpr_fixed"] = fixed["fpr"]
    frame = frame.reset_index()
    base = frame[frame["rho"] == 1].set_index(["bb", "category", "seed"])["fixed"]
    frame["vs_rho1"] = (
        frame["fixed"]
        - base.reindex(
            pd.MultiIndex.from_frame(frame[["bb", "category", "seed"]])
        ).values
    )
    full = at_budget(scores)
    levels = [(str(r), frame["rho"] == r) for r in LONGTAIL_RATIOS]
    top = max(LONGTAIL_RATIOS)
    levels.append(
        (f"{top}, at budget", (frame["rho"] == top) & frame["category"].isin(full))
    )
    rows = []
    for bb in frame["bb"].cat.categories:
        for level, chosen in levels:
            sub = frame[chosen & (frame["bb"] == bb)]
            if sub.empty:
                continue
            units = sub.groupby("category").mean(numeric_only=True)
            row = {"bb": bb, "rho": level, "n_categories": len(units)}
            for column in ("fixed", "selected", "vs_rho1"):
                row[column], row[f"{column}_sd"] = _mean_sd(sub, column)
            row["n_below"] = int((units["fixed"] < -0.1).sum())
            for column in (
                "image_auroc",
                "recall",
                "recall_fixed",
                "fpr",
                "fpr_fixed",
                "total",
            ):
                row[column] = float(units[column].mean())
            row["ratio_median"] = float(sub["realised_ratio"].median())
            row["ratio_min"] = float(sub["realised_ratio"].min())
            rows.append(row)
    return order_backbones(pd.DataFrame(rows))


def type_changes(scores: pd.DataFrame) -> pd.DataFrame:
    """Per defect type, run and operating point at rho = 3 and 10: the type's AUROC and recall
    minus the same type's in the same seed at rho = 1, with the count the type received."""
    types = scores[scores["defect_type"] != "all"]
    keys = ["bb", "category", "seed", "defect_type", "at"]
    base = types[types["rho"] == 1].set_index(keys)[["image_auroc", "recall"]]
    later = types[types["rho"] != 1].set_index(keys)
    out = later[["rho", "count", "head", "image_auroc", "recall"]].copy()
    out["d_auroc"] = out["image_auroc"] - base["image_auroc"].reindex(out.index).values
    out["d_recall"] = out["recall"] - base["recall"].reindex(out.index).values
    # a type that every image of separates perfectly at both ratios cannot change
    before = base["image_auroc"].reindex(out.index).values
    out["ceiling"] = (out["image_auroc"] >= 100.0 - 1e-9) & (before >= 100.0 - 1e-9)
    return order_backbones(out.reset_index())


def headroom_table(scores: pd.DataFrame, below: float | None = None) -> pd.DataFrame:
    """At lambda = 1, the types kNN does not separate perfectly: a type-run is kept when the
    type's kNN AUROC is below `below` (kNN scores the same at every ratio). Per backbone and
    support-count bin: the type's gain over kNN, and at rho > 1 its change from rho = 1. The
    first row of each backbone is rho = 1, where every type has 5 faults."""
    below = HEADROOM if below is None else below
    types = scores[scores["defect_type"] != "all"]
    keys = [*_RUN, "defect_type"]
    frame = types[types["at"] == "fixed"].set_index(keys)[["count"]].copy()
    frame["knn"] = types[types["at"] == "knn"].set_index(keys)["image_auroc"]
    frame["gain"] = _paired(types, "fixed")
    frame = frame.reset_index()
    frame = frame[frame["knn"] < below]
    same = ["bb", "category", "seed", "defect_type"]
    base = frame[frame["rho"] == 1].set_index(same)["gain"]
    later = frame[frame["rho"] != 1].copy()
    later["change"] = (
        later["gain"].values
        - base.reindex(pd.MultiIndex.from_frame(later[same])).values
    )
    later["bin"] = later["count"].map(_bin)
    first = frame[frame["rho"] == 1].assign(bin=FIRST_ROW, change=np.nan)
    order = [FIRST_ROW, *(_bin(lo) for lo, _ in COUNT_BINS)]
    rows = []
    for (bb, label), group in pd.concat([first, later]).groupby(
        ["bb", "bin"], observed=True
    ):
        row = {
            "bb": bb,
            "bin": label,
            "n": len(group),
            "n_categories": group["category"].nunique(),
            "knn": float(group["knn"].mean()),
        }
        row["gain"], row["gain_se"] = _mean_se_categories(group, "gain")
        row["change"], row["change_se"] = (
            (np.nan, np.nan)
            if label == FIRST_ROW
            else _mean_se_categories(group, "change")
        )
        rows.append(row)
    out = order_backbones(pd.DataFrame(rows))
    out["bin"] = pd.Categorical(out["bin"], categories=order, ordered=True)
    return out.sort_values(["bb", "bin"]).reset_index(drop=True)


def _bin(count: int) -> str:
    lo, hi = next((lo, hi) for lo, hi in COUNT_BINS if lo <= count <= hi)
    return str(lo) if lo == hi else f"{lo}–{hi}" if hi < 99 else f"≥ {lo}"


def type_table(changes: pd.DataFrame) -> pd.DataFrame:
    """Per backbone, operating point and support-count bin: the mean change from rho = 1 of a
    type's AUROC and recall (types averaged within a category and seed)."""
    changes = changes[changes["at"].isin(("fixed", "selected"))].copy()
    changes["bin"] = changes["count"].map(_bin)
    order = [_bin(lo) for lo, _ in COUNT_BINS]
    rows = []
    for (bb, at, label), group in changes.groupby(["bb", "at", "bin"], observed=True):
        n_categories = group["category"].nunique()
        auroc, auroc_sd = _mean_sd(group, "d_auroc")
        recall, recall_sd = _mean_sd(group, "d_recall")
        rows.append(
            {
                "bb": bb,
                "at": at,
                "bin": label,
                "n": len(group),
                "n_categories": n_categories,
                "ceiling": 100.0 * float(group["ceiling"].mean()),
                "d_auroc": auroc,
                "d_auroc_sd": auroc_sd,
                "d_recall": recall,
                "d_recall_sd": recall_sd,
            }
        )
    out = order_backbones(pd.DataFrame(rows))
    out["bin"] = pd.Categorical(out["bin"], categories=order, ordered=True)
    return out.sort_values(["bb", "at", "bin"]).reset_index(drop=True)


def head_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per backbone and ratio: the head type's share of the support, how often lambda = 0 is
    selected, and the change from lambda = 1 to the selected lambda in AUROC on the head type
    and on the tail types (at most `TAIL_MAX` faults)."""
    types = scores[scores["defect_type"] != "all"]
    keys = [*_RUN, "defect_type"]
    fixed = types[types["at"] == "fixed"].set_index(keys)
    chosen = types[types["at"] == "selected"].set_index(keys)
    frame = fixed[["count", "head", "head_share", "selected_lambda"]].copy()
    frame["d"] = chosen["image_auroc"] - fixed["image_auroc"]
    frame = frame.reset_index()
    rows = []
    for (bb, rho), group in frame.groupby(["bb", "rho"], observed=True):
        runs = group.drop_duplicates(_RUN)
        row = {
            "bb": bb,
            "rho": int(rho),
            "head_share": float(runs["head_share"].median()),
            "lambda_0": float((runs["selected_lambda"] == 0.0).mean() * 100.0),
            "lambda_median": float(runs["selected_lambda"].median()),
        }
        for name, part in (
            ("head", group["head"]),
            ("tail", group["count"] <= TAIL_MAX),
        ):
            chosen = group[part]
            if chosen.empty:
                row[name], row[f"{name}_sd"], row[f"{name}_n"] = np.nan, np.nan, 0
                continue
            row[name], row[f"{name}_sd"] = _mean_sd(chosen, "d")
            row[f"{name}_n"] = chosen["category"].nunique()
        rows.append(row)
    return order_backbones(pd.DataFrame(rows))


def type_figure(changes: pd.DataFrame):
    """Mean change of a type's image AUROC from rho = 1 against the support count it received,
    at lambda = 1 (solid) and at the selected lambda (dashed), one panel per backbone. Each point
    averages within categories first; the dotted line marks 5, the count at rho = 1."""
    backbones = [str(b) for b in changes["bb"].unique()]
    figure, axes = panels(len(backbones))
    counts = sorted(changes["count"].unique())
    lines = {
        (bb, at): changes[(changes["bb"] == bb) & (changes["at"] == at)]
        .groupby(["count", "category"])["d_auroc"]
        .mean()
        .groupby("count")
        .mean()
        for bb in backbones
        for at in ("fixed", "selected")
    }
    # one range for every panel, never tighter than +-Y_MIN_PT: the changes are small, and a
    # range fitted to each panel would make noise of a few hundredths look like an effect
    reach = max(Y_MIN_PT, max(float(line.abs().max()) for line in lines.values()) + 0.1)
    for axis, bb in zip(axes, backbones, strict=True):
        for at, style, label in (
            ("fixed", "-", r"$\lambda = 1$"),
            ("selected", "--", r"selected $\lambda$"),
        ):
            line = lines[(bb, at)]
            axis.plot(
                line.index,
                line.values,
                style,
                color=BACKBONE_COLOR[bb],
                marker="o",
                markersize=2.5,
                label=label,
            )
        axis.set_ylim(-reach, reach)
        reference_line(axis, 0.0)
        reference_line(axis, LONGTAIL_SHOTS, vertical=True)
        axis.set_xticks(counts, [str(c) for c in counts], fontsize=6.5)
        panel_title(axis, "abc"[backbones.index(bb)], BACKBONE_TITLE[bb])
        axis.set_xlabel("labelled faults of the type")
    axes[0].set_ylabel("type AUROC change from $\\rho = 1$ (pt)")
    axes[0].legend(loc="lower right")
    return figure


def _pm(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:+.{digits}f} ± {sd:.{digits}f}".replace("-", "−")


def write_longtail(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    scores = longtail_scores(longtail_runs(study))
    scores.to_csv(out_dir / "longtail_scores.csv", index=False)
    paths = [out_dir / "longtail_scores.csv"]

    table = ratio_table(scores)
    header = [
        "backbone",
        "ρ",
        "faults",
        "realised ratio",
        "kNN AUROC",
        "λ = 1 − kNN",
        "selected − kNN",
        "λ = 1 gain vs ρ = 1",
        "recall (kNN → λ = 1)",
        "FPR (kNN → λ = 1)",
    ]
    rows = []
    for r in table.to_dict("records"):
        first = r["rho"] == str(LONGTAIL_RATIOS[0])
        rows.append(
            [
                str(r["bb"]) if first else "",
                r["rho"]
                + (
                    ""
                    if r["rho"] != f"{max(LONGTAIL_RATIOS)}, at budget"
                    else f" ({r['n_categories']} cat.)"
                ),
                f"{r['total']:.1f}",
                f"{r['ratio_median']:.1f} (min {r['ratio_min']:.1f})",
                f"{r['image_auroc']:.2f}",
                _pm(r["fixed"], r["fixed_sd"]),
                _pm(r["selected"], r["selected_sd"]),
                "" if first else _pm(r["vs_rho1"], r["vs_rho1_sd"]),
                f"{r['recall']:.1f} → {r['recall_fixed']:.1f}",
                f"{r['fpr']:.1f} → {r['fpr_fixed']:.1f}",
            ]
        )
    note = (
        "Image AUROC in points on each category's common test set (the images that are support "
        "at no ratio). Mean over categories ± standard deviation over the 5 seeds of the mean over categories. λ in normalised units; kNN is λ = 0 in the same run. 'Faults' is the "
        "realised number of labelled faults per category (mean), 'realised ratio' the median "
        "over runs of the largest over the smallest count used. 'At budget': only the "
        "categories whose realised total at ρ = 10 is not below five faults per type."
    )
    paths += write_table(
        out_dir, "longtail_ratio", table, header, rows, "llrlrrrrrr", note
    )

    changes = type_changes(scores)
    types = type_table(changes)
    header = [
        "backbone",
        "λ",
        "faults of the type",
        "type-runs",
        "AUROC 100 at both",
        "type AUROC vs ρ = 1",
        "type recall vs ρ = 1",
    ]
    rows = []
    for r in types.to_dict("records"):
        first = r["bin"] == types["bin"].cat.categories[0]
        rows.append(
            [
                str(r["bb"]) if first and r["at"] == "fixed" else "",
                ("1" if r["at"] == "fixed" else "selected") if first else "",
                str(r["bin"]),
                str(r["n"]),
                f"{r['ceiling']:.0f} %",
                _pm(r["d_auroc"], r["d_auroc_sd"]),
                _pm(r["d_recall"], r["d_recall_sd"], 1),
            ]
        )
    note = (
        "Each defect type at ρ = 3 and 10 against the same type in the same seed at ρ = 1, "
        "where it had 5 labelled faults; the type order is random per category and seed, so "
        "a type is compared with itself. Type AUROC: the type's faults against all normal "
        "images of the common test set; recall at that λ's threshold τ_val. Mean over categories ± "
        "standard deviation over the 5 seeds of the mean over categories (types averaged within a category and seed). kNN's per-type values "
        "are the same at every ratio, so the changes are the negative bank's. 'AUROC 100 at "
        "both': type-runs whose type AUROC is 100 at ρ = 1 and at this ratio, which cannot "
        "change."
    )
    paths += write_table(
        out_dir, "longtail_types", types, header, rows, "lllrrrr", note
    )

    heads = head_table(scores)
    header = [
        "backbone",
        "ρ",
        "head share",
        "λ = 0 selected",
        "median selected λ",
        "head type: selected − λ = 1",
        "tail types: selected − λ = 1",
    ]
    rows = []
    for r in heads.to_dict("records"):
        rows.append(
            [
                str(r["bb"]) if r["rho"] == LONGTAIL_RATIOS[0] else "",
                str(r["rho"]),
                f"{100 * r['head_share']:.0f} %",
                f"{r['lambda_0']:.0f} %",
                f"{r['lambda_median']:g}",
                _pm(r["head"], r["head_sd"]),
                "—" if not r["tail_n"] else _pm(r["tail"], r["tail_sd"]),
            ]
        )
    note = (
        "The selected λ is chosen on leave-one-support-out folds, one per labelled fault, so "
        "the head type's share of the support is its share of the folds (median over runs). "
        f"Tail types: at most {TAIL_MAX} labelled faults. Image AUROC of the type at the "
        "selected λ minus at λ = 1; mean over categories ± standard deviation over the 5 seeds of the mean over categories."
    )
    paths += write_table(out_dir, "longtail_head", heads, header, rows, "llrrrrr", note)

    room = headroom_table(scores)
    header = [
        "backbone",
        "faults of the type",
        "type-runs",
        "categories",
        "kNN type AUROC",
        "λ = 1 − kNN",
        "change vs ρ = 1",
    ]
    rows = [
        [
            str(r["bb"]) if r["bin"] == FIRST_ROW else "",
            str(r["bin"]),
            str(r["n"]),
            str(r["n_categories"]),
            f"{r['knn']:.1f}",
            _pm(r["gain"], r["gain_se"]),
            "" if np.isnan(r["change"]) else _pm(r["change"], r["change_se"]),
        ]
        for r in room.to_dict("records")
    ]
    note = (
        f"Only the type-runs whose kNN type AUROC is below {HEADROOM:g}, the types kNN does "
        "not already separate; most types are at 100 and cannot gain. λ = 1. First row per "
        "backbone: ρ = 1, 5 faults of every type; other rows: ρ = 3 and 10 by the faults the "
        "type received, with the change of its gain from the same type and seed at ρ = 1. "
        "Unit: the category (type-runs averaged within it); mean ± standard error over "
        "categories. Unlike the other tables, not ± standard deviation over seeds: which "
        "types are kept differs from seed to seed, so a per-seed mean would weight a "
        "category by the number of seeds it appears in. Few categories per row, and the "
        "subset was chosen after the main results, so it explains them and is not a "
        "headline number."
    )
    paths += write_table(
        out_dir, "longtail_headroom", room, header, rows, "llrrrrr", note
    )
    return paths + [*save(type_figure(changes), out_dir, "longtail_types")]
