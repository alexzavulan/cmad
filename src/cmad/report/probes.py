"""The linear probes: a classifier trained on the same features and labels as the negative bank.

Three losses (`ce`, `wce`, `focal`); `wce` is the headline probe. Every probe run uses the split of
the negative-bank and kNN runs of the same category, seed and regime, so all comparisons are
paired on identical test images.

Readouts:
- few labelled faults per defect type (k = 1, 5, 10): image AUROC, recall and false-alarm rate at
  the validation threshold of the three probes, beside kNN and the negative bank with mask and
  with farthest-10 % negatives (each at its selected lambda). Mean over categories, +- the
  standard deviation over seeds of that mean. A category in which no defect type can spare k
  support images has neither a negative bank nor a trained probe at that k and is left out;
- one defect type held out: the `wce` probe on the unseen and the seen faults, beside kNN and the
  negative bank at lambda = 1 and at the selected lambda. The unit is the held-out type, as in
  the negative bank's own held-out-type table.

The two claim inputs (`claim_5_units`, `claim_6_units`) are used by the claims table.

Every reported probe is fitted with C = 1. The `wce` probe at C = 0.1 and 10 is written to a side
file only (`probe_c_sensitivity`), never to a reported table or a claim: it shows how much the
fixed C matters and is never used to choose it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cmad.data.regimes import GUARD_MARGIN, LODO_SHOTS
from cmad.report.few_abnormal import paired_runs
from cmad.report.lodo import N_FOLDS, _paired, lodo_curves, lodo_runs, subset_members
from cmad.report.runs import (
    BACKBONE_ORDER,
    PROBE_C,
    SEEDS,
    order_backbones,
    require_complete,
    seed_summary,
)
from cmad.report.tables import write_table

LOSSES = ("ce", "wce", "focal")
HEADLINE = "wce"
FEW_K = (1, 5, 10)
CLAIM_K = (1, 5)
SENSITIVITY_C = (0.1, 1.0, 10.0)
_KEYS = ["bb", "k", "category", "seed"]
_LODO_KEYS = ["bb", "category", "held_out", "seed"]
SEED_SD = "standard deviation over the 5 seeds of the mean over categories"


def has_probes(study: pd.DataFrame) -> bool:
    return bool((study["scorer"] == "probe").any())


def probe_few_runs(
    study: pd.DataFrame,
    loss: str,
    c: float = PROBE_C,
    knn: bool = False,
    ks: tuple[int, ...] = FEW_K,
) -> pd.DataFrame:
    """The few-abnormal probe runs of one loss and C at the given k, checked for completeness.
    `knn` selects the side variant with the kNN score as an extra input; it is never reported."""
    runs = study[
        (study["family"] == "few_abn")
        & (study["scorer"] == "probe")
        & (study["loss"] == loss)
        & (study["probe_c"] == c)
        & (study["probe_knn"] == knn)
        & study["k"].isin(ks)
    ]
    require_complete(runs, ["bb", "k"], what=f"few-abnormal probe ({loss}, C = {c:g})")
    return runs


def few_methods(
    study: pd.DataFrame,
    probes: list[tuple[str, float, bool]] | None = None,
    ks: tuple[int, ...] = FEW_K,
) -> pd.DataFrame:
    """One row per backbone, k, category, seed and method, with image AUROC, recall and FPR at
    tau_val. Methods: kNN, the negative bank with mask and with farthest-10 % negatives (each
    at its selected lambda, raw units), and the three probes. Only the categories with a
    negative bank at that k; the probe must share each run's split and be trained there."""
    masks = paired_runs(study)
    top = paired_runs(study, source="top_q", top_frac=0.10)
    metrics = ["image_auroc", "recall_at_tau_val", "fpr_at_tau_val"]
    frames = [
        masks[[*_KEYS, *(f"knn_{m}" for m in metrics)]].assign(method="kNN"),
        masks[[*_KEYS, *(f"neg_{m}" for m in metrics)]].assign(method="bank, masks"),
        top[[*_KEYS, *(f"neg_{m}" for m in metrics)]].assign(
            method="bank, farthest 10 %"
        ),
    ]
    frames = [
        f.rename(columns=lambda c: c.removeprefix("knn_").removeprefix("neg_"))
        for f in frames
    ]
    knn = study[(study["family"] == "few_abn") & (study["scorer"] == "knn")]
    knn = knn.set_index(_KEYS)["split_sha1"]
    with_bank = masks[_KEYS].drop_duplicates()
    for loss, c, with_knn in probes or [(loss, PROBE_C, False) for loss in LOSSES]:
        name = _probe_name(loss, c, with_knn)
        runs = probe_few_runs(study, loss, c, with_knn, ks).merge(with_bank, on=_KEYS)
        if not runs["probe_active"].astype(bool).all():
            raise ValueError(
                f"probe ({loss}) untrained in a category with a negative bank"
            )
        if (
            runs.set_index(_KEYS)["split_sha1"]
            != knn.reindex(runs.set_index(_KEYS).index)
        ).any():
            raise ValueError(
                f"a probe ({loss}) run and its kNN run used different splits"
            )
        frames.append(runs[[*_KEYS, *metrics]].assign(method=name))
    out = pd.concat(frames, ignore_index=True)
    return order_backbones(out[out["k"].isin(ks)])


def _probe_name(loss: str, c: float, with_knn: bool) -> str:
    name = f"probe, {loss}" + ("" if c == PROBE_C else f", C = {c:g}")
    return name + (" + kNN score" if with_knn else "")


def few_table(methods: pd.DataFrame) -> pd.DataFrame:
    order = list(dict.fromkeys(methods["method"]))
    table = seed_summary(
        methods,
        ["bb", "k", "method"],
        ["image_auroc", "recall_at_tau_val", "fpr_at_tau_val"],
    )
    table["method"] = pd.Categorical(table["method"], categories=order, ordered=True)
    return (
        order_backbones(table).sort_values(["bb", "k", "method"]).reset_index(drop=True)
    )


def probe_lodo_runs(
    study: pd.DataFrame, loss: str = HEADLINE, c: float = PROBE_C, knn: bool = False
) -> pd.DataFrame:
    """The probe runs with a held-out type, checked: all seeds once in each of the folds."""
    runs = study[
        (study["family"] == "lodo")
        & (study["scorer"] == "probe")
        & (study["loss"] == loss)
        & (study["probe_c"] == c)
        & (study["probe_knn"] == knn)
    ]
    if runs.empty:
        raise ValueError(f"no held-out-type probe runs ({loss}, C = {c:g})")
    problems = []
    for bb, group in runs.groupby("bb", observed=True):
        seeds = group.groupby(["category", "held_out"])["seed"].apply(sorted)
        bad = [fold for fold, found in seeds.items() if found != sorted(SEEDS)]
        if len(seeds) != N_FOLDS or bad:
            problems.append(f"{bb}: {len(seeds)} folds, wrong seeds in {bad}")
    if problems:
        raise ValueError(
            "held-out-type probe runs incomplete:\n  " + "\n  ".join(problems)
        )
    return runs


def score_probe_run(record: dict) -> list[dict]:
    """Image AUROC, recall and FPR at tau_val on the unseen and the seen subset of one run."""
    test, val = record["image_scores"]["test"], record["image_scores"]["val"]
    labels = np.array(test["label"])
    normal = labels == 0
    scores = np.array(test["score"])
    flagged = scores > np.max(val["score"])
    rows = []
    for subset, faults in subset_members(
        test, record["split"]["held_out_type"]
    ).items():
        if subset == "too_small" or not faults.any():
            continue
        keep = normal | faults
        rows.append(
            {
                "subset": subset,
                "image_auroc": 100.0 * roc_auc_score(labels[keep], scores[keep]),
                "recall": 100.0 * float(flagged[faults].mean()),
                "fpr": 100.0 * float(flagged[normal].mean()),
            }
        )
    return rows


def probe_lodo_scores(runs: pd.DataFrame) -> pd.DataFrame:
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
            "split_sha1": run.split_sha1,
            "marked": n_held < LODO_SHOTS + GUARD_MARGIN,
        }
        rows += [{**base, **row} for row in score_probe_run(record)]
    return order_backbones(pd.DataFrame(rows))


def lodo_methods(
    study: pd.DataFrame,
    cs: tuple[float, ...] = (PROBE_C,),
    with_knn: tuple[bool, ...] = (False,),
) -> pd.DataFrame:
    """One row per run, subset and method on the held-out-type folds: kNN (lambda = 0), the
    negative bank at lambda = 1 and at the selected lambda, and the `wce` probe. Marked folds
    (held-out type too small to give support anyway) are left out."""
    bank = lodo_runs(study)
    curves = lodo_curves(bank)
    probe = pd.concat(
        [
            probe_lodo_scores(probe_lodo_runs(study, c=c, knn=knn)).assign(
                method=_probe_name(HEADLINE, c, knn)
            )
            for c in cs
            for knn in with_knn
        ],
        ignore_index=True,
    )
    sha = bank.assign(bb=bank["bb"].astype(str)).set_index(_LODO_KEYS)["split_sha1"]
    mine = probe.drop_duplicates(_LODO_KEYS).set_index(_LODO_KEYS)["split_sha1"]
    if (mine != sha.reindex(mine.index)).any():
        raise ValueError("a probe run and its negative-bank run used different splits")
    frames = []
    for subset in ("unseen", "seen"):
        for compare, name in (
            ("fixed", "bank, λ = 1"),
            ("selected", "bank, selected λ"),
        ):
            paired = _paired(curves, subset, compare)
            if compare == "fixed":
                frames.append(
                    paired.assign(
                        subset=subset,
                        method="kNN",
                        image_auroc=paired["auroc_0"],
                        recall=paired["recall_0"],
                        fpr=paired["fpr_0"],
                    )
                )
            frames.append(
                paired.assign(subset=subset, method=name, image_auroc=paired["auroc"])
            )
    columns = [
        *_LODO_KEYS,
        "marked",
        "subset",
        "method",
        "image_auroc",
        "recall",
        "fpr",
    ]
    bank_rows = pd.concat([f[columns] for f in frames], ignore_index=True)
    probe_rows = probe[columns]
    out = pd.concat([bank_rows, probe_rows], ignore_index=True)
    return order_backbones(out[~out["marked"]])


def lodo_table(methods: pd.DataFrame) -> pd.DataFrame:
    """Per backbone, subset and method: mean over held-out types (seeds averaged first) with
    the standard error over types, as in the negative bank's held-out-type table."""
    order = list(
        dict.fromkeys(["kNN", "bank, λ = 1", "bank, selected λ", *methods["method"]])
    )
    rows = []
    for (bb, subset, method), group in methods.groupby(
        ["bb", "subset", "method"], observed=True
    ):
        units = group.groupby(["category", "held_out"])[
            ["image_auroc", "recall", "fpr"]
        ].mean()
        row = {"bb": bb, "subset": subset, "method": method, "n_types": len(units)}
        for column in ("image_auroc", "recall", "fpr"):
            row[column] = float(units[column].mean())
            row[f"{column}_se"] = float(units[column].std(ddof=1) / np.sqrt(len(units)))
        rows.append(row)
    table = order_backbones(pd.DataFrame(rows))
    table["method"] = pd.Categorical(table["method"], categories=order, ordered=True)
    table["subset"] = pd.Categorical(
        table["subset"], categories=["unseen", "seen"], ordered=True
    )
    return table.sort_values(["bb", "subset", "method"]).reset_index(drop=True)


def _per_category(frame: pd.DataFrame, value: str, levels: list[str]) -> pd.Series:
    keys = ["category", *levels]
    series = frame.groupby(keys)[value].mean()
    for _ in levels:
        series = series.groupby(level=list(range(series.index.nlevels - 1))).mean()
    return series


def claim_5_units(study: pd.DataFrame) -> dict[tuple[int, str], pd.Series]:
    """Per k (1, 5) and backbone: the negative bank with farthest-10 % negatives at its selected
    lambda minus the `wce` probe, image AUROC per category (seeds averaged first). Only the
    categories with a negative bank at that k."""
    methods = few_methods(study)
    wide = methods.pivot_table(
        index=_KEYS, columns="method", values="image_auroc", observed=True
    ).reset_index()
    wide["diff"] = wide["bank, farthest 10 %"] - wide[f"probe, {HEADLINE}"]
    return {
        (k, bb): _per_category(
            wide[(wide["k"] == k) & (wide["bb"] == bb)], "diff", ["seed"]
        )
        for k in CLAIM_K
        for bb in BACKBONE_ORDER
    }


def claim_6_units(study: pd.DataFrame) -> dict[str, pd.Series]:
    """Per backbone: the negative bank at fixed lambda = 1 (mask negatives, normalised units)
    minus the `wce` probe on the unseen subset, image AUROC per category (held-out types
    averaged within a category, seeds within a type first). Marked folds left out."""
    methods = lodo_methods(study)
    unseen = methods[methods["subset"] == "unseen"]
    wide = unseen.pivot_table(
        index=_LODO_KEYS, columns="method", values="image_auroc", observed=True
    ).reset_index()
    wide["diff"] = wide["bank, λ = 1"] - wide[f"probe, {HEADLINE}"]
    return {
        bb: _per_category(wide[wide["bb"] == bb], "diff", ["seed", "held_out"])
        for bb in BACKBONE_ORDER
    }


def _pm(mean: float, spread: float, digits: int) -> str:
    return f"{mean:.{digits}f} ± {spread:.{digits}f}"


def _not_converged(
    study: pd.DataFrame, reported: bool = True, knn: bool | None = None
) -> str:
    """The fits that stopped before the gradient tolerance, among the reported (C = 1) or the
    side (other C) probe runs. A fit can stop short of the tolerance without reaching the
    iteration cap, when the line search can make no more progress in float64."""
    probes = study[study["scorer"] == "probe"]
    side = (probes["probe_c"] != PROBE_C) | probes["probe_knn"].astype(bool)
    probes = probes[side != reported]
    if (
        knn is not None
    ):  # only the side runs with (True) or without (False) the kNN score
        probes = probes[probes["probe_knn"].astype(bool) == knn]
    bad = probes[probes["probe_converged"].eq(False)]  # None when there was no support
    if bad.empty:
        return "Every probe fit converged."
    cells = ", ".join(
        f"{r.bb} {r.category} {r.regime} seed {r.seed} {r.loss} C = {r.probe_c:g}"
        for r in bad.sort_values(["bb", "category", "regime", "seed"]).itertuples()
    )
    return (
        f"Probe fits that stopped before the gradient tolerance ({len(bad)}; none at the "
        f"iteration cap): {cells}."
    )


def write_probes(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    few = few_table(few_methods(study))
    header = [
        "backbone",
        "k",
        "method",
        "image AUROC",
        "recall at τ_val",
        "FPR at τ_val",
    ]
    rows = []
    for r in few.to_dict("records"):
        first = r["method"] == "kNN"
        rows.append(
            [
                str(r["bb"]) if first and r["k"] == FEW_K[0] else "",
                str(r["k"]) if first else "",
                str(r["method"]),
                _pm(r["image_auroc_mean"], r["image_auroc_std"], 2),
                _pm(r["recall_at_tau_val_mean"], r["recall_at_tau_val_std"], 1),
                _pm(r["fpr_at_tau_val_mean"], r["fpr_at_tau_val_std"], 1),
            ]
        )
    note = (
        "Few labelled faults per defect type. The negative bank at its selected λ (raw units). "
        "Probes: L2-regularised logistic regression (C = 1) on one global feature per image "
        "(DINOv2: class token and mean patch token; WRN-50: mean of the bank's patch "
        "features), fitted to convergence; image score = logit. τ_val = the largest score over the "
        "validation normals for every method. Categories where no defect type can spare k "
        f"support images are left out at that k. Mean over categories ± {SEED_SD}. "
        + _not_converged(study)
    )
    paths = write_table(out_dir, "probes_few", few, header, rows, "lllrrr", note)

    if (study["family"] == "lodo").any() and (
        (study["family"] == "lodo") & (study["scorer"] == "probe")
    ).any():
        table = lodo_table(lodo_methods(study))
        header = [
            "backbone",
            "faults",
            "method",
            "types",
            "image AUROC",
            "recall",
            "FPR",
        ]
        rows = []
        for r in table.to_dict("records"):
            first = r["method"] == "kNN"
            rows.append(
                [
                    str(r["bb"]) if first and r["subset"] == "unseen" else "",
                    str(r["subset"]) if first else "",
                    str(r["method"]),
                    str(r["n_types"]),
                    _pm(r["image_auroc"], r["image_auroc_se"], 2),
                    _pm(r["recall"], r["recall_se"], 1),
                    _pm(r["fpr"], r["fpr_se"], 1),
                ]
            )
        note = (
            "One defect type held out of the support in each run. Unseen: the held-out type's "
            "faults; seen: the types with support; both against all normal test images. Unit: "
            "the held-out type (seeds averaged first); mean ± standard error over types, the "
            "unit of the negative bank's held-out-type table. Folds whose held-out type is too "
            "small to give support anyway are left out. Recall and FPR at each method's own "
            "τ_val; the negative bank's λ in normalised units, mask negatives."
        )
        paths += write_table(
            out_dir, "probes_lodo", table, header, rows, "lllrrrr", note
        )
    return paths + write_c_sensitivity(study, out_dir) + write_knn_side(study, out_dir)


def write_c_sensitivity(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    """The `wce` probe at C = 0.1, 1 and 10, for the author and the critic only: not a report
    table, and never used to choose C."""
    probes = study[
        (study["scorer"] == "probe")
        & (study["loss"] == HEADLINE)
        & ~study["probe_knn"].astype(bool)
    ]
    have = set(probes["probe_c"].dropna())
    if not set(SENSITIVITY_C) <= have:
        print(f"C sensitivity: runs at C = {sorted(have)} only; skipped")
        return []
    few = few_table(few_methods(study, [(HEADLINE, c, False) for c in SENSITIVITY_C]))
    few = few[few["method"].astype(str).str.startswith("probe")]
    header = [
        "backbone",
        "k",
        "probe",
        "image AUROC",
        "recall at τ_val",
        "FPR at τ_val",
    ]
    rows = [
        [
            str(r["bb"]),
            str(r["k"]),
            str(r["method"]),
            _pm(r["image_auroc_mean"], r["image_auroc_std"], 2),
            _pm(r["recall_at_tau_val_mean"], r["recall_at_tau_val_std"], 1),
            _pm(r["fpr_at_tau_val_mean"], r["fpr_at_tau_val_std"], 1),
        ]
        for r in few.to_dict("records")
    ]
    note = (
        "NOT FOR THE REPORT. The wCE probe at C = 0.1 and 10 beside the reported C = 1, few "
        f"labelled faults per defect type. Mean over categories ± {SEED_SD}. Descriptive only: "
        "C is fixed at 1 in advance and never chosen from these rows. "
        + _not_converged(study, reported=False, knn=False)
    )
    paths = write_table(
        out_dir, "probe_c_sensitivity", few, header, rows, "lllrrr", note
    )
    lodo = probes[probes["family"] == "lodo"]
    if set(SENSITIVITY_C) <= set(lodo["probe_c"].dropna()):
        table = lodo_table(lodo_methods(study, SENSITIVITY_C))
        table = table[table["method"].astype(str).str.startswith("probe")]
        header = [
            "backbone",
            "faults",
            "probe",
            "types",
            "image AUROC",
            "recall",
            "FPR",
        ]
        rows = [
            [
                str(r["bb"]),
                str(r["subset"]),
                str(r["method"]),
                str(r["n_types"]),
                _pm(r["image_auroc"], r["image_auroc_se"], 2),
                _pm(r["recall"], r["recall_se"], 1),
                _pm(r["fpr"], r["fpr_se"], 1),
            ]
            for r in table.to_dict("records")
        ]
        note = (
            "NOT FOR THE REPORT. The wCE probe at C = 0.1, 1 and 10 with one defect type held "
            "out; unit = held-out type, mean ± standard error over types."
        )
        paths += write_table(
            out_dir, "probe_c_sensitivity_lodo", table, header, rows, "lllrrrr", note
        )
    return paths


KNN_SIDE_K = (1, 5)


def write_knn_side(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    """The `wce` probe (C = 1) with the image's kNN score as one extra input, beside kNN, the
    reported probe and the negative bank with masks. For the author's curiosity only: never a
    report table or a claim."""
    side = study[(study["scorer"] == "probe") & study["probe_knn"].astype(bool)]
    if side.empty:
        return []
    few = few_methods(
        study, [(HEADLINE, PROBE_C, False), (HEADLINE, PROBE_C, True)], ks=KNN_SIDE_K
    )
    few = few_table(few[few["method"] != "bank, farthest 10 %"])
    header = [
        "backbone",
        "k",
        "method",
        "image AUROC",
        "recall at τ_val",
        "FPR at τ_val",
    ]
    rows = [
        [
            str(r["bb"]),
            str(r["k"]),
            str(r["method"]),
            _pm(r["image_auroc_mean"], r["image_auroc_std"], 2),
            _pm(r["recall_at_tau_val_mean"], r["recall_at_tau_val_std"], 1),
            _pm(r["fpr_at_tau_val_mean"], r["fpr_at_tau_val_std"], 1),
        ]
        for r in few.to_dict("records")
    ]
    note = (
        "NOT FOR THE REPORT. The wCE probe (C = 1) with the image's one-class kNN score "
        "appended to its input (the backbone's own rule; training normals scored "
        "leave-one-image-out, without their own patches in the bank), beside kNN, the "
        "reported probe and the negative bank with mask negatives at its selected λ. "
        f"Mean over categories ± {SEED_SD}. "
        + _not_converged(study, reported=False, knn=True)
    )
    paths = write_table(out_dir, "probe_knn_side", few, header, rows, "lllrrr", note)
    if (side["family"] == "lodo").any():
        table = lodo_table(lodo_methods(study, with_knn=(False, True)))
        table = table[table["method"].astype(str) != "bank, selected λ"]
        header = [
            "backbone",
            "faults",
            "method",
            "types",
            "image AUROC",
            "recall",
            "FPR",
        ]
        rows = [
            [
                str(r["bb"]),
                str(r["subset"]),
                str(r["method"]),
                str(r["n_types"]),
                _pm(r["image_auroc"], r["image_auroc_se"], 2),
                _pm(r["recall"], r["recall_se"], 1),
                _pm(r["fpr"], r["fpr_se"], 1),
            ]
            for r in table.to_dict("records")
        ]
        note = (
            "NOT FOR THE REPORT. As above, with one defect type held out; the negative bank "
            "at λ = 1 (masks, normalised). Unit = held-out type, mean ± standard error over "
            "types."
        )
        paths += write_table(
            out_dir, "probe_knn_side_lodo", table, header, rows, "lllrrrr", note
        )
    return paths
