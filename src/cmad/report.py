"""Report v0: Gate 1 table -- our runs against the published PatchCore numbers.

python -m cmad.report --runs $CMAD_DATA_ROOT/results/runs --out report/generated
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "reference"
# Gate 1 (spec sec 14): mean over the 15 categories within these margins (percentage points)
GATE_IMAGE_AUROC_PT = 0.5
GATE_PRO_PT = 1.0
N_CATEGORIES = 15
AUPRO_COL = "aupro_0.3"
REFERENCE_FILES = {
    "image_auroc": "patchcore-mvtec-image-auroc.json",
    "pixel_auroc": "patchcore-mvtec-pixel-auroc.json",
    AUPRO_COL: "patchcore-mvtec-pro.json",
}


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per result file: factors flattened, metrics in percent (None stays NaN)."""
    rows = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        r = json.loads(path.read_text())
        row = {
            k: r[k]
            for k in (
                "config_hash",
                "timestamp",
                "dataset",
                "category",
                "regime",
                "seed",
                "backbone",
                "scorer",
                "res",
                "resize",
                "with_aupro",
            )
        }
        row["layers"] = "+".join(r["layers"])
        row["metric"] = r["scorer_params"]["metric"]
        row["coreset"] = r["scorer_params"]["coreset_fraction"]
        row["k"] = r["scorer_params"]["k"]
        for name, value in r["metrics"].items():
            row[name] = None if value is None else 100.0 * value
        row["fit_s"] = r["timing"]["fit_s"]
        rows.append(row)
    return pd.DataFrame(rows)


def variant_label(row: pd.Series) -> str:
    resize = "whole-image" if pd.isna(row["resize"]) else f"resize{int(row['resize'])}"
    return (
        f"{row['backbone']}/{row['layers']} {row['metric']} "
        f"coreset={row['coreset']:g} k={row['k']} res{int(row['res'])} {resize}"
    )


def best_per_factors(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicates of the same factors (e.g. a laptop run without AUPRO and the
    cluster run with it): prefer the run that has AUPRO, then the newest."""
    runs = runs.copy()
    runs["variant"] = runs.apply(variant_label, axis=1)
    keys = ["variant", "category", "regime", "seed", "dataset", "scorer"]
    has = runs[AUPRO_COL].notna() if AUPRO_COL in runs else False
    runs = runs.assign(_has=has).sort_values(["_has", "timestamp"])
    return runs.drop_duplicates(keys, keep="last").drop(columns="_has")


def load_reference(
    coreset_fraction: float, reference_dir: Path = REFERENCE_DIR
) -> dict[str, dict]:
    """metric -> {"mean": float, "per_category": {...}} for the matching coreset size."""
    percent = round(coreset_fraction * 100)
    out = {}
    for metric, filename in REFERENCE_FILES.items():
        data = json.loads((Path(reference_dir) / filename).read_text())
        for variant in data["variants"].values():
            if variant["coreset_percent"] == percent:
                out[metric] = variant
    return out


def _verdict(mean: float | None, ref: float, margin: float, n: int) -> str:
    if mean is None or pd.isna(mean):
        return "not evaluated"
    if n < N_CATEGORIES:
        return f"incomplete ({n}/{N_CATEGORIES})"
    return "PASS" if abs(mean - ref) <= margin else "FAIL"


def gate_table(runs: pd.DataFrame, reference: dict[str, dict]) -> tuple[str, dict]:
    """Markdown per-category table (ours | ref | delta) and the gate verdicts for one variant."""
    cols = [m for m in REFERENCE_FILES if m in runs and m in reference]
    runs = runs.set_index("category")
    cats = sorted(set(runs.index) | set(reference[cols[0]]["per_category"]))
    header = ["category"] + [f"{m} ours / ref / Δ" for m in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for cat in cats:
        cells = [cat]
        for m in cols:
            ref = reference[m]["per_category"].get(cat)
            ours = runs[m].get(cat) if cat in runs.index else None
            if ours is None or pd.isna(ours):
                cells.append(f"— / {ref:.1f} / —")
            else:
                cells.append(f"{ours:.1f} / {ref:.1f} / {ours - ref:+.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    verdicts = {}
    mean_cells = ["**mean**"]
    for m in cols:
        ours = runs[m].dropna()
        mean = float(ours.mean()) if len(ours) else None
        ref = reference[m]["mean"]
        margin = {"image_auroc": GATE_IMAGE_AUROC_PT, AUPRO_COL: GATE_PRO_PT}.get(m)
        verdict = (
            "no gate" if margin is None else _verdict(mean, ref, margin, len(ours))
        )
        verdicts[m] = {"mean": mean, "ref": ref, "n": len(ours), "verdict": verdict}
        cell = "—" if mean is None else f"{mean:.2f} / {ref:.1f} / {mean - ref:+.2f}"
        mean_cells.append(f"**{cell}** {verdict}")
    lines.append("| " + " | ".join(mean_cells) + " |")
    return "\n".join(lines), verdicts


def build_report(runs_dir: Path, reference_dir: Path = REFERENCE_DIR) -> str:
    runs = load_runs(runs_dir)
    if runs.empty:
        return f"No result files in {runs_dir}.\n"
    runs = best_per_factors(runs[runs["regime"] == "oneclass"])
    parts = ["# Gate 1: PatchCore reproduction\n"]
    parts.append(
        f"Pass: mean image AUROC within {GATE_IMAGE_AUROC_PT} pt and mean AUPRO@0.3 "
        f"within {GATE_PRO_PT} pt of the published PatchCore numbers "
        f"(15 categories, percent). Pixel AUROC is a debugging aid, not gated.\n"
    )
    for variant, group in runs.groupby("variant"):
        reference = load_reference(float(group["coreset"].iloc[0]), reference_dir)
        parts.append(f"## {variant}\n")
        if AUPRO_COL not in reference or "image_auroc" not in reference:
            parts.append("No published reference for this coreset size.\n")
            continue
        table, _ = gate_table(group, reference)
        parts.append(table + "\n")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cmad.report")
    parser.add_argument("--runs", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--out", type=Path, default=None, help="directory for gate1.md")
    args = parser.parse_args(argv)

    runs_dir = args.runs
    if runs_dir is None:
        root = os.environ.get("CMAD_DATA_ROOT")
        if not root:
            parser.error("pass --runs or set CMAD_DATA_ROOT")
        runs_dir = Path(root).expanduser() / "results" / "runs"
    text = build_report(runs_dir, args.reference)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "gate1.md").write_text(text)


if __name__ == "__main__":
    main()
