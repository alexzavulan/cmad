"""Gate 2 table: our `anomalydino` runs against the official code's runs and the published means.

python scripts/gate2_summary.py --runs $CMAD_DATA_ROOT/results/runs --official <results_MVTec/dinov2_vits14_448>
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

PUBLISHED = {1: 96.5, 16: 98.3}  # AnomalyDINO-S (448), image AUROC, Table in the paper
MARGIN = 1.0
SEEDS = (0, 1, 2)


def ours(runs_dir: Path, shots: int) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], tuple[str, float]] = {}
    for path in sorted(Path(runs_dir).glob("*.json")):
        r = json.loads(path.read_text())
        spec = r.get("regime_spec", {})
        if not (
            r.get("scorer") == "anomalydino"
            and r.get("backbone") == "dinov2_vits14"
            and r.get("roi") == "pca_official_mvtec"
            and r.get("reference_rotations")
            == [float(a) for a in (0, 45, 90, 135, 180, 225, 270, 315)]
            and spec.get("n_normal_shots") == shots
            and spec.get("shot_selection") == "contiguous_blocks"
        ):
            continue
        key = (r["category"], r["seed"])
        if key not in out or r["timestamp"] > out[key][0]:
            out[key] = (r["timestamp"], 100 * r["metrics"]["image_auroc"])
    return {k: v[1] for k, v in out.items()}


def official(root: Path, shots: int) -> dict[tuple[str, int], float]:
    out = {}
    for seed in SEEDS:
        f = (
            Path(root)
            / f"{shots}-shot_preprocess=agnostic"
            / f"metrics_seed={seed}.json"
        )
        if f.exists():
            for cat, m in json.loads(f.read_text()).items():
                if isinstance(m, dict):
                    out[(cat, seed)] = 100 * m["classification_AUROC"]
    return out


def summarise(runs_dir: Path, official_dir: Path | None) -> str:
    lines = []
    for shots, published in PUBLISHED.items():
        mine = ours(runs_dir, shots)
        ref = official(official_dir, shots) if official_dir else {}
        cats = sorted({c for c, _ in mine})
        lines += [f"## {shots}-shot (published {published}, margin {MARGIN} pt)", ""]
        lines += [
            "| category | ours (3-seed mean) | official code | delta |",
            "|---|---|---|---|",
        ]
        for cat in cats:
            a = [mine[(cat, s)] for s in SEEDS if (cat, s) in mine]
            b = [ref[(cat, s)] for s in SEEDS if (cat, s) in ref]
            ma, mb = st.mean(a), (st.mean(b) if b else None)
            tag = f"{len(a)}/3 seeds" if len(a) < 3 else ""
            lines.append(
                f"| {cat} | {ma:.1f} {tag} | {'-' if mb is None else f'{mb:.1f}'} | "
                f"{'-' if mb is None else f'{ma - mb:+.1f}'} |"
            )
        full = [s for s in SEEDS if all((c, s) in mine for c in cats)]
        if cats and full:
            per_seed = [st.mean(mine[(c, s)] for c in cats) for s in full]
            mean = st.mean(per_seed)
            verdict = (
                "PASS"
                if abs(mean - published) <= MARGIN
                and len(cats) == 15
                and len(full) == 3
                else ("incomplete" if len(cats) < 15 or len(full) < 3 else "FAIL")
            )
            off = [
                st.mean(ref[(c, s)] for c in cats)
                for s in full
                if all((c, s) in ref for c in cats)
            ]
            lines += [
                "",
                f"mean over {len(cats)} categories, seeds {full}: **{mean:.2f}** "
                f"(per seed {[round(x, 2) for x in per_seed]}); published {published}, "
                f"official code {st.mean(off):.2f}"
                if off
                else f"mean **{mean:.2f}**",
                f"verdict: **{verdict}**",
            ]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--official", type=Path, default=None)
    args = p.parse_args()
    print(summarise(args.runs, args.official))


if __name__ == "__main__":
    main()
