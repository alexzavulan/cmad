"""Build every table from the result files.

python -m cmad.report --runs $CMAD_DATA_ROOT/results/runs --out report/generated
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cmad.report.few_abnormal import write_few_abnormal
from cmad.report.gate import REFERENCE_DIR, build_report
from cmad.report.lambda_curves import write_lambda
from cmad.report.neg_source import write_neg_source
from cmad.report.normal_only import write_normal_only
from cmad.report.runs import load_records, study_runs
from cmad.report.shots import write_shots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cmad.report")
    parser.add_argument(
        "--runs", type=Path, default=None, help="directory of per-run JSON files"
    )
    parser.add_argument("--reference", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--out", type=Path, default=Path("report/generated"))
    args = parser.parse_args(argv)

    runs_dir = args.runs
    if runs_dir is None:
        root = os.environ.get("CMAD_DATA_ROOT")
        if not root:
            parser.error("pass --runs or set CMAD_DATA_ROOT")
        runs_dir = Path(root).expanduser() / "results" / "runs"

    args.out.mkdir(parents=True, exist_ok=True)
    written = [args.out / "gate1.md"]
    written[0].write_text(build_report(runs_dir, args.reference))

    records = load_records(runs_dir)
    study = study_runs(records)
    print(f"{len(records)} result files, {len(study)} study runs")
    # the DINOv2 middle block, run in the one-class setting only
    middle = study_runs(
        records, layers={"dinov2_vits14": "8_raw", "dinov2_vitb14": "8_raw"}
    )
    written += write_normal_only(study, middle[middle["layers"] == "8_raw"], args.out)
    written += write_few_abnormal(study, args.out)
    written += write_neg_source(study, args.out)
    written += write_shots(study, args.out)
    written += write_lambda(study, args.out)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
