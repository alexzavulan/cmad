import json
from pathlib import Path

from cmad.report import build_report, load_reference

CATS = ["bottle", "cable"]


def _run(
    tmp: Path, cat: str, auroc: float, aupro: float | None, stamp: str, name: str
) -> None:
    record = {
        "config_hash": name,
        "timestamp": stamp,
        "dataset": "mvtec_ad",
        "category": cat,
        "regime": "oneclass",
        "seed": 0,
        "backbone": "wrn50_in1k",
        "layers": ["cnn_l23_raw"],
        "scorer": "patchcore",
        "res": 224,
        "resize": 256,
        "with_aupro": aupro is not None,
        "scorer_params": {"metric": "euclidean", "coreset_fraction": 0.1, "k": 1},
        "metrics": {
            "image_auroc": auroc,
            "pixel_auroc": 0.98,
            "aupro_0.3": aupro,
        },
        "timing": {"fit_s": 1.0},
    }
    (tmp / f"{name}.json").write_text(json.dumps(record))


def test_reference_lookup_by_coreset_fraction() -> None:
    ref = load_reference(0.1)

    assert ref["image_auroc"]["mean"] == 99.0
    assert ref["aupro_0.3"]["mean"] == 93.5
    assert ref["image_auroc"]["per_category"]["bottle"] == 100.0


def test_incomplete_run_is_never_a_pass(tmp_path: Path) -> None:
    _run(tmp_path, "bottle", 1.0, 0.96, "2026-01-01", "a")

    text = build_report(tmp_path)

    assert "incomplete (1/15)" in text
    assert "PASS" not in text


def test_run_with_aupro_beats_a_newer_run_without(tmp_path: Path) -> None:
    _run(tmp_path, "bottle", 1.0, 0.961, "2026-01-01", "cluster")
    _run(tmp_path, "bottle", 1.0, None, "2026-02-01", "laptop")

    text = build_report(tmp_path)

    assert "96.1 / 96.1 / +0.0" in text
    assert "not evaluated" not in text


def test_full_pass_and_fail_use_the_gate_margins(tmp_path: Path) -> None:
    from cmad.report import REFERENCE_FILES

    ref = load_reference(0.1)
    cats = list(ref["image_auroc"]["per_category"])
    for i, cat in enumerate(cats):
        _run(
            tmp_path,
            cat,
            ref["image_auroc"]["per_category"][cat] / 100,
            ref["aupro_0.3"]["per_category"][cat] / 100 - 0.02,  # 2 pt low everywhere
            "2026-01-01",
            f"r{i}",
        )

    text = build_report(tmp_path)

    assert len(REFERENCE_FILES) == 3
    assert "PASS" in text  # image AUROC equals the reference
    assert "FAIL" in text  # AUPRO 2 pt low exceeds the 1 pt margin
