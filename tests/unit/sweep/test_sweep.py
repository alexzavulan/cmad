import json
from pathlib import Path

import pytest

from cmad import sweep
from cmad.sweep import (
    CONFIG_DIR,
    Experiment,
    Task,
    check_fingerprint,
    compose_run,
    expand,
    fingerprint,
    launch_slurm,
    load_experiment,
    parse_args,
    queued_jobs,
    render_sbatch,
    run_task,
    select,
)


def _config_dir(tmp_path: Path, experiment: str, categories: str = "[a, b, c]") -> Path:
    (tmp_path / "tier").mkdir()
    (tmp_path / "experiment").mkdir()
    (tmp_path / "tier" / "t.yaml").write_text(
        f"dataset: ds\ncategories: {categories}\n"
    )
    (tmp_path / "experiment" / "e.yaml").write_text(experiment)
    return tmp_path


EXPERIMENT = """
tier: t
walltime: "00:30:00"
overrides: [scorer=negbank]
grid:
  scorer.source: [gt_mask, top_q]
  seed: [0, 1, 2]
"""


def test_one_task_per_category_holding_the_whole_grid(tmp_path: Path) -> None:
    exp = load_experiment("e", _config_dir(tmp_path, EXPERIMENT))
    tasks = expand(exp)

    assert [t.category for t in tasks] == ["a", "b", "c"]
    assert all(len(t.runs) == 6 for t in tasks)
    assert tasks[1].runs[0] == (
        "dataset=ds",
        "category=b",
        "scorer=negbank",
        "scorer.source=gt_mask",
        "seed=0",
    )
    # the last grid factor varies fastest, and no two runs are alike
    assert [r[-1] for r in tasks[0].runs[:3]] == ["seed=0", "seed=1", "seed=2"]
    assert len(set(tasks[0].runs)) == 6
    assert exp.walltime == "00:30:00"


def test_more_categories_than_the_submit_limit_is_refused(tmp_path: Path) -> None:
    many = "[" + ", ".join(f"c{i}" for i in range(31)) + "]"
    exp = load_experiment("e", _config_dir(tmp_path, EXPERIMENT, many))
    with pytest.raises(ValueError, match="submit limit"):
        expand(exp)


@pytest.mark.parametrize(
    "text, message",
    [
        (
            "tier: t\nwalltime: '1:00:00'\noverrides: [seed=1]\ngrid:\n  seed: [0]\n",
            "both",
        ),
        ("tier: t\nwalltime: '1:00:00'\ngrid:\n  seed: []\n", "no values"),
        ("tier: t\nwalltime: '1:00:00'\ngrid:\n  seed: [[0, 1]]\n", "scalars"),
        ("tier: t\nwalltime: '1:00:00'\ngrid:\n  category: [x]\n", "tier"),
    ],
)
def test_malformed_experiments_are_rejected(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_experiment("e", _config_dir(tmp_path, text))


def test_select_filters_tasks_by_category() -> None:
    tasks = [Task("a", ()), Task("b", ()), Task("c", ())]
    assert [t.category for t in select(tasks, "b,c")] == ["b", "c"]
    assert select(tasks, None) == tasks
    with pytest.raises(ValueError, match="matches no task"):
        select(tasks, "z")


def test_the_shipped_experiment_composes_into_valid_run_configs() -> None:
    exp = load_experiment("negbank_dev")
    tasks = expand(exp)
    assert [t.category for t in tasks] == ["bottle", "screw", "carpet"]

    seen = set()
    for overrides in tasks[1].runs:
        cfg = compose_run(overrides)
        assert cfg["category"] == "screw" and cfg["dataset"] == "mvtec_ad"
        assert cfg["scorer"]["name"] == "negbank"
        assert cfg["scorer"]["lam"] == "select"
        seen.add((cfg["scorer"]["source"], cfg["regime"]["name"], cfg["seed"]))
    assert len(seen) == 18


def _record(auroc: float | None = 0.9) -> dict:
    return {"run_id": "r", "metrics": {"image_auroc": auroc}}


def test_run_task_counts_ran_and_cached_and_keeps_going_after_a_failure(
    capsys: pytest.CaptureFixture,
) -> None:
    exp = Experiment("e", "ds", ("a",), "1:00:00", (), {"seed": (0, 1, 2, 3)})
    task = expand(exp)[0]
    calls = []

    def fake_run(cfg: dict, root: Path) -> tuple[dict, bool]:
        calls.append(cfg["seed"])
        if cfg["seed"] == 1:
            raise RuntimeError("boom")
        return _record(), cfg["seed"] != 2

    summary = run_task(task, Path("."), run=fake_run)

    assert calls == [0, 1, 2, 3]  # the failure at seed 1 did not stop the task
    assert (summary.ran, summary.cached) == (2, 1)
    assert summary.failed == ["seed=1"]
    assert "FAILED" in capsys.readouterr().out


def test_the_shipped_template_fills_completely() -> None:
    text = render_sbatch(
        sweep.TEMPLATE.read_text(),
        {
            "experiment": "negbank_dev",
            "n_tasks_minus_1": 2,
            "walltime": "01:30:00",
            "results_dir": "/ceph/home/u/cmad-data/results",
            "data_root": "/ceph/home/u/cmad-data",
            "task_list": "/ceph/home/u/cmad-data/results/sweeps/negbank_dev/tasks.txt",
            "sweep_dir": "/ceph/home/u/cmad-data/results/sweeps/negbank_dev",
            "repo": "/ceph/home/u/cmad",
        },
    )

    assert "{{" not in text
    assert "#SBATCH --array=0-2%8" in text
    assert "#SBATCH --time=01:30:00" in text
    assert "#SBATCH --output=/ceph/home/u/cmad-data/results/logs/%x-%a.out" in text
    assert "cd /ceph/home/u/cmad && " in text
    assert (
        "launcher=local sweep_dir=/ceph/home/u/cmad-data/results/sweeps/negbank_dev $TASK"
        in text
    )


def test_an_unfilled_template_field_is_an_error() -> None:
    with pytest.raises(ValueError, match="unfilled"):
        render_sbatch("#SBATCH --time={{walltime}} {{oops}}", {"walltime": "1:00:00"})


def _slurm_setup(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _config_dir(
        cfg_dir, "tier: t\nwalltime: '00:30:00'\ngrid:\n  seed: [0, 1]\n", "[a, b]"
    )
    exp = load_experiment("e", cfg_dir)
    return exp, expand(exp), tmp_path / "data", cfg_dir


def _launch(tmp_path: Path, **kwargs):
    exp, tasks, root, cfg_dir = _slurm_setup(tmp_path)
    kwargs.setdefault("git_head", lambda: "abc123")
    return launch_slurm(exp, tasks, root, config_dir=cfg_dir, **kwargs), root, cfg_dir


def test_slurm_launcher_writes_files_but_does_not_submit_by_default(
    tmp_path: Path,
) -> None:
    def never(path: Path) -> str:
        raise AssertionError("submitted without submit=true")

    script, root, _ = _launch(tmp_path, sbatch=never)

    assert (
        root / "results" / "logs"
    ).is_dir()  # sbatch fails on a missing log directory
    assert script.read_text().count("#SBATCH --array=0-1%8") == 1
    assert (script.parent / "tasks.txt").read_text() == "category=a\ncategory=b\n"
    assert (
        json.loads((script.parent / "fingerprint.json").read_text())["git_head"]
        == "abc123"
    )


def test_slurm_launcher_submits_when_asked_and_there_is_room(tmp_path: Path) -> None:
    submitted = []

    _launch(
        tmp_path,
        submit=True,
        squeue=lambda: "1234\n",
        sbatch=lambda p: submitted.append(p) or "Submitted batch job 9",
    )

    assert len(submitted) == 1


def test_slurm_launcher_refuses_to_pass_the_submit_limit(tmp_path: Path) -> None:
    queued = "\n".join(f"55_{i}" for i in range(29))  # 29 + 2 new tasks > 30

    with pytest.raises(SystemExit, match="submit limit"):
        _launch(
            tmp_path,
            submit=True,
            squeue=lambda: queued,
            sbatch=lambda p: pytest.fail("submitted"),
        )


def test_queued_jobs_counts_every_job_of_the_user() -> None:
    assert queued_jobs(lambda: "") == 0
    assert queued_jobs(lambda: "12\n13\n77_4\n") == 3


def test_a_task_refuses_to_run_after_the_grid_or_the_code_changed(
    tmp_path: Path,
) -> None:
    script, _, cfg_dir = _launch(tmp_path)
    sweep_dir = script.parent
    same = fingerprint("e", cfg_dir, lambda: "abc123")
    check_fingerprint(sweep_dir, same)  # nothing changed: fine
    check_fingerprint(
        sweep_dir, fingerprint("e", cfg_dir, lambda: None)
    )  # no git: skipped

    with pytest.raises(SystemExit, match="git_head changed"):
        check_fingerprint(sweep_dir, fingerprint("e", cfg_dir, lambda: "def456"))

    exp_file = cfg_dir / "experiment" / "e.yaml"
    exp_file.write_text(exp_file.read_text() + "  # edited\n")
    with pytest.raises(SystemExit, match="config_sha1 changed"):
        check_fingerprint(sweep_dir, fingerprint("e", cfg_dir, lambda: "abc123"))


def test_a_changed_tier_also_counts_as_a_changed_grid(tmp_path: Path) -> None:
    script, _, cfg_dir = _launch(tmp_path)
    (cfg_dir / "tier" / "t.yaml").write_text("dataset: ds\ncategories: [a]\n")

    with pytest.raises(SystemExit, match="config_sha1 changed"):
        check_fingerprint(script.parent, fingerprint("e", cfg_dir, lambda: "abc123"))


def test_arguments_are_checked() -> None:
    assert parse_args(["experiment=e"]) == {"experiment": "e", "launcher": "local"}
    assert parse_args(["experiment=e", "sweep_dir=/x"])["sweep_dir"] == "/x"
    with pytest.raises(SystemExit):
        parse_args(["launcher=local"])  # no experiment
    with pytest.raises(SystemExit):
        parse_args(["experiment=e", "launcher=cloud"])
    with pytest.raises(SystemExit):
        parse_args(["experiment=e", "seeds=0-4"])
    assert CONFIG_DIR.is_dir()
