"""Expand an experiment into runs and execute them locally or as a Slurm job array.

    python -m cmad.sweep experiment=negbank_dev launcher=dry                 # list what would run
    python -m cmad.sweep experiment=negbank_dev launcher=local               # run everything here
    python -m cmad.sweep experiment=negbank_dev launcher=local category=screw   # one task's slice
    python -m cmad.sweep experiment=negbank_dev launcher=slurm               # write the array files
    python -m cmad.sweep experiment=negbank_dev launcher=slurm submit=true   # ... and sbatch them

An experiment (configs/experiment/<name>.yaml) names a tier (its dataset and categories), base
overrides, and a grid of factors to cross. One task is one category; the grid loops inside it, so
a job array never exceeds MAX_TASKS entries. Finished runs are found through run_one's own result
files, so a task killed by the walltime is resubmitted and continues where it stopped.

Everything except the local launcher needs only PyYAML, so `launcher=slurm` runs on the cluster
front-end with its plain python (hydra and torch are imported only when runs actually execute).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"
TEMPLATE = REPO / "slurm" / "array.sbatch"
MAX_TASKS = 30  # the cluster's per-user submit limit
ARGUMENTS = {"experiment", "launcher", "category", "data_root", "submit", "sweep_dir"}
LAUNCHERS = ("local", "slurm", "dry")


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str
    categories: tuple[str, ...]
    walltime: str
    overrides: tuple[str, ...]
    grid: Mapping[str, tuple]


@dataclass(frozen=True)
class Task:
    """All runs of one category: each run is the list of Hydra overrides that defines it."""

    category: str
    runs: tuple[tuple[str, ...], ...]


@dataclass
class Summary:
    ran: int = 0
    cached: int = 0
    failed: list[str] = field(default_factory=list)


def _override_key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+~")


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError(f"grid values must be plain scalars, got {value!r}")


def load_experiment(name: str, config_dir: Path = CONFIG_DIR) -> Experiment:
    path = Path(config_dir) / "experiment" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no experiment {name!r}: {path} does not exist")
    cfg = yaml.safe_load(path.read_text())
    tier = yaml.safe_load(
        (Path(config_dir) / "tier" / f"{cfg['tier']}.yaml").read_text()
    )
    overrides = tuple(str(o) for o in cfg.get("overrides") or [])
    grid = {k: tuple(v) for k, v in (cfg.get("grid") or {}).items()}
    for key, values in grid.items():
        if not values:
            raise ValueError(f"grid factor {key!r} has no values")
        for v in values:
            _format(v)
    base_keys = {_override_key(o) for o in overrides}
    if clash := base_keys & grid.keys():
        raise ValueError(f"{sorted(clash)} are both a base override and a grid factor")
    if reserved := (base_keys | grid.keys()) & {"category", "dataset"}:
        raise ValueError(f"{sorted(reserved)} come from the tier, not the experiment")
    return Experiment(
        name=name,
        dataset=tier["dataset"],
        categories=tuple(tier["categories"]),
        walltime=str(cfg["walltime"]),
        overrides=overrides,
        grid=grid,
    )


def expand(exp: Experiment) -> list[Task]:
    """One task per category, each holding the full grid in a fixed order."""
    if len(exp.categories) > MAX_TASKS:
        raise ValueError(
            f"{len(exp.categories)} categories exceed the {MAX_TASKS}-task submit limit"
        )
    keys = list(exp.grid)
    combos = [
        [f"{k}={_format(v)}" for k, v in zip(keys, values)]
        for values in itertools.product(*(exp.grid[k] for k in keys))
    ]
    tasks = []
    for category in exp.categories:
        head = [f"dataset={exp.dataset}", f"category={category}", *exp.overrides]
        tasks.append(Task(category, tuple(tuple(head + combo) for combo in combos)))
    return tasks


def select(tasks: Sequence[Task], category: str | None) -> list[Task]:
    if category is None:
        return list(tasks)
    chosen = [t for t in tasks if t.category in category.split(",")]
    if not chosen:
        raise ValueError(f"category {category!r} matches no task")
    return chosen


def compose_run(overrides: Sequence[str], config_dir: Path = CONFIG_DIR) -> dict:
    """The fully resolved run config for one override list (what `python -m cmad.run` builds)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def run_task(
    task: Task,
    data_root: Path,
    config_dir: Path = CONFIG_DIR,
    run: Callable[[dict, Path], tuple[dict, bool]] | None = None,
) -> Summary:
    """Run every run of a task in this process. One failing run is reported and skipped."""
    if run is None:
        from cmad.run import run_one as run
    summary = Summary()
    for i, overrides in enumerate(task.runs, 1):
        label = " ".join(overrides[2:])
        try:
            record, ran = run(compose_run(overrides, config_dir), data_root)
        except Exception:  # noqa: BLE001 - one bad run must not stop the rest of the task
            traceback.print_exc()
            _free_gpu_memory()
            summary.failed.append(label)
            print(f"[{task.category} {i}/{len(task.runs)}] FAILED {label}", flush=True)
            continue
        summary.ran += ran
        summary.cached += not ran
        auroc = record["metrics"]["image_auroc"]
        print(
            f"[{task.category} {i}/{len(task.runs)}] {'ran' if ran else 'cached'} "
            f"{record['run_id']} image_auroc={'n/a' if auroc is None else f'{auroc:.4f}'} {label}",
            flush=True,
        )
    return summary


def _free_gpu_memory() -> None:
    """After a failed run (typically out of memory) hand the cached GPU blocks back."""
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fingerprint(
    name: str,
    config_dir: Path = CONFIG_DIR,
    git_head: Callable[[], str | None] = _git_head,
) -> dict:
    """What a sweep was planned from: the experiment and tier files, and the code version."""
    path = Path(config_dir) / "experiment" / f"{name}.yaml"
    text = path.read_bytes()
    tier = yaml.safe_load(text)["tier"]
    text += (Path(config_dir) / "tier" / f"{tier}.yaml").read_bytes()
    return {"config_sha1": hashlib.sha1(text).hexdigest(), "git_head": git_head()}


def check_fingerprint(sweep_dir: Path, now: Mapping) -> None:
    """Refuse to run a task whose grid or code differs from what was submitted."""
    planned = json.loads((Path(sweep_dir) / "fingerprint.json").read_text())
    for key, value in planned.items():
        if (
            now.get(key) is None
        ):  # e.g. no git inside the container: cannot compare, not a change
            continue
        if now[key] != value:
            raise SystemExit(
                f"{key} changed since the sweep was submitted ({value} -> {now[key]}); "
                "resubmit the sweep instead of running this task"
            )


def render_sbatch(template: str, values: Mapping[str, object]) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    if "{{" in text:
        left = text[text.index("{{") : text.index("{{") + 40]
        raise ValueError(f"unfilled template field: {left!r}")
    return text


def queued_jobs(squeue: Callable[[], str]) -> int:
    """How many jobs (array tasks counted one by one) this user has pending or running."""
    return len(squeue().split())


def _squeue() -> str:
    return subprocess.run(
        ["squeue", "--me", "-h", "-r", "-o", "%i"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _sbatch(path: Path) -> str:
    return subprocess.run(
        ["sbatch", str(path)], capture_output=True, text=True, check=True
    ).stdout.strip()


def launch_slurm(
    exp: Experiment,
    tasks: Sequence[Task],
    data_root: Path,
    submit: bool = False,
    template: Path = TEMPLATE,
    squeue: Callable[[], str] = _squeue,
    sbatch: Callable[[Path], str] = _sbatch,
    config_dir: Path = CONFIG_DIR,
    git_head: Callable[[], str | None] = _git_head,
) -> Path:
    """Write the task list and the filled sbatch file; only sbatch them if `submit`."""
    results = Path(data_root) / "results"
    out_dir = results / "sweeps" / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (results / "logs").mkdir(parents=True, exist_ok=True)
    task_list = out_dir / "tasks.txt"
    task_list.write_text("".join(f"category={t.category}\n" for t in tasks))
    (out_dir / "fingerprint.json").write_text(
        json.dumps(fingerprint(exp.name, config_dir, git_head))
    )
    script = out_dir / "array.sbatch"
    script.write_text(
        render_sbatch(
            template.read_text(),
            {
                "experiment": exp.name,
                "n_tasks_minus_1": len(tasks) - 1,
                "walltime": exp.walltime,
                "results_dir": results,
                "data_root": data_root,
                "task_list": task_list,
                "sweep_dir": out_dir,
                "repo": REPO,
            },
        )
    )
    print(f"{len(tasks)} tasks, {sum(len(t.runs) for t in tasks)} runs -> {script}")
    if not submit:
        print(f"not submitted; to submit: sbatch {script}")
        return script
    queued = queued_jobs(squeue)
    if queued + len(tasks) > MAX_TASKS:
        raise SystemExit(
            f"{queued} jobs of yours are queued or running and this adds {len(tasks)}: "
            f"over the {MAX_TASKS}-job submit limit; wait before submitting"
        )
    print(sbatch(script))
    return script


def parse_args(argv: Sequence[str]) -> dict[str, str]:
    args: dict[str, str] = {}
    for item in argv:
        key, sep, value = item.partition("=")
        if not sep or key not in ARGUMENTS:
            raise SystemExit(
                f"bad argument {item!r}; expected key=value with key in {sorted(ARGUMENTS)}"
            )
        args[key] = value
    if "experiment" not in args:
        raise SystemExit("experiment=<name> is required")
    if args.setdefault("launcher", "local") not in LAUNCHERS:
        raise SystemExit(f"launcher must be one of {LAUNCHERS}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    exp = load_experiment(args["experiment"])
    tasks = select(expand(exp), args.get("category"))
    launcher = args["launcher"]
    if launcher == "dry":
        for t in tasks:
            print(f"{t.category}: {len(t.runs)} runs")
            for r in t.runs:
                print("   ", " ".join(r[2:]))
        return 0
    root = args.get("data_root") or os.environ.get("CMAD_DATA_ROOT")
    if not root:
        raise SystemExit("CMAD_DATA_ROOT is not set; export it or pass data_root=...")
    root = Path(root).expanduser()
    if launcher == "slurm":
        launch_slurm(exp, tasks, root, submit=args.get("submit") == "true")
        return 0
    if "sweep_dir" in args:
        check_fingerprint(Path(args["sweep_dir"]), fingerprint(exp.name))
    total = Summary()
    for t in tasks:
        s = run_task(t, root)
        total.ran += s.ran
        total.cached += s.cached
        total.failed += s.failed
    print(f"== {total.ran} ran, {total.cached} cached, {len(total.failed)} failed")
    for label in total.failed:
        print("   failed:", label)
    return 1 if total.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
