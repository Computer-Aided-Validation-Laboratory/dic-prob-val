"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Execution runners, parallel local runner, caching, and run records.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import time
from typing import Sequence
import numpy as np
from .model import IModel
from .param import ParamValues
from .result import ResultField, ResultScalar, SimResult


class ERunStatus(Enum):
    """Lifecycle execution status of a single simulation evaluation."""

    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


@dataclass(slots=True)
class RunResult:
    """Execution outcome and artifacts for one simulation run."""

    run_id: str
    params: ParamValues
    status: ERunStatus
    work_dir: Path
    exit_code: int | None
    message: str | None
    result: SimResult | None
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class RunSet:
    """Collection of executed simulation run records."""

    runs: tuple[RunResult, ...]

    def __init__(self, runs: Sequence[RunResult]) -> None:
        self.runs = tuple(runs)

    def get_completed_results(self) -> list[SimResult]:
        """Get list of successfully completed SimResult objects."""
        return [r.result for r in self.runs if r.result is not None]

    def get_num_complete(self) -> int:
        """Count number of completed runs."""
        return sum(1 for r in self.runs if r.status == ERunStatus.complete)

    def get_num_failed(self) -> int:
        """Count number of failed runs."""
        return sum(1 for r in self.runs if r.status == ERunStatus.failed)


class IRunner(ABC):
    """Abstract execution runner interface."""

    @abstractmethod
    def run_samples(
        self,
        model: IModel,
        samples: ParamValues,
        work_dir: Path,
    ) -> RunSet:
        """Execute a batch of deterministic samples."""
        ...


def _format_param_summary(param_values: ParamValues) -> str:
    """Format short human-readable string of key parameter values."""
    param_dict = param_values.extract_dict(sample_idx=0)
    items = []
    for k, v in param_dict.items():
        if abs(v) >= 1e3:
            items.append(f"{k}={v/1e3:.2f}k")
        elif abs(v) < 1e-2 and abs(v) > 0:
            items.append(f"{k}={v:.2e}")
        else:
            items.append(f"{k}={v:.3f}")
    return ", ".join(items[:4]) + ("..." if len(items) > 4 else "")


def _execute_single_run(
    model: IModel,
    run_id: str,
    single_param: ParamValues,
    run_dir: Path,
    restart: bool,
    verbose: bool = True,
    sample_index: int = 0,
    total_samples: int = 1,
) -> RunResult:
    """Execute or load a single cached simulation run with progress logging."""
    result_cache_file = run_dir / "sim_result.npz"
    param_str = _format_param_summary(single_param)

    if restart and result_cache_file.exists():
        try:
            cached_data = np.load(result_cache_file, allow_pickle=True)
            scalars = [
                ResultScalar(name=str(n), value=float(v))
                for n, v in zip(
                    cached_data["scalar_names"], cached_data["scalar_values"]
                )
            ]
            fields = []
            if "field_names" in cached_data:
                for f_name in cached_data["field_names"]:
                    fields.append(
                        ResultField(
                            name=str(f_name),
                            values=cached_data[f"field_vals_{f_name}"],
                            coords=cached_data[f"field_coords_{f_name}"],
                        )
                    )
            if verbose:
                print(
                    f"[simorc] [{sample_index + 1}/{total_samples}] "
                    f"({run_id}) Loaded from cache ({param_str})"
                )
            return RunResult(
                run_id=run_id,
                params=single_param,
                status=ERunStatus.complete,
                work_dir=run_dir,
                exit_code=0,
                message="Loaded from cache",
                result=SimResult(scalars=scalars, fields=fields),
                elapsed_seconds=0.0,
            )
        except Exception:
            pass

    t_start = time.perf_counter()
    if verbose:
        print(
            f"[simorc] [{sample_index + 1}/{total_samples}] "
            f"({run_id}) Launching simulation ({param_str})"
        )

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        sim_res = model.run(single_param, run_dir)
        elapsed = time.perf_counter() - t_start

        scalar_names = [s.name for s in sim_res.scalars]
        scalar_vals = [s.value for s in sim_res.scalars]
        cache_dict = {
            "scalar_names": np.array(scalar_names),
            "scalar_values": np.array(scalar_vals, dtype=np.float64),
        }
        if sim_res.fields:
            field_names = [f.name for f in sim_res.fields]
            cache_dict["field_names"] = np.array(field_names)
            for f in sim_res.fields:
                cache_dict[f"field_vals_{f.name}"] = f.values
                cache_dict[f"field_coords_{f.name}"] = f.coords

        np.savez_compressed(result_cache_file, **cache_dict)

        if verbose:
            print(
                f"[simorc] [{sample_index + 1}/{total_samples}] "
                f"({run_id}) Completed in {elapsed:.2f} s"
            )

        return RunResult(
            run_id=run_id,
            params=single_param,
            status=ERunStatus.complete,
            work_dir=run_dir,
            exit_code=0,
            message="Success",
            result=sim_res,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        if verbose:
            print(
                f"[simorc] [{sample_index + 1}/{total_samples}] "
                f"({run_id}) FAILED in {elapsed:.2f} s: {exc}"
            )
        return RunResult(
            run_id=run_id,
            params=single_param,
            status=ERunStatus.failed,
            work_dir=run_dir,
            exit_code=1,
            message=str(exc),
            result=None,
            elapsed_seconds=elapsed,
        )


@dataclass(slots=True)
class RunnerLocal(IRunner):
    """Local runner with parallel execution and live console tracking."""

    num_workers: int = 4
    num_threads_per_sim: int = 1
    restart: bool = True
    verbose: bool = True

    def run_samples(
        self,
        model: IModel,
        samples: ParamValues,
        work_dir: Path,
    ) -> RunSet:
        """Run parameter samples locally with controlled parallel workers."""
        work_dir.mkdir(parents=True, exist_ok=True)
        num_samples = samples.get_num_samples()
        results: list[RunResult | None] = [None] * num_samples

        # Configure within-simulation threads if supported
        if hasattr(model, "num_threads"):
            model.num_threads = self.num_threads_per_sim

        max_cores = os.cpu_count() or 4
        total_requested = self.num_workers * self.num_threads_per_sim
        if self.verbose:
            print(
                f"[simorc] Starting batch execution of {num_samples} runs:\n"
                f"  - Over-simulation workers   : {self.num_workers}\n"
                f"  - Within-simulation threads : {self.num_threads_per_sim}\n"
                f"  - Total core allocation     : {total_requested} "
                f"(System available: {max_cores})"
            )

        tasks = []
        for idx in range(num_samples):
            run_id = f"run_{idx:06d}"
            run_dir = work_dir / run_id
            single_param = ParamValues(
                names=samples.names,
                values=samples.values[idx : idx + 1, :],
            )
            tasks.append((idx, run_id, single_param, run_dir))

        if self.num_workers <= 1:
            for idx, run_id, single_param, run_dir in tasks:
                results[idx] = _execute_single_run(
                    model,
                    run_id,
                    single_param,
                    run_dir,
                    self.restart,
                    self.verbose,
                    idx,
                    num_samples,
                )
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        _execute_single_run,
                        model,
                        run_id,
                        single_param,
                        run_dir,
                        self.restart,
                        self.verbose,
                        idx,
                        num_samples,
                    ): idx
                    for idx, run_id, single_param, run_dir in tasks
                }
                for future in as_completed(futures):
                    orig_idx = futures[future]
                    results[orig_idx] = future.result()

        valid_runs = [r for r in results if r is not None]
        run_set = RunSet(runs=valid_runs)

        if self.verbose:
            print(
                f"[simorc] Batch complete: {run_set.get_num_complete()}/"
                f"{num_samples} runs succeeded."
            )

        return run_set
