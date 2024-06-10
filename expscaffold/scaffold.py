from math import ceil
from pathlib import Path
from typing import Callable, Iterable

import joblib
import pandas as pd

from .data import ExperimentResult

# TODO: this is available in newer Python. Upgrade when suitable
def batched(iterable: Iterable, n_batches: int):
    iterable = list(iterable)
    n_per_batch = ceil(len(iterable) / n_batches)
    for i in range(n_batches):
        lower = i*n_per_batch
        upper = min(len(iterable), (i+1)*n_per_batch)
        yield iterable[lower:upper]

def run_experiment(
    exp_func: Callable,
    param_names: Iterable[str],
    param_vals: Iterable
):
    return ExperimentRunner(exp_func, param_names, param_vals).run()

class ExperimentRunner:
    def __init__(
        self,
        exp_func: Callable,
        param_names: Iterable[str],
        param_vals: Iterable
    ):
        self.data = {}
        self.exp_func = exp_func
        self.param_names = param_names
        self.param_vals = list(param_vals)

        for param_name in param_names:
            self.data[param_name] = []

        self.n_exp_between_save: int | None = None
        self.checkpoint_dir: Path | None = None

        self.n_jobs = 1

    def configure_autosave(
        self,
        n_exp_between_save: int,
        checkpoint_dir: Path | str
    ):
        self.n_exp_between_save = n_exp_between_save

        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

        return self

    def configure_parallelism(self, n_jobs: int):
        self.n_jobs = n_jobs
        return self

    def run(self):
        def job(param):
            result = ExperimentResult()
            self.exp_func(result, *param)
            for name, val in zip(self.param_names, param):
                setattr(result, name, val)
            return result

        if self.n_exp_between_save is not None:
            n_batches = len(self.param_vals) // self.n_exp_between_save
            assert self.checkpoint_dir is not None
        else:
            n_batches = 1

        total_exps = 0
        for param_vals in batched(self.param_vals, n_batches):
            results = joblib.Parallel(self.n_jobs)(
                joblib.delayed(job)(param) for param in param_vals
            )

            for result in results:
                for key, val in result.__dict__.items():
                    if key not in self.data:
                        self.data[key] = [val]
                    else:
                        self.data[key].append(val)

            total_exps += len(param_vals)
            if self.checkpoint_dir is not None:
                fname = self.checkpoint_dir / f'{total_exps}.csv'
                pd.DataFrame(self.data).to_csv(fname, index=False)

        return pd.DataFrame(self.data)
