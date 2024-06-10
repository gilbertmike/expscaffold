from pathlib import Path
from typing import Callable, Iterable

import joblib
import pandas as pd

from .data import ExperimentResult

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
        self.param_vals = param_vals

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

        results = joblib.Parallel(self.n_jobs)(
            joblib.delayed(job)(param) for param in self.param_vals
        )

        for result in results:
            for key, val in result.__dict__.items():
                if key not in self.data:
                    self.data[key] = [val]
                else:
                    self.data[key].append(val)

        return pd.DataFrame(self.data)
