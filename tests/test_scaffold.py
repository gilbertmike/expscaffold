import unittest
from itertools import product
from pathlib import Path

import pandas as pd

from expscaffold import ExperimentResult, ExperimentRunner, run_experiment


class TestResult(unittest.TestCase):
    def test_result(self):
        result = ExperimentResult()

        result.colname = 1
        self.assertEqual(result.colname, 1)

        self.assertRaises(AttributeError, lambda: print(result.unset_name))


class TestScaffoldSumExperiment(unittest.TestCase):
    def test_with_product(self):
        collected_result = run_experiment(
            TestScaffoldSumExperiment._experiment,
            ['var1', 'var2'],
            product([1, 2, 3], [1, 2, 3])
        )
        self._check_result(collected_result, 9)

    def test_with_zip(self):
        collected_result = run_experiment(
            TestScaffoldSumExperiment._experiment,
            ['var1', 'var2'],
            zip([1, 2, 3], [1, 2, 3])
        )
        self._check_result(collected_result, 3)

    def test_parallel(self):
        collected_result = ExperimentRunner(
            TestScaffoldSumExperiment._experiment,
            ['var1', 'var2'],
            zip([1, 2, 3], [1, 2, 3])
        ).configure_parallelism(3).run()

        self._check_result(collected_result, 3)

    def test_checkpointing(self):
        CHECKPOINT_PATH = Path(__file__).parent / 'tmp'

        collected_result = ExperimentRunner(
            TestScaffoldSumExperiment._experiment,
            ['var1', 'var2'],
            zip([1, 2, 3], [1, 2, 3])
        )\
        .configure_autosave(1, CHECKPOINT_PATH)\
        .run()

        self._check_result(collected_result, 3)

        checkpoint_fnames = list(CHECKPOINT_PATH.iterdir())
        self.assertEqual(len(checkpoint_fnames), 3)

        for fname in checkpoint_fnames:
            n_exps = int(str(fname.parts[-1]).split('.')[0])
            df = pd.read_csv(fname)
            self.assertEqual(len(df), n_exps)


    @staticmethod
    def _experiment(result: ExperimentResult, free_var1, free_var2):
        result.total = free_var1 + free_var2
    
    def _check_result(self, result, n_data_points):
        self.assertEqual(n_data_points, len(result['var1']))
        self.assertEqual(n_data_points, len(result['var2']))
        self.assertEqual(n_data_points, len(result['total']))
        self.assertTrue(
            (result.var1 + result.var2 == result.total).all()
        )
