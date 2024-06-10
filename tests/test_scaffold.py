import unittest

from itertools import product

from expscaffold import ExperimentResult, ExperimentRunner, run_experiment


class TestResult(unittest.TestCase):
    def test_result(self):
        result = ExperimentResult()

        result.colname = 1
        self.assertEqual(result.colname, 1)

        self.assertRaises(AttributeError, lambda: print(result.unset_name))

class TestScaffoldSumExperiment(unittest.TestCase):
    def test_scaffold_product(self):
        collected_result = run_experiment(
            TestScaffoldSumExperiment._experiment,
            ['var1', 'var2'],
            product([1, 2, 3], [1, 2, 3])
        )
        self._check_result(collected_result, 9)

    def test_scaffold_zip(self):
        collected_result = run_experiment(
            TestScaffoldSumExperiment._experiment,
            ['var1', 'var2'],
            zip([1, 2, 3], [1, 2, 3])
        )
        self._check_result(collected_result, 3)

    def test_scaffold_parallel(self):
        collected_result = ExperimentRunner(
            TestScaffoldSumExperiment._experiment,
            ['var1', 'var2'],
            zip([1, 2, 3], [1, 2, 3])
        ).configure_parallelism(3).run()

        self._check_result(collected_result, 3)

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
