import unittest, os
import pandas as pd
import numpy as np
from .. import stat_utils

class TestStatUtils(unittest.TestCase):

    def test_normalize(self):
        y = pd.DataFrame(np.array([-2, 0, 0, 0]))
        mean = -0.5
        var = 1
        normalized_y = stat_utils.normalize(y, mean, var)
        self.assertTrue(normalized_y.equals(pd.DataFrame(np.array([-1.5, 0.5, 0.5, 0.5]))))

    def test_compute_stats(self):
        y = pd.DataFrame(np.array([[-2, 0, 0, 0]]))
        (mean, var) = stat_utils.compute_stats(y)
        self.assertEqual(mean[0], -0.5)
        self.assertEqual(var[0], 1)

    # testing an example held-out error calculation with a single predictor
    def test_compute_error_single_predictor(self):
        sample_names = ['S0', 'S1', 'S2', 'S3']
        X = pd.DataFrame(np.array([[3, 0, 1, 1]]), index=['TF1'], columns = sample_names)
        Y = pd.DataFrame(np.array([[1, 7, 2, 3],[8, 2, 0, 1]]), index=['G0', 'G1'], columns = sample_names)

        thresholded_matrix = pd.DataFrame(np.array([[1], [1]]), index=['G0', 'G1'])

        # holding out the first sample, S0
        held_out_X = pd.DataFrame(X.iloc[:,0])
        held_out_Y = pd.DataFrame(Y.iloc[:,0])
        (train, test) = stat_utils.compute_error(X.iloc[:,1:], Y.iloc[:,1:], thresholded_matrix, held_out_X, held_out_Y)
        # Expected train error for G0: 0^2 + (1/(2 sqrt(7)))^2 + (1/(2 sqrt(7)))^2 = 0.0714...
        # Expected test error for G0: (15/ (2 sqrt(7)))^2 = 8.035...
        expected_train_error = {'G0': 0.07142857142857141, 'G1': 0.5, 'counts': {'G0': 3, 'G1': 3}}
        expected_test_error = {'G0': 8.0357142857142954, 'G1': 110.25000000000004, 'counts': {'G0': 1, 'G1': 1}}
        np.testing.assert_allclose(expected_train_error['G0'], train['G0'])
        np.testing.assert_allclose(expected_test_error['G0'], test['G0'])
        np.testing.assert_allclose(expected_train_error['G1'], train['G1'])
        np.testing.assert_allclose(expected_test_error['G1'], test['G1'])