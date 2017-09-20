import unittest, os
import pandas as pd
import numpy as np
from .. import stat_utils
from scipy import linalg
import copy

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

    # testing an example held-out error calculation with two predictors and activity
    def test_compute_error_two_predictors_from_activity(self):
        sample_names = ['S0', 'S1', 'S2', 'S3']
        gene_names = ['G0', 'G1']
        tf_names = ['TF0', 'TF1']
        P = pd.DataFrame(np.array([[1, 0],[1, 1]]), index=gene_names, columns=tf_names)
        Y = pd.DataFrame(np.array([[1, 7, 2, 3],[8, 2, 0, 1]]), index=gene_names, columns = sample_names)
        thresholded_matrix = pd.DataFrame(np.array([[1], [1]]), index=gene_names)
        activities = np.matrix(linalg.pinv2(P)) * np.matrix(Y)
        X = pd.DataFrame(activities, index=tf_names, columns = sample_names)

        # holding out the first sample, S0
        held_out_X = pd.DataFrame(X.iloc[:,0])
        held_out_Y = pd.DataFrame(Y.iloc[:,0])
        (train, test) = stat_utils.compute_error(X.iloc[:,1:], Y.iloc[:,1:], thresholded_matrix, held_out_X, held_out_Y)
        # Expected train error for G0: 0, test error: 0 (this is because the TF's activity is entirely based off of G0's expression)
        expected_train_error = {'G0': 0, 'G1': 0.21428571428571436}
        expected_test_error = {'G0': 0, 'G1': 65.147959183673464}

        np.testing.assert_allclose(expected_train_error['G0'], train['G0'], atol=1e-30)
        np.testing.assert_allclose(expected_test_error['G0'], test['G0'], atol=1e-30)
        np.testing.assert_allclose(expected_train_error['G1'], train['G1'])
        np.testing.assert_allclose(expected_test_error['G1'], test['G1'])


    # testing 100 random X held-out error calculation with a single predictor
    def test_compute_error_single_predictor_from_activity_every_fold(self):
        sample_names = ['S0', 'S1', 'S2', 'S3']
        gene_names = ['G0', 'G1']
        tf_names = ['TF0']
        P = pd.DataFrame(np.array([[1], [1]]), index=gene_names, columns=tf_names)
        Y = pd.DataFrame(np.array([[1, 7, 2, 3],[8, 2, 0, 1]]), index=gene_names, columns = sample_names)
        thresholded_matrix = pd.DataFrame(np.array([[1], [1]]), index=gene_names)
        activities = np.matrix(linalg.pinv2(P)) * np.matrix(Y)
        X = pd.DataFrame(activities, index=tf_names, columns = sample_names)
        # this is what the values of the activities of TF0 look like
        # (it's the average of the two target genes' expression)
        # TF0  4.5  4.5  1.0  2.0
        for i in sample_names:
            X_for_fold = copy.deepcopy(X)
            Y_for_fold = copy.deepcopy(Y)
            held_out_X = pd.DataFrame(X_for_fold.pop(i))
            held_out_Y = pd.DataFrame(Y_for_fold.pop(i))
            (train, test) = stat_utils.compute_error(X_for_fold, Y_for_fold, thresholded_matrix, held_out_X, held_out_Y)
            
            # The sample S2&S3 have a higher training error than test error (by orders of magnitude)
            if i == 'S2':
                expected_train_error = {'G1': 1.2558139534883712, 'G0': 1.928571428571429}
                expected_test_error = {'G1': 0.025116279069767614, 'G0': 0.038571428571428701}
                np.testing.assert_allclose(expected_train_error['G0'], train['G0'])
                np.testing.assert_allclose(expected_test_error['G0'], test['G0'])
                np.testing.assert_allclose(expected_train_error['G1'], train['G1'])
                np.testing.assert_allclose(expected_test_error['G1'], test['G1'])
            elif i == 'S3':
                expected_train_error = {'G1': 1.0384615384615381, 'G0': 1.741935483870968}
                expected_test_error = {'G1': 0.010596546310832037, 'G0': 0.017774851876234295}
                np.testing.assert_allclose(expected_train_error['G0'], train['G0'])
                np.testing.assert_allclose(expected_test_error['G0'], test['G0'])
                np.testing.assert_allclose(expected_train_error['G1'], train['G1'])
                np.testing.assert_allclose(expected_test_error['G1'], test['G1'])
            else:
                self.assertTrue(train['G1'] < test['G1'])
                self.assertTrue(train['G0'] < test['G0'])


    # testing 100 random Y held-out error calculation with a single predictor
    def single_predictor_test_error_vs_train_every_fold(self):
        sample_names = ['S0', 'S1', 'S2', 'S3']
        gene_names = ['G0', 'G1']
        tf_names = ['TF0']
        Y = pd.DataFrame(np.array([[1, 7, 2, 3],[8, 2, 0, 1]]), index=gene_names, columns = sample_names)
        thresholded_matrix = pd.DataFrame(np.array([[1], [1]]), index=gene_names)
        for _ in range(10):
            X = pd.DataFrame(np.matrix(np.array(np.random.choice(range(10), size=4, replace = False)).transpose()), index=tf_names, columns = sample_names)
            MSE = {}
            MSE['train'] = {}
            MSE['test'] = {}
            for g in gene_names:
                MSE['train'][g] = 0
                MSE['test'][g] = 0
            for i in sample_names:
                X_for_fold = copy.deepcopy(X)
                Y_for_fold = copy.deepcopy(Y)
                held_out_X = pd.DataFrame(X_for_fold.pop(i))
                held_out_Y = pd.DataFrame(Y_for_fold.pop(i))
                (train, test) = stat_utils.compute_error(X_for_fold, Y_for_fold, thresholded_matrix, held_out_X, held_out_Y)
                for g in gene_names:
                    MSE['train'][g] = MSE['train'][g] + train[g] / float(train['counts'][g])
                    MSE['test'][g] = MSE['test'][g] + test[g] / float(test['counts'][g])
            for g in gene_names:
                for t in ['test', 'train']:
                    MSE[t][g] = MSE[t][g] / float(len(sample_names))
                self.assertTrue(MSE['train'][g] < MSE['test'][g])

       # testing an example held-out error calculation with a single predictor
    def test_single_predictor_test_error_vs_train_every_fold_with_activitie(self):
        sample_names = ['S0', 'S1', 'S2', 'S3']
        gene_names = ['G0', 'G1']
        tf_names = ['TF0']
        P = pd.DataFrame(np.array([[1], [1]]), index=gene_names, columns=tf_names)
        for _ in range(10):
            Y = pd.DataFrame([np.array(np.random.choice(range(10), size=4, replace = False)).transpose(),
                    np.array(np.random.choice(range(10), size=4, replace = False)).transpose()], index=gene_names, columns = sample_names)
            thresholded_matrix = pd.DataFrame(np.array([[1], [1]]), index=gene_names)
            activities = np.matrix(linalg.pinv2(P)) * np.matrix(Y)
            X = pd.DataFrame(activities, index=tf_names, columns = sample_names)
            MSE = {}
            MSE['train'] = {}
            MSE['test'] = {}
            for g in gene_names:
                MSE['train'][g] = 0
                MSE['test'][g] = 0
            for i in sample_names:
                X_for_fold = copy.deepcopy(X)
                Y_for_fold = copy.deepcopy(Y)
                held_out_X = pd.DataFrame(X_for_fold.pop(i))
                held_out_Y = pd.DataFrame(Y_for_fold.pop(i))
                (train, test) = stat_utils.compute_error(X_for_fold, Y_for_fold, thresholded_matrix, held_out_X, held_out_Y)
                for g in gene_names:
                    MSE['train'][g] = MSE['train'][g] + train[g] / float(train['counts'][g])
                    MSE['test'][g] = MSE['test'][g] + test[g] / float(test['counts'][g])
            for g in gene_names:
                for t in ['test', 'train']:
                    MSE[t][g] = MSE[t][g] / float(len(sample_names))
                self.assertTrue(MSE['train'][g] < MSE['test'][g])
            
            