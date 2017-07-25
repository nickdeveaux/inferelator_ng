"""
Run BSubtilis Network Inference with TFA BBSR. 
"""

import numpy as np
import os
from workflow import WorkflowBase
import design_response_R
from tfa import TFA
from results_processor import ResultsProcessor
import stat_utils
import mi_R
import bbsr_R
import datetime
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class BBSR_TFA_Workflow(WorkflowBase):

    def partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        np.random.seed(self.random_seed)
        self.num_folds = 10
        self.mi_clr_driver = mi_R.MIDriver()
        self.regression_driver = bbsr_R.BBSR_driver()
        self.design_response_driver = design_response_R.DRDriver()

        self.get_data()
        
        # store total data sets (without hold outs) in "archived" data frames
        self.archived_expression_matrix = self.expression_matrix
        self.archived_meta_data = self.meta_data
        self.compute_common_data()
        self.archived_activity = self.compute_activity()

        # Set up a K-fold partition of the samples, i.e. the expression columns
        total_test_error = {}
        total_train_error = {}
        fold_indices = self.expression_matrix.columns.tolist()
        random.shuffle(fold_indices)
        partitioned_fold_indices = self.partition(fold_indices, self.num_folds)\

        for fold in range(self.num_folds):
            excluded_samples = partitioned_fold_indices[fold]
            self.expression_matrix = stat_utils.filter_out(self.archived_expression_matrix, excluded_samples)
            self.meta_data = self.archived_meta_data.loc[~self.archived_meta_data.condName.isin(excluded_samples) , :]
            self.compute_common_data()
            self.activity = self.compute_activity()

            betas = []
            rescaled_betas = []
    
            for idx, bootstrap in enumerate(self.get_bootstraps()):
                print('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps))
                X = self.activity.ix[:, bootstrap]
                Y = self.response.ix[:, bootstrap]
                print('Calculating MI, Background MI, and CLR Matrix')
                (self.clr_matrix, self.mi_matrix) = self.mi_clr_driver.run(X, Y)
                print('Calculating betas using BBSR')
                current_betas, current_rescaled_betas = self.regression_driver.run(X, Y, self.clr_matrix, self.priors_data)
                betas.append(current_betas)
                rescaled_betas.append(current_rescaled_betas)
            thresholded_matrix = ResultsProcessor(betas, rescaled_betas).threshold_and_summarize()
            held_out_data = self.archived_expression_matrix[excluded_samples]
            held_out_meta_data = self.archived_meta_data.loc[self.archived_meta_data.condName.isin(excluded_samples) , :]
            (self.design, held_out_response) = self.design_response_driver.run(held_out_data, held_out_meta_data)
            (self.design, self.half_tau_response) = self.design_response_driver.run(held_out_data, held_out_meta_data)
            held_out_activity = self.compute_activity()
            (train_error, test_error) = self.compute_error_unnormalized_y(self.activity, self.response, thresholded_matrix, \
                                             held_out_activity, held_out_response)

            total_test_error[fold] = test_error
            total_train_error[fold] = train_error

        # boxplot of each gene
        for gene in total_test_error[0].keys():
            test_error = [total_test_error[i][gene] for i in range(self.num_folds) if gene in total_test_error[i].keys()]
            train_error = [total_train_error[i][gene] for i in range(self.num_folds) if gene in total_train_error[i].keys()]
            fig = plt.figure(1, figsize=(10, 8))
            plt.boxplot([train_error, test_error])
            plt.ylabel('Mean Squared Error (SSE/N)')
            plt.xticks([1, 2], ['train error', 'test error'])
            plt.title('{}-fold validation ters'.format(self.num_folds))
            plt.savefig('{}.png'.format(gene))
            fig.clear()
            plt.close()
        flat_list_test_error = []
        for k in total_test_error.keys():
            for z in total_test_error[k]:
                total_test_error[k]
        flat_list_test_error = [total_test_error[k][gene] for k in total_test_error.keys() for gene in total_test_error[k]]
        flat_list_train_error = [total_train_error[k][gene] for k in total_train_error.keys() for gene in total_train_error[k]]
        fig = plt.figure(1, figsize=(10, 8))
        plt.boxplot([flat_list_train_error, flat_list_test_error])
        plt.ylabel('Mean Squared Error (SSE/N)')
        plt.xticks([1, 2], ['train error', 'test error'])
        plt.title('{}-fold validation ters'.format(self.num_folds))
        plt.savefig('{}.png'.format('All_Genes'))
        import pdb; pdb.set_trace() 


    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        print('Computing Transcription Factor Activity ... ')
        TFA_calculator = TFA(self.priors_data, self.design, self.half_tau_response)
        return TFA_calculator.compute_transcription_factor_activity()

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(output_dir)
        self.results_processor = ResultsProcessor(betas, rescaled_betas)
        self.results_processor.summarize_network(output_dir, gold_standard, priors)

    def compute_error(self, X, Y, thresholded_matrix, held_out_X, held_out_Y):
        """
        Does a linear fit using sklearn, on the non-zero predictors. 
        Returns the mean squared error on training data per gene and test data per gene
        We will normalize the testdata and the testactivity with the mean and std of the train, as described in
        https://stats.stackexchange.com/questions/174823/how-to-apply-standardization-normalization-to-train-and-testset-if-prediction-i
        """
        test_error = {}
        train_error = {}

        (X_mu, X_sigma_squared) = stat_utils.compute_stats(X)
        X_normalized = stat_utils.normalize(X, X_mu, X_sigma_squared)
        (Y_mu, Y_sigma_squared) = stat_utils.compute_stats(Y)
        Y_normalized = stat_utils.normalize(Y, Y_mu, Y_sigma_squared)

        held_out_Y_normalized = stat_utils.normalize(held_out_Y, Y_mu, Y_sigma_squared)
        held_out_X_normalized = stat_utils.normalize(held_out_X, X_mu, X_sigma_squared)
        ols = LinearRegression(normalize=False, fit_intercept=True)
        for gene_name, y_normalized in Y_normalized.iterrows():

            nonzero  = thresholded_matrix.loc[gene_name,:].nonzero()[0]
            #only compute betas if there was found to be a predictive TF for this target gene
            if len(nonzero) > 1:
                nonzero_X_normalized = X_normalized.iloc[nonzero,:].transpose()
                n = len(y_normalized)
                if n < 1:
                    import pdb; pdb.set_trace()
                if nonzero_X_normalized.shape[1] < 1:
                    import pdb; pdb.set_trace()
                ols.fit(nonzero_X_normalized, y_normalized)
                train_error[gene_name] = np.sum((ols.predict(nonzero_X_normalized) - y_normalized) ** 2) / n
    
                held_out_nonzero_X_normalized = held_out_X_normalized.iloc[nonzero,:].transpose()
                n = len(held_out_Y_normalized.loc[gene_name,:])
                test_error[gene_name] = np.sum((ols.predict(held_out_nonzero_X_normalized) - held_out_Y_normalized.loc[gene_name,:]) ** 2) / n
        return (train_error, test_error)

    def compute_error_unnormalized_y(self, X, Y, thresholded_matrix, held_out_X, held_out_Y):
        """
        Does a linear fit using sklearn, on the non-zero predictors. 
        Returns the sum of squared error on training data per gene and test data per gene
        We will normalize the testdata and the testactivity with the mean and std of the train, as described in
        https://stats.stackexchange.com/questions/174823/how-to-apply-standardization-normalization-to-train-and-testset-if-prediction-i
        """
        test_error = {}
        train_error = {}

        (X_mu, X_sigma_squared) = stat_utils.compute_stats(X)
        X_normalized = stat_utils.normalize(X, X_mu, X_sigma_squared)

        held_out_X_normalized = stat_utils.normalize(held_out_X, X_mu, X_sigma_squared)
        ols = LinearRegression(normalize=False, fit_intercept=True)
        for gene_name, y in Y.iterrows():

            nonzero  = thresholded_matrix.loc[gene_name,:].nonzero()[0]
            #only compute betas if there was found to be a predictive TF for this target gene
            if len(nonzero) > 1:
                nonzero_X_normalized = X_normalized.iloc[nonzero,:].transpose()
                n = len(y)
                if n < 1:
                    import pdb; pdb.set_trace()
                if nonzero_X_normalized.shape[1] < 1:
                    import pdb; pdb.set_trace()
                ols.fit(nonzero_X_normalized, y)
                train_error[gene_name] = np.sum((ols.predict(nonzero_X_normalized) - y) ** 2) / n
    
                held_out_nonzero_X_normalized = held_out_X_normalized.iloc[nonzero,:].transpose()
                n = len(held_out_Y.loc[gene_name,:])
                test_error[gene_name] = np.sum((ols.predict(held_out_nonzero_X_normalized) - held_out_Y.loc[gene_name,:]) ** 2) / n
        return (train_error, test_error)
        

        
        

        
        

        
        
        
        
        
        

        
        
        



