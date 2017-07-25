"""
Run HIV DC Network Inference with TFA BBSR. 
"""

import numpy as np
import os
from workflow import WorkflowBase
import design_response_R
from tfa import TFA
import mi_R
import bbsr_R
import datetime
import random
import pandas as pd
from results_processor import ResultsProcessor

class Hiv_Dc_Bbsr_Workflow(WorkflowBase):

    def __init__(self):
        # Do nothing (all configuration is external to init)
        pass

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.gold_standard = None
        self.gold_standard_file = None
        
        self.mi_clr_driver = mi_R.MIDriver()
        self.regression_driver = bbsr_R.BBSR_driver()
        self.design_response_driver = design_response_R.DRDriver()

        self.get_data()
        self.compute_common_data()
        self.compute_activity()
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
        self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        print('Computing Transcription Factor Activity using {} subsamples ... '.format(str(self.num_subsamples)))
        bootstrapped_activities = []
        number_of_nonzeros = np.count_nonzero(self.priors_data.values)
        flat_nonzero_index = np.flatnonzero(self.priors_data.values)
        print ('Of the {} non-zero indices, a fraction of {} will be randomly sampled'.format(str(number_of_nonzeros), str(self.frac_subsamples)))
        for i in range(self.num_subsamples):
            temp_prior = self.priors_data.copy()
            sample = random.sample(xrange(number_of_nonzeros), int(float(number_of_nonzeros)*(1 - self.frac_subsamples)))
            temp_prior.values[np.unravel_index(flat_nonzero_index[sample], self.priors_data.shape)] = 0
            pseudoinverse = np.linalg.pinv(temp_prior)
            TFA_calculator = TFA(temp_prior, self.design, self.half_tau_response)
            bootstrapped_activities.append( TFA_calculator.compute_transcription_factor_activity())
        stack = np.dstack([b.values for b in bootstrapped_activities])
        median_activity = pd.DataFrame(np.median(stack, axis = 2), index=self.priors_data.columns, columns = self.expression_matrix.columns)
        self.activity = median_activity

    def filter_expression_and_priors(self):
        """
        Guarantee that each row of the prior is in the expression and vice versa.
        Also filter the priors to only includes columns, transcription factors and targets that meet filter criteria.
        """
        # filter the expression matrix so that at least one rld expression value per gene is greater than 0
        filtered_xpn = self.expression_matrix[self.expression_matrix.max(axis=1) > 0]
        targets = set.intersection(set(filtered_xpn.index.tolist()), set(self.priors_data.index.tolist()))
        predictors = set.intersection(set(filtered_xpn.index.tolist()), set(self.priors_data.columns.tolist()))
        self.expression_matrix = filtered_xpn.loc[targets, :]
        self.priors_data = self.priors_data.loc[targets, predictors]
        print('Filtering expression and priors down to {} tfs and {} targets ... '.format(str(len(predictors)), str(len(targets))))

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(output_dir)
        self.activity.to_csv(os.path.join(output_dir, 'median_activities.tsv'), sep = '\t')
        self.results_processor = ResultsProcessor(betas, rescaled_betas)
        self.results_processor.summarize_network(output_dir, gold_standard, priors)
