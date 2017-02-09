from inferelator_ng.yeast_bbsr_workflow import Yeast_Bbsr_Workflow

import unittest
from .. import condition
from .. import time_series
from .. import gene_model
import pandas as pd
import numpy as np

class TestEndToEnd(unittest.TestCase):

    def test_yeast(self):
        workflow = Yeast_Bbsr_Workflow()
        # Common configuration parameters
        workflow.input_dir = 'data/yeast'
        workflow.num_bootstraps = 1
        workflow.delTmax = 110
        workflow.delTmin = 0
        workflow.tau = 45
        aupr_result = workflow.run()
        print(aupr_result)
        self.assertTrue(aupr_result > 0.59)