from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
from inferelator_ng.prior_gs_split_workflow import PriorGoldStandardSplitWorkflowBase

class BBSR_TFA_Workflow_with_Prior_GS_split(BBSR_TFA_Workflow, PriorGoldStandardSplitWorkflowBase):
    """ 
        The class BBSR_TFA_Workflow_with_Prior_GS_split is a case of multiple inheritance,
        as it inherits both from BBSR_TFA_Workflow and PriorGoldStandardSplitWorkflowBase      
    """

workflow = BBSR_TFA_Workflow_with_Prior_GS_split()
# Common configuration parameters
workflow.input_dir = 'data/dmel_TF_knn_experiment/'
workflow.num_bootstraps = 20
workflow.delTmax = 110
workflow.delTmin = 0
workflow.expression_matrix_file = 'wtJ_TF_cells_k_3.tsv'
workflow.tf_names_file = 'Marbach_tf_names.tsv'
workflow.tau = 45
workflow.random_seed = 1
workflow.priors_file = 'Marbach_gold_standard.tsv'
workflow.run()
