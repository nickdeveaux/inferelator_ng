from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow

workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/dmel_wtF_and_wtJ_big_network_combined_without_GC'
workflow.num_bootstraps = 20 
workflow.delTmax = 110
workflow.delTmin = 0
workflow.expression_matrix_file = 'expression.tsv'
workflow.tau = 45
workflow.random_seed = 1
workflow.priors_file = 'prior_with_tfs_filtered_by_one_percent_expression.tsv'
workflow.run() 

