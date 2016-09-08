from inferelator_ng.hiv_dc_bbsr_workflow import Hiv_Dc_Bbsr_Workflow

workflow = Hiv_Dc_Bbsr_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/hiv_dc'
workflow.priors_file = 'combined_priors.tsv'
workflow.expression_matrix_file = 'batch_corrected_expression.tsv'
workflow.num_bootstraps = 2
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.random_seed = 999
workflow.num_subsamples = 4
workflow.frac_subsamples = 0.66
workflow.run() 
