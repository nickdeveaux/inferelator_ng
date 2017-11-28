from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow

workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/hiv_dc'
workflow.priors_file = 'priors.tsv'
workflow.num_bootstraps = 100
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.run() 
