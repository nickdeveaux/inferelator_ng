from inferelator_ng.hiv_dc_bbsr_workflow import Hiv_Dc_Bbsr_Workflow

workflow = Hiv_Dc_Bbsr_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/hiv_dc'
workflow.priors_file = 'priors.tsv'
workflow.num_bootstraps = 20
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.run() 
