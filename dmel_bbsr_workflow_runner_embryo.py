from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/dmel_embryo/'
workflow.num_bootstraps = 2
workflow.delTmax = 110
workflow.delTmin = 0
workflow.expression_matrix_file = 'GSE95025_high_quality_cells_digital_expression_Fbgnids.txt'
workflow.tf_names_file = 'Marbach_and_fly_factor_survey_TFs.txt'
workflow.tau = 45
workflow.random_seed = 1
workflow.priors_file = 'Marbach_gold_standard.tsv'
workflow.gold_standard_file = workflow.priors_file
workflow.run()
