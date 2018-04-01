from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/dmel_TF_knn_experiment/'
workflow.num_bootstraps = 60
workflow.delTmax = 110
workflow.delTmin = 0
workflow.expression_matrix_file = 'combined_wtJ_and_wtF_TF_cells_smoothed_separately_with_k_3.tsv'
workflow.tf_names_file = 'tf_names_all_in_fly_factor.tsv'
workflow.tau = 45
workflow.random_seed = 1
workflow.priors_file = 'prior_with_tfs_filtered_by_one_percent_expression.tsv'
workflow.run()
