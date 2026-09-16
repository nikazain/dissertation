Sequential Fine-Tuning of a Machine-Generated Text Detector Across Language Model Generations

 All results in the report come from experiment2/; experiment1/ is an exploration experiment kept for the record. Run everything from inside experiment2/ after pip install -r requirements.txt. Checkpoints and data/ are not committed; probs/ holds the stored probabilities every decision-level table is computed from.

experiment2
File	What it does
prepare_data.py	Builds the frozen splits into data/
data_loading.py	Loads the splits and tokenises
config.py	Training configuration
train.py	Trains one candidate for one stage
metrics.py	AUROC, accuracy and recall functions
seeding.py	Fixes the random seed
calibrate.py	Training-set cap calibration at 5k, 10k and 20k texts per class
verify_splits.py	Checks the split counts before an experiment
smoke.py	Quick end-to-end check on a small subset
run_when_free.sh	Waits for a free GPU and launches a run
run_sequential.py	Baseline chain, learning rate selected by validation AUROC → results.json
run_cumulative.py	Retention-aware chain, learning rate selected on all stages seen so far → results_cumulative.json
run_sequential_acc.py	Baseline chain with selection by validation accuracy → results_sequential_acc.json
run_cumulative_acc.py	Retention-aware chain with selection by validation accuracy → results_cumulative_acc.json
pooled.py	Reference model trained on all five stages at once → results_pooled.json
score_validation.py	Stores validation probabilities for threshold fitting
threshold_analysis.py	Deployed and refit threshold regimes → results_thresholds.json
truncate_eval.py	Evaluates every model at each input token budget
truncation_analysis.py	Accuracy and recalls by budget → results_truncation.json
error_analysis.py	False positive and false negative rates, macro-F1 → results_errors.json
human_luck_check.py	Re-scores M5 and the pooled model on a fresh human test sample → results_human_luck.json
artifact_audit.py	Checks the data for preprocessing artefacts
bootstrap_ci.py	95% bootstrap intervals for every matrix cell → results_ci.json
displacement.py	Weight distance between consecutive checkpoints
make_data_figures.py	All figures and LaTeX tables in the report, from the results_*.json files
