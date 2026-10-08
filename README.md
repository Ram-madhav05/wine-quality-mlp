# Wine Quality MLP Architecture Study

## Run the project
Use Python 3.12. Create a virtual environment if desired, install the dependencies,
then run from this folder:

```sh
pip install -r requirements.txt
python train_mlp.py
```

The original UCI red-wine CSV is included. If missing, the script downloads the
official archive. Outputs go to `results/` and plots to `figures/`.

## What was measured
Eight architectures, five initialization seeds each, one fixed stratified
60/20/20 split after exact duplicate removal. StandardScaler is fitted only on
training data. Lowest validation log loss defines each restored checkpoint;
mean validation macro F1 selects the architecture. Seed 11 is fixed in advance
for final testing. The selected model is not refitted on validation data.

The five seeds measure optimization variability on one split, not uncertainty
over different datasets. Test data do not choose architectures or seeds.

## Contents
- `train_mlp.py`: complete runnable experiment and figure generation.
- `data/`: original CSV and UCI data description.
- `results/all_runs.csv`: every measured run.
- `results/architecture_summary.csv`: architecture means and sample SDs.
- `results/split_manifest.csv`: original zero-based source row and split.
- `results/training_histories.json`: epoch-by-epoch curves for all runs.
- `results/test_predictions.csv`: every test label and model prediction.
- `results/selected_model.joblib`: scaler, model, features, source audit.
- `results/execution.log`: output captured during the report's training run.
- `results/*.html`: readable views of captured program output.
- `figures/`: measured plots, diagram images, and output screenshots.

Timing is environment-dependent. Re-running may produce tiny floating-point
differences on different hardware or library versions. The output captures are
rendered directly from the saved real program output, not images of an IDE session.

## Data citation and license
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009).
Wine Quality [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C56S3T
Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Changes: exact duplicates removed in memory for the experiment. Included CSV
bytes are unchanged; their SHA-256 is recorded in results/data_audit.json.
