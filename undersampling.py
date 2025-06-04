from LR import train_evaluate_lr
from RF import train_evaluate_rf
from MLP import train_evaluate_mlp
from helper import load_raw_data, load_scaled_data, load_processed_data, print_metrics, load_json, save_json
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import json
from collections import Counter
from sklearn.model_selection import train_test_split

SEED = 42

x_train, x_test_1, x_test_2_labelled, x_test_2_unlabelled, y_train, y_test_2 = load_processed_data()

# Split train into holdout
x_train_main, x_train_holdout, y_train_main, y_train_holdout = train_test_split(
    x_train, y_train, random_state=SEED, test_size=0.2, stratify=y_train
)

print("Original class distribution:", dict(zip(*np.unique([int(x) for x in y_train_main], return_counts=True))))

orig_counts = Counter(y_train_main)
maj_class, maj_count = orig_counts.most_common(1)[0]

# target size = 20% of original majority
target_n = int(0.20 * maj_count)

undersampler = RandomUnderSampler(random_state=0, sampling_strategy={maj_class: target_n})
x_train_res, y_train_res = undersampler.fit_resample(x_train_main, y_train_main)

print("Modified class distribution:", dict(zip(*np.unique([int(x) for x in y_train_res], return_counts=True))))

# Load existing data or initialize
current_metrics = load_json("metrics/undersampling.json")
current_best_params = load_json("best_params/undersampling.json")

# Evaluate Logistic Regression
if "lr" not in current_metrics:
    lr_metrics, lr_best_params = train_evaluate_lr(x_train_res, x_train_holdout, y_train_res, y_train_holdout)
else:
    lr_metrics = current_metrics["lr"]
    lr_best_params = current_best_params["lr"]

# Evaluate Random Forests
if "rf" not in current_metrics:
    rf_metrics, rf_best_params = train_evaluate_rf(x_train_res, x_train_holdout, y_train_res, y_train_holdout)
else:
    rf_metrics = current_metrics["rf"]
    rf_best_params = current_best_params["rf"]

# Evaluate MLP
if "mlp" not in current_metrics:
    mlp_metrics, mlp_best_params = train_evaluate_mlp(
        x_train_res, x_train_holdout, y_train_res, y_train_holdout, dataset_type='undersampling'
    )
else:
    mlp_metrics = current_metrics["mlp"]
    mlp_best_params = current_best_params["mlp"]

# Collate metrics and best parameters
all_metrics = {
    "lr": lr_metrics,
    "rf": rf_metrics,
    "mlp": mlp_metrics
}

all_best_params = {
    "lr": lr_best_params,
    "rf": rf_best_params,
    "mlp": mlp_best_params
}

# Save Results
save_json("metrics/undersampling.json", all_metrics)
save_json("best_params/undersampling.json", all_best_params)

