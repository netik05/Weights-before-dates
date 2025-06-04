from MLP import train_evaluate_mlp
from helper import load_raw_data, load_scaled_data, load_processed_data, print_metrics, load_json, save_json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed 

SEED = 42

x_train, x_test_1, x_test_2_labelled, x_test_2_unlabelled, y_train, y_test_2 = load_processed_data()

# Split train into holdout
x_train_main, x_train_holdout, y_train_main, y_train_holdout = train_test_split(
    x_train, y_train, random_state=SEED, test_size=0.2, stratify=y_train
)

# Load existing data or initialize
current_metrics = load_json("metrics/smote_knn_tuning.json.json")
current_best_params = load_json("best_params/smote_knn_tuning.json.json")

all_metrics = {}
all_best_params = {}

def run_oversampled_mlp(i):
    run_name = f"{i}_nearest_neightbours_with_mlp"

    if run_name not in current_metrics:
        smote = SMOTE(random_state=0, k_neighbors=i)
        x_train_res, y_train_res = smote.fit_resample(x_train_main, y_train_main)
            
        # Evaluate MLP
        mlp_metrics, mlp_best_params = train_evaluate_mlp(
            x_train_res, x_train_holdout, y_train_res, y_train_holdout, dataset_type='SMOTE'
        )
    else:
        mlp_metrics = current_metrics[run_name]
        mlp_best_params = current_best_params[run_name]

    all_metrics[run_name] = mlp_metrics
    all_best_params[run_name] = mlp_best_params

    return run_name, mlp_metrics, mlp_best_params

# # Single Thread Execution
# for i in range(1, 5):
#     run_name, mlp_metrics, mlp_best_params = run_oversampled_mlp(i)
#     all_metrics[run_name] = mlp_metrics
#     all_best_params[run_name] = mlp_best_params

# Parallel Execution
N_JOBS = 4
results = Parallel(n_jobs=N_JOBS)(
    delayed(run_oversampled_mlp)(i) for i in range(1, 5)
)
for run_name, metrics, best_params in results:
    all_metrics[run_name] = metrics
    all_best_params[run_name] = best_params

# Save Results
save_json("metrics/smote_knn_tuning.json", all_metrics)
save_json("best_params/smote_knn_tuning.json", all_best_params)

