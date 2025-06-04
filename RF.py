import numpy as np
import optuna
import json
from helper import load_raw_data, load_scaled_data, load_processed_data, evaluate_metrics, print_metrics, weighted_log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

NUM_CLASSES = 28 
SEED = 42

def train_evaluate_rf(x_train_main, x_train_holdout, y_train_main, y_train_holdout):
    # Find optimised paramters for Random Forests
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 18, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            "min_samples_split": trial.suggest_int("min_samples_split", 6, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": SEED,
        }

        model = RandomForestClassifier(**params)
        model.fit(x_train_main, y_train_main)

        pred_probs = model.predict_proba(x_train_holdout)
        labels = y_train_holdout

        one_hot_labels = []
        for label in labels:
            one_hot_array = np.zeros(NUM_CLASSES)
            one_hot_array[label] = 1
            one_hot_labels.append(one_hot_array)

        loss = weighted_log_loss(one_hot_labels, pred_probs)
        return np.mean(loss)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # Train Logistic Regression with best params
    best_params = study.best_params
    final_model = RandomForestClassifier(**best_params, random_state=SEED)
    final_model.fit(x_train_main, y_train_main)

    # Evaluate
    preds = final_model.predict(x_train_holdout)
    pred_probs = final_model.predict_proba(x_train_holdout)
    metrics = evaluate_metrics(preds, pred_probs, y_train_holdout)

    return metrics, best_params

if __name__ == "__main__":
    # # Raw Data
    # x_train, x_test_1, x_test_2_labelled, x_test_2_unlabelled, y_train, y_test_2 = load_raw_data()

    # # Split train into holdout
    # x_train_main, x_train_holdout, y_train_main, y_train_holdout = train_test_split(
    #     x_train, y_train, random_state=SEED, test_size=0.2, stratify=y_train
    # )
    
    # # Evaluate
    # raw_metrics, raw_best_params = train_evaluate_rf(x_train_main, x_train_holdout, y_train_main, y_train_holdout)

    # # Scaled Data
    # x_train, x_test_1, x_test_2_labelled, x_test_2_unlabelled, y_train, y_test_2 = load_scaled_data()
    
    # # Split train into holdout
    # x_train_main, x_train_holdout, y_train_main, y_train_holdout = train_test_split(
    #     x_train, y_train, random_state=SEED, test_size=0.2, stratify=y_train
    # )

    # # Evaluate
    # scaled_metrics, scaled_best_params = train_evaluate_rf(x_train_main, x_train_holdout, y_train_main, y_train_holdout)

    # Processed Data
    x_train, x_test_1, x_test_2_labelled, x_test_2_unlabelled, y_train, y_test_2 = load_processed_data()

    # Split train into holdout
    x_train_main, x_train_holdout, y_train_main, y_train_holdout = train_test_split(
        x_train, y_train, random_state=SEED, test_size=0.2, stratify=y_train
    )
    
    # Evaluate
    processed_metrics, processed_best_params = train_evaluate_rf(x_train_main, x_train_holdout, y_train_main, y_train_holdout)

    all_metrics = {
        # "raw": raw_metrics,
        # "scaled": scaled_metrics,
        "processed": processed_metrics
    }

    all_best_params = {
        # "raw": raw_best_params,
        # "scaled": scaled_best_params,
        "processed": processed_best_params
    }

    with open("metrics/rf.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    with open("best_params/rf.json", "w") as f:
        json.dump(all_best_params, f, indent=4)
