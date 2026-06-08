# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from xgboost import XGBClassifier

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Data Paths
Xtrain_path = "hf://datasets/rajmayank092018/pred-engine-maintenance/Xtrain.csv"
Xtest_path = "hf://datasets/rajmayank092018/pred-engine-maintenance/Xtest.csv"
ytrain_path = "hf://datasets/rajmayank092018/pred-engine-maintenance/ytrain.csv"
ytest_path = "hf://datasets/rajmayank092018/pred-engine-maintenance/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)

# FIX 1: Flatten targets to 1D arrays to prevent scikit-learn fit failure errors
y_train = pd.read_csv(ytrain_path).values.ravel()
y_test = pd.read_csv(ytest_path).values.ravel()

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

params = {
    "Decision Tree": { 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10] },
    "Random Forest": { 'n_estimators': [100, 150], 'max_depth': [5, 10] },
    "AdaBoost": { 'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1] },
    "Gradient Boosting": { 'n_estimators': [100, 150], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5] },
    "Bagging": { 'n_estimators': [10, 50] },
    "XGBoost": { 'n_estimators': [100, 150], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5] }
}

best_models = {}
results = []

# Start a parent MLflow run to group the entire multi-model tournament
with mlflow.start_run(run_name="Ensemble_Comparison_Experiment"):
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create a nested run for each specific algorithm family
        with mlflow.start_run(run_name=name, nested=True):
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params[name],
                cv=3,
                scoring='f1',
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            # Store the best estimator locally
            best_model_obj = grid_search.best_estimator_
            best_models[name] = best_model_obj 

            # Predictions
            preds = best_model_obj.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            # --- 1. Log Best Parameters to MLflow ---
            clean_name_prefix = name.replace(' ', '_').lower()
            mlflow.log_params({f"{clean_name_prefix}__{k}": v for k, v in grid_search.best_params_.items()})

            # --- 2. Log Metrics to MLflow ---
            mlflow.log_metrics({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1-score": f1
            })

            # --- 3. Log the Best Model Artifact ---
            mlflow.sklearn.log_model(best_model_obj, name=f"{clean_name_prefix}_best_model")

            # Store results locally for DataFrame display
            results.append({ 
                "Model": name,
                "Best Parameters": grid_search.best_params_,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
            
            print("Best Parameters:", grid_search.best_params_)
            print("F1 Score:", f1)
            print("-" * 50)

    # --- Post-Training Evaluation and File Serialization ---
    results_df = pd.DataFrame(results)
    print("\nAll Model Results Summary:")
    print(results_df)

    # Identify best model based on F1 Score
    best_model_row = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]
    best_model_name = best_model_row['Model']
    print(f"\nWinner Model: {best_model_name}")

    # FIX 2: Extract the actual model object from the dictionary, NOT the string name
    final_model_object = best_models[best_model_name]

    # Save the real model object locally
    model_path = "pred_maintenance_model_v1.joblib"
    joblib.dump(final_model_object, model_path)

    # Log the model file as a generic artifact to the parent run
    mlflow.log_artifact(model_path, artifact_path="final_production_model")
    print(f"Model successfully saved as artifact at: {model_path}")

# --- Uploading clean model file to Hugging Face ---
repo_id = "rajmayank092018/pred-engine-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repository '{repo_id}' created.")

print("Uploading production binary file to Hugging Face Hub...")
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="pred_maintenance_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type
)
print("Upload Complete!")
