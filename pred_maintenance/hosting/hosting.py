from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="pred_maintenance/deployment",     # the local folder containing your files
    repo_id="rajmayank092018/predictive-maintenance-engine-failure-predictor",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
