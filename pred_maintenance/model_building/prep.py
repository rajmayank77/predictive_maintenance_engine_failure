from sklearn.model_selection import train_test_split
import pandas as pd
from huggingface_hub import hf_hub_download
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
import os

api = HfApi(token=os.getenv('HF_TOKEN'))
REPO_ID = "rajmayank092018/pred_engine_maintenance"

try:

    local_file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="data/engine_data.csv",
        repo_type="dataset",
        token=os.getenv('HF_TOKEN')
    )

    data_df = pd.read_csv(local_file_path)

    engine_data_df = data_df.copy()

    print("Success! File downloaded locally and loaded into data_df.")
    print(engine_data_df.head())

except Exception as e:
    print(f"An error occurred: {e}")

engine_data_df.columns = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature",
    "Engine_Condition"
]

X = engine_data_df.drop('Engine_Condition', axis=1)
y = engine_data_df['Engine_Condition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="rajmayank092018/pred-engine-maintenance",
        repo_type="dataset",
    )
