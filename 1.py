from huggingface_hub import HfApi, upload_folder
import os

# Set your Hugging Face username and repo name
USERNAME = "Tirth2102"  # Change to your Hugging Face username
REPO_NAME = "Job-Recommendation-System"
MODEL_PATH = r"C:\Users\Admin\Desktop\New folder\output\model-last"

# Initialize the HfApi client
api = HfApi()

# 1. Create a new Hugging Face repository (if not already created)
def create_hf_repo():
    repo_id = f"{USERNAME}/{REPO_NAME}"
    api.create_repo(
        repo_id=repo_id,
        private=True,             # Set to False if you want it public
        repo_type="model",
        exist_ok=True             # Avoid error if the repo already exists
    )
    print(f"✅ Repository '{repo_id}' created (or already exists).")
    return repo_id

# 2. Upload the model folder to Hugging Face Hub
def upload_model_to_hf(repo_id):
    if not os.path.exists(MODEL_PATH):
        raise Exception(f"❌ Model folder '{MODEL_PATH}' not found.")
    
    upload_folder(
        folder_path=MODEL_PATH,
        path_in_repo="model-last",   # Folder name in Hugging Face repo
        repo_id=repo_id,
        commit_message="Upload spaCy model"
    )
    print(f"🚀 Model uploaded to Hugging Face Hub at '{repo_id}'.")

if __name__ == "__main__":
    repo_id = create_hf_repo()
    upload_model_to_hf(repo_id)
