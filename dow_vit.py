from huggingface_hub import snapshot_download

snapshot_download(repo_id="google/vit-base-patch16-224", local_dir="./vit_model")
