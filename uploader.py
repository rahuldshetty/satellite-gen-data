import os
import json
import shutil
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, upload_folder
from datasets import Dataset, Features, Image, Value


def load_jsonl(jsonl_path: str) -> List[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_task_dataset(image_dir: str, jsonl_path: str, task_type: str) -> Dataset:
    """Create a `datasets.Dataset` for a specific task type.

    Each task has different fields, so features are built dynamically.
    """
    records = load_jsonl(jsonl_path)
    if not records:
        print(f"  No records found in {jsonl_path}")
        return None

    # Resolve absolute image paths
    for rec in records:
        rec["image"] = os.path.abspath(rec["image_path"])

    # Build features based on task type
    features_dict = {
        "scenario": Value("string"),
        "prompt": Value("string"),
        "image": Image(),
        "metadata": Value("string"),
    }

    if task_type == "caption":
        features_dict["caption"] = Value("string")
    elif task_type == "vqa":
        features_dict["question"] = Value("string")
        features_dict["answer"] = Value("string")
    elif task_type == "segmentation":
        features_dict["segmentation_mask"] = Value("string")

    # Convert metadata dict to JSON string
    for rec in records:
        rec["metadata"] = json.dumps(rec.get("metadata", {}))

    return Dataset.from_dict({k: [r[k] for r in records] for k in features_dict.keys()}, features=Features(features_dict))


def push_dataset_to_hub(
    repo_id: str,
    dataset: Dataset,
    token: str,
    private: bool = False,
    tags: List[str] | None = None,
    commit_message: str = "Add dataset",
    repo_suffix: str = "",
) -> str:
    """Upload a `datasets.Dataset` to the Hugging Face Hub.

    Returns the full dataset URL.
    """
    api = HfApi(token=token)
    full_repo_id = f"{repo_id}{repo_suffix}" if repo_suffix else repo_id

    # Create the repo if it does not exist
    if not api.repo_exists(full_repo_id, repo_type="dataset"):
        api.create_repo(repo_id=full_repo_id, token=token, private=private, repo_type="dataset", exist_ok=True)
        if tags:
            api.add_tags(full_repo_id, tags=tags, repo_type="dataset")

    # Save dataset locally in a temporary folder
    tmp_dir = Path(".temp_hf_dataset")
    tmp_dir.mkdir(exist_ok=True)
    dataset.save_to_disk(tmp_dir)

    # Upload the whole folder
    upload_folder(
        repo_id=full_repo_id,
        folder_path=str(tmp_dir),
        token=token,
        repo_type="dataset",
        commit_message=commit_message,
    )
    shutil.rmtree(tmp_dir)

    return f"https://huggingface.co/datasets/{full_repo_id}"


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not found in .env")

    parser = argparse.ArgumentParser(description="Upload generated satellite dataset to Hugging Face Hub")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")
    args = parser.parse_args()

    # Load config to get upload settings and output paths
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    upload_cfg = cfg.get("upload", {})
    output_cfg = cfg.get("output", {})

    if not upload_cfg.get("enabled", True):
        print("Upload is disabled in config. Skipping.")
        exit(0)

    repo_id = upload_cfg.get("repo_id")
    private = upload_cfg.get("private", False)
    tags = upload_cfg.get("tags", [])

    # Define task datasets to upload
    tasks = {
        "caption": output_cfg.get("caption_path"),
        "vqa": output_cfg.get("vqa_path"),
        "segmentation": output_cfg.get("segmentation_path"),
    }

    image_dir = output_cfg.get("image_dir", "output/images")

    uploaded = []
    for task_type, jsonl_path in tasks.items():
        if not jsonl_path or not os.path.exists(jsonl_path):
            print(f"Skipping {task_type}: {jsonl_path} not found")
            continue

        print(f"\nBuilding {task_type} dataset from {jsonl_path}...")
        ds = build_task_dataset(image_dir, jsonl_path, task_type)
        if ds is None:
            continue

        url = push_dataset_to_hub(
            repo_id=repo_id,
            dataset=ds,
            token=HF_TOKEN,
            private=private,
            tags=tags,
            commit_message=f"Add {task_type} dataset",
            repo_suffix=f"-{task_type}",
        )
        uploaded.append(url)
        print(f"  Uploaded: {url}")

    if uploaded:
        print(f"\n{len(uploaded)} dataset(s) uploaded successfully.")
    else:
        print("No datasets to upload.")
