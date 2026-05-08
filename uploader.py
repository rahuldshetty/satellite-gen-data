import os
import json
from pathlib import Path
from typing import List

from huggingface_hub import Repository, HfApi, create_commit, upload_folder
from datasets import Dataset, DatasetDict, Features, Image, Value


def load_metadata(metadata_path: str) -> List[dict]:
    """Load metadata records from a JSONL file."""
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(image_dir: str, metadata_path: str) -> Dataset:
    """Create a `datasets.Dataset` where each example contains the image and its metadata.

    The image column uses the `Image` feature type, allowing automatic loading of image files.
    """
    records = load_metadata(metadata_path)
    # Resolve absolute image paths
    for rec in records:
        rec["image"] = os.path.abspath(rec["image_path"])  # rename for HF Image feature
    features = Features({
        "scenario": Value("string"),
        "prompt": Value("string"),
        "image": Image(),
        "caption": Value("string"),
        "vqa_answer": Value("string"),
        "segmentation_mask": Value("string", nullable=True),
        "metadata": Value("string"),  # store as JSON string for simplicity
    })
    # Convert metadata dict to JSON string
    for rec in records:
        rec["metadata"] = json.dumps(rec.get("metadata", {}))
    return Dataset.from_dict({k: [r[k] for r in records] for k in records[0].keys()}, features=features)


def push_dataset_to_hub(
    repo_id: str,
    dataset: Dataset,
    token: str,
    private: bool = False,
    tags: List[str] | None = None,
    commit_message: str = "Add synthetic satellite dataset",
) -> None:
    """Upload a `datasets.Dataset` to the Hugging Face Hub.

    Parameters
    ----------
    repo_id: str
        The identifier of the repository, e.g. ``username/dataset-name``.
    dataset: Dataset
        The dataset object to upload.
    token: str
        HF access token with write permissions.
    private: bool
        Whether the repo should be private.
    tags: List[str] | None
        Optional list of tags to attach to the repo.
    commit_message: str
        Message for the initial commit.
    """
    api = HfApi(token=token)
    # Create the repo if it does not exist
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(repo_id=repo_id, token=token, private=private, repo_type="dataset", exist_ok=True)
        if tags:
            api.update_repo_visibility(repo_id=repo_id, private=private, repo_type="dataset")
            api.add_tags(repo_id, tags=tags, repo_type="dataset")

    # Save dataset locally in a temporary folder
    tmp_dir = Path(".temp_hf_dataset")
    tmp_dir.mkdir(exist_ok=True)
    dataset.save_to_disk(tmp_dir)

    # Upload the whole folder
    upload_folder(
        repo_id=repo_id,
        folder_path=str(tmp_dir),
        token=token,
        repo_type="dataset",
        commit_message=commit_message,
    )
    # Clean up temporary directory
    import shutil
    shutil.rmtree(tmp_dir)

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

    ds = build_dataset(
        image_dir=output_cfg.get("image_dir", "output/images"),
        metadata_path=output_cfg.get("metadata_path", "output/metadata.jsonl"),
    )
    push_dataset_to_hub(
        repo_id=upload_cfg.get("repo_id"),
        dataset=ds,
        token=HF_TOKEN,
        private=upload_cfg.get("private", False),
        tags=upload_cfg.get("tags", []),
    )

    print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{upload_cfg.get('repo_id')}")
