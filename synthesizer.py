import yaml
import json
import random
from pathlib import Path
from typing import List, Dict, Union

from PIL import Image
from datasets import load_dataset
from huggingface_hub import list_repo_files
from hf_client import Text2ImageClient, VisionClient


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("text2image", {})
    cfg["text2image"].setdefault("image_width", 512)
    cfg["text2image"].setdefault("image_height", 512)
    cfg.setdefault("variations", {})
    cfg.setdefault("hf_img_datasets", [])
    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image(image: Union[bytes, Image.Image], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if isinstance(image, Image.Image):
        image.save(output_path)
    else:
        output_path.write_bytes(image)


def write_jsonl(record: dict, jsonl_path: Path) -> None:
    """Append a single JSON record to a JSONL file."""
    ensure_dir(jsonl_path.parent)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _apply_variations(prompt: str, variations: dict) -> str:
    """Replace placeholders like {weather} with random values."""
    result = prompt
    for var_key, var_options in variations.items():
        placeholder = f"{{{var_key}}}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(var_options))
    return result


def _load_hf_images(hf_cfg: dict, count: int, output_cfg: dict) -> List[Dict]:
    """Load and randomly sample images from a Hugging Face dataset using streaming.

    Prefers parquet files from refs/convert/parquet for efficient streaming.
    Falls back to default streaming if parquet files aren't available.
    """
    repo_id = hf_cfg["repo_id"]
    split = hf_cfg.get("split", "train")
    image_column = hf_cfg.get("image_column", "image")
    name = hf_cfg.get("name", repo_id.split("/")[-1])

    # Check for parquet files (refs/convert/parquet) for efficient streaming
    parquet_files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith("refs/convert/parquet/") and f.endswith(".parquet")
    ]

    if parquet_files:
        print(f"  Using parquet streaming: {len(parquet_files)} files from refs/convert/parquet")
        ds = load_dataset(
            repo_id,
            data_files={"train": parquet_files},
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
    else:
        print(f"  Streaming HF dataset: {repo_id} (split={split}, column={image_column})")
        ds = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=True)

    # Shuffle and select only the images we need (lazy loading)
    shuffled = ds.shuffle(seed=random.randint(0, 2**31))
    selected = list(shuffled.take(count))
    print(f"  Sampled {len(selected)} images (streaming)")

    images = []
    for i, row in enumerate(selected):
        img = row[image_column]
        if not isinstance(img, Image.Image):
            print(f"  Warning: skipping row {i} — not a PIL Image")
            continue
        img_name = f"hf_{name}_{i+1}.png"
        img_path = Path(output_cfg["image_dir"]).joinpath(img_name)
        save_image(img, img_path)
        images.append({
            "source": "hf",
            "source_name": name,
            "index": i + 1,
            "name": img_name,
            "path": str(img_path),
            "prompt": f"Satellite image from {name} dataset",
        })
        print(f"  Loaded: {img_name}")

    return images


def _generate_images(scenario: dict, txt2img_cfg: dict, variations: dict, num_images: int) -> List[Dict]:
    """Generate all images for a scenario and return list of image info dicts."""
    provider = txt2img_cfg.get("provider")
    txt2img_client = Text2ImageClient(
        model=scenario.get("model", txt2img_cfg.get("model")),
        provider=provider
    )

    width = txt2img_cfg.get("image_width", 512)
    height = txt2img_cfg.get("image_height", 512)
    base_prompt = scenario["prompt"]
    images = []

    for i in range(num_images):
        prompt = _apply_variations(base_prompt, variations)
        image = txt2img_client.generate(prompt=prompt, width=width, height=height)
        img_name = f"{scenario['name']}_{i+1}.png"
        img_path = Path(output_cfg["image_dir"]).joinpath(img_name)
        save_image(image, img_path)
        images.append({
            "source": "generated",
            "source_name": scenario["name"],
            "index": i + 1,
            "name": img_name,
            "path": str(img_path),
            "prompt": prompt,
        })
        print(f"  Generated: {img_name}")

    return images


def _run_caption_task(images: List[Dict], itt2t_client: VisionClient, caption_path: Path) -> None:
    """Run captioning on all images and write separate JSONL."""
    total = 0
    for img_info in images:
        captions = itt2t_client.caption(img_info["path"], num=3)
        for idx, caption in enumerate(captions):
            record = {
                "task_type": "caption",
                "source": img_info["source"],
                "source_name": img_info["source_name"],
                "prompt": img_info["prompt"],
                "image_path": img_info["path"],
                "caption": caption,
                "metadata": {"image_index": img_info["index"], "caption_index": idx + 1},
            }
            write_jsonl(record, caption_path)
            total += 1
    print(f"  Wrote {total} caption records -> {caption_path}")


def _run_vqa_task(images: List[Dict], itt2t_client: VisionClient, vqa_path: Path) -> None:
    """Run VQA on all images and write separate JSONL."""
    total = 0
    for img_info in images:
        qa_pairs = itt2t_client.vqa(img_info["path"], num=3)
        for idx, pair in enumerate(qa_pairs):
            record = {
                "task_type": "vqa",
                "source": img_info["source"],
                "source_name": img_info["source_name"],
                "prompt": img_info["prompt"],
                "image_path": img_info["path"],
                "question": pair.get("question", ""),
                "answer": pair.get("answer", ""),
                "metadata": {"image_index": img_info["index"], "qa_index": idx + 1},
            }
            write_jsonl(record, vqa_path)
            total += 1
    print(f"  Wrote {total} VQA records -> {vqa_path}")


def _run_segmentation_task(images: List[Dict], seg_client: VisionClient, seg_path: Path) -> None:
    """Run segmentation on all images and write separate JSONL."""
    total = 0
    for img_info in images:
        seg_results = seg_client.segment(img_info["path"])
        mask_path = None
        if isinstance(seg_results, Image.Image):
            mask_name = f"{img_info['source_name']}_{img_info['index']}_mask.png"
            mask_path = Path(output_cfg["image_dir"]).joinpath(mask_name)
            save_image(seg_results, mask_path)
        elif isinstance(seg_results, list) and len(seg_results) > 0:
            first_mask = seg_results[0].get("mask")
            if isinstance(first_mask, Image.Image):
                mask_name = f"{img_info['source_name']}_{img_info['index']}_mask.png"
                mask_path = Path(output_cfg["image_dir"]).joinpath(mask_name)
                save_image(first_mask, mask_path)

        record = {
            "task_type": "segmentation",
            "source": img_info["source"],
            "source_name": img_info["source_name"],
            "prompt": img_info["prompt"],
            "image_path": img_info["path"],
            "segmentation_mask": str(mask_path) if mask_path else None,
            "metadata": {"image_index": img_info["index"]},
        }
        if mask_path:
            print(f"  Generated mask: {mask_name}")
        write_jsonl(record, seg_path)
        total += 1
    print(f"  Wrote {total} segmentation records -> {seg_path}")


def process_hf_datasets(hf_datasets: list, output_cfg: dict, vision_cfg: dict, tasks: dict) -> None:
    """Process images from HF datasets through enabled vision tasks.

    hf_datasets should already be filtered for enabled=True.
    """
    if not hf_datasets:
        return

    # Task flags
    do_caption = tasks.get("caption", False)
    do_vqa = tasks.get("vqa", False)
    do_segmentation = tasks.get("segmentation", False)

    if not any([do_caption, do_vqa, do_segmentation]):
        print("No tasks enabled. Skipping.")
        return

    # Create vision clients as needed
    if do_caption or do_vqa:
        itt2t_client = VisionClient(model=vision_cfg["image_text_to_text_model"])
    else:
        itt2t_client = None

    if do_segmentation:
        seg_model = vision_cfg.get("segmentation_model")
        seg_client = VisionClient(model=seg_model) if seg_model else None
    else:
        seg_client = None

    # Process each HF dataset
    for hf_cfg in hf_datasets:
        dataset_name = hf_cfg.get("name", hf_cfg["repo_id"].split("/")[-1])
        caption_count = output_cfg.get("caption_count", 50)
        vqa_count = output_cfg.get("vqa_count", 50)
        seg_count = output_cfg.get("segmentation_count", 50)

        # Determine how many images to sample (use max of enabled task counts)
        counts = []
        if do_caption:
            counts.append(caption_count)
        if do_vqa:
            counts.append(vqa_count)
        if do_segmentation:
            counts.append(seg_count)
        num_images = max(counts) if counts else 50

        print(f"\nProcessing HF dataset: {dataset_name}")
        images = _load_hf_images(hf_cfg, num_images, output_cfg)

        if do_caption:
            _run_caption_task(images, itt2t_client, Path(output_cfg["caption_path"]))
        if do_vqa:
            _run_vqa_task(images, itt2t_client, Path(output_cfg["vqa_path"]))
        if do_segmentation and seg_client:
            _run_segmentation_task(images, seg_client, Path(output_cfg["segmentation_path"]))


def process_scenarios(scenarios: list, txt2img_cfg: dict, variations: dict, output_cfg: dict, vision_cfg: dict, tasks: dict) -> None:
    """Process text-to-image scenarios (synthetic generation)."""
    if not scenarios:
        return

    # Task flags
    do_caption = tasks.get("caption", False)
    do_vqa = tasks.get("vqa", False)
    do_segmentation = tasks.get("segmentation", False)

    if not any([do_caption, do_vqa, do_segmentation]):
        print("No tasks enabled. Skipping.")
        return

    # Create vision clients as needed
    if do_caption or do_vqa:
        itt2t_client = VisionClient(model=vision_cfg["image_text_to_text_model"])
    else:
        itt2t_client = None

    if do_segmentation:
        seg_model = vision_cfg.get("segmentation_model")
        seg_client = VisionClient(model=seg_model) if seg_model else None
    else:
        seg_client = None

    for scenario in scenarios:
        if not scenario.get("enabled", True):
            print(f"Skipping disabled scenario: {scenario['name']}")
            continue

        scenario_name = scenario["name"]
        print(f"\nProcessing scenario: {scenario_name}")

        # Generate images
        images = _generate_images(scenario, txt2img_cfg, variations, 5)

        if do_caption:
            _run_caption_task(images, itt2t_client, Path(output_cfg["caption_path"]))
        if do_vqa:
            _run_vqa_task(images, itt2t_client, Path(output_cfg["vqa_path"]))
        if do_segmentation and seg_client:
            _run_segmentation_task(images, seg_client, Path(output_cfg["segmentation_path"]))


def run_pipeline(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    txt2img_cfg = cfg["text2image"]
    vision_cfg = cfg["vision"]
    output_cfg = cfg.get("output", {})
    variations = cfg.get("variations", {})
    hf_datasets = cfg.get("hf_img_datasets", [])
    scenarios = txt2img_cfg.get("scenarios", [])
    tasks = vision_cfg.get("tasks", {"caption": True, "vqa": True, "segmentation": True})

    # Check which sources are enabled
    synth_enabled = txt2img_cfg.get("enabled", True)
    hf_enabled = [ds for ds in hf_datasets if ds.get("enabled", True)]

    # Validate at least one source is enabled
    has_enabled_scenarios = any(s.get("enabled", True) for s in scenarios)
    if not synth_enabled or not has_enabled_scenarios:
        synth_enabled = False

    if not synth_enabled and not hf_enabled:
        print("Error: No image source enabled. Enable at least one of:")
        print("  - text2image.enabled + a scenario with enabled: true")
        print("  - hf_img_datasets with enabled: true")
        exit(1)

    # Set default output paths per task
    output_cfg.setdefault("caption_path", "output/captions.jsonl")
    output_cfg.setdefault("vqa_path", "output/vqa.jsonl")
    output_cfg.setdefault("segmentation_path", "output/segmentation.jsonl")

    # Ensure output directories exist
    Path(output_cfg.get("image_dir", "output/images")).mkdir(parents=True, exist_ok=True)
    Path(output_cfg.get("caption_path", "output/captions.jsonl")).parent.mkdir(parents=True, exist_ok=True)
    Path(output_cfg.get("vqa_path", "output/vqa.jsonl")).parent.mkdir(parents=True, exist_ok=True)
    Path(output_cfg.get("segmentation_path", "output/segmentation.jsonl")).parent.mkdir(parents=True, exist_ok=True)

    # Process HF dataset images first
    if hf_enabled:
        process_hf_datasets(hf_enabled, output_cfg, vision_cfg, tasks)

    # Then process synthetic scenarios
    if synth_enabled:
        process_scenarios(scenarios, txt2img_cfg, variations, output_cfg, vision_cfg, tasks)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Satellite Dataset Generator")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")
    args = parser.parse_args()
    run_pipeline(config_path=args.config)
