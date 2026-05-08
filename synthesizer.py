import yaml
import json
import random
from pathlib import Path
from typing import List, Dict, Union

from PIL import Image
from hf_client import Text2ImageClient, VisionClient


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("text2image", {})
    cfg["text2image"].setdefault("image_width", 512)
    cfg["text2image"].setdefault("image_height", 512)
    cfg.setdefault("variations", {})
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


def _generate_images(scenario: dict, txt2img_cfg: dict, variations: dict, output_cfg: dict, num_images: int) -> List[Dict]:
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
            "index": i + 1,
            "name": img_name,
            "path": str(img_path),
            "prompt": prompt,
        })
        print(f"  Generated: {img_name}")

    return images


def _run_caption_task(images: List[Dict], itt2t_client: VisionClient, scenario_name: str, caption_path: Path) -> None:
    """Run captioning on all images and write separate JSONL."""
    for img_info in images:
        caption = itt2t_client.caption(img_info["path"])
        record = {
            "task_type": "caption",
            "scenario": scenario_name,
            "prompt": img_info["prompt"],
            "image_path": img_info["path"],
            "caption": caption,
            "metadata": {"image_index": img_info["index"]},
        }
        write_jsonl(record, caption_path)
    print(f"  Wrote {len(images)} caption records -> {caption_path}")


def _run_vqa_task(images: List[Dict], itt2t_client: VisionClient, scenario_name: str, vqa_path: Path) -> None:
    """Run VQA on all images and write separate JSONL."""
    question = "What is the main object in the image?"
    for img_info in images:
        answer = itt2t_client.vqa(img_info["path"], question=question)
        record = {
            "task_type": "vqa",
            "scenario": scenario_name,
            "prompt": img_info["prompt"],
            "image_path": img_info["path"],
            "question": question,
            "answer": answer,
            "metadata": {"image_index": img_info["index"]},
        }
        write_jsonl(record, vqa_path)
    print(f"  Wrote {len(images)} VQA records -> {vqa_path}")


def _run_segmentation_task(images: List[Dict], seg_client: VisionClient, scenario_name: str, seg_path: Path) -> None:
    """Run segmentation on all images and write separate JSONL."""
    for img_info in images:
        seg_results = seg_client.segment(img_info["path"])
        mask_path = None
        if isinstance(seg_results, Image.Image):
            mask_name = f"{scenario_name}_{img_info['index']}_mask.png"
            mask_path = Path(output_cfg["image_dir"]).joinpath(mask_name)
            save_image(seg_results, mask_path)
        elif isinstance(seg_results, list) and len(seg_results) > 0:
            first_mask = seg_results[0].get("mask")
            if isinstance(first_mask, Image.Image):
                mask_name = f"{scenario_name}_{img_info['index']}_mask.png"
                mask_path = Path(output_cfg["image_dir"]).joinpath(mask_name)
                save_image(first_mask, mask_path)

        record = {
            "task_type": "segmentation",
            "scenario": scenario_name,
            "prompt": img_info["prompt"],
            "image_path": img_info["path"],
            "segmentation_mask": str(mask_path) if mask_path else None,
            "metadata": {"image_index": img_info["index"]},
        }
        if mask_path:
            print(f"  Generated mask: {mask_name}")
        write_jsonl(record, seg_path)
    print(f"  Wrote {len(images)} segmentation records -> {seg_path}")


def process_scenario(scenario: dict, txt2img_cfg: dict, vision_cfg: dict, variations: dict, output_cfg: dict, num_images: int) -> None:
    """Process a single scenario: generate images, then run each enabled task."""
    scenario_name = scenario["name"]
    print(f"\nProcessing scenario: {scenario_name}")

    # Task flags (default true if not specified)
    tasks = vision_cfg.get("tasks", {"caption": True, "vqa": True, "segmentation": True})

    # Phase 1: Generate all images (batch)
    images = _generate_images(scenario, txt2img_cfg, variations, output_cfg, num_images)

    # Phase 2: Run vision tasks separately
    if tasks.get("caption") or tasks.get("vqa"):
        itt2t_client = VisionClient(model=vision_cfg["image_text_to_text_model"])
        if tasks.get("caption"):
            _run_caption_task(images, itt2t_client, scenario_name, Path(output_cfg["caption_path"]))
        if tasks.get("vqa"):
            _run_vqa_task(images, itt2t_client, scenario_name, Path(output_cfg["vqa_path"]))
    else:
        itt2t_client = None

    if tasks.get("segmentation"):
        seg_model = vision_cfg.get("segmentation_model")
        seg_client = VisionClient(model=seg_model) if seg_model else None
        if seg_client:
            _run_segmentation_task(images, seg_client, scenario_name, Path(output_cfg["segmentation_path"]))


def run_pipeline(config_path: str = "config.yaml", num_per_scenario: int = 5) -> None:
    cfg = load_config(config_path)
    txt2img_cfg = cfg["text2image"]
    vision_cfg = cfg["vision"]
    output_cfg = cfg.get("output", {})
    variations = cfg.get("variations", {})

    # Set default output paths per task
    output_cfg.setdefault("caption_path", "output/captions.jsonl")
    output_cfg.setdefault("vqa_path", "output/vqa.jsonl")
    output_cfg.setdefault("segmentation_path", "output/segmentation.jsonl")

    # Ensure output directories exist
    Path(output_cfg.get("image_dir", "output/images")).mkdir(parents=True, exist_ok=True)
    Path(output_cfg.get("caption_path", "output/captions.jsonl")).parent.mkdir(parents=True, exist_ok=True)
    Path(output_cfg.get("vqa_path", "output/vqa.jsonl")).parent.mkdir(parents=True, exist_ok=True)
    Path(output_cfg.get("segmentation_path", "output/segmentation.jsonl")).parent.mkdir(parents=True, exist_ok=True)

    for scenario in txt2img_cfg.get("scenarios", []):
        if not scenario.get("enabled", True):
            print(f"Skipping disabled scenario: {scenario['name']}")
            continue
        process_scenario(scenario, txt2img_cfg, vision_cfg, variations, output_cfg, num_per_scenario)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Satellite Dataset Generator")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")
    parser.add_argument("--num", type=int, default=5, help="Number of images per scenario")
    args = parser.parse_args()
    run_pipeline(config_path=args.config, num_per_scenario=args.num)
