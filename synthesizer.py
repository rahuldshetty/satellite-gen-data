import yaml
import json
import os
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


def write_metadata(record: dict, metadata_path: Path) -> None:
    ensure_dir(metadata_path.parent)
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def process_scenario(scenario: dict, txt2img_cfg: dict, vision_cfg: dict, variations: dict, output_cfg: dict, num_images: int) -> None:
    """Process a single scenario, generate images and extract metadata."""
    provider = txt2img_cfg.get("provider")
    
    txt2img_client = Text2ImageClient(
        model=scenario.get("model", txt2img_cfg.get("model")),
        provider=provider
    )
    
    # Vision clients for different tasks
    caption_client = VisionClient(model=vision_cfg["caption_model"])
    detection_client = VisionClient(model=vision_cfg["detection_model"])
    segmentation_client = VisionClient(model=vision_cfg.get("segmentation_model")) if vision_cfg.get("segmentation_model") else None

    width = txt2img_cfg.get("image_width", 512)
    height = txt2img_cfg.get("image_height", 512)

    for i in range(num_images):
        prompt = scenario["prompt"]
        # Apply random variations placeholders
        for var_key, var_options in variations.items():
            placeholder = f"{{{var_key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, random.choice(var_options))
        
        # Generate image
        image = txt2img_client.generate(prompt=prompt, width=width, height=height)
        img_name = f"{scenario['name']}_{i+1}.png"
        img_path = Path(output_cfg["image_dir"]).joinpath(img_name)
        save_image(image, img_path)

        # Vision tasks
        caption = caption_client.caption(img_path)
        vqa_answer = detection_client.vqa(img_path, question="What is the main object in the image?")
        
        mask_path = None
        if segmentation_client:
            seg_results = segmentation_client.segment(img_path)
            # seg_results can be a PIL Image or a list of dicts with 'mask'
            if isinstance(seg_results, Image.Image):
                mask_path = Path(output_cfg["image_dir"]).joinpath(f"{scenario['name']}_{i+1}_mask.png")
                save_image(seg_results, mask_path)
            elif isinstance(seg_results, list) and len(seg_results) > 0:
                # Save the first mask for simplicity in this pipeline
                first_mask = seg_results[0].get("mask")
                if isinstance(first_mask, Image.Image):
                    mask_path = Path(output_cfg["image_dir"]).joinpath(f"{scenario['name']}_{i+1}_mask.png")
                    save_image(first_mask, mask_path)

        record = {
            "scenario": scenario["name"],
            "prompt": prompt,
            "image_path": str(img_path),
            "caption": caption,
            "vqa_answer": vqa_answer,
            "metadata": {"image_index": i + 1},
        }
        if mask_path:
            record["segmentation_mask"] = str(mask_path)
            
        write_metadata(record, Path(output_cfg["metadata_path"]))


def run_pipeline(config_path: str = "config.yaml", num_per_scenario: int = 5) -> None:
    cfg = load_config(config_path)
    txt2img_cfg = cfg["text2image"]
    vision_cfg = cfg["vision"]
    output_cfg = cfg["output"]
    variations = cfg.get("variations", {})

    # Ensure output directories exist
    Path(output_cfg.get("image_dir", "output/images")).mkdir(parents=True, exist_ok=True)
    Path(output_cfg.get("metadata_path", "output/metadata.jsonl")).parent.mkdir(parents=True, exist_ok=True)

    for scenario in txt2img_cfg.get("scenarios", []):
        process_scenario(scenario, txt2img_cfg, vision_cfg, variations, output_cfg, num_per_scenario)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Satellite Dataset Generator")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")
    parser.add_argument("--num", type=int, default=5, help="Number of images per scenario")
    args = parser.parse_args()
    run_pipeline(config_path=args.config, num_per_scenario=args.num)
