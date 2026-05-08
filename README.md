# Satellite Dataset Synthesizer

## Overview

A Python utility that generates synthetic satellite imagery using Hugging Face **text‑to‑image** models and enriches each image with metadata (captions, VQA answers, segmentation masks, etc.) via Vision models.  The generated assets are packaged into a `datasets` dataset and can be pushed directly to a Hugging Face Hub repository.

---

## Quick Start

```bash
# Clone / initialise the project directory
cd c:/Files/Projects/sat-lite/data-gen

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your HF token
echo HF_TOKEN=YOUR_HF_TOKEN > .env
```

### Generate a dataset

```bash
python synthesizer.py --config config.yaml --num 5
```

This will:
1. Generate 5 images for each scenario defined in `config.yaml`.
2. Run captioning, VQA, and optional segmentation.
3. Store images under `output/images` and metadata in `output/metadata.jsonl`.

### Upload to Hugging Face Hub

```bash
python uploader.py --config config.yaml
```

The script creates (or updates) the repository specified in `upload.repo_id` and publishes the dataset.

---

## Configuration (`config.yaml`)

The YAML file controls the entire pipeline.  See the file itself for a full example.  Key sections:

- **text2image** – model ID and a list of scenarios with custom prompts.
- **vision** – models for captioning, segmentation, and object detection/VQA.
- **upload** – Hub repository details, privacy flag, and tags.
- **output** – directories for generated images and the JSON‑Lines metadata file.

Edit the prompts or swap models to tailor the synthetic data to your needs.

---

## Project Structure

```
.
├── .env                # HF token (keep secret)
├── .gitignore          # ignores .env, output/, __pycache__/, *.pyc
├── config.yaml         # pipeline configuration
├── requirements.txt    # Python dependencies
├── hf_client.py        # HF inference wrapper
├── synthesizer.py      # dataset generation logic
├── uploader.py         # build and push dataset to Hub
└── output/            # generated images & metadata (auto‑created)
```

---

## Extending the Pipeline

- **Add new vision tasks** – implement additional methods in `VisionClient` and call them from `process_scenario`.
- **Custom post‑processing** – modify `write_metadata` to include extra fields (e.g., geocoordinates, timestamps).
- **Parallel generation** – replace the simple loop with `concurrent.futures` for faster batch creation.

---

## License & Contributions

Feel free to fork, adapt, and contribute back via pull requests.  The code is released under the MIT license.


## Reference

- https://huggingface.co/docs/inference-providers/tasks/index
