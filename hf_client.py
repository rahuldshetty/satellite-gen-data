import os
import io
import base64
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables (HF token)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class Text2ImageClient:
    """Client for text-to-image generation using huggingface_hub.InferenceClient."""

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        # provider and api_key must be passed to the constructor in InferenceClient
        self.client = InferenceClient(model=model, provider=provider, token=HF_TOKEN, api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """
        Generates an image from a prompt.
        Returns a PIL.Image object.
        """
        # We don't pass provider/api_key here as they are already in the client
        return self.client.text_to_image(prompt, **kwargs)

class VisionClient:
    """Client for vision tasks using huggingface_hub.InferenceClient."""

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.client = InferenceClient(model=model, provider=provider, token=HF_TOKEN, api_key=api_key)

    def caption(self, image_path: Union[str, Path, bytes]) -> str:
        """Generates a caption for an image (image-to-text task)."""
        result = self.client.image_to_text(image_path)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                return result[0].get("generated_text", "")
            return str(result[0])
        elif isinstance(result, dict):
            return result.get("generated_text", "")
        return str(result)

    def vqa(self, image_path: Union[str, Path, bytes], question: str) -> str:
        """Performs Visual Question Answering."""
        results = self.client.visual_question_answering(image_path, question)
        if results and isinstance(results, list):
            return results[0].get("answer", "")
        return str(results)

    def segment(self, image_path: Union[str, Path, bytes]) -> Any:
        """Performs image segmentation."""
        return self.client.image_segmentation(image_path)

    def detect(self, image_path: Union[str, Path, bytes]) -> Any:
        """Performs object detection."""
        return self.client.object_detection(image_path)
