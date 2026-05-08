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


def _image_to_base64(image_path: Union[str, Path, bytes]) -> str:
    """Convert an image path or bytes to a base64 data URI string."""
    if isinstance(image_path, (str, Path)):
        with open(image_path, "rb") as f:
            data = f.read()
    elif isinstance(image_path, bytes):
        data = image_path
    else:
        buf = io.BytesIO()
        image_path.save(buf, format="PNG")
        data = buf.getvalue()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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

    def _chat(self, image_path: Union[str, Path, bytes], prompt: str) -> str:
        """Send a multimodal chat completion request (image-text-to-text)."""
        b64 = _image_to_base64(image_path)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": b64}},
                    ],
                }
            ],
        )
        return completion.choices[0].message.content

    def caption(self, image_path: Union[str, Path, bytes], num: int = 1) -> list:
        """Generate N different captions for an image.

        Returns a list of caption strings.
        """
        b64 = _image_to_base64(image_path)
        prompt = (
            "You are a satellite imagery captioning assistant. "
            "Look at the image carefully and generate a concise, descriptive caption. "
            "Focus on what is visible: landscapes, structures, terrain, water, vegetation, etc."
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": b64}},
                    ],
                }
            ],
            temperature=0.7 + (0.1 * num),  # increase creativity per sample
            n=num,
        )
        return [choice.message.content for choice in completion.choices]

    def vqa(self, image_path: Union[str, Path, bytes], num: int = 1) -> list:
        """Generate N Q&A pairs for an image.

        The model acts as a VQA generator: it creates both questions and answers
        about the image content.

        Returns a list of dicts: [{"question": "...", "answer": "..."}, ...]
        """
        b64 = _image_to_base64(image_path)
        prompt = (
            "You are a VQA (Visual Question Answering) dataset generator. "
            "Look at the image and generate exactly the requested number of diverse Q&A pairs. "
            "Each pair should ask something specific about the image and provide a factual answer. "
            "Vary the questions: ask about objects, colors, layout, relationships, weather, etc. "
            "Output format: a JSON array of objects with 'question' and 'answer' keys. "
            "Example: [{\"question\": \"What is in the center?\", \"answer\": \"A runway\"}]"
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": b64}},
                    ],
                }
            ],
            temperature=0.8,
            n=1,
        )
        response = completion.choices[0].message.content
        # Try to parse JSON array from response
        try:
            import json
            pairs = json.loads(response)
            if isinstance(pairs, list) and len(pairs) >= num:
                return pairs[:num]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: create a single Q&A from the text response
        return [{"question": "What is in this image?", "answer": response.strip()}]

    def segment(self, image_path: Union[str, Path, bytes]) -> Any:
        """Performs image segmentation."""
        return self.client.image_segmentation(image_path)

    def detect(self, image_path: Union[str, Path, bytes]) -> Any:
        """Performs object detection."""
        return self.client.object_detection(image_path)
