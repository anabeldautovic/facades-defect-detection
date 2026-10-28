import os
from PIL import Image
from typing import List

def get_image_files(directory: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> List[str]:
    """
    recursively find all image files in a directory with given extensions.
    """
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from path and convert to RGB.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
