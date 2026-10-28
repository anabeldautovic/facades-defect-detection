import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F

class GlassDetector:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP model and processor.
        """
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Define the classes we want to differentiate
        self.classes = ["a photo of a glass surface", "a photo of a surface with no glass"]
        self.glass_index = 0  # Index of the positive class

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        """
        Predict if the image contains a glass surface.
        Returns a dictionary with 'is_glass', 'score', and 'probs'.
        """
        if image is None:
            return None

        # Prepare inputs
        inputs = self.processor(text=self.classes, images=image, return_tensors="pt", padding=True).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        # Get probability of glass
        glass_prob = probs[0][self.glass_index].item()
        
        return {
            "is_glass": glass_prob > threshold,
            "score": glass_prob,
            "probs": {
                "glass": glass_prob,
                "no_glass": probs[0][1].item()
            }
        }
