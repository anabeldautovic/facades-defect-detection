import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch.nn.functional as F

class GlassDetectorV2:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        """
        Initialize the SigLIP model.
        """
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Ensemble prompts
        self.positive_prompts = [
            "a photo of a glass surface",
            "transparent glass",
            "reflective glass window",
            "a windshield",
            "clear glass"
        ]
        
        self.negative_prompts = [
            "a photo of a wood surface",
            "metal surface",
            "plastic object",
            "concrete wall",
            "rock",
            "fabric",
            "opaque material",
            "something that is not glass"
        ]
        
        # Pre-compute classifier weights (average embeddings)
        self._build_classifier()
        self._build_crack_classifier()

    def _build_classifier(self):
        print("Building prompt ensemble classifier (Glass vs No Glass)...")
        # Get embeddings for all prompts
        all_texts = self.positive_prompts + self.negative_prompts
        self.classifier = self._create_ensemble_weights(all_texts, len(self.positive_prompts))

    def _build_crack_classifier(self):
        print("Building prompt ensemble classifier (Cracked vs Intact)...")
        positive_crack = [
            "cracked glass",
            "broken glass",
            "shattered glass",
            "glass with fissures",
            "spiderweb crack",
            "damaged glass surface"
        ]
        negative_crack = [
            "intact glass",
            "clean glass",
            "unbroken glass",
            "perfect condition glass",
            "smooth glass surface"
        ]
        all_texts = positive_crack + negative_crack
        self.crack_classifier = self._create_ensemble_weights(all_texts, len(positive_crack))

    def _create_ensemble_weights(self, texts, n_pos):
        inputs = self.processor(text=texts, return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
        pos_embeddings = text_embeddings[:n_pos]
        neg_embeddings = text_embeddings[n_pos:]
        
        img_pos_embed = pos_embeddings.mean(dim=0)
        img_neg_embed = neg_embeddings.mean(dim=0)
        
        classifier = torch.stack([img_pos_embed, img_neg_embed])
        return classifier / classifier.norm(p=2, dim=-1, keepdim=True)

    def _predict_with_classifier(self, image_features, classifier, threshold):
        # Cosine similarity (logits)
        logits = torch.matmul(image_features, classifier.T)
        
        if hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale.exp() + self.model.logit_bias if hasattr(self.model, "logit_bias") else 0
            
        probs = logits.softmax(dim=1)
        
        pos_prob = probs[0][0].item()
        neg_prob = probs[0][1].item()
        
        return {
            "result": pos_prob > threshold,
            "score": pos_prob,
            "probs": {"pos": pos_prob, "neg": neg_prob}
        }

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        """
        Predict if the image contains a glass surface.
        """
        if image is None: return None
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
             image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        res = self._predict_with_classifier(image_features, self.classifier, threshold)
        return {
            "is_glass": res["result"],
            "score": res["score"],
            "probs": {"glass": res["probs"]["pos"], "no_glass": res["probs"]["neg"]},
            "features": image_features # Return features to reuse for crack detection
        }

    def detect_crack(self, image_features, threshold: float = 0.5) -> dict:
        """
        Predict if the glass is cracked.
        """
        res = self._predict_with_classifier(image_features, self.crack_classifier, threshold)
        return {
            "is_cracked": res["result"],
            "score": res["score"],
            "probs": {"cracked": res["probs"]["pos"], "intact": res["probs"]["neg"]}
        }
