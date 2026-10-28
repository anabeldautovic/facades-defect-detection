import os
import argparse
from src.utils import get_image_files, load_image
from src.detector_v2 import GlassDetectorV2

# Project data folder, so the defaults work no matter where the script is run from
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "compressed")

def check_path(path: str) -> str:
    """
    Check if path exists. If not, try navigating up one level.
    Returns the valid path or the original if neither exists.
    """
    if os.path.exists(path):
        return path
    
    # Try ../path
    alt_path = os.path.join("..", path)
    if os.path.exists(alt_path):
        print(f"Note: '{path}' not found, using '{alt_path}' instead.")
        return alt_path
        
    return path

def main():
    parser = argparse.ArgumentParser(description="Glass Surface Detector V2 (SigLIP)")
    parser.add_argument("--negative_dir", type=str, default=os.path.join(DATA_DIR, "negative"), help="Path to negative images folder")
    parser.add_argument("--positive_dir", type=str, default=os.path.join(DATA_DIR, "positive"), help="Path to positive images folder")
    parser.add_argument("--threshold", type=float, default=0.98, help="Probability threshold for glass detection")
    args = parser.parse_args()

    # robust path checking
    neg_dir = check_path(args.negative_dir)
    pos_dir = check_path(args.positive_dir)

    # Initialize detector v2
    detector = GlassDetectorV2()
    
    directories = [neg_dir, pos_dir]
    
    print(f"\nScanning directories: {directories}")
    print(f"Detection Threshold: {args.threshold}\n")
    print("-" * 65)
    print(f"{'File':<40} | {'Score':<10} | {'Result':<10}")
    print("-" * 65)

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
            
        image_files = get_image_files(directory)
        
        for img_path in image_files:
            image = load_image(img_path)
            if image is None:
                continue
                
            result = detector.predict(image, threshold=args.threshold)
            
            filename = os.path.basename(img_path)
            score = result['score']
            is_glass = result['is_glass']
            
            # Using colors or just consistent symbols
            status = "GLASS" if is_glass else "No Glass"
            print(f"{filename:<40} | {score:.4f}     | {status}")

if __name__ == "__main__":
    main()
