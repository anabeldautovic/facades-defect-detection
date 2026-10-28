import os
import argparse
from src.utils import get_image_files, load_image
from src.detector import GlassDetector

# Project data folder, so the defaults work no matter where the script is run from
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "compressed")

def main():
    parser = argparse.ArgumentParser(description="Glass Surface Detector")
    parser.add_argument("--negative_dir", type=str, default=os.path.join(DATA_DIR, "negative"), help="Path to negative images folder")
    parser.add_argument("--positive_dir", type=str, default=os.path.join(DATA_DIR, "positive"), help="Path to positive images folder")
    parser.add_argument("--threshold", type=float, default=0.4, help="Probability threshold for glass detection")
    args = parser.parse_args()

    # Initialize detector
    detector = GlassDetector()
    
    directories = [args.negative_dir, args.positive_dir]
    
    print(f"\nScanning directories: {directories}")
    print(f"Detection Threshold: {args.threshold}\n")
    print("-" * 60)
    print(f"{'File':<40} | {'Score':<10} | {'Result':<10}")
    print("-" * 60)

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
            
            # Print only found glass or maybe all? User asked: "only get True for those that have glass"
            # So I will print everything but mark the ones that are True clearly, or maybe just list the True ones as final output.
            # But during execution it's nice to see progress.
            # Let's print all but highlight the ones found.
            
            filename = os.path.basename(img_path)
            score = result['score']
            is_glass = result['is_glass']
            
            status = "GLASS" if is_glass else "No Glass"
            print(f"{filename:<40} | {score:.4f}     | {status}")

if __name__ == "__main__":
    main()
