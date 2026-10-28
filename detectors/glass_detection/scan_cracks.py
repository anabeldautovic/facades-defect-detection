import os
import argparse
from src.utils import get_image_files, load_image
from src.detector_v2 import GlassDetectorV2

# Project data folder, so the defaults work no matter where the script is run from
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "compressed")

def check_path(path: str) -> str:
    if os.path.exists(path): return path
    alt_path = os.path.join("..", path)
    if os.path.exists(alt_path): return alt_path
    return path

def main():
    parser = argparse.ArgumentParser(description="Glass Crack Detector")
    parser.add_argument("--negative_dir", type=str, default=os.path.join(DATA_DIR, "negative"), help="Path to negative images folder")
    parser.add_argument("--positive_dir", type=str, default=os.path.join(DATA_DIR, "positive"), help="Path to positive images folder")
    parser.add_argument("--glass_threshold", type=float, default=0.9, help="Threshold for glass detection")
    parser.add_argument("--crack_threshold", type=float, default=0.5, help="Threshold for crack detection")
    args = parser.parse_args()

    neg_dir = check_path(args.negative_dir)
    pos_dir = check_path(args.positive_dir)

    detector = GlassDetectorV2()
    
    directories = [neg_dir, pos_dir]
    
    print(f"\nScanning for CRACKS...")
    print(f"Glass Threshold: {args.glass_threshold}")
    print(f"Crack Threshold: {args.crack_threshold}\n")
    print("-" * 80)
    print(f"{'File':<40} | {'Glass Score':<12} | {'Crack Score':<12} | {'Result':<10}")
    print("-" * 80)

    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        image_files = get_image_files(directory)
        
        for img_path in image_files:
            image = load_image(img_path)
            if image is None: continue
                
            # Step 1: Detect Glass
            glass_res = detector.predict(image, threshold=args.glass_threshold)
            
            if glass_res['is_glass']:
                # Step 2: Detect Crack (using features from step 1)
                crack_res = detector.detect_crack(glass_res['features'], threshold=args.crack_threshold)
                
                filename = os.path.basename(img_path)
                g_score = glass_res['score']
                c_score = crack_res['score']
                
                status = "CRACKED" if crack_res['is_cracked'] else "Intact"
                
                # Highlight cracks
                if crack_res['is_cracked']:
                     print(f"{filename:<40} | {g_score:.4f}       | {c_score:.4f}       | {status} <---")
                else:
                     print(f"{filename:<40} | {g_score:.4f}       | {c_score:.4f}       | {status}")
            else:
                # Skip non-glass or print simple output
                # print(f"{os.path.basename(img_path):<40} | {glass_res['score']:.4f}       | N/A          | Not Glass")
                pass

if __name__ == "__main__":
    main()
