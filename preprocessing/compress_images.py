import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

# ============================
# CONFIG
# ============================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INPUT_ROOT = DATA_DIR / "raw" / "TRAIN"
OUTPUT_ROOT = DATA_DIR / "compressed"

CLASSES = ["positive", "negative"]

MAX_SIDE = 1200        # Resize longest side (None to disable)
JPEG_QUALITY = 75      # 60–80 is usually a good range

SUPPORTED_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}

# ============================
# IMAGE LOADING
# ============================

def load_image(path: Path) -> np.ndarray:
    ext = path.suffix.lower()

    if ext in {".heic", ".heif"}:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)   # Fix iPhone rotation
        img = img.convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(path))

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    return img

# ============================
# IMAGE COMPRESSION
# ============================

def compress_and_save(img: np.ndarray, out_path: Path):
    h, w = img.shape[:2]

    if MAX_SIDE is not None and max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(out_path),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

# ============================
# MAIN PIPELINE
# ============================

def process_class(class_name: str):
    in_dir = INPUT_ROOT / class_name
    out_dir = OUTPUT_ROOT / class_name

    files = [
        p for p in in_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    print(f"\nProcessing {class_name}: {len(files)} images")

    for i, path in enumerate(files, 1):
        try:
            img = load_image(path)
            out_name = path.stem + ".jpg"
            out_path = out_dir / out_name
            compress_and_save(img, out_path)

            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {out_name}")

        except Exception as e:
            print(f"  ERROR processing {path.name}: {e}")

def main():
    print("Starting dataset compression...")
    for cls in CLASSES:
        process_class(cls)
    print("\nDone!")

if __name__ == "__main__":
    main()
