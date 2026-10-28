# exmpandamo dataset -> rotiranje, proširenje slika.... ugl manpilulacije sa slikama da proširimo dataest

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image
from typing import Optional

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False


@dataclass
class MissingSheetResult:
    image_path: str
    missing_sheet_metal: bool
    dark_ratio: float
    largest_dark_blob_ratio: float
    roi_fraction_used: float


def load_image_any(path: str) -> Optional[np.ndarray]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img
    if ext in [".heic", ".heif"]:
        if not HEIC_OK:
            raise RuntimeError("pillow-heif not available. Install: pip install pillow-heif")
        pil_img = Image.open(str(p)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported extension: {ext}")


def largest_component_stats(mask_255: np.ndarray):
    mask = (mask_255 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n <= 1:
        return 0, (0, 0)

    # najveća ne-background komponenta
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    area = int(stats[idx, cv2.CC_STAT_AREA])
    width = int(stats[idx, cv2.CC_STAT_WIDTH])
    height = int(stats[idx, cv2.CC_STAT_HEIGHT])

    return area, (width, height)



def build_dark_mask(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Dark/black plate detection (robust to lighting):
    - low Value (V) and low Saturation (S) in HSV
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # "Black" ~ low V; also keep S low to avoid colored dark areas
    # Tune if needed:
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([179, 80, 85], dtype=np.uint8)  # S<=80, V<=85
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up
    k = max(3, int(min(bgr.shape[:2]) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def build_dark_panel_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Kao build_dark_mask, ali agresivnije spaja rascjepkane crne dijelove
    u jednu 'panel' komponentu (sive letvice/refleksije će napraviti rupe).
    """
    mask = build_dark_mask(bgr)

    # jači CLOSE da popuni rupe i spoji dijelove
    k = max(5, int(min(bgr.shape[:2]) * 0.03) | 1)  # ~3% min dim
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def build_grey_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Sive/metalne letvice: niska saturacija + srednje/visoka vrijednost.
    Radi dobro u HSV.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # sivo: S mali, V dovoljno velik (ali ne bijelo nužno)
    lower = np.array([0, 0, 90], dtype=np.uint8)
    upper = np.array([179, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = max(3, int(min(bgr.shape[:2]) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_component_props(mask_255: np.ndarray):
    """Returns (area, w, h, rectangularity) for largest component in binary mask."""
    mask = (mask_255 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0, 0, 0, 0.0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    rect = area / max(1, (w * h))
    return area, w, h, float(rect)


def build_orange_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Orange mask tuned for 'tile orange' sheet metal."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Orange range (OpenCV H: 0..179)
    lower = np.array([5, 60, 40], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = max(3, int(min(bgr.shape[:2]) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def largest_component_bbox(mask_255: np.ndarray):
    """
    Returns (area, x, y, w, h) for the largest component in a binary mask.
    """
    mask = (mask_255 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0, 0, 0, 0, 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    return area, x, y, w, h

def check_missing_sheet(
    bgr: np.ndarray,
    use_bottom_fraction: float = 0.60,
    crop_border_frac: float = 0.03,
    exclude_bottom_frac: float = 0.20,
    thresh_dark_ratio: float = 0.10,
    thresh_largest_blob: float = 0.06,
) -> Tuple[MissingSheetResult, np.ndarray, np.ndarray]:

    H, W = bgr.shape[:2]

    # --- focus region: bottom part only ---
    y0 = int(H * (1.0 - use_bottom_fraction))
    region = bgr[y0:H, :, :].copy()

    # --- ignore borders (frame rails) ---
    bh = int(region.shape[0] * crop_border_frac)
    bw = int(region.shape[1] * crop_border_frac)

    y1 = int(region.shape[0] * (1.0 - exclude_bottom_frac))  # odreži dno
    region2 = region[bh:y1, bw:region.shape[1] - bw].copy()

    total = region2.shape[0] * region2.shape[1]

    # ---------- 1) GATE: je li prisutan lim (velika narančasta ploha)? ----------
    mask_orange = build_orange_mask_hsv(region2)

    o_area, o_w, o_h, o_rect = largest_component_props(mask_orange)
    o_ratio = o_area / max(1, total)
    o_aspect = o_w / max(1, o_h)

    ORANGE_BIG = (o_ratio >= 0.06) and (o_rect >= 0.55) and (0.4 <= o_aspect <= 5.0)

    if ORANGE_BIG:
        overlay = region2.copy()
        green = np.zeros_like(region2, dtype=np.uint8)
        green[:] = (0, 255, 0)
        overlay[mask_orange > 0] = cv2.addWeighted(
            overlay[mask_orange > 0], 0.55, green[mask_orange > 0], 0.45, 0
        )

        res = MissingSheetResult(
            image_path="",
            missing_sheet_metal=False,
            dark_ratio=0.0,
            largest_dark_blob_ratio=0.0,
            roi_fraction_used=float(use_bottom_fraction),
        )
        return res, mask_orange, overlay

    # ---------- 2) DARK signal (tvoj stari način) ----------
    mask_dark = build_dark_mask(region2)

    dark_pixels = int((mask_dark > 0).sum())
    dark_ratio = dark_pixels / max(1, total)

    largest_area, (bw2, bh2) = largest_component_stats(mask_dark)
    largest_ratio = largest_area / max(1, total)

    d_area, d_w, d_h, d_rect = largest_component_props(mask_dark)
    d_ratio = d_area / max(1, total)
    d_aspect = d_w / max(1, d_h)

    # Stari kriterij (radi, ali promaši kad se crna ploha "raspadne" zbog letvice/refleksije)
    DARK_BIG_RECT = (d_ratio >= 0.08) and (d_rect >= 0.55) and (0.4 <= d_aspect <= 5.0)

    # ---------- 2B) NOVI kriterij: "kvadratast" crni panel + siva/srebrna linija unutra ----------
    # Panel maska spaja rascjepkane crne dijelove u jednu komponentu
    mask_panel = build_dark_panel_mask(region2)
    p_area, px, py, pw, ph = largest_component_bbox(mask_panel)

    p_ratio = p_area / max(1, total)
    p_aspect = pw / max(1, ph)
    # rectangularity na panel maski (da bude "ploha", a ne šum)
    p_area2, p_w2, p_h2, p_rect = largest_component_props(mask_panel)

    # "bliže kvadratu" (slobodno kasnije stisni/raširi)
    SQUAREISH = (0.60 <= p_aspect <= 1.80)

    # mora biti dovoljno velika i relativno pravokutna ploha
    PANEL_BIG = (p_ratio >= 0.06) and (p_rect >= 0.40)

    # siva linija unutar panela:
    #  - uzmi grey mask
    #  - gledaj samo unutar bbox najvećeg panela
    grey = build_grey_mask(region2)

    # sigurnost granica
    px2 = max(0, px); py2 = max(0, py)
    pw2 = max(0, min(region2.shape[1] - px2, pw))
    ph2 = max(0, min(region2.shape[0] - py2, ph))

    grey_roi = grey[py2:py2 + ph2, px2:px2 + pw2]
    panel_roi = (mask_panel[py2:py2 + ph2, px2:px2 + pw2] > 0).astype(np.uint8)

    # sivo "unutar panela" (da ignoriramo pozadinu)
    grey_in_panel = ((grey_roi > 0) & (panel_roi > 0))
    grey_inside = int(grey_in_panel.sum())
    panel_pixels = int(panel_roi.sum())
    grey_ratio_inside = grey_inside / max(1, panel_pixels)

    # dodatno: da to izgleda kao LINIJa (ne samo šum) -> tražimo jak red ili stupac
    # projekcije sive po redovima i stupcima
    row_sums = grey_in_panel.sum(axis=1) if ph2 > 0 and pw2 > 0 else np.array([0])
    col_sums = grey_in_panel.sum(axis=0) if ph2 > 0 and pw2 > 0 else np.array([0])
    max_row = int(row_sums.max()) if row_sums.size else 0
    max_col = int(col_sums.max()) if col_sums.size else 0

    # linija "značajna" ako pokriva bar ~20% širine ili visine ROI
    has_grey_line = (max_row >= int(0.20 * pw2)) or (max_col >= int(0.20 * ph2))

    # final novi kandidat
    DARK_SQUARE_PANEL_WITH_LINE = PANEL_BIG and SQUAREISH and (grey_ratio_inside >= 0.003) and has_grey_line

    # ---------- 3) Ako nema ni velike narančaste ni "dovoljno uvjerljive" crne plohe -> preskoči ----------
    # (ovdje čuvamo tvoju anti-false-positive logiku)
    if not (DARK_BIG_RECT or DARK_SQUARE_PANEL_WITH_LINE):
        overlay = region2.copy()
        res = MissingSheetResult(
            image_path="",
            missing_sheet_metal=False,
            dark_ratio=float(dark_ratio),
            largest_dark_blob_ratio=float(largest_ratio),
            roi_fraction_used=float(use_bottom_fraction),
        )
        return res, mask_dark, overlay

    # ---------- 4) Velika crna ploha -> DEFECT (tvoj stari threshold signal ostaje) ----------
    missing = (dark_ratio >= thresh_dark_ratio) and (largest_ratio >= thresh_largest_blob)

    # Debug overlay:
    overlay = region2.copy()
    red = np.zeros_like(region2, dtype=np.uint8); red[:] = (0, 0, 255)
    overlay[mask_dark > 0] = cv2.addWeighted(overlay[mask_dark > 0], 0.55, red[mask_dark > 0], 0.45, 0)

    # ako je novi kriterij aktivan, označi bbox panela i grey liniju (za lakši tuning)
    if DARK_SQUARE_PANEL_WITH_LINE and pw2 > 0 and ph2 > 0:
        cv2.rectangle(overlay, (px2, py2), (px2 + pw2, py2 + ph2), (0, 255, 255), 2)
        # oboji grey liniju unutar panela u plavo
        blue = np.zeros_like(region2, dtype=np.uint8); blue[:] = (255, 0, 0)
        # maska grey_in_panel u koordinatama region2
        tmp = np.zeros((region2.shape[0], region2.shape[1]), dtype=np.uint8)
        tmp[py2:py2 + ph2, px2:px2 + pw2] = grey_in_panel.astype(np.uint8) * 255
        overlay[tmp > 0] = cv2.addWeighted(overlay[tmp > 0], 0.55, blue[tmp > 0], 0.45, 0)

    res = MissingSheetResult(
        image_path="",
        missing_sheet_metal=bool(missing),
        dark_ratio=float(dark_ratio),
        largest_dark_blob_ratio=float(largest_ratio),
        roi_fraction_used=float(use_bottom_fraction),
    )
    return res, mask_dark, overlay



def iter_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    # npr. data/raw/lim_data/positive ili data/raw/test
    data_dir = Path(__file__).resolve().parent.parent / "data"
    ap.add_argument("--input_dir", type=str, default=str(data_dir / "raw" / "lim_data" / "positive"))
    args = ap.parse_args()

    input_dir = Path(args.input_dir)

    imgs = iter_images(input_dir)
    if not imgs:
        raise RuntimeError(f"No images found in {input_dir}")

    for p in imgs:
        bgr = load_image_any(str(p))
        res, mask, overlay = check_missing_sheet(bgr)

        # ispis samo za FAIL (missing_sheet == True)
        if res.missing_sheet_metal:
            print(f"Slika: {p.name}")
            print("Status: FAIL")
            print("Detektirani defekti:")
            print("nedostaje lim")
            print()  # prazna linija između rezultata


if __name__ == "__main__":
    main()
