# =====================================================================
# ACNE DETECTION PIPELINE
# ---------------------------------------------------------------------
# Single-file, configurable pipeline merging the two experimental
# notebook cells (heatmap masking + object scoring + zone analysis).
#
# Path resolution:
#   ROOT is derived from this file's location (script located anywhere
#   under ROOT works). If the image is not found, all candidate paths
#   are printed so the problem is obvious.
#
# Outputs per run are saved to:
#   ROOT/Models/cv/runs/run_<timestamp>/   (result.jpg, mask.jpg,
#                                           enhanced.jpg)
#
# Main control switches (see PipelineConfig / __main__ section):
#   base_mask_mode : "percentile" -> adaptive percentile threshold of
#                                     the heatmap (notebook v2)
#                    "fixed"      -> fixed red threshold (notebook v1)
#   annotation_mode: "boxes"     -> scored bounding boxes (v2 style)
#                    "contours"  -> precise lesion outlines (v1 style)
#                    "zones"     -> intensity-zone overlay + stats (v1)
#   use_enhancement: apply CLAHE + bilateral pre-processing branch
#
# Shared project modules are used when available, with built-in
# fallbacks so the script also runs standalone:
#   utils.create_heatmap          -> create_acne_heatmap (local)
#   src.image_preprocess.enhance_details -> enhance_details (local)
#
# All tunable thresholds are exposed in PipelineConfig.
# =====================================================================

import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =====================================================================
# 1. PATH RESOLUTION
# =====================================================================

def find_project_root() -> Path:
    """Walk upwards from this file until a plausible project root is
    found. Accepts any directory containing 'Models' or 'images'.
    Falls back to three levels up, then to the current working dir.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Models").is_dir() or (parent / "images").is_dir():
            return parent
    # Fallback: same assumption as parents[3] in the original script.
    return here.parents[3] if len(here.parents) > 3 else Path.cwd()


ROOT = find_project_root()
print(f"[init] Project root resolved to: {ROOT}")
# Primary input image (can be overridden in PipelineConfig).
DEFAULT_IMAGE = ROOT / "images" / "levle1_271.jpg"

# Per-run output directory: Models/cv/runs/run_YYYYmmdd_HHMMSS
RUN_DIR = (ROOT / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = RUN_DIR / "result.jpg"
OUTPUT_MASK = RUN_DIR / "mask.jpg"
OUTPUT_ENHANCED = RUN_DIR / "enhanced.jpg"

# Project module search paths (shared helpers, YOLO models, src pkg).
SHARED_DIR = ROOT / "Models" / "shared"
SRC_DIR = ROOT / "Models" / "shared" / "src"
YOLO_DIR = ROOT / "Models" / "yolo"
for _p in (SHARED_DIR, SRC_DIR, YOLO_DIR):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))


def load_image(path: Path):
    """Load the input image with a helpful diagnostic on failure."""
    if not path.is_file():
        candidates = [
            DEFAULT_IMAGE,
            ROOT / "Models" / "shared" / "images" / path.name,
            ROOT / "images" / path.name,
            Path.cwd() / path.name,
        ]
        msg = [f"Image not found: {path}\nTried / existing candidates:"]
        for c in candidates:
            tag = "OK " if c.is_file() else "-- "
            msg.append(f"  [{tag}] {c}")
        # Show what actually exists next to the expected location.
        img_dir = path.parent
        if img_dir.is_dir():
            msg.append(f"'{img_dir}' contains: "
                       + ", ".join(sorted(p.name for p in img_dir.iterdir())))
        else:
            msg.append(f"(directory does not exist: {img_dir})")
        raise FileNotFoundError("\n".join(msg))
    return cv2.imread(str(path))


# Optional shared modules (used when present, local fallbacks otherwise).
try:
    from utils import create_heatmap as shared_create_heatmap  # type: ignore
except ImportError:
    shared_create_heatmap = None

try:
    from src.image_preprocess import enhance_details as shared_enhance_details  # type: ignore
except ImportError:
    shared_enhance_details = None


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

@dataclass
class PipelineConfig:
    """Central configuration for the whole detection pipeline."""

    # --- input / model paths -----------------------------------------
    image_path: Path = DEFAULT_IMAGE
    model_path: Path = ROOT / "face_landmarker.task"

    # --- face cropping (MediaPipe) ------------------------------------
    top_head_expansion: float = 0.30   # extra space above the forehead
    chin_padding: float = 0.10         # extra space below the chin
    side_padding: float = 0.08         # extra space on left/right
    target_aspect_ratio: float = 0.75  # width/height of the final crop

    # --- skin colour segmentation (HSV + YCrCb union) ------------------
    skin_h_min: int = 0
    skin_h_max: int = 30
    skin_s_min: int = 15
    skin_cr_min: int = 130
    skin_cr_max: int = 190
    skin_cb_min: int = 70
    skin_cb_max: int = 140

    # --- adaptive scaling reference ------------------------------------
    ref_diag: float = 1414.0  # diagonal of the ~1000x1000 reference frame

    # --- detail enhancement branch (CLAHE) ------------------------------
    use_enhancement: bool = True
    clahe_clip_l: float = 1.2
    clahe_clip_a: float = 1.1
    clahe_grid: tuple = (8, 8)
    clahe_median_blur: int = 5

    # --- base mask construction ----------------------------------------
    base_mask_mode: str = "percentile"    # "percentile" | "fixed"
    base_mask_percentile: float = 85.0
    fixed_red_threshold: int = 150
    sobel_contrast_threshold: int = 35
    yellow_relative_threshold: float = 0.70

    # --- object scoring -------------------------------------------------
    ideal_radius_frac: float = 0.0035
    max_area_frac: float = 0.0015
    min_area_frac: float = 0.000001
    strong_red_intensity: float = 0.65
    strong_red_min_conf: float = 0.85
    w_intensity: float = 0.70
    w_circularity: float = 0.20
    w_size: float = 0.10

    # --- visualization ---------------------------------------------------
    annotation_mode: str = "zones"   # "boxes" | "contours" | "zones"
    min_confidence_draw: float = 0.15
    box_padding_factor: float = 1.2
    red_zone_threshold: int = 150
    orange_zone_threshold: int = 120
    yellow_zone_threshold: int = 90
    blue_dominance_limit: float = 80.0
    yellow_dominance_limit: float = 50.0
    orange_dominance_limit: float = 40.0

    # --- debugging / saving ----------------------------------------------
    show_intermediate: bool = True
    verbose: bool = True
    save_outputs: bool = True  # write result/mask/enhanced to RUN_DIR


# =====================================================================
# 3. UTILITIES
# =====================================================================

def show_image(img, title="Image", skin_mask=None):
    """Display an image with matplotlib, optionally masked to the skin."""
    if img is None:
        print(f"[viz] '{title}' skipped (None)")
        return
    if skin_mask is not None:
        img = cv2.bitwise_and(img, img, mask=skin_mask)
    plt.figure(figsize=(3.5, 3.5), dpi=300)
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=9, fontweight="light")
    plt.axis("off")
    plt.show()


def image_diagonal(img):
    """Diagonal length of the image, used for size-relative scaling."""
    h, w = img.shape[:2]
    return float(np.hypot(h, w))


def oddify(v):
    """Clamp to >= 3 and force an odd value (OpenCV kernel sizes)."""
    return max(3, int(v) | 1)


# =====================================================================
# 4. FACE LOCALIZATION AND CROPPING (MediaPipe)
# =====================================================================

# Official FaceMesh topology: landmark indices forming the face oval.
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Polygonal indices of key facial regions (visualization only).
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
        318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
        269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157,
            158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388,
             387, 386, 385, 384, 398]


def create_landmarker(model_path: Path) -> vision.FaceLandmarker:
    """Build a MediaPipe FaceLandmarker detecting a single face."""
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def crop_face_with_mediapipe(image, model_path, cfg):
    """Locate the face and return a padded, aspect-normalized crop."""
    h, w, _ = image.shape
    if not model_path.exists():
        print(f"[crop] WARNING: model missing at {model_path}, "
              "using the full image.")
        return image, image.copy()

    detector = create_landmarker(model_path)
    # MediaPipe expects RGB; OpenCV loads BGR, hence the conversion.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        print("[crop] No face detected by MediaPipe. Using full image.")
        return image, image.copy()

    landmarks = detection_result.face_landmarks[0]

    # --- visualization: face oval, eye/lip contours, key points -------
    viz = image.copy()
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    n = len(FACE_OVAL_INDICES)
    for i in range(n):
        cv2.line(viz, pts[FACE_OVAL_INDICES[i]],
                 pts[FACE_OVAL_INDICES[(i + 1) % n]], (0, 255, 255), 2)

    def draw_poly(indices, color):
        poly = np.array([pts[i] for i in indices], dtype=np.int32)
        cv2.polylines(viz, [poly], True, color, 1, cv2.LINE_AA)

    draw_poly(LIPS, (0, 0, 255))
    draw_poly(LEFT_EYE, (255, 255, 0))
    draw_poly(RIGHT_EYE, (255, 255, 0))

    for idx in [1, 10, 152, 234, 454]:
        cv2.circle(viz, pts[idx], 6, (255, 0, 0), -1)
        cv2.putText(viz, str(idx), (pts[idx][0] + 8, pts[idx][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)

    # --- compute the crop rectangle ------------------------------------
    # Bounding box of all landmarks plus directional padding: more room
    # on top, less below the chin.
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    face_w, face_h = x_max - x_min, y_max - y_min

    cx1 = x_min - face_w * cfg.side_padding
    cx2 = x_max + face_w * cfg.side_padding
    cy1 = y_min - face_h * cfg.top_head_expansion
    cy2 = y_max + face_h * cfg.chin_padding

    # Enforce the target aspect ratio by growing the narrower dimension
    # (extra height split 70/30 towards the top so the face stays centred).
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw / ch < cfg.target_aspect_ratio:
        diff = ch * cfg.target_aspect_ratio - cw
        cx1 -= diff / 2
        cx2 += diff / 2
    else:
        diff = cw / cfg.target_aspect_ratio - ch
        cy1 -= diff * 0.7
        cy2 += diff * 0.3

    x1 = int(max(0, cx1)); x2 = int(min(w, cx2))
    y1 = int(max(0, cy1)); y2 = int(min(h, cy2))
    return image[y1:y2, x1:x2], viz


# =====================================================================
# 5. SKIN SEGMENTATION
# =====================================================================

def create_skin_mask(image, cfg):
    """Union of HSV and YCrCb chroma-range masks, cleaned morphologically.

    The union (OR) of two colour models is deliberately permissive:
    recall matters more than precision, because the redness analysis is
    relative (median subtraction) and stray false-positive pixels are
    mostly harmless.
    """
    diag = image_diagonal(image)
    k_close = oddify(diag * 0.003)   # close gaps between skin patches
    k_open = oddify(diag * 0.0015)   # remove speckle noise

    # Model 1: HSV hue/saturation/value ranges typical of skin.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([cfg.skin_h_min, cfg.skin_s_min, 40]),   # V floor = 40
        np.array([cfg.skin_h_max, 255, 255]),
    )

    # Model 2: Cr/Cb chroma plane of YCrCb (robust to lighting changes).
    _, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
    cr_mask = cv2.inRange(cr, cfg.skin_cr_min, cfg.skin_cr_max)
    cb_mask = cv2.inRange(cb, cfg.skin_cb_min, cfg.skin_cb_max)
    ycrcb_mask = cv2.bitwise_and(cr_mask, cb_mask)

    skin_mask = cv2.bitwise_or(hsv_mask, ycrcb_mask)

    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8),
        iterations=2)
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8),
        iterations=1)
    return skin_mask


# =====================================================================
# 6. ACNE HEATMAP (relative redness map in CIELAB)
# =====================================================================

def create_acne_heatmap(image, skin_mask, cfg):
    """Local fallback for the shared `create_heatmap` utility.

    Lesion redness is measured RELATIVE to the median skin tone of the
    photo (compensating for skin tone / white-balance differences):
      1. take the CIELAB `a` channel (green(-) .. red(+)),
      2. bilateral-filter it -> denoise, keep lesion edges,
      3. subtract median skin `a` value; clip negatives at 0,
      4. clip outliers at the 98th percentile, renormalize to 0..255,
      5. Gaussian smooth + JET colormap; zero outside the skin mask.
    """
    diag = image_diagonal(image)
    scale = diag / cfg.ref_diag
    bilateral_d = oddify(9 * scale)
    blur_ksize = oddify(15 * scale)

    _, a_channel, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    a_denoised = cv2.bilateralFilter(a_channel, d=bilateral_d,
                                     sigmaColor=75, sigmaSpace=75)

    skin_pixels = a_denoised[skin_mask > 0]
    if len(skin_pixels) == 0:
        print("[heatmap] Empty skin mask - returning black heatmap.")
        return np.zeros_like(image)

    median_skin_redness = np.median(skin_pixels)
    a_diff = np.maximum(0, a_denoised.astype(np.float32) - median_skin_redness)
    a_diff[skin_mask == 0] = 0

    non_zero = a_diff[a_diff > 0]
    if len(non_zero) == 0:
        return np.zeros_like(image)

    p98 = np.percentile(non_zero, 98)
    a_clipped = np.clip(a_diff, 0, p98)
    redness_norm = cv2.normalize(a_clipped, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    redness_smooth = cv2.GaussianBlur(redness_norm, (blur_ksize, blur_ksize), 0)
    heatmap = cv2.applyColorMap(redness_smooth, cv2.COLORMAP_JET)
    heatmap[skin_mask == 0] = 0
    return heatmap


def get_heatmap_fn(cfg):
    """Prefer the shared project implementation, fall back to local."""
    if shared_create_heatmap is not None:
        print("[heatmap] Using shared utils.create_heatmap")
        return shared_create_heatmap
    print("[heatmap] Shared utils not found - using local implementation")
    return lambda img, mask: create_acne_heatmap(img, mask, cfg)


# =====================================================================
# 7. DETAIL ENHANCEMENT BRANCH (optional)
# =====================================================================

def enhance_details_local(img, cfg):
    """Local fallback for shared enhance_details (CLAHE variant)."""
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    l, a, b = cv2.split(cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB))

    clahe_l = cv2.createCLAHE(clipLimit=cfg.clahe_clip_l,
                              tileGridSize=cfg.clahe_grid)
    clahe_a = cv2.createCLAHE(clipLimit=cfg.clahe_clip_a,
                              tileGridSize=cfg.clahe_grid)
    l_enhanced = clahe_l.apply(l)
    a_enhanced = clahe_a.apply(a)

    a_enhanced = cv2.medianBlur(a_enhanced, cfg.clahe_median_blur)
    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a_enhanced, b)),
                            cv2.COLOR_LAB2BGR)
    return enhanced, None


def get_enhance_fn(cfg):
    """Prefer the shared project implementation, fall back to local."""
    if shared_enhance_details is not None:
        print("[enhance] Using shared src.image_preprocess.enhance_details")
        return shared_enhance_details
    print("[enhance] Shared module not found - using local implementation")
    return lambda img: enhance_details_local(img, cfg)


# =====================================================================
# 8. BASE MASK FROM HEATMAP (two selectable strategies)
# =====================================================================

def create_base_mask_from_heatmap(heatmap, skin_mask, cfg):
    """Binarize the heatmap into a candidate-lesion mask.

    "fixed" (v1):  constant absolute threshold on the grey heatmap
                   (>= cfg.fixed_red_threshold).
    "percentile" (v2): threshold = percentile of skin-pixel brightness,
                   adapting to per-photo intensity spread.

    The weaker "yellow" band enters only when supported by a strong
    Sobel gradient (sharp colour boundary). Strong lesions are filled
    morphologically and bypass size limits; only plausible-size weaker
    components survive the final filtering.
    """
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    skin_pixels = gray[skin_mask > 0]
    if len(skin_pixels) == 0:
        return np.zeros_like(gray)

    # ---- 1. threshold for the strong (red) core -----------------------
    if cfg.base_mask_mode == "percentile":
        base_thr = float(np.percentile(skin_pixels, cfg.base_mask_percentile))
    else:  # "fixed"
        base_thr = float(cfg.fixed_red_threshold)

    _, strong_mask = cv2.threshold(gray, base_thr, 255, cv2.THRESH_BINARY)

    # Fill interior holes of strong lesions (kernel scaled by diagonal).
    diag = image_diagonal(gray)
    k_close = oddify(diag * 0.002)
    strong_mask = cv2.morphologyEx(
        strong_mask, cv2.MORPH_CLOSE,
        np.ones((k_close, k_close), np.uint8), iterations=2)

    # ---- 2. gradient-supported weaker (yellow) regions -----------------
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)   # d/dx
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)   # d/dy
    grad_norm = cv2.normalize(cv2.magnitude(sobel_x, sobel_y),
                              None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, contrast_mask = cv2.threshold(grad_norm,
                                     cfg.sobel_contrast_threshold,
                                     255, cv2.THRESH_BINARY)
    _, yellow_potential = cv2.threshold(
        gray, base_thr * cfg.yellow_relative_threshold, 255,
        cv2.THRESH_BINARY)
    yellow_supported = cv2.bitwise_and(yellow_potential, contrast_mask)

    # ---- 3. combine and restrict to skin -------------------------------
    base_mask = cv2.bitwise_or(strong_mask, yellow_supported)
    base_mask = cv2.bitwise_and(base_mask, skin_mask)

    # ---- 4. coarse size sanity filtering -------------------------------
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = base_mask.shape[:2]
    total_area = h * w
    max_area = total_area * cfg.max_area_frac
    min_area = max(2, int(total_area * cfg.min_area_frac))

    filtered = np.zeros_like(base_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:
            continue  # pure pixel noise

        probe = np.zeros(base_mask.shape, dtype=np.uint8)
        cv2.drawContours(probe, [cnt], -1, 255, -1)
        max_val = np.max(gray[probe > 0])

        if max_val >= base_thr:
            # Strong red focus: always keep, no size limits.
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
        elif min_area <= area <= max_area:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    return filtered


# =====================================================================
# 9. OBJECT EXTRACTION AND CONFIDENCE SCORING
# =====================================================================

def extract_and_score_objects(mask, heatmap, cfg):
    """Convert mask contours into scored candidate objects.

    Confidence = w_i * intensity^2 + w_c * circularity + w_s * size-fit.
    Strong red candidates (raw intensity >= 0.65) receive a confidence
    floor and are exempt from the minimum-area requirement.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    total_pixels = h * w
    diag = image_diagonal(mask)
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    ideal_radius = diag * cfg.ideal_radius_frac

    objects = []
    for obj_id, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area <= 0 or perimeter == 0:
            continue

        (_, _), radius = cv2.minEnclosingCircle(cnt)

        probe = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(probe, [cnt], -1, 255, -1)
        pixels = heatmap_gray[probe > 0]
        if len(pixels) == 0:
            continue

        raw_intensity = min(1.0, np.max(pixels) / 255.0)
        intensity_score = raw_intensity ** 2.0
        is_strong_red = raw_intensity >= cfg.strong_red_intensity

        # Weak microscopic object -> noise, discard (strong red exempt).
        if not is_strong_red and area < max(2, int(total_pixels
                                                 * cfg.min_area_frac)):
            continue
        # Pathologically large weak object -> not a lesion, discard.
        if not is_strong_red and area > total_pixels * cfg.max_area_frac:
            continue

        # Circularity: 1.0 for a perfect circle (standard shape metric).
        circularity = min(1.0, max(0.0, (4 * np.pi * area)
                                   / (perimeter ** 2)))
        # Size fit: linear decay to 0 at twice the reference deviation.
        size_score = max(0.0, 1.0 - (abs(radius - ideal_radius)
                                     / (ideal_radius * 2.0)))

        confidence = (cfg.w_intensity * intensity_score
                      + cfg.w_circularity * circularity
                      + cfg.w_size * size_score)
        if is_strong_red:
            confidence = max(cfg.strong_red_min_conf, confidence)

        x, y, bw, bh = cv2.boundingRect(cnt)
        objects.append({
            "id": obj_id,
            "contour": cnt,
            "confidence": float(confidence),
            "raw_intensity": float(raw_intensity),
            "intensity_score": float(intensity_score),
            "circularity_score": float(circularity),
            "size_score": float(size_score),
            "box": (x, y, bw, bh),
        })
    return objects


def print_debug_table(objects, title, verbose=True):
    """Pretty-print a pandas table of all scored detections."""
    if not verbose:
        return
    if not objects:
        print(f"\n--- {title} ---\nNo objects detected.")
        return
    df = pd.DataFrame([{
        "ID": o["id"],
        "Confidence": round(o["confidence"], 3),
        "Raw Intensity": round(o["raw_intensity"], 3),
        "Intensity Score": round(o["intensity_score"], 3),
        "Circularity": round(o["circularity_score"], 3),
        "Size Score": round(o["size_score"], 3),
    } for o in objects])
    print(f"\n--- {title} ---")
    print(df.to_string(index=False))


# =====================================================================
# 10. ANNOTATION BACK-ENDS
# =====================================================================

def draw_scored_boxes(image, objects, cfg):
    """Annotation mode 'boxes': padded squares with ID + confidence."""
    result = image.copy()
    h, w = image.shape[:2]
    diag = image_diagonal(image)
    thickness = max(1, int(diag * 0.0008))
    count = 0

    for obj in objects:
        if obj["confidence"] < cfg.min_confidence_draw:
            continue
        x, y, bw, bh = obj["box"]
        conf = obj["confidence"]

        # Dynamic padding: tiny boxes get relatively more margin.
        object_size = max(5, max(bw, bh))
        pad = max(1.1, min(2.5, cfg.box_padding_factor
                           + (10.0 / object_size)))
        half = int(object_size * pad / 2)
        cx, cy = x + bw // 2, y + bh // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)

        # BGR colour encodes confidence: green (high) -> red (low).
        color = (0, int(255 * (1 - conf)), int(255 * conf))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness,
                      cv2.LINE_AA)
        cv2.putText(result, f"ID:{obj['id']} ({conf:.2f})",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                    1, cv2.LINE_AA)
        count += 1
    return result, count


def draw_lesion_contours(image, objects, cfg):
    """Annotation mode 'contours': precise outlines instead of boxes."""
    result = image.copy()
    diag = image_diagonal(image)
    thickness = max(1, int(diag * 0.001))

    for obj in objects:
        if obj["confidence"] < cfg.min_confidence_draw:
            continue
        conf = obj["confidence"]
        color = (0, int(255 * (1 - conf)), int(255 * conf))
        cv2.drawContours(result, [obj["contour"]], -1, color, thickness,
                         cv2.LINE_AA)

        M = cv2.moments(obj["contour"])
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(result, f"#{obj['id']}", (cX - 10, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
                        1, cv2.LINE_AA)
    return result, sum(1 for o in objects
                       if o["confidence"] >= cfg.min_confidence_draw)


def show_zone_colorbar(cfg):
    """Render the annotated JET colour bar explaining the zone system."""
    gradient = np.tile(np.linspace(0, 255, 500, dtype=np.uint8), (40, 1))
    jet_rgb = cv2.cvtColor(cv2.applyColorMap(gradient, cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)

    r, o, y = (cfg.red_zone_threshold, cfg.orange_zone_threshold,
               cfg.yellow_zone_threshold)
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)
    ax.imshow(jet_rgb, aspect="auto", extent=[0, 255, 0, 1])
    ax.set_title("Intensity zone scale (global thresholds)",
                 fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Heat-map pixel value (0 - 255)", fontsize=9)

    for xpos, ypos, label, txt_col, box_col in [
        (r, 0.8, f"Red zone (>= {r})", "white", "darkred"),
        (o, 0.5, f"Orange zone ({o}-{r})", "black", "orange"),
        (y, 0.2, f"Yellow zone ({y}-{o})", "black", "yellow"),
    ]:
        ax.axvline(x=xpos, color="white", linestyle="--", linewidth=1.5)
        ax.text(xpos + 2, ypos, label, color=txt_col, fontsize=7,
                fontweight="bold",
                bbox=dict(facecolor=box_col, alpha=0.7))
    plt.tight_layout()
    plt.show()


def analyze_zones_and_draw(image, heatmap, skin_mask, cfg):
    """Annotation mode 'zones': whole-skin intensity zone segmentation."""
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    skin_pixels = gray[skin_mask > 0]
    total = len(skin_pixels)
    if total == 0:
        return image.copy(), {}
    pct = lambda n: n / total * 100.0  # noqa: E731

    red_pct = pct(np.sum(skin_pixels >= cfg.red_zone_threshold))
    orange_pct = pct(np.sum((skin_pixels >= cfg.orange_zone_threshold)
                            & (skin_pixels < cfg.red_zone_threshold)))
    yellow_pct = pct(np.sum((skin_pixels >= cfg.yellow_zone_threshold)
                            & (skin_pixels < cfg.orange_zone_threshold)))
    blue_pct = pct(np.sum((skin_pixels < cfg.yellow_zone_threshold)
                          & (skin_pixels > 0)))

    # Dominance guard: when neutral background dominates, weak zones
    # are artifacts of a globally brighter map -> suppress drawing.
    draw_yellow = yellow_pct < cfg.yellow_dominance_limit
    draw_orange = orange_pct < cfg.orange_dominance_limit

    h, w = image.shape[:2]
    diag = image_diagonal(image)
    thickness = max(1, int(diag * 0.0008))

    zones = {
        "Czerwona (Zapalna)": {
            "mask": cv2.bitwise_and(
                (gray >= cfg.red_zone_threshold).astype(np.uint8) * 255,
                skin_mask),
            "color": (0, 0, 255), "conf": 0.95, "pct": red_pct,
            "active": True,  # red zone is ALWAYS outlined
        },
        "Pomarańczowa": {
            "mask": cv2.bitwise_and(
                ((gray >= cfg.orange_zone_threshold)
                 & (gray < cfg.red_zone_threshold)).astype(np.uint8) * 255,
                skin_mask),
            "color": (0, 165, 255), "conf": 0.70, "pct": orange_pct,
            "active": draw_orange,
        },
        "Żółta": {
            "mask": cv2.bitwise_and(
                ((gray >= cfg.yellow_zone_threshold)
                 & (gray < cfg.orange_zone_threshold)).astype(np.uint8) * 255,
                skin_mask),
            "color": (0, 255, 255),
            "conf": 0.35 if draw_yellow else 0.0, "pct": yellow_pct,
            "active": draw_yellow,
        },
        "Niebieska (Neutralna)": {
            "mask": cv2.bitwise_and(
                ((gray < cfg.yellow_zone_threshold)
                 & (gray > 0)).astype(np.uint8) * 255,
                skin_mask),
            "color": (255, 0, 0), "conf": 0.0, "pct": blue_pct,
            "active": False,  # neutral background: never drawn
        },
    }

    result = image.copy()
    for name, data in zones.items():
        if not data["active"] or data["conf"] == 0.0:
            continue
        contours, _ = cv2.findContours(data["mask"], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 2:
                cv2.drawContours(result, [cnt], -1, data["color"],
                                 thickness, cv2.LINE_AA)
    return result, zones


def print_zone_report(zones, verbose=True):
    """Print the legend/percentage table for zone analysis."""
    if not verbose or not zones:
        return
    print("\n--- ZONE REPORT ---")
    for name, data in zones.items():
        status = ("Active (outlined)"
                  if data["active"] and data["conf"] > 0
                  else "Neutral / suppressed (conf: 0)")
        print(f"Zone: {name:<24} | Share: {data['pct']:5.2f}% "
              f"| Confidence: {data['conf']:<4} | {status}")


# =====================================================================
# 11. MAIN PIPELINE
# =====================================================================

def run_pipeline(cfg: PipelineConfig):
    """Execute the full detection pipeline according to the config."""
    # ---- 0. load -------------------------------------------------------
    image = load_image(cfg.image_path)

    # ---- 1. face detection and cropping --------------------------------
    cropped, crop_viz = crop_face_with_mediapipe(image, cfg.model_path, cfg)
    if cfg.show_intermediate:
        show_image(crop_viz, "MediaPipe Landmarks")
        show_image(cropped, "Cropped Face")

    # ---- 2. skin segmentation -------------------------------------------
    skin_mask = create_skin_mask(cropped, cfg)
    if cfg.show_intermediate:
        show_image(skin_mask, "Skin Mask")

    # ---- 3. optional enhancement branch ----------------------------------
    if cfg.use_enhancement:
        enh_fn = get_enhance_fn(cfg)
        enhanced, _ = enh_fn(cropped)
        process_source = enhanced
        skin_for_heat = create_skin_mask(enhanced, cfg)
        if cfg.show_intermediate:
            show_image(enhanced, "Enhanced Image")
            show_image(skin_for_heat, "Skin Mask (Enhanced)")
    else:
        process_source, skin_for_heat = cropped, skin_mask

    # ---- 4. heatmap -------------------------------------------------------
    heat_fn = get_heatmap_fn(cfg)
    heatmap = heat_fn(process_source, skin_for_heat)
    if cfg.show_intermediate:
        show_image(heatmap, f"Heatmap ({cfg.base_mask_mode} mask mode)")

    # ---- 5. base mask binaryzation ---------------------------------------
    base_mask = create_base_mask_from_heatmap(heatmap, skin_for_heat, cfg)
    if cfg.show_intermediate:
        show_image(base_mask, "Base Mask", skin_mask=skin_for_heat)

    # ---- 6. object extraction + scoring ------------------------------------
    objects = extract_and_score_objects(base_mask, heatmap, cfg)
    print_debug_table(objects, f"Detections - {cfg.base_mask_mode} mode",
                      cfg.verbose)

    # ---- 7. annotation ------------------------------------------------------
    zones = None
    if cfg.annotation_mode == "boxes":
        final, count = draw_scored_boxes(cropped, objects, cfg)
        print(f"Detected lesions (boxes): {count}")
        show_image(final, "Final detections (boxes)")
    elif cfg.annotation_mode == "contours":
        final, count = draw_lesion_contours(cropped, objects, cfg)
        print(f"Detected lesions (contours): {count}")
        show_image(final, "Final detections (contours)")
    elif cfg.annotation_mode == "zones":
        show_zone_colorbar(cfg)
        final, zones = analyze_zones_and_draw(cropped, heatmap,
                                              skin_for_heat, cfg)
        show_image(final, "Zone analysis overlay")
        print_zone_report(zones, cfg.verbose)
    else:
        raise ValueError(f"Unknown annotation_mode: {cfg.annotation_mode}")

    # ---- 8. persist run outputs -------------------------------------------
    if cfg.save_outputs:
        cv2.imwrite(str(OUTPUT_IMAGE), final)
        cv2.imwrite(str(OUTPUT_MASK), base_mask)
        if cfg.use_enhancement and process_source is not cropped:
            cv2.imwrite(str(OUTPUT_ENHANCED), process_source)
        print(f"[save] Outputs written to: {RUN_DIR}")

    return {"objects": objects, "zones": zones, "heatmap": heatmap,
            "base_mask": base_mask}


# =====================================================================
# 12. ENTRY POINT - experiment matrix
# =====================================================================

if __name__ == "__main__":
    
    # Override the input image here if needed (e.g. sys.argv[1]):
    image_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_IMAGE)

    # Sweep the four main combinations (mask mode x enhancement)?
    RUN_EXPERIMENT_MATRIX = False

    if RUN_EXPERIMENT_MATRIX:
        for mode in ("percentile", "fixed"):
            for enh in (False, True):
                print(f"\n{'=' * 60}\nEXPERIMENT: mask={mode}, "
                      f"enhancement={enh}\n{'=' * 60}")
                cfg = PipelineConfig(
                    image_path=Path(image_arg),
                    base_mask_mode=mode,
                    use_enhancement=enh,
                    annotation_mode="boxes",
                )
                run_pipeline(cfg)
    else:
        cfg = PipelineConfig(
            image_path=Path(image_arg),
            base_mask_mode="percentile",   # "percentile" | "fixed"
            use_enhancement=True,
            annotation_mode="zones",       # "boxes" | "contours" | "zones"
        )
        run_pipeline(cfg)