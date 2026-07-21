"""
OpenCV acne detection pipeline

Automatically detects, isolates, and analyzes red regions or inflammation markers 
in an input image using OpenCV, and generates a visual Markdown execution report.

Processing Steps:
- Step 0 (Load): Imports input image and creates a timestamped output directory.
- Step 1 (Preprocessing): Enhances contrast and details using CLAHE.
- Step 1.5 (Heatmap): Generates a heatmap highlighting red/inflammatory areas.
- Step 2 (Segmentation): Combines HSV and LAB color spaces for precise red masking.
- Step 3 (Mask Cleaning): Applies morphological operations to remove noise and fill gaps.
- Step 4 (Extraction): Filters contours by size/shape, draws bounding boxes on valid lesions.
- Step 5 (Reporting): Saves steps and compiles a REPORT.md summary.

Prerequisites:
- Required input image: Models/shared/images/levle1_271.jpg
- Local helper modules: utils.py and src/image_preprocess.py in the workspace.

Outputs:
Saved inside a timestamped run folder: Models/cv/runs/run_YYYYMMDD_HHMMSS/
- original.jpg, enhanced.jpg, heatmap.jpg
- raw_mask.jpg, mask.jpg (binary masks)
- result.jpg (final image with bounding boxes)
- REPORT.md (complete pipeline report)

TODO: Add mask based on heatmap to further impove detection accuracy. 
    To cosider: combining heatmap and color segmentation masks.

"""

import cv2
from matplotlib import image
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
INPUT_IMAGE = ROOT / "Models" / "shared" / "images" / "levle1_271.jpg"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "Models" / "cv" / "runs" / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = RUN_DIR / "result.jpg"
OUTPUT_MASK = RUN_DIR / "mask.jpg"
OUTPUT_ENHANCED = RUN_DIR / "enhanced.jpg"

SHARED_DIR = ROOT / "Models" / "shared"
YOLO_DIR = ROOT / "Models" / "yolo"
if str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))
if str(YOLO_DIR) not in sys.path:
    sys.path.append(str(YOLO_DIR))

from utils import create_heatmap # type: ignore
from src.image_preprocess import enhance_details  # type: ignore


def detect_red_regions(image):
    """
    Step 1: Color segmentation.
    Uses a hybrid approach:
    - HSV: Robust for color hue/saturation detection.
    - LAB: 'a' channel for highlighting red vs green color.
    - Combining masks from both color spaces to increase precision.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lower_red1 = np.array([0, 40, 50])
    upper_red1 = np.array([20, 255, 255])

    lower_red2 = np.array([160, 40, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_hsv = (cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2))

    l, a, b = cv2.split(lab)
    _, mask_lab = cv2.threshold(a, 150, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(mask_hsv, mask_lab)

    return mask



def clean_mask(mask, img_w, img_h):
    """
    Step 2: Clean the mask.
    The kernel size changes based on image resolution.

    """
    # Calculate a dynamic kernel size (0.1% of image width/height)
    # Ensuring the result is at least 7x7 to make sense
    dynamic_size = int(max(img_w, img_h) * 0.001)
    kernel_size = max(7, dynamic_size)
    # print("Using kernel size:", kernel_size)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Remove random tiny pixel spots (noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Fill in small holes inside the detected shapes
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def extract_boxes(mask, image, img_w, img_h):
    """
    Step 3: Object isolation.
    Find right shapes in the mask and remove outliers based on size and shape.
    """
    # 1. Find boundaries of all white shapes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No shapes detected.")
        return []
    
    # Calculate a dynamic minimum area based on image size
    min_area_limit = (img_w * img_h) * 0.00001

    raw_candidates = []
    
    # 2. Initial filter: remove extremely small noise and long, thin lines
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate how square or round the shape is
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        # Skip very small dots or very long thin lines
        if area <= min_area_limit or aspect_ratio > 4.0:
            continue
            
        raw_candidates.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})

    if not raw_candidates:
        print("\n--- Filter Report ---")
        print("No candidates passed initial size/shape filter.")
        return []

    # 3. Dynamic filter: remove anomalies based on the size of all found objects
    all_areas = np.array([c['area'] for c in raw_candidates])
    
    # Set limits based on the 5th and 95th percentile (ignores smallest and largest)
    min_area_threshold = np.percentile(all_areas, 5)
    max_area_threshold = np.percentile(all_areas, 95)
    
    # Fallback: if thresholds are too close, use median to estimate safe size range
    median_area = np.median(all_areas)
    if max_area_threshold <= min_area_threshold:
        min_area_threshold = max(1.0, median_area * 0.3)
        max_area_threshold = median_area * 5.0

    boxes = []
    rejected_small = 0
    rejected_large = 0

    # 4. Final selection: keep only objects within calculated size range
    for cand in raw_candidates:
        area = cand['area']
        x, y, w, h = cand['bbox']
        
        if area < min_area_threshold:
            rejected_small += 1
            continue
            
        if area > max_area_threshold:
            rejected_large += 1
            continue

        boxes.append((x, y, w, h))

        # Draw boxes on the final image for visualization
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 5. Print a summary of what was filtered out
    filter_report_lines = [
            "### Filter Statistics",
            f"- **Total raw shapes found:** {len(contours)}",
            f"- **Shapes after basic filter:** {len(raw_candidates)}",
            f"- **Final objects kept:** {len(boxes)}",
            f"- **Rejected as too small (noise):** {rejected_small}",
            f"- **Rejected as too large (anomaly):** {rejected_large}"
        ]

    print("\n----------------- Filter Report -----------------")
    print(f"Total raw shapes found:           {len(contours)}")
    print(f"Shapes after basic filter:        {len(raw_candidates)}")
    print(f"Final objects kept:               {len(boxes)}")
    print(f"Rejected as too small (noise):    {rejected_small}")
    print(f"Rejected as too large (anomaly):  {rejected_large}")
    print("---------------------------------------------------\n")

    return boxes, filter_report_lines

def main():
    # 1. Load the input image
    image = cv2.imread(str(INPUT_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Could not load image from: {INPUT_IMAGE}")

    height, width = image.shape[:2]
    print(f"Processing image: {width}x{height} pixels.")

    # Prepare lines for the Markdown report
    report_lines = [
        f"# Pipeline Execution Report",
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source Image:** `{INPUT_IMAGE.name}`",
        f"**Image Dimensions:** `{width} x {height} px`",
        "---",
        "## Step 0: Original Input Image",
        "The raw image before any processing.",
        f"![Original Image](original.jpg)\n"
    ]
    
    cv2.imwrite(str(RUN_DIR / "original.jpg"), image)

    # 2. Preprocessing: Enhance contrast and colors
    enhanced_image = enhance_details(image, clip_l=1.2, clip_a=1.1, grid_size=(8, 8))
    cv2.imwrite(str(OUTPUT_ENHANCED), enhanced_image)
    
    report_lines.extend([
        "## Step 1: Image Preprocessing",
        "Improve image contrast using CLAHE on color channels.",
        f"![Enhanced Image]({OUTPUT_ENHANCED.name})\n"
    ])
    
    # 3. Heatmap: Visualize red inflammatory markers
    heatmap_img = create_heatmap(image)
    heatmap_path = RUN_DIR / "heatmap.jpg"
    cv2.imwrite(str(heatmap_path), heatmap_img)
    
    report_lines.extend([
        "## Step 1.5: Heatmap Generation",
        "Highlight red areas to help identify potential inflammation.",
        f"![Heatmap]({heatmap_path.name})\n"
    ])

    # 4. Segmentation: Create mask for red regions
    # raw_mask = detect_red_regions(enhanced_image, heatmap_img)
    raw_mask = detect_red_regions(enhanced_image)
    raw_mask_path = RUN_DIR / "raw_mask.jpg"
    cv2.imwrite(str(raw_mask_path), raw_mask)
    
    report_lines.extend([
        "## Step 2: Red Region Detection",
        "Filter image by color range (HSV + LAB) to create a binary mask.",
        f"![Raw Mask]({raw_mask_path.name})\n"
    ])

    # 5. Cleaning: Remove noise from the mask
    mask = clean_mask(raw_mask, width, height)
    cv2.imwrite(str(OUTPUT_MASK), mask)
    
    report_lines.extend([
        "## Step 3: Mask Refinement",
        "Remove tiny noise particles and fill small gaps within shapes.",
        f"![Clean Mask]({OUTPUT_MASK.name})\n"
    ])

    # 6. Extraction: Filter objects and draw final boxes
    boxes, filter_report_lines = extract_boxes(mask, image, width, height)
    cv2.imwrite(str(OUTPUT_IMAGE), image)
    
    report_lines.extend([
        "## Step 4: Final Detection Results",
        "Filter out small or odd-shaped objects and draw boxes on valid lesions.",
        f"**Total detected: {len(boxes)}**",
        ""
    ])

    report_lines.extend(filter_report_lines)
    report_lines.extend([
        "",
        f"![Final Result]({OUTPUT_IMAGE.name})\n"
    ])

    # 7. Save the final report
    report_path = RUN_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Detected lesions: {len(boxes)}")
    print(f"Results saved. Report generated at: {RUN_DIR}")

if __name__ == "__main__":
    main()