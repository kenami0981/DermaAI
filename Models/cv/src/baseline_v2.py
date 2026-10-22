import cv2
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import blob_log  # type: ignore

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Set up project directory paths dynamically
CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = CURRENT_DIR / "src"
SHARED_DIR = PROJECT_ROOT / "shared"
YOLO_DIR = PROJECT_ROOT / "Models" / "yolo"

for path_dir in [SRC_DIR, SHARED_DIR, YOLO_DIR]:
    if str(path_dir) not in sys.path:
        sys.path.insert(0, str(path_dir))

IMAGES_DIR = PROJECT_ROOT / "images" / "zuza" / "inzynierka" / "IMG_20251129_093304954.jpg"

image = cv2.imread(str(IMAGES_DIR))
if image is None:
    raise FileNotFoundError(f"Could not load image from:\n{IMAGES_DIR}")

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

MODEL_PATH = Path("face_landmarker.task")

SKIN_H_MIN = 0
SKIN_H_MAX = 30      
SKIN_S_MIN = 15      
SKIN_S_MAX = 255     
SKIN_V_MIN = 40      
SKIN_CR_MIN = 130    
SKIN_CR_MAX = 190    
SKIN_CB_MIN = 70     
SKIN_CB_MAX = 140    

REF_DIAG = 1414.0    
BASE_BLUR = 5        
BASE_ADAPTIVE_BLOCK = 31 
BASE_MORPH_KERNEL = 5 

def create_landmarker(model_path: Path) -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def crop_face_with_mediapipe(image, model_path, top_head_expansion=0.3, chin_padding=0.1, 
                           side_padding=0.08, target_aspect_ratio=0.75):
    if not model_path.exists():
        print(f"Warning: MediaPipe model does not exist at {model_path}. Skipping crop.")
        return image

    h, w, _ = image.shape
    detector = create_landmarker(model_path)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        print("No face detected by MediaPipe. Using full image.")
        return image

    landmarks = detection_result.face_landmarks[0]

    img_visualization = image.copy()
    num_pts = len(FACE_OVAL_INDICES)
    for i in range(num_pts):
        idx1 = FACE_OVAL_INDICES[i]
        idx2 = FACE_OVAL_INDICES[(i + 1) % num_pts]
        pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
        pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
        cv2.line(img_visualization, pt1, pt2, (0, 255, 255), 2)

    nose_lm, left_lm, right_lm = landmarks[1], landmarks[234], landmarks[454]
    pt_nose = (int(nose_lm.x * w), int(nose_lm.y * h))
    pt_left = (int(left_lm.x * w), int(left_lm.y * h))
    pt_right = (int(right_lm.x * w), int(right_lm.y * h))

    cv2.line(img_visualization, pt_left, pt_right, (255, 255, 0), 1)
    cv2.circle(img_visualization, pt_left, 6, (255, 0, 0), -1)    
    cv2.circle(img_visualization, pt_right, 6, (0, 0, 255), -1)   
    cv2.circle(img_visualization, pt_nose, 8, (0, 255, 0), -1)    

    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    face_w = x_max - x_min
    face_h = y_max - y_min

    crop_x1 = x_min - (face_w * side_padding)
    crop_x2 = x_max + (face_w * side_padding)
    crop_y1 = y_min - (face_h * top_head_expansion)  
    crop_y2 = y_max + (face_h * chin_padding)        

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    current_ratio = crop_w / crop_h

    if current_ratio < target_aspect_ratio:
        new_w = crop_h * target_aspect_ratio
        diff_w = new_w - crop_w
        crop_x1 -= diff_w / 2
        crop_x2 += diff_w / 2
    else:
        new_h = crop_w / target_aspect_ratio
        diff_h = new_h - crop_h
        crop_y1 -= diff_h * 0.7  
        crop_y2 += diff_h * 0.3  

    x1 = int(max(0, crop_x1))
    x2 = int(min(w, crop_x2))
    y1 = int(max(0, crop_y1))
    y2 = int(min(h, crop_y2))

    cropped_face = image[y1:y2, x1:x2]
    return cropped_face, img_visualization

def get_adaptive_params(image):
    h, w = image.shape[:2]
    diag = np.sqrt(h**2 + w**2)
    scale = diag / REF_DIAG
    blur_k = max(3, int(BASE_BLUR * scale) | 1)
    adaptive_block = max(5, int(BASE_ADAPTIVE_BLOCK * scale) | 1)
    k_close = max(3, int(diag * 0.003) | 1)
    k_open = max(3, int(diag * 0.0015) | 1)
    return blur_k, adaptive_block, k_close, k_open

def create_skin_mask_exp_a(image):
    h, w = image.shape[:2]
    blur_k, adaptive_block, k_close, k_open = get_adaptive_params(image)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([SKIN_H_MIN, SKIN_S_MIN, SKIN_V_MIN]),
        np.array([SKIN_H_MAX, SKIN_S_MAX, 255])
    )

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    cr_mask = cv2.inRange(cr, SKIN_CR_MIN, SKIN_CR_MAX)
    cb_mask = cv2.inRange(cb, SKIN_CB_MIN, SKIN_CB_MAX)
    ycrcb_mask = cv2.bitwise_and(cr_mask, cb_mask)

    skin_mask = cv2.bitwise_or(hsv_mask, ycrcb_mask)

    kernel_close = np.ones((k_close, k_close), np.uint8)
    kernel_open = np.ones((k_open, k_open), np.uint8)
    
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return skin_mask

def create_acne_heatmap(image, skin_mask):
    h, w = image.shape[:2]
    diag = np.sqrt(h**2 + w**2)
    scale = diag / 1414.0
    bilateral_d = max(3, int(9 * scale) | 1)
    blur_ksize = max(3, int(15 * scale) | 1)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(lab)
    
    a_denoised = cv2.bilateralFilter(a_channel, d=bilateral_d, sigmaColor=75, sigmaSpace=75)
    
    a_skin_pixels = a_denoised[skin_mask > 0]
    if len(a_skin_pixels) == 0:
        return np.zeros_like(image)
    
    median_skin_redness = np.median(a_skin_pixels)
    a_diff = np.maximum(0, a_denoised.astype(np.float32) - median_skin_redness)
    a_diff[skin_mask == 0] = 0
    
    non_zero_diffs = a_diff[a_diff > 0]
    if len(non_zero_diffs) > 0:
        p99 = np.percentile(non_zero_diffs, 98)
        a_clipped = np.clip(a_diff, 0, p99)
        redness_norm = cv2.normalize(a_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    else:
        redness_norm = np.zeros_like(a_denoised, dtype=np.uint8)
    
    redness_smooth = cv2.GaussianBlur(redness_norm, (blur_ksize, blur_ksize), 0)
    heatmap_color = cv2.applyColorMap(redness_smooth, cv2.COLORMAP_JET)
    heatmap_color[skin_mask == 0] = 0
    
    return heatmap_color

def enhance_details(img, clip_l=2.0, clip_a=3.0, grid_size=(8, 8), blur_kernel=5):
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe_l = cv2.createCLAHE(clipLimit=clip_l, tileGridSize=grid_size)
    l_enhanced = clahe_l.apply(l)
    
    clahe_a = cv2.createCLAHE(clipLimit=clip_a, tileGridSize=grid_size)
    a_enhanced = clahe_a.apply(a)
    
    redness_map = cv2.subtract(a_enhanced, 128)
    redness_map = cv2.normalize(redness_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    a_enhanced = cv2.medianBlur(a_enhanced, blur_kernel)
    enhanced_img = cv2.merge((l_enhanced, a_enhanced, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)
    
    return enhanced_img, redness_map


def create_base_mask_from_heatmap(perfect_heatmap, skin_mask, percentile=85):
    """
    Poprawiona maska bazowa:
    1. Czerwone punkty (najwyższe wartości w skali szarości heatmapy) wchodzą BEZWZGLĘDNIE na stałe.
    2. Obszary pośrednie (żółte) wchodzą tylko wtedy, gdy towarzyszy im nagły kontrast krawędzi.
    """
    gray = cv2.cvtColor(perfect_heatmap, cv2.COLOR_BGR2GRAY)
    skin_pixels_gray = gray[skin_mask > 0]
    
    if len(skin_pixels_gray) == 0:
        return np.zeros_like(gray)
        
    # 1. Bezwzględne wpuszczenie najsilniejszych (czerwonych) zmian bez dziur
    p_val = np.percentile(skin_pixels_gray, percentile)
    _, permanent_red_mask = cv2.threshold(gray, p_val, 255, cv2.THRESH_BINARY)
    
    # Domknięcie ewentualnych dziur wewnątrz mocnych czerwonych plam
    h, w = gray.shape[:2]
    diag = np.sqrt(h**2 + w**2)
    k_size = max(3, int(diag * 0.002) | 1)
    kernel = np.ones((k_size, k_size), np.uint8)
    permanent_red_mask = cv2.morphologyEx(permanent_red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 2. Wykrywanie kontrastu dla obszarów pośrednich (żółtych)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, contrast_mask = cv2.threshold(gradient_norm, 35, 255, cv2.THRESH_BINARY)
    _, yellow_potential = cv2.threshold(gray, p_val * 0.70, 255, cv2.THRESH_BINARY)
    yellow_with_contrast = cv2.bitwise_and(yellow_potential, contrast_mask)
    
    # 3. Połączenie warstw
    base_mask = cv2.bitwise_or(permanent_red_mask, yellow_with_contrast)
    base_mask = cv2.bitwise_and(base_mask, skin_mask)
    
    return base_mask


def filter_mask_variations(mask, image, skin_mask, perfect_heatmap):
    h, w = mask.shape[:2]
    total_pixels = h * w
    diag = np.sqrt(h**2 + w**2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []
    
    heatmap_gray = cv2.cvtColor(perfect_heatmap, cv2.COLOR_BGR2GRAY)
    ideal_radius = diag * 0.0035  
    
    obj_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area <= 0 or perimeter == 0:
            continue
            
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        
        max_allowed_area = total_pixels * 0.0015 
        min_allowed_area = max(2, int(total_pixels * 0.000001))
        
        if area > max_allowed_area:
            continue  
            
        mask_single = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_single, [cnt], -1, 255, -1)
        
        # POPRAWKA: zmieniono > 255 na > 0 oraz użyto np.max dla czerwonych punktów
        heatmap_pixels = heatmap_gray[mask_single > 0]
        max_intensity = np.max(heatmap_pixels) if len(heatmap_pixels) > 0 else 0.0
        
        raw_intensity = min(1.0, max_intensity / 255.0) 
        intensity_score = raw_intensity ** 2.0 
        
        is_strong_red = (raw_intensity >= 0.65)
        
        if not is_strong_red and area < min_allowed_area:
            continue
            
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity_score = min(1.0, max(0.0, circularity))
        
        size_diff = abs(radius - ideal_radius)
        size_score = max(0.0, 1.0 - (size_diff / (ideal_radius * 2.0)))
        
        if is_strong_red:
            confidence = max(0.85, (0.7 * intensity_score) + (0.2 * circularity_score) + (0.1 * size_score))
        else:
            confidence = (0.7 * intensity_score) + (0.2 * circularity_score) + (0.1 * size_score)
        
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        
        detected_objects.append({
            'id': obj_id,
            'contour': cnt,
            'confidence': float(confidence),
            'raw_intensity': float(raw_intensity),
            'intensity_score': float(intensity_score),
            'circularity_score': float(circularity_score),
            'size_score': float(size_score),
            'box': (x, y, w_box, h_box)
        })
        obj_id += 1
        
    return detected_objects


def get_binary_mask_from_objects(mask_shape, detected_objects):
    res_mask = np.zeros(mask_shape, dtype=np.uint8)
    for obj in detected_objects:
        cv2.drawContours(res_mask, [obj['contour']], -1, 255, -1)
    return res_mask


def print_debug_dataframe(detected_objects, title="Tabela debugowania wykryć"):
    if not detected_objects:
        print(f"Brak wykrytych obiektów dla: {title}")
        return
    
    df_data = []
    for obj in detected_objects:
        df_data.append({
            'ID': obj['id'],
            'Confidence': round(obj['confidence'], 3),
            'Raw Intensity': round(obj['raw_intensity'], 3),
            'Intensity Score': round(obj['intensity_score'], 3),
            'Circularity Score': round(obj['circularity_score'], 3),
            'Size Score': round(obj['size_score'], 3)
        })
    df = pd.DataFrame(df_data)
    print(f"\n--- {title} ---")
    print(df.to_string(index=False))


def detect_and_draw_scored_boxes(image, detected_objects, padding_factor=1.2, min_confidence=0.08):
    result_img = image.copy()
    h, w = image.shape[:2]
    diag = np.sqrt(h**2 + w**2)
    line_thickness = max(1, int(diag * 0.0008))
    
    ideal_radius = diag * 0.0035  
    detected_count = 0
    
    for obj in detected_objects:
        if obj['confidence'] < min_confidence:
            continue
            
        x, y, w_box, h_box = obj['box']
        conf = obj['confidence']
        obj_id = obj['id']
        cnt = obj['contour']
        
        _, radius = cv2.minEnclosingCircle(cnt)
        size_error = abs(radius - ideal_radius) / ideal_radius  
        
        cx = x + w_box // 2
        cy = y + h_box // 2
        
        object_size = max(w_box, h_box)
        safe_object_size = max(5, object_size)
        dynamic_padding = max(1.1, min(2.5, padding_factor + (10.0 / safe_object_size)))
        
        size = object_size * dynamic_padding
        half_size = int(size / 2)
        
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        
        if size_error <= 0.25:
            color = (0, 255, 255)  # Żółty
        else:
            color = (0, int(255 * conf), int(255 * (1 - conf)))
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
        
        label = f"ID:{obj_id} ({conf:.2f})"
        cv2.putText(result_img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        detected_count += 1

    return result_img, detected_count


def main():
    cropped_image, img_visualization = crop_face_with_mediapipe(image, MODEL_PATH)
    skin_mask = create_skin_mask_exp_a(cropped_image)
    perfect_heatmap = create_acne_heatmap(cropped_image, skin_mask)

    enhanced_image, redness_map = enhance_details(cropped_image, clip_l=1.2, clip_a=1.1, grid_size=(8, 8))

    perfect_heatmap_enhanced = create_acne_heatmap(enhanced_image, skin_mask)

    base_mask_enhanced = create_base_mask_from_heatmap(perfect_heatmap_enhanced, skin_mask, percentile=85)

    filtered_objects_enhanced = filter_mask_variations(base_mask_enhanced, cropped_image, skin_mask, perfect_heatmap_enhanced)

    print_debug_dataframe(filtered_objects_enhanced, "Wykrycia - Obraz Enhanced")

    final_boxes_image_enh, count_enh = detect_and_draw_scored_boxes(cropped_image, filtered_objects_enhanced, padding_factor=1.2, min_confidence=0.1)
    print(f"Liczba wykrytych zmian (Enhanced): {count_enh}")


