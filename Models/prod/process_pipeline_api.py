import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

"""
Integration pipeline connecting acne detection models with DermaAPI.

Fetches pending analyses (optionally filtered by --analysis-id), gets each analysis
by ID, resolves its input image, runs model inference, and submits an AnalysisResult 
with detections to DermaAPI.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{function:<24}:{line:<3}</cyan> | <level>{message}</level>",
)

LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "pipeline.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="14 days",
    encoding="utf-8",
)

for env_candidate in (MODELS_DIR / ".env", PROJECT_ROOT / ".env"):
    if env_candidate.is_file():
        load_dotenv(env_candidate, override=False)
load_dotenv(override=False)


def _clean_path(raw_path: str | Path, base_dir: Path | None = None) -> Path:
    """Normalize a path from configuration or API and resolve relative paths against base_dir."""
    cleaned = str(raw_path).strip().strip('"').strip("'")
    cleaned = cleaned.replace("/", os.sep)
    path = Path(cleaned)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


# API connection and authentication settings
BASE_URL = os.getenv("DERMA_API_URL", "").strip().rstrip("/")
API_KEY = os.getenv("DERMA_API_KEY", "").strip()

# Optional endpoint override. Default is POST /api/AnalysisResult.
RESULT_SUBMIT_ENDPOINT = os.getenv("RESULT_SUBMIT_ENDPOINT", "").strip()
RESULT_SUBMIT_METHOD = os.getenv("RESULT_SUBMIT_METHOD", "POST").strip().upper()

# Portable filesystem paths relative to project layout; override in .env if needed.
MODEL_SRC_DIR = _clean_path(
    os.getenv("MODEL_SRC_DIR", Path("Models") / "yolo" / "src"),
    PROJECT_ROOT,
)
YOLO_RUNS_DIR = _clean_path(
    os.getenv("YOLO_RUNS_DIR", Path("Models") / "yolo" / "runs"),
    PROJECT_ROOT,
)
SESSION_DIR = _clean_path(
    os.getenv(
        "SESSION_DIR",
        Path("Models") / "yolo" / "runs" / "predict" / "pipeline_session",
    ),
    PROJECT_ROOT,
)
API_INPUTS_DIR = _clean_path(
    os.getenv(
        "API_INPUTS_DIR",
        Path("Models") / "yolo" / "runs" / "predict" / "api_inputs",
    ),
    PROJECT_ROOT,
)
UPLOADS_DIR = _clean_path(
    os.getenv("UPLOADS_DIR", Path("Backend") / "DermaAPI" / "Uploads"),
    PROJECT_ROOT,
)
OUTPUTS_DIR = _clean_path(
    os.getenv("OUTPUTS_DIR", Path("Backend") / "DermaAPI" / "Outputs"),
    PROJECT_ROOT,
)

# ==============================================================================
# FASTER R-CNN & DETECTION VOTING CONFIGURATION (PLACEHOLDER)
# ==============================================================================
FASTER_RCNN_WEIGHTS_PATH = os.getenv("FASTER_RCNN_WEIGHTS_PATH", "").strip()
VOTING_ENABLED = os.getenv("VOTING_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
VOTING_IOU_THRESHOLD = float(os.getenv("VOTING_IOU_THRESHOLD", "0.2"))
# ==============================================================================

FORCE_REPROCESS = True


def log_configured_paths() -> None:
    """Log configured paths and API settings."""
    logger.info("=== Configured Paths & Environment ===")
    logger.info(f"PROJECT_ROOT             : {PROJECT_ROOT}")
    logger.info(f"DERMA_API_URL            : {BASE_URL}")
    logger.info(f"MODEL_SRC_DIR            : {MODEL_SRC_DIR.resolve()}")
    logger.info(f"YOLO_RUNS_DIR            : {YOLO_RUNS_DIR.resolve()}")
    logger.info(f"SESSION_DIR              : {SESSION_DIR.resolve()}")
    logger.info(f"API_INPUTS_DIR           : {API_INPUTS_DIR.resolve()}")
    logger.info(f"UPLOADS_DIR              : {UPLOADS_DIR.resolve()}")
    logger.info(f"OUTPUTS_DIR              : {OUTPUTS_DIR.resolve()}")
    logger.info(f"LOG_DIR                  : {LOG_DIR.resolve()}")
    logger.info(
        f"RESULT_SUBMIT_ENDPOINT   : "
        f"{RESULT_SUBMIT_ENDPOINT or '/api/AnalysisResult'}"
    )
    logger.info(f"RESULT_SUBMIT_METHOD     : {RESULT_SUBMIT_METHOD}")
    logger.info(f"API_KEY set              : {bool(API_KEY)}")
    logger.info(
        f"FASTER_RCNN_WEIGHTS_PATH : {FASTER_RCNN_WEIGHTS_PATH or '<not configured>'}"
    )
    logger.info(f"VOTING_ENABLED           : {VOTING_ENABLED}")
    logger.info("======================================")


log_configured_paths()

from utils import save_drawn_detections

for path_candidate in (str(MODEL_SRC_DIR.resolve()), str(SCRIPT_DIR)):
    if path_candidate not in sys.path:
        sys.path.insert(0, path_candidate)

from predict import load_model as load_yolo_model, run_inference as run_yolo_inference  # noqa: E402

# ==============================================================================
# FASTER R-CNN INFERENCE & VOTING PLACEHOLDERS
# ==============================================================================
def load_faster_rcnn_model(weights_path: str | Path | None = None):
    """Placeholder to load the Faster R-CNN model."""
    target = Path(weights_path) if weights_path else (
        Path(FASTER_RCNN_WEIGHTS_PATH) if FASTER_RCNN_WEIGHTS_PATH else None
    )
    if not target:
        return None

    logger.info(f"[PLACEHOLDER] Faster R-CNN weights path set: {target}")
    return None


def run_faster_rcnn_inference(image_path: Path, faster_rcnn_model) -> list[dict]:
    """Placeholder to run Faster R-CNN inference and return detections."""
    if faster_rcnn_model is None:
        return []

    # TODO: Run Faster R-CNN on image_path and return [{"class":..., "confidence": ..., "bbox": ...}]
    logger.info(f"[PLACEHOLDER] Faster R-CNN inference skipped for {image_path}")
    return []


def vote_model_detections(
    yolo_detections: list[dict],
    faster_rcnn_detections: list[dict],
    iou_threshold: float = VOTING_IOU_THRESHOLD,
) -> list[dict]:
    """Placeholder to vote/merge detections from YOLO and Faster R-CNN."""
    if not VOTING_ENABLED or not faster_rcnn_detections:
        return list(yolo_detections)

    logger.debug(
        f"[PLACEHOLDER] Voting passthrough (iou={iou_threshold}, "
        f"yolo={len(yolo_detections)}, frcnn={len(faster_rcnn_detections)})"
    )
    return list(yolo_detections)

# ==============================================================================

def _build_headers() -> dict:
    """Build HTTP headers with optional API key."""
    headers = {
        "Accept": "application/json",
    }
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _fetch_pending_analyses(headers: dict, analysis_id: str | None = None) -> list[dict]:
    """Fetch pending analyses, optionally filtered by a specific analysis_id."""
    url = f"{BASE_URL}/api/SkinAnalysis"
    logger.info(f"Fetching skin analyses from GET {url}...")
    response = requests.get(url, headers=headers, timeout=30)
    logger.info(f"GET /api/SkinAnalysis status: {response.status_code}")
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items") or data.get("data") or []
    else:
        items = []

    pending = []
    accepted = {0, "0", "pending", "new", 3, "3", "failed"}

    for item in items:
        item_id = str(item.get("id") or item.get("Id") or "").strip()

        if analysis_id:
            if item_id.lower() != analysis_id.strip().lower():
                continue
            pending.append(item)
            continue

        status = item.get("status", item.get("Status"))
        if status in accepted:
            pending.append(item)

    logger.info(
        f"Found {len(pending)} pending analysis record(s) "
        f"out of {len(items)} returned item(s)."
    )
    return pending


def _fetch_and_display_analysis(
    analysis_id: str, headers: dict, fallback_record: dict
) -> dict:
    """Fetch and print one SkinAnalysis record, including its current result."""
    url = f"{BASE_URL}/api/SkinAnalysis/{analysis_id}"
    logger.info(f"Fetching analysis details from GET {url}...")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        logger.info(
            f"GET /api/SkinAnalysis/{analysis_id} status: {response.status_code}"
        )
        response.raise_for_status()
        details = response.json()

        result = details.get("result") if isinstance(details, dict) else None
        if result:
            detections = result.get("detections") or []
            logger.info(
                f"Existing result: acneScore={result.get('acneScore')}, "
                f"acneCount={result.get('acneCount')}, "
                f"detections={len(detections)}"
            )
            for index, detection in enumerate(detections, start=1):
                logger.info(
                    f"Detection #{index}: "
                    f"class={detection.get('className')}, "
                    f"bbox={detection.get('bbox')}, "
                    f"confidence={detection.get('confidence')}"
                )

        return details if isinstance(details, dict) else fallback_record

    except Exception as exc:
        logger.warning(f"Could not fetch analysis {analysis_id}: {exc}")
        print(json.dumps(fallback_record, indent=2, ensure_ascii=False))
        return fallback_record


def _resolve_image_path(
    image_url: str, analysis_id: str, headers: dict
) -> Path | None:
    """Find the input image locally or download it from the API."""
    if not image_url:
        return None

    local_candidate = _clean_path(image_url, PROJECT_ROOT)
    logger.debug(f"Checking local image path: {local_candidate}")
    if local_candidate.is_file():
        return local_candidate

    relative_path = str(image_url).strip().lstrip("/\\").replace("/", os.sep)

    upload_name_candidate = _clean_path(UPLOADS_DIR / Path(relative_path).name)
    logger.debug(f"Checking uploads filename: {upload_name_candidate}")
    if upload_name_candidate.is_file():
        return upload_name_candidate

    upload_relative_candidate = _clean_path(UPLOADS_DIR / relative_path)
    logger.debug(f"Checking uploads relative path: {upload_relative_candidate}")
    if upload_relative_candidate.is_file():
        return upload_relative_candidate

    if image_url.startswith(("http://", "https://")):
        candidates = [image_url]
    else:
        url_relative = str(image_url).lstrip("/")
        candidates = [
            f"{BASE_URL}/{url_relative}",
            f"{BASE_URL}/uploads/{url_relative}",
            f"{BASE_URL}/api/SkinAnalysis/{analysis_id}/image",
        ]

    API_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(relative_path).suffix or ".jpg"
    downloaded_file = API_INPUTS_DIR / f"{analysis_id}{suffix}"

    for download_url in dict.fromkeys(candidates):
        logger.info(f"Trying image download from {download_url}...")
        try:
            response = requests.get(download_url, headers=headers, timeout=30)
            if response.ok and response.content:
                downloaded_file.write_bytes(response.content)
                logger.info(f"Downloaded image to {downloaded_file.resolve()}")
                return downloaded_file
            logger.debug(
                f"Image URL returned status {response.status_code}: {download_url}"
            )
        except requests.RequestException as exc:
            logger.debug(f"Image download failed for {download_url}: {exc}")

    logger.warning(f"Image not found locally or through API: {image_url}")
    return None


def _save_output_image(
    yolo_output_raw: str | Path,
    input_image_path: Path,
    detections: list[dict] | None = None,
    bbox_format: str = "xywh",
) -> Path:
    """Save final image to OUTPUTS_DIR; if detections are given, draw on original image."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = (OUTPUTS_DIR / input_image_path.name).resolve()

    if detections is not None:
        return save_drawn_detections(
            image_path=input_image_path,
            output_path=dest_path,
            detections=detections,
            bbox_format=bbox_format,
        )

    source_path = _clean_path(yolo_output_raw, YOLO_RUNS_DIR).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"YOLO output image not found: {source_path}")

    shutil.copy2(source_path, dest_path)
    return dest_path


def _normalize_bbox(raw_bbox) -> list[float]:
    """Convert a YOLO bounding box to a list of floats for the API."""
    parsed = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox

    if isinstance(parsed, dict):
        keys = ("x1", "y1", "x2", "y2")
        if any(parsed.get(key) is None for key in keys):
            raise ValueError(f"Incomplete bbox dictionary: {parsed}")
        return [float(parsed[key]) for key in keys]

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Unsupported bbox format: {type(parsed).__name__}")

    return [float(coordinate) for coordinate in parsed]


def _submit_result(
    analysis_id: str, request_payload: dict, headers: dict
) -> requests.Response:
    """
    Submit the AnalysisResult DTO.

    Important: the API schema expects AnalysisResult fields at the JSON root.
    Do NOT wrap the payload in {"analysisResult": ...}.
    """
    json_headers = {**headers, "Content-Type": "application/json"}

    if RESULT_SUBMIT_ENDPOINT:
        endpoint = RESULT_SUBMIT_ENDPOINT.format(
            id=analysis_id, analysisId=analysis_id
        )
        target_url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        method = RESULT_SUBMIT_METHOD
    else:
        target_url = f"{BASE_URL}/api/AnalysisResult"
        method = "POST"

    logger.info(f"Submitting AnalysisResult using {method} {target_url}")
    response = requests.request(
        method,
        target_url,
        headers=json_headers,
        json=request_payload,
        timeout=30,
    )
    return response


def _set_analysis_status(analysis_id: str, status: int, headers: dict) -> None:
    """Update SkinAnalysis status via existing PUT /api/SkinAnalysis/{id}."""
    url = f"{BASE_URL}/api/SkinAnalysis/{analysis_id}"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        data["id"] = analysis_id
        data["status"] = status  # 0=Pending, 1=Processing, 2=Completed, 3=Failed

        response = requests.put(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json=data,
            timeout=30,
        )
        response.raise_for_status()
        logger.info(f"Status of analysis {analysis_id} set to {status}.")
    except (requests.RequestException, ValueError) as exc:
        logger.warning(
            f"Could not set status={status} for analysis {analysis_id}: {exc}"
        )


def run_pipeline(analysis_id: str | None = None) -> None:
    """Fetch pending analyses, run model inference, and submit results to DermaAPI."""
    logger.info("Starting the DermaAPI + detection integration pipeline...")

    headers = _build_headers()

    try:
        pending_records = _fetch_pending_analyses(headers, analysis_id)
    except Exception as exc:
        logger.exception(f"Could not fetch pending analyses: {exc}")
        return

    if not pending_records:
        logger.warning("No pending analyses found.")
        return

    logger.success(f"Found {len(pending_records)} pending analysis record(s).")

    logger.info("Loading detection models...")
    try:
        yolo_model = load_yolo_model()
        faster_rcnn_model = load_faster_rcnn_model()
    except Exception as exc:
        logger.exception(f"Could not load detection models: {exc}")
        return

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using session directory: {SESSION_DIR.resolve()}")

    use_tta = True
    use_sahi = False
    use_preprocess = True

    for record in pending_records:
        raw_id = record.get("id") or record.get("Id")
        analysis_id = str(raw_id) if raw_id else ""
        if not analysis_id:
            logger.warning("Skipping a record with no analysis ID.")
            continue

        logger.info(f"--- Processing analysis ID: {analysis_id} ---")
        _set_analysis_status(analysis_id, 1, headers)   # Pending -> Processing
        details = _fetch_and_display_analysis(analysis_id, headers, record)

        old_result = (
            details.get("result") or details.get("Result")
            if isinstance(details, dict) else None
        )

        if old_result:
            if not FORCE_REPROCESS:
                logger.warning(
                    f"Analysis {analysis_id} already has a result - skipping."
                )
                _set_analysis_status(analysis_id, 3, headers)   # Processing -> Failed
                continue

            old_result_id = old_result.get("id") or old_result.get("Id")
            try:
                response = requests.delete(
                    f"{BASE_URL}/api/AnalysisResult/{old_result_id}",
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                logger.info(
                    f"Removed previous result {old_result_id} "
                    f"for analysis {analysis_id}."
                )
            except requests.RequestException as exc:
                logger.warning(
                    f"Could not delete previous result {old_result_id} "
                    f"for analysis {analysis_id}: {exc} - skipping."
                )
                continue

        image_url = (
            details.get("imageUrl")
            or details.get("ImageUrl")
            or record.get("imageUrl")
            or record.get("ImageUrl")
            or ""
        )
        notes = (
            details.get("notes")
            or details.get("Notes")
            or record.get("notes")
            or record.get("Notes")
            or ""
        )
        logger.info(f"Input image: {image_url} | Notes: {notes}")

        image_path = _resolve_image_path(image_url, analysis_id, headers)
        if not image_path or not image_path.is_file():
            logger.error(
                f"Image is unavailable for analysis {analysis_id}: {image_url}"
            )
            continue

        logger.info(f"Resolved input image: {image_path.resolve()}")
        logger.info("Running model inference...")

        try:
            # 1. Run YOLO inference
            yolo_result = run_yolo_inference(
                image_path=image_path,
                model=yolo_model,
                session_dir=SESSION_DIR,
                preprocess=use_preprocess,
                tta=use_tta,
                sahi=use_sahi,
            )

            if not yolo_result:
                logger.error(f"YOLO returned an empty result for {image_path}")
                _set_analysis_status(analysis_id, 3, headers)   # Processing -> Failed
                continue

            # 2. Run Faster R-CNN inference & vote detections (placeholder)
            faster_rcnn_detections = run_faster_rcnn_inference(
                image_path=image_path,
                faster_rcnn_model=faster_rcnn_model,
            )

            yolo_detections = list(yolo_result.get("detections") or [])
            
            final_detections = vote_model_detections(
                yolo_detections=yolo_detections,
                faster_rcnn_detections=faster_rcnn_detections,
            )
            yolo_result["detections"] = final_detections

            logger.success("Model inference completed.")
        except Exception as exc:
            logger.exception(
                f"Inference failed for analysis {analysis_id}: {exc}"
            )
            _set_analysis_status(analysis_id, 3, headers)   # Processing -> Failed
            continue

        try:
            acne_score = float(yolo_result["score"])
            detections_list = yolo_result.get("detections") or []
            acne_count = len(detections_list)
            tta_applied = bool(yolo_result["tta_applied"])
            sahi_applied = bool(yolo_result["sahi_applied"])

            output_image_relative = yolo_result["image"]["output"]
            saved_output_path = _save_output_image(
                output_image_relative,
                image_path,
                detections=detections_list,
            )
            output_image_abs = str(saved_output_path)

            result_id = str(uuid.uuid4())
            detections_payload = []

            for detection in detections_list:
                detections_payload.append(
                    {
                        "id": str(uuid.uuid4()),
                        "analysisResultId": result_id,
                        "className": str(detection["class"]),
                        "bbox": _normalize_bbox(detection["bbox"]),
                        "confidence": float(detection["confidence"]),
                    }
                )

            request_payload = {
                "id": result_id,
                "skinAnalysisId": analysis_id,
                "acneScore": acne_score,
                "acneCount": acne_count,
                "outputImageUrl": output_image_abs,
                "tta_applied": tta_applied,
                "sahi_applied": sahi_applied,
                "detections": detections_payload,
            }

            logger.info(
                f"[RESULT] Score={acne_score} | detections={acne_count} | "
                f"TTA={tta_applied} | SAHI={sahi_applied}"
            )
            logger.info(f"[OUTPUT] {output_image_abs}")

        except (KeyError, TypeError, ValueError, OSError) as exc:
            logger.exception(
                f"Could not build AnalysisResult for {analysis_id}: {exc}"
            )
            _set_analysis_status(analysis_id, 3, headers)   # Processing -> Failed
            continue

        try:
            response = _submit_result(analysis_id, request_payload, headers)
            logger.info(
                f"Submit response: {response.status_code} | {response.text}"
            )
            response.raise_for_status()
            logger.success(
                f"Saved result and {acne_count} detection(s) "
                f"for analysis {analysis_id}."
            )
            _set_analysis_status(analysis_id, 2, headers)   # Processing -> Completed
        except requests.RequestException as exc:
            logger.exception(
                f"Failed to submit result for analysis {analysis_id}: {exc}"
            )
            _set_analysis_status(analysis_id, 3, headers)   # Processing -> Failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--analysis-id",
        required=False,
        default=None,
        help="ID analizy do przetworzenia",
    )

    args = parser.parse_args()

    run_pipeline(args.analysis_id)