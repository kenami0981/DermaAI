import sqlite3
import json
from pathlib import Path
from loguru import logger
import uuid
import sys

"""
Integration pipeline script connecting the YOLO model prediction workflow with the SQLite database.
"""

# Add the model src folder to Python path to import predict.py
MODEL_SRC_DIR = Path(r"D:\studia\DermaAI\Models\yolo\src")
if str(MODEL_SRC_DIR) not in sys.path:
    sys.path.append(str(MODEL_SRC_DIR))

# Import functions and configuration from predict.py
from predict import load_model, load_sahi_model, run_inference

# Path to the .NET / SQLite database
DB_PATH = Path(r"D:\studia\DermaAI\Backend\DermaAPI\derma.db")

def get_table_columns(cursor, table_name):
    """
    Helper function to fetch the list of columns for a given table from the SQLite database.
    """
    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row[1].lower(): row[1] for row in cursor.fetchall()}

def run_pipeline():
    """
    Executes the integration pipeline, fetching pending skin analyses,
    running YOLO inference, and storing results back into the database.
    """
    logger.info("Starting the integration pipeline with the real YOLO model...")

    if not DB_PATH.exists():
        logger.error(f"Database not found at path: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Fetch records with status 0 (pending)
        cursor.execute("""
            SELECT sa.Id, sa.ImageUrl, sa.Notes 
            FROM SkinAnalysis sa
            WHERE sa.Status = 0
        """)
            

        pending_records = cursor.fetchall()

        if not pending_records:
            logger.warning("No new analyses to process (status = 0).")
            conn.close()
            return

        logger.success(f"Found {len(pending_records)} record(s) waiting for analysis.")

        # Fetch column schema to avoid errors
        ar_cols = get_table_columns(cursor, "AnalysisResult")
        det_cols = get_table_columns(cursor, "Detection")

        # Initialize the YOLO model once (standard or SAHI depending on preference)
        # Here we use standard load_model (you can change to load_sahi_model() if SAHI is default)
        logger.info("Loading YOLO model into memory...")
        model = load_model()
        
        # Create a session folder for YOLO prediction results matching the project architecture
        session_dir = Path(r"D:\studia\DermaAI\Models\yolo\runs\predict\pipeline_session")
        session_dir.mkdir(parents=True, exist_ok=True)

        for row in pending_records:
            analysis_id, image_path_str, notes = row
            logger.info(f"--- Processing analysis ID: {analysis_id} ---")
            logger.info(f"Input image path: {image_path_str}")

            image_path = Path(image_path_str)
            if not image_path.exists():
                logger.error(f"Image file does not exist on disk: {image_path}")
                continue

            # ==========================================
            # 2. RUN REAL PREDICT.PY
            # ==========================================
            logger.info("Running YOLO prediction via function from predict.py...")
            
            # You can control tta and sahi flags here (e.g., tta=True, sahi=False)
            use_tta = True
            use_sahi = False
            use_preprocess = True

            try:
                # If you want to use SAHI, the model must be loaded via load_sahi_model()
                if use_sahi:
                    from predict import load_sahi_model, run_inference as run_sahi_inference
                    sahi_model = load_sahi_model()
                    yolo_res = run_sahi_inference(image_path, sahi_model, session_dir, preprocess=use_preprocess, tta=use_tta, sahi=True)
                else:
                    yolo_res = run_inference(image_path, model, session_dir, preprocess=use_preprocess, tta=use_tta, sahi=False)
                
                if not yolo_res:
                    logger.error(f"Model returned no results for image: {image_path}")
                    continue

                logger.success("YOLO model completed analysis successfully!")
            except Exception as e:
                logger.exception(f"Error executing YOLO model: {e}")
                continue

            # Extract data from the dictionary returned by predict.py
            acne_score = yolo_res["score"]
            detections_list = yolo_res["detections"]
            acne_count = len(detections_list)
            tta_applied = bool(yolo_res["tta_applied"])
            sahi_applied = bool(yolo_res["sahi_applied"])
            
            # Determine absolute path to the output image (e.g., image0.jpg from output folder)
            # yolo_res["image"]["output"] returns a relative path, so we build the full absolute path:
            output_img_relative = yolo_res["image"]["output"]
            output_image_abs = str((Path(r"D:\studia\DermaAI\Models\yolo\runs") / output_img_relative).resolve())

            logger.info(f"[RESULT] Score: {acne_score} | Detections count: {acne_count} | TTA: {tta_applied} | SAHI: {sahi_applied}")
            logger.info(f"[OUTPUT] Absolute path of output image: {output_image_abs}")

            # ==========================================
            # 3. CHECK AND CLEAR OLD RESULTS IN THE DATABASE
            # ==========================================
            cursor.execute("SELECT Id FROM AnalysisResult WHERE SkinAnalysisId = ?", (analysis_id,))
            existing_result = cursor.fetchone()

            if existing_result:
                result_id = existing_result[0]
                logger.warning(f"Detected old AnalysisResult (ID: {result_id}). Clearing old detections...")
                cursor.execute("DELETE FROM Detection WHERE AnalysisResultId = ?", (result_id,))
            else:
                result_id = str(uuid.uuid4()).upper()
                logger.info(f"Creating new AnalysisResult ID: {result_id}")

            # ==========================================
            # 4. SAVE TO AnalysisResult (including tta and sahi)
            # ==========================================
            ar_data = {}
            if 'id' in ar_cols: ar_data[ar_cols['id']] = result_id
            if 'skinanalysisid' in ar_cols: ar_data[ar_cols['skinanalysisid']] = analysis_id
            if 'acnescore' in ar_cols: ar_data[ar_cols['acnescore']] = acne_score
            if 'acnecount' in ar_cols: ar_data[ar_cols['acnecount']] = acne_count
            if 'outputimageurl' in ar_cols: ar_data[ar_cols['outputimageurl']] = output_image_abs
            # if 'ttaapplied' in ar_cols: ar_data[ar_cols['ttaapplied']] = int(tta_applied)
            # if 'sahiapplied' in ar_cols: ar_data[ar_cols['sahiapplied']] = int(sahi_applied)
            for col_variant in ['tta_applied', 'ttaapplied']:
                if col_variant.lower() in ar_cols:
                    ar_data[ar_cols[col_variant.lower()]] = int(tta_applied)
                    break
                    
            for col_variant in ['sahi_applied', 'sahiapplied']:
                if col_variant.lower() in ar_cols:
                    ar_data[ar_cols[col_variant.lower()]] = int(sahi_applied)
                    break

            ar_keys = list(ar_data.keys())
            ar_values = list(ar_data.values())
            ar_placeholders = ", ".join(["?"] * len(ar_keys))
            ar_columns_str = ", ".join(ar_keys)

            cursor.execute(f"""
                INSERT OR REPLACE INTO AnalysisResult ({ar_columns_str})
                VALUES ({ar_placeholders})
            """, ar_values)

            # ==========================================
            # 5. SAVE TO Detection
            # ==========================================
            for det in detections_list:
                det_id = str(uuid.uuid4()).upper()
                bbox_str = json.dumps(det["bbox"])

                det_data = {}
                if 'id' in det_cols: det_data[det_cols['id']] = det_id
                if 'analysisresultid' in det_cols: det_data[det_cols['analysisresultid']] = result_id
                if 'classname' in det_cols: det_data[det_cols['classname']] = det["class"]
                if 'bbox' in det_cols: det_data[det_cols['bbox']] = bbox_str
                if 'confidence' in det_cols: det_data[det_cols['confidence']] = det["confidence"]

                det_keys = list(det_data.keys())
                det_values = list(det_data.values())
                det_placeholders = ", ".join(["?"] * len(det_keys))
                det_columns_str = ", ".join(det_keys)

                cursor.execute(f"""
                    INSERT INTO Detection ({det_columns_str})
                    VALUES ({det_placeholders})
                """, det_values)

            # ==========================================
            # 6. UPDATE SkinAnalysis STATUS TO 1
            # ==========================================
            cursor.execute("""
                UPDATE SkinAnalysis 
                SET Status = 1 
                WHERE Id = ?
            """, (analysis_id,))


            conn.commit()
            logger.success(f"Saved real YOLO results for analysis ID: {analysis_id} [OK]!\n")

    except Exception as e:
        logger.exception(f"A critical error occurred in the pipeline: {e}")
        conn.rollback()
    finally:
        conn.close()
        logger.info("Closed database connection.")

if __name__ == "__main__":
    run_pipeline()