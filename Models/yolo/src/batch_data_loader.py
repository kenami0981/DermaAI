import sqlite3
import json
from pathlib import Path
from loguru import logger
import uuid

# Path to the SQLite database in the .NET backend
DB_PATH = Path(r"D:\studia\DermaAI\Backend\DermaAPI\derma.db")

# Root directory containing dated photo folders
PHOTOS_ROOT = Path(r"D:\studia\DermaAI\Models\yolo\images\my\dermaai_v2\dated_fotoes")

# Path to the external JSON file containing structured lifestyle data
JSON_DATA_PATH = Path(r"D:\studia\DermaAI\Models\yolo\images\my\dermaai_v2\lifestyle_data.json")

# Single User ID for all batch-inserted records
USER_ID = "1a2c5462-4439-46ce-9a44-d025be764238"


def load_lifestyle_data() -> dict:
    """Loads lifestyle data from the external JSON file."""
    if not JSON_DATA_PATH.exists():
        logger.error(f"Lifestyle JSON file not found at path: {JSON_DATA_PATH}")
        return {}
    
    try:
        with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded lifestyle data for {len(data)} dates from JSON.")
            return data
    except Exception as e:
        logger.exception(f"Failed to parse JSON file: {e}")
        return {}


def batch_insert_lifestyle_photos():
    """
    Scans dated photo folders, matches them with structured lifestyle data loaded from JSON,
    and performs batch insertion into SkinAnalysis and SkinAnalysisTag tables
    for a single specified user ID. Notes and tags are added only to the first
    photo encountered per day.
    """
    logger.info("Starting batch insertion of lifestyle photos...")

    if not DB_PATH.exists():
        logger.error(f"Database file not found at path: {DB_PATH}")
        return

    if not PHOTOS_ROOT.exists():
        logger.error(f"Photos root directory not found at path: {PHOTOS_ROOT}")
        return

    # Load lifestyle data from the external file
    lifestyle_data = load_lifestyle_data()
    if not lifestyle_data:
        logger.warning("Lifestyle data is empty. Proceeding with default fallbacks.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    analysis_count = 0
    tag_count = 0

    try:
        # Iterate through dated directories (e.g., 2026-10-25)
        for date_folder in sorted(PHOTOS_ROOT.iterdir()):
            if not date_folder.is_dir():
                continue

            date_str = date_folder.name
            
            # Retrieve lifestyle data for the specific date, fallback if missing
            day_data = lifestyle_data.get(date_str, {"notes": "No daily notes", "tags": {}})

            # Flag to check if the daily note has already been assigned to an image
            notes_added_for_day = False

            # Iterate through side subdirectories (left, front, right)
            for side_folder in date_folder.iterdir():
                if not side_folder.is_dir():
                    continue

                face_side = side_folder.name.capitalize()

                # Iterate through image files in the side directory
                for img_file in side_folder.glob("*.*"):
                    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                        continue

                    analysis_id = str(uuid.uuid4()).upper()
                    image_url = str(img_file.resolve())
                    
                    # Note assigned ONLY to the FIRST encountered photo of the day
                    if not notes_added_for_day:
                        combined_notes = f"Side: {face_side} | Notes: {day_data['notes']}"
                        current_day_tags = day_data['tags']
                        notes_added_for_day = True  # Mark that notes and tags are used for this day
                    else:
                        combined_notes = f"Side: {face_side}"
                        current_day_tags = {}  # No tags for subsequent photos of the same day

                    upload_date = f"{date_str} 12:00:00.000"
                    status = 0  # 0 = Pending analysis status

                    # 1. Insert main SkinAnalysis record
                    cursor.execute("""
                        INSERT INTO SkinAnalysis (Id, Status, imageUrl, notes, uploadDate, userId)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (analysis_id, status, image_url, combined_notes, upload_date, USER_ID))

                    analysis_count += 1
                    logger.debug(f"Inserted analysis ID {analysis_id} for image {img_file.name} ({face_side})")

                    # 2. Insert associated lifestyle tags (only for the first photo of the day)
                    for tag_key, tag_value in current_day_tags.items():
                        tag_id = str(uuid.uuid4()).upper()
                        tag_enum = int(tag_key)

                        cursor.execute("""
                            INSERT INTO SkinAnalysisTag (Id, SkinAnalysisId, Tag, Value)
                            VALUES (?, ?, ?, ?)
                        """, (tag_id, analysis_id, tag_enum, tag_value))

                        tag_count += 1

        conn.commit()
        logger.success(f"Batch insertion completed successfully! Inserted {analysis_count} analyses and {tag_count} tags for User ID: {USER_ID}")

    except Exception as e:
        logger.exception(f"An error occurred during batch insertion: {e}")
        conn.rollback()
    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    batch_insert_lifestyle_photos()