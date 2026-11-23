from pathlib import Path
import shutil


def setup_yolo_empty_labels(input_dir: str | Path, output_dir: str | Path, 
                            prefix: str = "levle00_", 
                            allowed_extensions: tuple = (".png", ".jpg", ".jpeg")) -> None:
    """
    Copies images from input_dir to output_dir/images with an optional prefix,
    and generates empty .txt files in output_dir/labels for YOLO background images.
    
    input_dir: Path to the folder containing source images.
    output_dir: Destination path (will create images/ and labels/ subdirectories).
    prefix: File prefix to prepend (e.g., 'levle00_').
    allowed_extensions: Tuple of supported image extensions.
    """

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    # 1. Create output directories
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 2. Collect image files
    image_files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ]

    print(f"Found {len(image_files)} image(s) in '{input_path}'.")

    processed_count = 0

    # 3. Copy/rename images and create corresponding empty label files
    for img_file in image_files:
        # Construct new filenames with the prefix
        new_stem = f"{prefix}{img_file.stem}"
        new_img_name = f"{new_stem}{img_file.suffix.lower()}"
        new_label_name = f"{new_stem}.txt"

        target_img_path = images_dir / new_img_name
        target_label_path = labels_dir / new_label_name

        # Copy the image
        shutil.copy2(img_file, target_img_path)

        # Create an empty text file (YOLO background image)
        target_label_path.touch(exist_ok=True)

        processed_count += 1

    print("=" * 60)
    print(f"Successfully processed files: {processed_count}")
    print(f"Images saved to:  {images_dir.resolve()}")
    print(f"Labels saved to: {labels_dir.resolve()}")
    print("=" * 60)



if __name__ == "__main__":

    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        SCRIPT_DIR = Path.cwd()

    INPUT_FOLDER = (SCRIPT_DIR / ".." / "data" / "clear-faces").resolve()
    OUTPUT_FOLDER = (SCRIPT_DIR / ".." / "data" / "dataset-clear-faces").resolve()

    setup_yolo_empty_labels(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        prefix="levle00_" 
    )