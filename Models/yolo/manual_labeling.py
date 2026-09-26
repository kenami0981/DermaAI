"""
YOLO Bounding Box Editor
A lightweight OpenCV tool to view, add, delete, and modify YOLO format bboxes.

Controls:
  LMB drag   - Add new bounding box
  RMB click  - Delete target bounding box
  MMB drag   - Pan / move image
  Mouse Wheel- Zoom in / out
  N / P      - Next / Previous image
  S          - Save current changes
  D          - Delete image and label file
  ESC        - Exit (auto-saves before exit)
"""

from pathlib import Path
import cv2
import numpy as np
import copy

IMAGES_DIR = Path("Models/yolo/data/dataset_final_preprocessed_tiled_v2/train/images")
LABELS_DIR = Path("Models/yolo/data/dataset_final_preprocessed_tiled_v2/train/labels")
DEFAULT_CLASS_ID = 0

class YOLOEditor:
    def __init__(self, img_dir, lbl_dir, file_list=None):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        
        # Load given list or scan all directory files
        if file_list:
            self.image_files = []
            for f in file_list:
                p = Path(f)
                if p.suffix.lower() == ".txt":
                    p = p.with_suffix(".jpg")
                if not p.is_absolute():
                    p = self.img_dir / p.name
                if p.exists() and p.suffix.lower() in valid_exts:
                    self.image_files.append(p)
        else:
            self.image_files = sorted([f for f in self.img_dir.iterdir() if f.suffix.lower() in valid_exts], key=lambda x: x.name)

        if not self.image_files:
            print("No images found")
            exit()

        # State initialization
        self.current_idx = 0
        self.image = None
        self.orig_w = self.orig_h = 0
        self.boxes, self.old_boxes = [], []
        self.zoom, self.min_zoom, self.max_zoom = 1.0, 0.1, 6.0
        self.offset_x, self.offset_y = 0, 0
        self.drag_box, self.drag_pan = False, False
        self.start_x, self.start_y = 0, 0
        self.mouse_img_x, self.mouse_img_y = 0, 0
        self.pan_x, self.pan_y = 0, 0
        self.window = "YOLO BBox Editor"
        self.canvas_w, self.canvas_h, self.info_w = 1600, 900, 350

    def load_current_data(self):
        """Load active image and parse corresponding YOLO txt label."""
        img_path = self.image_files[self.current_idx]
        self.image = cv2.imread(str(img_path))
        if self.image is None:
            return False
        self.orig_h, self.orig_w = self.image.shape[:2]
        self.boxes = []
        txt_path = self.lbl_dir / f"{img_path.stem}.txt"
        
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 5:
                        cls = int(p[0])
                        xc, yc, bw, bh = map(float, p[1:5])
                        x1 = int((xc - bw / 2) * self.orig_w)
                        y1 = int((yc - bh / 2) * self.orig_h)
                        x2 = int((xc + bw / 2) * self.orig_w)
                        y2 = int((yc + bh / 2) * self.orig_h)
                        self.boxes.append([cls, x1, y1, x2, y2])
        self.old_boxes = copy.deepcopy(self.boxes)
        self.zoom = 1.0
        self.offset_x = self.offset_y = 0
        return True

    def has_changes(self):
        return self.boxes != self.old_boxes

    def save_current_data(self):
        """Save bounding boxes back to YOLO txt file if modified."""
        if not self.has_changes():
            return
        img_path = self.image_files[self.current_idx]
        txt_path = self.lbl_dir / f"{img_path.stem}.txt"
        lines = []
        
        for cls, x1, y1, x2, y2 in self.boxes:
            x1, y1 = max(0, min(x1, self.orig_w)), max(0, min(y1, self.orig_h))
            x2, y2 = max(0, min(x2, self.orig_w)), max(0, min(y2, self.orig_h))
            if x2 <= x1 or y2 <= y1:
                continue
            bw, bh = (x2 - x1) / self.orig_w, (y2 - y1) / self.orig_h
            xc, yc = ((x1 + x2) / 2) / self.orig_w, ((y1 + y2) / 2) / self.orig_h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Saved: {txt_path.name} | boxes {len(self.old_boxes)} -> {len(self.boxes)}")
        self.old_boxes = copy.deepcopy(self.boxes)

    def image_to_screen(self, x, y):
        return int(x * self.zoom - self.offset_x), int(y * self.zoom - self.offset_y)

    def screen_to_image(self, x, y):
        return int((x + self.offset_x) / self.zoom), int((y + self.offset_y) / self.zoom)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse drag, wheel zoom, and box creation/deletion."""
        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom
            self.zoom *= 1.15 if flags > 0 else (1 / 1.15)
            self.zoom = max(self.min_zoom, min(self.zoom, self.max_zoom))
            img_x, img_y = (x + self.offset_x) / old_zoom, (y + self.offset_y) / old_zoom
            self.offset_x = img_x * self.zoom - x
            self.offset_y = img_y * self.zoom - y

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.drag_pan, self.pan_x, self.pan_y = True, x, y

        elif event == cv2.EVENT_MBUTTONUP:
            self.drag_pan = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_pan:
                self.offset_x -= (x - self.pan_x)
                self.offset_y -= (y - self.pan_y)
                self.pan_x, self.pan_y = x, y
            if self.drag_box:
                self.mouse_img_x, self.mouse_img_y = self.screen_to_image(x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drag_box = True
            self.start_x, self.start_y = self.screen_to_image(x, y)
            self.mouse_img_x, self.mouse_img_y = self.start_x, self.start_y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_box:
                self.drag_box = False
                end_x, end_y = self.screen_to_image(x, y)
                x1, x2 = min(self.start_x, end_x), max(self.start_x, end_x)
                y1, y2 = min(self.start_y, end_y), max(self.start_y, end_y)
                if x2 - x1 > 3 and y2 - y1 > 3:
                    self.boxes.append([DEFAULT_CLASS_ID, x1, y1, x2, y2])
                    print("Added box:", x1, y1, x2, y2)

        elif event == cv2.EVENT_RBUTTONDOWN:
            ix, iy = self.screen_to_image(x, y)
            remove_id, min_dist = -1, float("inf")
            for i, (_, x1, y1, x2, y2) in enumerate(self.boxes):
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    remove_id = i
                    break
                dist = ((x1 + x2) / 2 - ix) ** 2 + ((y1 + y2) / 2 - iy) ** 2
                if dist < min_dist:
                    min_dist, remove_id = dist, i
            if remove_id != -1:
                print("Deleted box:", self.boxes.pop(remove_id))

    def draw_image(self):
        """Render image viewport and right sidebar HUD."""
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        image_area_w = self.canvas_w - self.info_w
        frame = cv2.resize(self.image, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)

        # Draw boxes
        for _, x1, y1, x2, y2 in self.boxes:
            sx1, sy1 = int(x1 * self.zoom), int(y1 * self.zoom)
            sx2, sy2 = int(x2 * self.zoom), int(y2 * self.zoom)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

        if self.drag_box:
            x1, y1 = int(self.start_x * self.zoom), int(self.start_y * self.zoom)
            x2, y2 = int(self.mouse_img_x * self.zoom), int(self.mouse_img_y * self.zoom)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Crop frame to canvas viewport
        h, w = frame.shape[:2]
        x1, y1 = int(self.offset_x), int(self.offset_y)
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(w, x1 + image_area_w), min(h, y1 + self.canvas_h)
        dx1, dy1 = max(0, -x1), max(0, -y1)

        if sx2 > sx1 and sy2 > sy1:
            canvas[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = frame[sy1:sy2, sx1:sx2]

        # Draw UI overlay panel
        cv2.rectangle(canvas, (image_area_w, 0), (self.canvas_w, self.canvas_h), (0, 0, 0), -1)
        texts = [
            f"Image {self.current_idx + 1}/{len(self.image_files)}",
            self.image_files[self.current_idx].name,
            "",
            f"Zoom: {self.zoom:.2f}",
            "",
            "CONTROLS:",
            "LMB drag  - add bbox",
            "RMB click - delete bbox",
            "MMB drag  - move image",
            "Wheel     - zoom",
            "",
            "N - next image",
            "P - previous image",
            "S - save",
            "D - delete",
            "ESC - exit"
        ]

        y = 40
        for t in texts:
            text_w = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(canvas, t, (1600 - text_w - 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 32

        return canvas

    def delete_current_image(self):
        """Remove active image and corresponding label file from disk."""
        img = self.image_files[self.current_idx]
        txt = self.lbl_dir / f"{img.stem}.txt"

        if img.exists():
            img.unlink()
            print("Deleted image:", img.name)
        if txt.exists():
            txt.unlink()
            print("Deleted label:", txt.name)

        self.image_files.pop(self.current_idx)
        if not self.image_files:
            return False

        if self.current_idx >= len(self.image_files):
            self.current_idx = len(self.image_files) - 1

        self.load_current_data()
        return True

    def run(self):
        """Main window loop."""
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        if not self.load_current_data():
            return

        while True:
            frame = self.draw_image()
            cv2.imshow(self.window, frame)
            key = cv2.waitKey(30) & 0xFF

            if key in [ord("n"), ord("N")]:
                self.save_current_data()
                self.current_idx = (self.current_idx + 1) % len(self.image_files)
                self.load_current_data()

            elif key in [ord("p"), ord("P")]:
                self.save_current_data()
                self.current_idx = (self.current_idx - 1) % len(self.image_files)
                self.load_current_data()

            elif key in [ord("s"), ord("S")]:
                self.save_current_data()

            elif key in [ord("d"), ord("D")]:
                if not self.delete_current_image():
                    break

            elif key == 27:
                self.save_current_data()
                break

        cv2.destroyAllWindows()





if __name__ == "__main__":
    
    # Specific list of files to load
    top_20_files = [
        "levle3_102.txt",
        "levle3_89_1.txt",
        "levle3_89_2.txt",
        "levle3_89_7.txt",
        "levle2_8.txt",
        "levle3_89_4.txt",
        "levle1_83_6.txt",
        "levle2_94_7.txt",
        "levle3_104.txt",
        "levle2_138_8.txt",
        "levle3_101_1.txt",
        "levle3_112_7.txt",
        "levle1_178_4.txt",
        "levle2_159_7.txt",
        "levle3_117.txt",
        "levle0_411_7.txt",
        "levle2_121_3.txt",
        "levle3_39.txt",
        "levle2_158_7.txt",
        "levle1_177_5.txt"
    ]

    # OPTION 1: Load all images from directory
    editor = YOLOEditor(IMAGES_DIR, LABELS_DIR)

    # OPTION 2: Load specific file list
    # editor = YOLOEditor(IMAGES_DIR, LABELS_DIR, file_list=top_20_files)

    editor.run()