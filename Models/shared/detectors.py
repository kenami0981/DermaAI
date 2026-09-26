from abc import ABC, abstractmethod
import os
import contextlib
from ultralytics import YOLO

class BaseDetector(ABC):
    """
    Abstract base class establishing a common interface for all detection models.
    """
    @abstractmethod
    def predict(self, image):
        """
        Runs model inference.

        Output:
            Standardized detection results:
            - Bounding boxes in [x1, y1, x2, y2] format.
            - Optional confidence scores depending on the model type.
        """
        pass


class CVDetector(BaseDetector):
    """
    Wrapper adapter for traditional OpenCV computer vision processing.
    Output:
        Bounding boxes detected by OpenCV in format (list of lists): [x1, y1, x2, y2]
    
    """
    def __init__(self, detect_fn, clean_fn, extract_fn, silence=True):
        self.detect_fn = detect_fn
        self.clean_fn = clean_fn
        self.extract_fn = extract_fn
        self.silence = silence

    def predict(self, image):
        """
        Runs the OpenCV pipeline and converts outputs to float [x1, y1, x2, y2] coordinates.
        """
        img_h, img_w = image.shape[:2]
        mask = self.detect_fn(image)
        mask = self.clean_fn(mask, img_w, img_h)
        
        img_tmp = image.copy()
        
        if self.silence:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    cv_boxes, _ = self.extract_fn(mask, img_tmp, img_w, img_h)
        else:
            cv_boxes, _ = self.extract_fn(mask, img_tmp, img_w, img_h)
                
        return [[float(x), float(y), float(x + w), float(y + h)] for (x, y, w, h) in cv_boxes]


class YOLODetector(BaseDetector):
    """
    Wrapper adapter for YOLO models.
    Output:
        Model predictions converted into the common detection format:
        - Bounding boxes: [x1, y1, x2, y2]
        - Optional confidence scores
    """
    def __init__(self, model_path, default_conf=0.001):
        self.model = YOLO(str(model_path))
        self.default_conf = default_conf

    def predict(self, image):
        """
        Runs YOLO model inference and returns raw predictions with confidences.
        """
        results = self.model.predict(source=image, conf=self.default_conf, verbose=False)
        yolo_boxes, yolo_confs = [], []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                yolo_boxes.append([x1, y1, x2, y2])
                yolo_confs.append(float(box.conf[0]))
                
        return yolo_boxes, yolo_confs
    
    
# PLACEHOLDER for future models     
class NewModelDetector(BaseDetector):
    """
    Wrapper adapter for a new detection model.
    """
    def __init__(self, model_path):
        # Load the new model here
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # Implement model loading logic
        pass

    def predict(self, image):
        """
        Runs the new model inference and returns predictions.
        """
        # Implement prediction logic for the new model
        pass