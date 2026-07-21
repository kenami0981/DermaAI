class EnsembleEngine:
    """
    Ensemble detection coordinator

    Coordinates the execution of YOLO and traditional Computer Vision (OpenCV)
    detectors and combines their predictions into a single detection pipeline.

    The engine uses the OpenCV detector as an additional verification step.
    If a YOLO detection is confirmed by a nearby OpenCV detection,
    its confidence score is increased (confidence boosting). Otherwise, the
    original YOLO confidence is preserved.
    
    The primary objective of this strategy is to improve the detection recall
    by reinforcing potentially correct YOLO predictions, even if this comes
    at the cost of a slight reduction in precision.
    """

    def __init__(self, yolo_detector, cv_detector):
        self.yolo = yolo_detector
        self.cv = cv_detector

    def run_ensemble_boosting(self, image, dynamic_tolerance_pct=0.10, boost_value=0.08):
        """
        Executes predictions from both YOLO and OpenCV pipelines
        and performs proximity verification.
        """
        # Execute individual pipeline predictions
        cv_boxes = self.cv.predict(image)
        yolo_boxes, yolo_confs = self.yolo.predict(image)
        
        boosted_confs = []
        verification_statuses = []

        for y_box, y_conf in zip(yolo_boxes, yolo_confs):
            yx1, yy1, yx2, yy2 = y_box
            
            # Calculate center points and dynamic scaling margin
            y_cx = (yx1 + yx2) / 2
            y_cy = (yy1 + yy2) / 2
            y_w = yx2 - yx1
            y_h = yy2 - yy1
            
            pad_x = y_w * dynamic_tolerance_pct
            pad_y = y_h * dynamic_tolerance_pct
            
            has_cv_confirmation = False

            for c_box in cv_boxes:
                cx1, cy1, cx2, cy2 = c_box
                
                extended_cx1 = cx1 - pad_x
                extended_cy1 = cy1 - pad_y
                extended_cx2 = cx2 + pad_x
                extended_cy2 = cy2 + pad_y
                
                # Check spatial overlap
                if (extended_cx1 <= y_cx <= extended_cx2) and (extended_cy1 <= y_cy <= extended_cy2):
                    has_cv_confirmation = True
                    break

            if has_cv_confirmation:
                boosted_confs.append(min(1.0, y_conf + boost_value))
                verification_statuses.append("YOLO + CV Boost")
            else:
                boosted_confs.append(y_conf)
                verification_statuses.append("YOLO Only")

        return yolo_boxes, boosted_confs, cv_boxes, verification_statuses