import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger("FOD.Detector")


class YOLODetector:
    """
    Wrapper for YOLO model to perform object detection
    """
    # Class-level dictionary for class names - moved from instance method to class level
    CLASS_NAMES = {
        0: "AdjustableWrench_01", 1: "AdjustableWrench", 2: "Battery",
        3: "Bolt", 4: "BoltNutSet", 5: "BoltWasher",
        6: "ClampPart", 7: "Cutter", 8: "FuelCap",
        9: "Hammer", 10: "Hose", 11: "Label",
        12: "LuggagePart", 13: "LuggageTag", 14: "MetalPart",
        15: "MetalSheet", 16: "Nail", 17: "Nut",
        18: "PaintChip", 19: "Pen", 20: "PlasticPart",
        21: "Pliers", 22: "Rock", 23: "Screw",
        24: "Screwdriver", 25: "SodaCan", 26: "Tape",
        27: "Washer", 28: "Wire", 29: "Wood",
        30: "Wrench", 31: "Copper", 32: "Metallic shine",
        33: "Eyebolt", 34: "AsphaltCrack", 35: "FaucetHandle",
        36: "Tie-Wrap", 37: "Pitot cover", 38: "Scissors",
        39: "NutShell"
    }

    # Make sure the YOLODetector.__init__ in core/detector.py looks like this:

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25,
                 use_gpu: bool = True, classes_of_interest: Optional[List[int]] = None,
                 class_manager=None):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model file
            confidence: Confidence threshold for detections
            use_gpu: Whether to use GPU acceleration
            classes_of_interest: List of class IDs to detect (None = all classes)
            class_manager: Class manager instance for dynamic class mapping
        """
        self.model_path = model_path
        self.confidence = confidence
        self.use_gpu = use_gpu
        self.classes_of_interest = classes_of_interest
        self.model = None
        self.class_manager = class_manager
        self._dynamic_class_names = {}

        # Update dynamic class names if class manager is provided
        if self.class_manager:
            self._update_dynamic_class_names()

            # Register for class changes if class manager supports it
            if hasattr(self.class_manager, "add_listener"):
                self.class_manager.add_listener(self._handle_class_change)

        # Only try to load the model if a valid path is provided
        if model_path:
            self.load_model()

    def set_class_manager(self, class_manager):
        """Set the class manager for dynamic class handling"""
        self.class_manager = class_manager
        self._update_dynamic_class_names()

        # Register for class changes
        if hasattr(self.class_manager, "add_listener"):
            self.class_manager.add_listener(self._handle_class_change)

    def _handle_class_change(self, event):
        """Handle class change events"""
        # Update dynamic class names when classes change
        if event.action in ["add", "update", "delete", "import", "model_update"]:
            self._update_dynamic_class_names()
            logger.info("Updated detector class mappings due to class changes")

    def _update_dynamic_class_names(self):
        """Update dynamic class names from class manager"""
        if not self.class_manager:
            return

        try:
            # Get class names from class manager
            self._dynamic_class_names = {}

            # Get all classes
            for class_info in self.class_manager.get_all_classes():
                class_id = class_info["class_id"]
                self._dynamic_class_names[class_id] = class_info["class_name"]

            logger.info(f"Updated dynamic class names, found {len(self._dynamic_class_names)} classes")
        except Exception as e:
            logger.error(f"Error updating dynamic class names: {e}")

    @classmethod
    def get_class_names(cls) -> Dict[int, str]:
        """Class method to get the class names dictionary without initializing a model"""
        return cls.CLASS_NAMES

    def get_dynamic_class_names(self) -> Dict[int, str]:
        """Instance method to get the dynamic class names from class manager or model"""
        # If we have dynamic class names from class manager, use those
        if self._dynamic_class_names:
            return self._dynamic_class_names

        # If model is loaded, try to get class names from model
        if self.model and hasattr(self.model, 'names'):
            return self.model.names

        # Fall back to class-level dictionary
        return self.CLASS_NAMES

    # Update the load_model method to populate dynamic class names from model:

    def load_model(self):
        """Load YOLO model with appropriate device selection"""
        if not self.model_path:
            logger.warning("No model path provided, skipping model loading")
            return

        try:
            self.model = YOLO(self.model_path)

            if self.use_gpu and torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("YOLO Model loaded on GPU")
            else:
                self.model.to("cpu")
                logger.info("YOLO Model loaded on CPU")

            # Update dynamic class names from model if available
            if hasattr(self.model, 'names'):
                model_classes = {}
                for idx, name in self.model.names.items():
                    model_classes[int(idx)] = name

                # Only update if we don't have class manager
                if not self.class_manager:
                    self._dynamic_class_names = model_classes
                    logger.info(f"Updated class names from model: {len(model_classes)} classes")

                # If we have class manager, update it with model classes
                elif hasattr(self.class_manager, "update_from_model"):
                    # Extract model name from path
                    import os
                    model_name = os.path.splitext(os.path.basename(self.model_path))[0]

                    # Update class manager
                    self.class_manager.update_from_model(model_name, len(model_classes), model_classes)
                    logger.info(f"Updated class manager with {len(model_classes)} classes from model")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    # Update the detect method to use dynamic class names:

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform object detection on a frame

        Args:
            frame: The image frame to detect objects in

        Returns:
            List of detections, each with class_id, confidence, bbox, etc.
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []

        try:
            results = self.model(frame, conf=self.confidence, verbose=False)[0]
            detections = []

            if results.boxes is not None:
                boxes_data = results.boxes.data.cpu().numpy()

                # Get class names - prefer dynamic names from class manager
                class_names = self.get_dynamic_class_names()

                for box in boxes_data:
                    x1, y1, x2, y2, conf, class_id = box
                    class_id = int(class_id)

                    # Skip if not in classes of interest
                    if self.classes_of_interest is not None and class_id not in self.classes_of_interest:
                        continue

                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    detection = {
                        "class_id": class_id,
                        "class_name": class_names.get(class_id, f"Unknown-{class_id}"),
                        "confidence": float(conf),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "center": (center_x, center_y)
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]],
                        highlight_in_roi: Optional[List[int]] = None) -> np.ndarray:
        """
        Draw detection boxes on frame

        Args:
            frame: The original frame
            detections: List of detection dictionaries
            highlight_in_roi: List of indices of detections that are inside ROIs

        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            conf = det["confidence"]

            # Use green for detections in ROI, gray for others
            color = (0, 255, 0) if highlight_in_roi is not None and i in highlight_in_roi else (200, 200, 200)

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output_frame