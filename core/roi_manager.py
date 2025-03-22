import cv2
import numpy as np
import json
import logging
import os
from typing import List, Dict, Tuple, Any, Optional, Union

logger = logging.getLogger("FOD.ROIManager")


class ROI:
    """
    Region of Interest with detection settings
    """

    def __init__(self, name: str, points: List[Tuple[int, int]],
                 threshold: float = 1,
                 cooldown: int = 60,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 classes_of_interest: Optional[List[int]] = None):
        """
        Initialize a Region of Interest

        Args:
            name: Name of the ROI
            points: List of (x, y) points defining the ROI polygon
            threshold: Minimum number of objects to trigger an alert
            cooldown: Cooldown period between alerts (seconds)
            color: BGR color for ROI display
            classes_of_interest: List of class IDs for this ROI (None = use global)
        """
        self.name = name
        self.points = points
        self.threshold = threshold
        self.cooldown = cooldown
        self.color = color
        self.classes_of_interest = classes_of_interest

        # Runtime state
        self.last_alert_time = 0
        self.class_counts = {}
        self.current_detections = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert ROI to dictionary for serialization"""
        return {
            "name": self.name,
            "points": self.points,
            "threshold": self.threshold,
            "cooldown": self.cooldown,
            "color": self.color,
            "classes_of_interest": self.classes_of_interest
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ROI':
        """Create ROI from dictionary"""
        return cls(
            name=data.get("name", "Unnamed ROI"),
            points=data.get("points", []),
            threshold=data.get("threshold", 1),
            cooldown=data.get("cooldown", 60),
            color=data.get("color", (255, 0, 0)),
            classes_of_interest=data.get("classes_of_interest", None)
        )

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if the ROI contains a point"""
        if len(self.points) < 3:
            return False

        contour = np.array(self.points, dtype=np.int32)
        result = cv2.pointPolygonTest(contour, point, False)
        return result >= 0

    def get_center(self) -> Tuple[int, int]:
        """Get the center point of the ROI"""
        pts = np.array(self.points)
        center = np.mean(pts, axis=0)
        return (int(center[0]), int(center[1]))

    def reset_counts(self):
        """Reset object counts"""
        self.class_counts = {}
        self.current_detections = []

    def update_class_counts(self, detection: Dict[str, Any]):
        """Add a detection to the ROI counts"""
        class_id = detection["class_id"]
        self.class_counts[class_id] = self.class_counts.get(class_id, 0) + 1
        self.current_detections.append(detection)

    def should_alert(self, current_time: float) -> bool:
        """Check if this ROI should trigger an alert"""
        if not self.class_counts:
            return False

        total_objects = sum(self.class_counts.values())
        cooldown_elapsed = (current_time - self.last_alert_time) >= self.cooldown

        return total_objects >= self.threshold and cooldown_elapsed


class ROIManager:
    """
    Manages multiple Regions of Interest and their configurations
    """

    def __init__(self, config_file: str = "rois_config.json", class_manager=None):
        """
        Initialize the ROI Manager

        Args:
            config_file: Path to ROI configuration file
            class_manager: ClassManager instance for class lookups
        """
        self.config_file = config_file
        self.rois: List[ROI] = []
        self.class_manager = class_manager

        # Runtime state for ROI editing
        self.current_roi_points = []
        self.edit_mode = False
        self.selected_roi_index = None
        self.selected_point_index = None
        self.dragging = False
        self.resizing_mode = False
        self.resizing_roi_index = None
        self.initial_mouse_pos = None
        self.initial_roi_points = None

        # Add listener for class changes if class manager is provided
        if self.class_manager:
            self.class_manager.add_listener(self._handle_class_change)

        # Load existing ROIs if available
        self.load_config()

    def set_class_manager(self, class_manager):
        """Set the class manager for dynamic class handling"""
        self.class_manager = class_manager
        # Register for class change events
        if hasattr(class_manager, "add_listener"):
            class_manager.add_listener(self._handle_class_change)

    def _handle_class_change(self, event):
        """Handle class change events"""
        # Handle model update events to update ROIs
        if event.action == "model_update" and 'model_name' in event.data:
            model_name = event.data['model_name']
            self._update_rois_for_model_change(model_name)

    def _update_rois_for_model_change(self, new_model_name):
        """Update ROIs when the model changes"""
        if not hasattr(self, 'class_manager') or not self.class_manager:
            return

        for roi in self.rois:
            if roi.classes_of_interest:
                self._map_roi_classes(roi, new_model_name)

        # Save changes
        self.save_config()

    def _map_roi_classes(self, roi, target_model_name):
        """Map ROI classes to new model classes"""
        if not hasattr(self, 'class_manager') or not self.class_manager or not hasattr(self.class_manager, 'mapper'):
            return

        # Get all mappings that target the current model
        mapped_classes = []
        unmapped_classes = []

        for class_id in roi.classes_of_interest:
            mapped = False

            # Check all mapping sources
            for mapping_key in self.class_manager.mapper.mappings:
                if ":" in mapping_key and mapping_key.split(":")[1] == target_model_name:
                    source_model = mapping_key.split(":")[0]
                    mapped_id = self.class_manager.mapper.get_mapped_id(
                        source_model, target_model_name, class_id)

                    if mapped_id is not None and mapped_id >= 0:
                        mapped_classes.append(mapped_id)
                        mapped = True
                        break

            if not mapped:
                unmapped_classes.append(class_id)

        # Update ROI's classes of interest if we found mappings
        if mapped_classes:
            # Remove duplicates that might occur from multiple mappings
            all_classes = list(set(mapped_classes + unmapped_classes))
            roi.classes_of_interest = all_classes

    def add_roi(self, roi: ROI) -> int:
        """
        Add a new ROI

        Args:
            roi: The ROI object to add

        Returns:
            Index of the added ROI
        """
        self.rois.append(roi)
        logger.info(f"Added ROI '{roi.name}'. Total ROI count: {len(self.rois)}")
        return len(self.rois) - 1

    def remove_roi(self, index: int) -> bool:
        """
        Remove an ROI by index

        Args:
            index: Index of the ROI to remove

        Returns:
            True if removed successfully, False otherwise
        """
        if 0 <= index < len(self.rois):
            removed = self.rois.pop(index)
            logger.info(f"Removed ROI '{removed.name}'")
            return True
        return False

    def clear_all_rois(self):
        """Remove all ROIs"""
        self.rois = []
        logger.info("Cleared all ROIs")

    def load_config(self) -> bool:
        """
        Load ROI configuration from file

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.config_file):
            logger.warning(f"ROI configuration file {self.config_file} not found")
            return False

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.rois = [ROI.from_dict(roi_data) for roi_data in data]
            logger.info(f"Loaded {len(self.rois)} ROIs from {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ROI configuration: {e}")
            return False

    def save_config(self) -> bool:
        """
        Save ROI configuration to file

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = [roi.to_dict() for roi in self.rois]
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            logger.info(f"Saved {len(self.rois)} ROIs to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ROI configuration: {e}")
            return False

    def process_detections(self, detections: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
        """
        Process detections against all ROIs

        Args:
            detections: List of detection dictionaries from the detector

        Returns:
            Tuple containing:
            - List of detection indices that are inside any ROI
            - List of ROI indices that have detections
        """
        # Reset all ROI counts first
        for roi in self.rois:
            roi.reset_counts()

        detections_in_roi = []
        rois_with_detections = set()

        for i, detection in enumerate(detections):
            center_point = detection["center"]

            for roi_idx, roi in enumerate(self.rois):
                if roi.contains_point(center_point):
                    # Check if this class is of interest for this ROI
                    if roi.classes_of_interest is None or detection["class_id"] in roi.classes_of_interest:
                        detections_in_roi.append(i)
                        rois_with_detections.add(roi_idx)
                        roi.update_class_counts(detection)

        return detections_in_roi, list(rois_with_detections)

    # Update the draw_rois method in ROIManager class in core/roi_manager.py:

    def draw_rois(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """
        Draw all ROIs on the frame

        Args:
            frame: The frame to draw on
            show_labels: Whether to show ROI labels and detection counts

        Returns:
            Frame with ROIs drawn
        """
        output_frame = frame.copy()

        for idx, roi in enumerate(self.rois):
            # Draw the ROI polygon
            pts = np.array(roi.points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output_frame, [pts], isClosed=True, color=roi.color, thickness=2)

            if show_labels:
                # Draw ROI name and detection counts
                x, y, w, h = cv2.boundingRect(pts)

                # Create info text with detection counts
                roi_info = [f"{roi.name}"]

                if roi.class_counts:
                    # Get class names from class manager if available, otherwise use detector
                    if hasattr(self, 'class_manager') and self.class_manager:
                        class_names = {}
                        for class_info in self.class_manager.get_all_classes():
                            class_id = class_info["class_id"]
                            class_names[class_id] = class_info["class_name"]
                    else:
                        # Fallback to detector's class names
                        from core.detector import YOLODetector
                        class_names = YOLODetector.get_class_names()

                    for cls, count in roi.class_counts.items():
                        class_name = class_names.get(cls, str(cls))
                        roi_info.append(f"{class_name}: {count}")
                else:
                    roi_info.append("No detections")

                info_text = " | ".join(roi_info)

                # Draw semi-transparent background for text
                text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_frame, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y), (0, 0, 0, 128), -1)

                # Draw text
                cv2.putText(output_frame, info_text, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi.color, 2)

        # Draw ROI being created (if any)
        if self.current_roi_points and not self.edit_mode:
            pts = np.array(self.current_roi_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

            # Draw points
            for pt in self.current_roi_points:
                cv2.circle(output_frame, pt, 5, (0, 255, 255), -1)

        return output_frame

    def on_frame_clicked(self, point: Tuple[int, int]):
        """
        Handle frame clicks for ROI creation/editing

        Args:
            point: (x, y) coordinates of the click
        """
        # If we're creating a new ROI, add this point to the current points
        if hasattr(self, 'current_roi_points'):
            self.current_roi_points.append(point)
            logger.info(f"Added point {point} to current ROI")