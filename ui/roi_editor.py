import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QDialog, QFormLayout, QLineEdit, QDoubleSpinBox,
                             QColorDialog, QSpinBox, QMessageBox, QSplitter,
                             QGroupBox, QCheckBox, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from core.video_source import VideoSource
from core.roi_manager import ROIManager, ROI
from core.detector import YOLODetector  # Import for class names only

logger = logging.getLogger("FOD.ROIEditor")


class ROIEditorWidget(QWidget):
    """
    Widget for creating and editing Regions of Interest (ROIs)
    """

    # Signal emitted when ROIs are changed
    rois_changed = pyqtSignal()

    def __init__(self, video_source: VideoSource, roi_manager: ROIManager, parent=None):
        super().__init__(parent)

        self.video_source = video_source
        self.roi_manager = roi_manager

        # ROI editing state
        self.creating_roi = False
        self.editing_roi = False
        self.selected_roi_index = None
        self.selected_point_index = None
        self.dragging = False
        self.resizing = False

        # Display frame and temporary frame for editing
        self.current_frame = None
        self.editor_frame = None

        # Setup UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        # Create main layout
        main_layout = QHBoxLayout(self)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: ROI list and properties
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # ROI list
        roi_group = QGroupBox("ROI List")
        roi_list_layout = QVBoxLayout(roi_group)

        self.roi_list = QListWidget()
        self.roi_list.setMinimumWidth(200)
        self.roi_list.currentRowChanged.connect(self.on_roi_selected)
        roi_list_layout.addWidget(self.roi_list)

        # ROI list buttons
        list_buttons_layout = QHBoxLayout()

        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.clicked.connect(self.start_roi_creation)
        list_buttons_layout.addWidget(self.add_roi_button)

        self.edit_roi_button = QPushButton("Edit")
        self.edit_roi_button.clicked.connect(self.edit_selected_roi)
        list_buttons_layout.addWidget(self.edit_roi_button)

        self.delete_roi_button = QPushButton("Delete")
        self.delete_roi_button.clicked.connect(self.delete_selected_roi)
        list_buttons_layout.addWidget(self.delete_roi_button)

        roi_list_layout.addLayout(list_buttons_layout)
        left_layout.addWidget(roi_group)

        # ROI properties
        properties_group = QGroupBox("ROI Properties")
        properties_layout = QFormLayout(properties_group)

        self.roi_name_edit = QLineEdit()
        properties_layout.addRow("Name:", self.roi_name_edit)

        self.roi_threshold_spin = QDoubleSpinBox()
        self.roi_threshold_spin.setMinimum(0.1)
        self.roi_threshold_spin.setMaximum(100)
        self.roi_threshold_spin.setSingleStep(0.1)
        self.roi_threshold_spin.setValue(1.0)
        properties_layout.addRow("Threshold:", self.roi_threshold_spin)

        self.roi_cooldown_spin = QSpinBox()
        self.roi_cooldown_spin.setMinimum(1)
        self.roi_cooldown_spin.setMaximum(3600)
        self.roi_cooldown_spin.setSingleStep(5)
        self.roi_cooldown_spin.setValue(60)
        properties_layout.addRow("Cooldown (s):", self.roi_cooldown_spin)

        self.roi_color_button = QPushButton()
        self.roi_color_button.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.roi_color_button.clicked.connect(self.select_roi_color)
        properties_layout.addRow("Color:", self.roi_color_button)

        # Classes filter
        classes_layout = QVBoxLayout()
        self.use_global_classes = QCheckBox("Use global classes")
        self.use_global_classes.setChecked(True)
        self.use_global_classes.stateChanged.connect(self.toggle_classes_filter)
        classes_layout.addWidget(self.use_global_classes)

        self.classes_combo = QComboBox()
        self.classes_combo.setEnabled(False)
        self.classes_combo.addItem("All Classes")

        # Get class names dynamically
        self.update_class_combo()

        classes_layout.addWidget(self.classes_combo)

        if hasattr(self.roi_manager, 'class_manager') and self.roi_manager.class_manager:
            class_dict = {}

            # Get all classes from class manager
            for class_info in self.roi_manager.class_manager.get_all_classes():
                class_id = class_info["class_id"]
                class_dict[class_id] = class_info["class_name"]
        else:
            # Fallback to hard-coded class names if class manager not available
            class_dict = YOLODetector.get_class_names()

        # Sort by class_id
        for class_id, class_name in sorted(class_dict.items()):
            self.classes_combo.addItem(f"{class_id}: {class_name}")

        # Add class names using the class method instead of creating an instance
        class_dict = YOLODetector.get_class_names()
        classes_layout.addWidget(self.classes_combo)

        self.add_class_button = QPushButton("Add Class")
        self.add_class_button.setEnabled(False)
        self.add_class_button.clicked.connect(self.add_roi_class)
        classes_layout.addWidget(self.add_class_button)

        self.selected_classes_list = QListWidget()
        self.selected_classes_list.setEnabled(False)
        self.selected_classes_list.setMaximumHeight(100)
        classes_layout.addWidget(self.selected_classes_list)

        properties_layout.addRow("Classes:", classes_layout)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_roi_properties)
        properties_layout.addRow("", self.apply_button)

        left_layout.addWidget(properties_group)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.clear_button = QPushButton("Clear All ROIs")
        self.clear_button.clicked.connect(self.clear_all_rois)
        controls_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save ROIs")
        self.save_button.clicked.connect(self.save_rois)
        controls_layout.addWidget(self.save_button)

        left_layout.addLayout(controls_layout)

        # Right panel: Camera view
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Instructions
        instructions = QLabel("Left-click to add points. Right-click to complete ROI.")
        instructions.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(instructions)

        # Camera view
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(960, 540)
        self.camera_view.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.camera_view)

        # Editing controls
        edit_controls = QHBoxLayout()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_roi_edit)
        edit_controls.addWidget(self.cancel_button)

        self.complete_button = QPushButton("Complete ROI")
        self.complete_button.clicked.connect(self.complete_roi)
        edit_controls.addWidget(self.complete_button)

        edit_controls.addStretch()

        self.mode_label = QLabel("Mode: Viewing")
        edit_controls.addWidget(self.mode_label)

        right_layout.addLayout(edit_controls)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([300, 700])

        # Make the camera view accept mouse events
        self.camera_view.setMouseTracking(True)
        self.camera_view.mousePressEvent = self.on_mouse_press
        self.camera_view.mouseMoveEvent = self.on_mouse_move
        self.camera_view.mouseReleaseEvent = self.on_mouse_release

        # Update the ROI list
        self.refresh_roi_list()

    def update_class_combo(self):
        """Update the class combo box with current class names"""
        # Save current selection
        current_text = self.classes_combo.currentText() if self.classes_combo.count() > 0 else ""

        # Clear combo
        self.classes_combo.clear()
        self.classes_combo.addItem("All Classes")

        # Get class names
        class_dict = {}

        # Try to get class names from class manager
        if hasattr(self.roi_manager, 'class_manager') and self.roi_manager.class_manager:
            for class_info in self.roi_manager.class_manager.get_all_classes():
                class_id = class_info["class_id"]
                class_dict[class_id] = class_info["class_name"]
        else:
            # Fallback to detector's class names
            class_dict = YOLODetector.get_class_names()

        # Add all classes, sorted by ID
        for class_id, class_name in sorted(class_dict.items()):
            self.classes_combo.addItem(f"{class_id}: {class_name}")

        # Restore selection if possible
        if current_text:
            index = self.classes_combo.findText(current_text)
            if index >= 0:
                self.classes_combo.setCurrentIndex(index)

    def update_frame(self, frame):
        """
        Update the displayed frame

        Args:
            frame: The new frame to display
        """
        if frame is None:
            return

        self.current_frame = frame.copy()

        # Draw current ROI points or editing state
        self.editor_frame = self.current_frame.copy()

        if self.creating_roi and self.roi_manager.current_roi_points:
            # Draw ROI being created
            points = np.array(self.roi_manager.current_roi_points, np.int32)
            cv2.polylines(self.editor_frame, [points.reshape((-1, 1, 2))],
                          False, (0, 255, 255), 2)

            # Draw points
            for pt in self.roi_manager.current_roi_points:
                cv2.circle(self.editor_frame, pt, 5, (0, 255, 255), -1)

        elif self.editing_roi and self.selected_roi_index is not None:
            # Draw ROI being edited
            roi = self.roi_manager.rois[self.selected_roi_index]
            points = np.array(roi.points, np.int32)
            cv2.polylines(self.editor_frame, [points.reshape((-1, 1, 2))],
                          True, roi.color, 2)

            # Draw points (larger for editing)
            for i, pt in enumerate(roi.points):
                # Highlight selected point
                color = (0, 0, 255) if i == self.selected_point_index else (255, 255, 0)
                cv2.circle(self.editor_frame, pt, 8, color, -1)

        # Display the frame
        self.display_frame(self.editor_frame)

    def display_frame(self, frame):
        """
        Convert and display a frame in the camera view

        Args:
            frame: The frame to display
        """
        # Convert the frame to QImage
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_image = QImage(frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Create a pixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)

        # Get current size of the label
        label_size = self.camera_view.size()

        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Set the pixmap to the label
        self.camera_view.setPixmap(scaled_pixmap)

    def refresh_roi_list(self):
        """Update the ROI list widget"""
        self.roi_list.clear()

        for i, roi in enumerate(self.roi_manager.rois):
            item = QListWidgetItem(f"{i + 1}: {roi.name}")
            # Set background color similar to ROI color
            color = QColor(*roi.color)
            # Make it lighter for visibility
            color.setAlpha(100)
            item.setBackground(color)
            self.roi_list.addItem(item)

    def start_roi_creation(self):
        """Start creating a new ROI"""
        self.creating_roi = True
        self.editing_roi = False
        self.roi_manager.current_roi_points = []
        self.mode_label.setText("Mode: Creating ROI")

    def start_edit_mode(self):
        """Start ROI edit mode"""
        self.editing_roi = True
        self.creating_roi = False
        self.mode_label.setText("Mode: Editing ROIs")

    def stop_edit_mode(self):
        """Stop ROI edit mode"""
        self.editing_roi = False
        self.creating_roi = False
        self.selected_roi_index = None
        self.selected_point_index = None
        self.mode_label.setText("Mode: Viewing")

    def on_roi_selected(self, row):
        """
        Handle ROI list selection

        Args:
            row: Selected row index
        """
        if row < 0 or row >= len(self.roi_manager.rois):
            self.selected_roi_index = None
            return

        self.selected_roi_index = row
        roi = self.roi_manager.rois[row]

        # Update properties panel
        self.roi_name_edit.setText(roi.name)
        self.roi_threshold_spin.setValue(roi.threshold)
        self.roi_cooldown_spin.setValue(roi.cooldown)

        # Update color button
        r, g, b = roi.color
        self.roi_color_button.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

        # Update classes list
        self.selected_classes_list.clear()
        if roi.classes_of_interest is not None:
            self.use_global_classes.setChecked(False)
            self.selected_classes_list.setEnabled(True)
            self.classes_combo.setEnabled(True)
            self.add_class_button.setEnabled(True)

            # Add classes to list
            class_dict = YOLODetector.get_class_names()
            for class_id in roi.classes_of_interest:
                class_name = class_dict.get(class_id, f"Unknown-{class_id}")
                self.selected_classes_list.addItem(f"{class_id}: {class_name}")
        else:
            self.use_global_classes.setChecked(True)

    def edit_selected_roi(self):
        """Start editing the selected ROI"""
        if self.selected_roi_index is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to edit.")
            return

        self.editing_roi = True
        self.creating_roi = False
        self.mode_label.setText(f"Mode: Editing ROI {self.selected_roi_index + 1}")

    def delete_selected_roi(self):
        """Delete the selected ROI"""
        if self.selected_roi_index is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to delete.")
            return

        reply = QMessageBox.question(self, "Delete ROI",
                                     f"Are you sure you want to delete ROI {self.selected_roi_index + 1}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.roi_manager.remove_roi(self.selected_roi_index)
            self.refresh_roi_list()
            self.selected_roi_index = None
            self.selected_point_index = None
            self.rois_changed.emit()

    def clear_all_rois(self):
        """Clear all ROIs"""
        if not self.roi_manager.rois:
            return

        reply = QMessageBox.question(self, "Clear All ROIs",
                                     "Are you sure you want to clear all ROIs?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.roi_manager.clear_all_rois()
            self.refresh_roi_list()
            self.selected_roi_index = None
            self.selected_point_index = None
            self.rois_changed.emit()

    def save_rois(self):
        """Save ROI configuration to file"""
        success = self.roi_manager.save_config()
        if success:
            QMessageBox.information(self, "ROI Configuration", "ROI configuration saved successfully.")
        else:
            QMessageBox.warning(self, "ROI Configuration", "Failed to save ROI configuration.")

    def apply_roi_properties(self):
        """Apply ROI properties from the form"""
        if self.selected_roi_index is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to apply properties.")
            return

        roi = self.roi_manager.rois[self.selected_roi_index]

        # Update ROI properties
        roi.name = self.roi_name_edit.text()
        roi.threshold = self.roi_threshold_spin.value()
        roi.cooldown = self.roi_cooldown_spin.value()

        # Update classes of interest
        if self.use_global_classes.isChecked():
            roi.classes_of_interest = None
        else:
            roi.classes_of_interest = []
            for i in range(self.selected_classes_list.count()):
                item_text = self.selected_classes_list.item(i).text()
                class_id = int(item_text.split(":")[0])
                roi.classes_of_interest.append(class_id)

        # Refresh the list to update display
        self.refresh_roi_list()
        self.rois_changed.emit()

        QMessageBox.information(self, "ROI Properties", f"Properties for ROI {roi.name} applied successfully.")

    def select_roi_color(self):
        """Open color selection dialog"""
        if self.selected_roi_index is None:
            return

        roi = self.roi_manager.rois[self.selected_roi_index]
        current_color = QColor(*roi.color)

        color = QColorDialog.getColor(current_color, self, "Select ROI Color")
        if color.isValid():
            roi.color = (color.red(), color.green(), color.blue())
            self.roi_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.refresh_roi_list()

    def toggle_classes_filter(self, state):
        """Toggle between global and custom classes for ROI"""
        is_custom = not bool(state)
        self.selected_classes_list.setEnabled(is_custom)
        self.classes_combo.setEnabled(is_custom)
        self.add_class_button.setEnabled(is_custom)

    def add_roi_class(self):
        """Add a class to the ROI's classes of interest"""
        if self.classes_combo.currentIndex() == 0:
            # "All Classes" selected
            return

        # Get selected class
        class_text = self.classes_combo.currentText()

        # Check if already in list
        for i in range(self.selected_classes_list.count()):
            if self.selected_classes_list.item(i).text() == class_text:
                return

        # Add to list
        self.selected_classes_list.addItem(class_text)

    def on_mouse_press(self, event):
        """
        Handle mouse press events

        Args:
            event: Mouse event
        """
        if self.current_frame is None:
            return

        # Calculate position in image
        x, y = self.get_image_position(event)

        if event.button() == Qt.LeftButton:
            if self.creating_roi:
                # Add point to current ROI being created
                self.roi_manager.current_roi_points.append((x, y))
                logger.info(f"Added ROI point: ({x}, {y})")
            elif self.editing_roi and self.selected_roi_index is not None:
                # Check if clicking on an existing point
                roi = self.roi_manager.rois[self.selected_roi_index]

                for i, pt in enumerate(roi.points):
                    # Check if point is close to click
                    if np.linalg.norm(np.array(pt) - np.array((x, y))) < 10:
                        self.selected_point_index = i
                        self.dragging = True
                        logger.info(f"Selected point {i} of ROI {self.selected_roi_index}")
                        return

                # If not clicking on a point, check if inside ROI for resizing
                roi = self.roi_manager.rois[self.selected_roi_index]
                if roi.contains_point((x, y)):
                    self.resizing = True
                    self.resizing_start_point = (x, y)
                    self.resizing_original_points = roi.points.copy()
                    self.resizing_center = roi.get_center()
                    logger.info(f"Starting resize of ROI {self.selected_roi_index}")
        elif event.button() == Qt.RightButton:
            if self.creating_roi:
                # Complete ROI with at least 3 points
                if len(self.roi_manager.current_roi_points) >= 3:
                    self.complete_roi()
                else:
                    QMessageBox.warning(self, "ROI Creation", "ROI must have at least 3 points.")

    def on_mouse_move(self, event):
        """
        Handle mouse move events

        Args:
            event: Mouse event
        """
        if self.current_frame is None:
            return

        # Calculate position in image
        x, y = self.get_image_position(event)

        if self.dragging and self.selected_roi_index is not None and self.selected_point_index is not None:
            # Move the selected point
            self.roi_manager.rois[self.selected_roi_index].points[self.selected_point_index] = (x, y)
        elif self.resizing and self.selected_roi_index is not None:
            # Resize the ROI
            roi = self.roi_manager.rois[self.selected_roi_index]

            # Calculate distance from center to current point vs. start point
            current_dist = np.linalg.norm(np.array((x, y)) - np.array(self.resizing_center))
            start_dist = np.linalg.norm(np.array(self.resizing_start_point) - np.array(self.resizing_center))

            # Calculate scale factor
            if start_dist > 0:
                scale = current_dist / start_dist

                # Apply scaling to all points
                new_points = []
                for pt in self.resizing_original_points:
                    # Get vector from center to point
                    vector = np.array(pt) - np.array(self.resizing_center)
                    # Scale vector
                    scaled_vector = vector * scale
                    # Calculate new point
                    new_pt = np.array(self.resizing_center) + scaled_vector
                    new_points.append((int(new_pt[0]), int(new_pt[1])))

                roi.points = new_points

    def on_mouse_release(self, event):
        """
        Handle mouse release events

        Args:
            event: Mouse event
        """
        if self.dragging or self.resizing:
            self.dragging = False
            self.resizing = False
            self.resizing_start_point = None
            self.resizing_original_points = None
            self.selected_point_index = None

            # Emit signal that ROIs changed
            self.rois_changed.emit()

    def complete_roi(self):
        """Complete the current ROI creation"""
        if not self.creating_roi or len(self.roi_manager.current_roi_points) < 3:
            return

        # Create a dialog to get ROI properties
        dialog = QDialog(self)
        dialog.setWindowTitle("New ROI Properties")

        dialog_layout = QFormLayout(dialog)

        # ROI name
        name_edit = QLineEdit(f"ROI {len(self.roi_manager.rois) + 1}")
        dialog_layout.addRow("Name:", name_edit)

        # Threshold
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setMinimum(0.1)
        threshold_spin.setMaximum(100)
        threshold_spin.setSingleStep(0.1)
        threshold_spin.setValue(1.0)
        dialog_layout.addRow("Threshold:", threshold_spin)

        # Cooldown
        cooldown_spin = QSpinBox()
        cooldown_spin.setMinimum(1)
        cooldown_spin.setMaximum(3600)
        cooldown_spin.setSingleStep(5)
        cooldown_spin.setValue(60)
        dialog_layout.addRow("Cooldown (s):", cooldown_spin)

        # Color
        color_button = QPushButton()
        color_button.setStyleSheet("background-color: rgb(255, 0, 0);")
        color = QColor(255, 0, 0)

        def select_color():
            nonlocal color
            new_color = QColorDialog.getColor(color, dialog, "Select ROI Color")
            if new_color.isValid():
                color = new_color
                color_button.setStyleSheet(f"background-color: {color.name()};")

        color_button.clicked.connect(select_color)
        dialog_layout.addRow("Color:", color_button)

        # Buttons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        buttons_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_button)

        dialog_layout.addRow("", buttons_layout)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Create ROI
            roi = ROI(
                name=name_edit.text(),
                points=self.roi_manager.current_roi_points.copy(),
                threshold=threshold_spin.value(),
                cooldown=cooldown_spin.value(),
                color=(color.red(), color.green(), color.blue())
            )

            # Add to manager
            self.roi_manager.add_roi(roi)

            # Reset current points
            self.roi_manager.current_roi_points = []

            # Update UI
            self.refresh_roi_list()
            self.creating_roi = False
            self.mode_label.setText("Mode: Viewing")

            # Emit signal that ROIs changed
            self.rois_changed.emit()
        else:
            # Just reset current points
            self.roi_manager.current_roi_points = []
            self.creating_roi = False
            self.mode_label.setText("Mode: Viewing")

    def cancel_roi_edit(self):
        """Cancel the current ROI creation or editing"""
        if self.creating_roi:
            self.roi_manager.current_roi_points = []
            self.creating_roi = False

        if self.editing_roi:
            self.editing_roi = False
            self.selected_point_index = None

        self.mode_label.setText("Mode: Viewing")


    def get_image_position(self, event):
        """
        Convert mouse position to image coordinates

        Args:
            event: Mouse event

        Returns:
            Tuple of (x, y) coordinates in the image
        """
        if self.current_frame is None:
            return (0, 0)

        # Get image dimensions
        height, width = self.current_frame.shape[:2]

        # Get label dimensions
        label_width = self.camera_view.width()
        label_height = self.camera_view.height()

        # Get pixmap dimensions (the scaled image)
        pixmap = self.camera_view.pixmap()
        if pixmap is None:
            return (0, 0)

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Calculate margins (if the image is centered in the label)
        margin_x = (label_width - pixmap_width) // 2
        margin_y = (label_height - pixmap_height) // 2

        # Adjust for margin
        pos_x = event.x() - margin_x
        pos_y = event.y() - margin_y

        # Check if click is within the image area
        if pos_x < 0 or pos_x >= pixmap_width or pos_y < 0 or pos_y >= pixmap_height:
            return (0, 0)

        # Convert to original image coordinates using ratio
        image_x = int(pos_x * (width / pixmap_width))
        image_y = int(pos_y * (height / pixmap_height))

        # Clamp to image bounds
        image_x = max(0, min(width - 1, image_x))
        image_y = max(0, min(height - 1, image_y))

        return (image_x, image_y)