import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QComboBox, QSplitter,
                             QMenu, QAction, QToolButton)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QCursor

from core.camera_manager import CameraManager
from ui.camera_view import CameraViewWidget

logger = logging.getLogger("FOD.MultiCameraView")


class CameraGridWidget(QWidget):
    """
    Widget that displays multiple cameras in a grid layout
    """

    # Signal when a camera is selected
    camera_selected = pyqtSignal(str)  # camera_id

    # Signal when a frame is clicked
    frame_clicked = pyqtSignal(str, int, int)  # camera_id, x, y

    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)

        self.camera_manager = camera_manager
        self.camera_views = {}  # Dictionary of camera_id -> CameraViewWidget
        self.layout_mode = "grid"  # "grid" or "single"
        self.grid_columns = 2
        self.selected_camera_id = None

        # Initialize UI
        self.init_ui()

        # Add connection listener
        self.camera_manager.add_connection_listener(self.on_camera_connection_changed)

    def init_ui(self):
        """Initialize the user interface"""
        # Create main layout
        self.main_layout = QVBoxLayout(self)

        # Create toolbar
        toolbar_layout = QHBoxLayout()

        # Layout selection dropdown
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Grid View", "grid")
        self.layout_combo.addItem("Single View", "single")
        self.layout_combo.currentIndexChanged.connect(self.change_layout_mode)
        toolbar_layout.addWidget(QLabel("View Mode:"))
        toolbar_layout.addWidget(self.layout_combo)

        # Grid size selection (for grid layout)
        self.grid_size_combo = QComboBox()
        self.grid_size_combo.addItem("2×2", 2)
        self.grid_size_combo.addItem("3×3", 3)
        self.grid_size_combo.addItem("4×4", 4)
        self.grid_size_combo.currentIndexChanged.connect(self.change_grid_size)
        toolbar_layout.addWidget(QLabel("Grid Size:"))
        toolbar_layout.addWidget(self.grid_size_combo)

        # Camera selection dropdown (for single view)
        self.camera_combo = QComboBox()
        self.update_camera_combo()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected_from_combo)
        toolbar_layout.addWidget(QLabel("Camera:"))
        toolbar_layout.addWidget(self.camera_combo)

        toolbar_layout.addStretch()

        self.main_layout.addLayout(toolbar_layout)

        # Create grid container
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(4)
        self.main_layout.addWidget(self.grid_container)

        # Create single view container
        self.single_container = QWidget()
        self.single_layout = QVBoxLayout(self.single_container)
        self.main_layout.addWidget(self.single_container)

        # Initially hide the single view container
        self.single_container.hide()

        # Set initial layout mode
        self.set_layout_mode("grid")

    def update_camera_combo(self):
        """Update the camera selection combo box"""
        # Save current selection
        current_camera_id = self.camera_combo.currentData()

        # Clear combo
        self.camera_combo.clear()

        # Add all cameras
        for camera_id, info in self.camera_manager.get_all_cameras().items():
            name = info["name"]
            status = " (Connected)" if info["connected"] else " (Disconnected)"
            self.camera_combo.addItem(f"{name}{status}", camera_id)

        # Restore selection if possible
        if current_camera_id:
            index = self.camera_combo.findData(current_camera_id)
            if index >= 0:
                self.camera_combo.setCurrentIndex(index)

    def set_layout_mode(self, mode: str):
        """
        Set layout mode ('grid' or 'single')

        Args:
            mode: Layout mode
        """
        if mode not in ["grid", "single"]:
            return

        self.layout_mode = mode

        # Update combo selection
        index = self.layout_combo.findData(mode)
        if index >= 0:
            self.layout_combo.setCurrentIndex(index)

        # Update visibility
        if mode == "grid":
            self.grid_container.show()
            self.single_container.hide()
            self.camera_combo.setEnabled(False)
            self.grid_size_combo.setEnabled(True)
            self.refresh_grid()
        else:  # single
            self.grid_container.hide()
            self.single_container.show()
            self.camera_combo.setEnabled(True)
            self.grid_size_combo.setEnabled(False)
            self.refresh_single_view()

    def change_layout_mode(self, index: int):
        """
        Handle layout mode selection change

        Args:
            index: Selected index in combo box
        """
        mode = self.layout_combo.itemData(index)
        self.set_layout_mode(mode)

    def change_grid_size(self, index: int):
        """
        Handle grid size selection change

        Args:
            index: Selected index in combo box
        """
        self.grid_columns = self.grid_size_combo.itemData(index)
        if self.layout_mode == "grid":
            self.refresh_grid()

    # In MultiCameraView class
    def refresh_grid(self):
        """Refresh the grid layout with current cameras"""
        # Stop processing during refresh
        processing_was_active = False
        if hasattr(self, 'processing_timer') and self.processing_timer.isActive():
            processing_was_active = True
            self.processing_timer.stop()

        # Clear grid layout in the main thread
        self._clear_layout(self.grid_layout)

        # Get all cameras
        cameras = self.camera_manager.get_all_cameras()
        if not cameras:
            # Add placeholder if no cameras
            placeholder = QLabel("No cameras available. Add cameras in the Camera Manager tab.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
            self.grid_layout.addWidget(placeholder, 0, 0)
            return

        # Add cameras to grid in the main thread
        row = 0
        col = 0
        for camera_id, camera_info in cameras.items():
            # Check if we need a new camera view
            create_new_view = False
            if camera_id not in self.camera_views:
                create_new_view = True
            else:
                # Check if the existing view is still valid
                try:
                    view = self.camera_views[camera_id]
                    # Try to access a property to verify the widget still exists
                    _ = view.parent()
                except RuntimeError:
                    # Widget has been deleted
                    create_new_view = True

            if create_new_view:
                # Create new camera view widget
                camera = self.camera_manager.get_camera(camera_id)
                if camera:
                    view = CameraViewWidget(camera, None)
                    view.frame_clicked.connect(lambda x, y, cid=camera_id: self.frame_clicked.emit(cid, x, y))
                    view.setMinimumSize(320, 240)

                    # Add context menu
                    view.setContextMenuPolicy(Qt.CustomContextMenu)
                    view.customContextMenuRequested.connect(
                        lambda pos, cid=camera_id: self._show_camera_context_menu(pos, cid))

                    # Set click handler
                    original_mouse_press = view.image_label.mousePressEvent

                    def new_mouse_press(event, cid=camera_id, original=original_mouse_press):
                        # Call the original handler first
                        original(event)
                        # Then handle our selection logic
                        if event.button() == Qt.LeftButton:
                            self.select_camera(cid)

                    view.image_label.mousePressEvent = new_mouse_press

                    self.camera_views[camera_id] = view

            # Get the view widget
            view = self.camera_views.get(camera_id)
            if view:
                try:
                    # Add the view to the grid
                    frame = QWidget()
                    frame_layout = QVBoxLayout(frame)
                    frame_layout.setContentsMargins(2, 2, 2, 2)

                    # Add camera view
                    frame_layout.addWidget(view)

                    # Add camera name label
                    name_label = QLabel(camera_info["name"])
                    name_label.setAlignment(Qt.AlignCenter)
                    name_label.setStyleSheet("background-color: rgba(0, 0, 0, 50%); color: white; padding: 4px;")
                    frame_layout.addWidget(name_label)

                    # Add to grid
                    self.grid_layout.addWidget(frame, row, col)

                    # Update position
                    col += 1
                    if col >= self.grid_columns:
                        col = 0
                        row += 1
                except RuntimeError as e:
                    # Handle case where adding widget fails
                    logger.warning(f"Failed to add camera view for {camera_id}: {e}")
                    if camera_id in self.camera_views:
                        del self.camera_views[camera_id]

        # Resume processing if it was active
        if processing_was_active:
            self.processing_timer.start()

    def refresh_single_view(self):
        """Refresh the single camera view"""
        # Clear single view layout
        self._clear_layout(self.single_layout)

        # If no camera is selected, use the active camera
        if not self.selected_camera_id:
            self.selected_camera_id = self.camera_manager.active_camera_id

        # If still no camera, show placeholder
        if not self.selected_camera_id:
            placeholder = QLabel("No camera selected")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
            self.single_layout.addWidget(placeholder)
            return

        # Get or create camera view
        camera_id = self.selected_camera_id
        create_new_view = False

        if camera_id not in self.camera_views:
            create_new_view = True
        else:
            # Check if the existing view is still valid
            try:
                view = self.camera_views[camera_id]
                # Try to access a property to verify the widget still exists
                _ = view.parent()
            except RuntimeError:
                create_new_view = True

        if create_new_view:
            camera = self.camera_manager.get_camera(camera_id)
            if camera:
                view = CameraViewWidget(camera, None)
                view.frame_clicked.connect(lambda x, y: self.frame_clicked.emit(camera_id, x, y))
                self.camera_views[camera_id] = view

        # Add the view
        view = self.camera_views.get(camera_id)
        if view:
            try:
                self.single_layout.addWidget(view)
            except RuntimeError as e:
                logger.warning(f"Failed to add camera view to single view: {e}")
                if camera_id in self.camera_views:
                    del self.camera_views[camera_id]

    def select_camera(self, camera_id: str):
        """
        Select a camera (make it the active one)

        Args:
            camera_id: ID of camera to select
        """
        if camera_id not in self.camera_manager.cameras:
            return

        # Update selected camera
        self.selected_camera_id = camera_id

        # Update camera manager's active camera
        self.camera_manager.set_active_camera(camera_id)

        # Update combo box selection
        index = self.camera_combo.findData(camera_id)
        if index >= 0:
            self.camera_combo.setCurrentIndex(index)

        # Switch to single view if in grid mode
        if self.layout_mode == "grid":
            self.set_layout_mode("single")
        else:
            # Just refresh the single view
            self.refresh_single_view()

        # Emit signal
        self.camera_selected.emit(camera_id)

    def on_camera_selected_from_combo(self, index: int):
        """
        Handle camera selection from combo box

        Args:
            index: Selected index in combo box
        """
        if index < 0:
            return

        camera_id = self.camera_combo.itemData(index)
        if camera_id and camera_id != self.selected_camera_id:
            self.select_camera(camera_id)

    def on_camera_connection_changed(self, camera_id: str, is_connected: bool):
        """
        Handle camera connection status changes

        Args:
            camera_id: ID of the camera
            is_connected: New connection state
        """
        # Update camera combo
        self.update_camera_combo()

        # Refresh views if needed
        if self.layout_mode == "grid":
            self.refresh_grid()
        elif self.layout_mode == "single" and camera_id == self.selected_camera_id:
            self.refresh_single_view()

    def update_frame(self, camera_id: str, frame: np.ndarray):
        """
        Update the frame for a specific camera

        Args:
            camera_id: ID of the camera
            frame: The new frame
        """
        view = self.camera_views.get(camera_id)
        if view and hasattr(view, 'update_frame'):
            try:
                view.update_frame(frame)
            except (RuntimeError, AttributeError) as e:
                # Handle case where view was deleted
                logger.debug(f"Failed to update frame for camera {camera_id}: {e}")
                # Remove the invalid view from the dictionary
                if camera_id in self.camera_views:
                    del self.camera_views[camera_id]

    def update_all_frames(self):
        """Update frames for all cameras"""
        for camera_id, view in self.camera_views.items():
            camera = self.camera_manager.get_camera(camera_id)
            if camera and camera.connection_ok:
                frame = camera.get_frame()
                view.update_frame(frame)

    def _clear_layout(self, layout):
        """
        Clear all widgets from a layout

        Args:
            layout: Layout to clear
        """
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                # If item is a layout
                if item.layout():
                    self._clear_layout(item.layout())

    def _show_camera_context_menu(self, pos: QPoint, camera_id: str):
        """
        Show context menu for a camera

        Args:
            pos: Position to show menu
            camera_id: ID of the camera
        """
        menu = QMenu(self)

        # Get camera info
        camera_info = self.camera_manager.get_all_cameras().get(camera_id, {})
        if not camera_info:
            return

        # Camera name heading
        title_action = QAction(f"Camera: {camera_info['name']}", self)
        title_action.setEnabled(False)
        title_font = title_action.font()
        title_font.setBold(True)
        title_action.setFont(title_font)
        menu.addAction(title_action)
        menu.addSeparator()

        # Select action
        select_action = QAction("Select Camera", self)
        select_action.triggered.connect(lambda: self.select_camera(camera_id))
        menu.addAction(select_action)

        # Connection actions
        if camera_info.get("connected", False):
            disconnect_action = QAction("Disconnect", self)
            disconnect_action.triggered.connect(lambda: self.camera_manager.disconnect_camera(camera_id))
            menu.addAction(disconnect_action)
        else:
            connect_action = QAction("Connect", self)
            connect_action.triggered.connect(lambda: self.camera_manager.connect_camera(camera_id))
            menu.addAction(connect_action)

        menu.addSeparator()

        # View settings
        view = self.camera_views.get(camera_id)
        if view:
            if view.show_detections:
                hide_detections_action = QAction("Hide Detections", self)
                hide_detections_action.triggered.connect(lambda: self._toggle_detections(camera_id, False))
                menu.addAction(hide_detections_action)
            else:
                show_detections_action = QAction("Show Detections", self)
                show_detections_action.triggered.connect(lambda: self._toggle_detections(camera_id, True))
                menu.addAction(show_detections_action)

            if view.show_rois:
                hide_rois_action = QAction("Hide ROIs", self)
                hide_rois_action.triggered.connect(lambda: self._toggle_rois(camera_id, False))
                menu.addAction(hide_rois_action)
            else:
                show_rois_action = QAction("Show ROIs", self)
                show_rois_action.triggered.connect(lambda: self._toggle_rois(camera_id, True))
                menu.addAction(show_rois_action)

        menu.addSeparator()

        # Take snapshot
        snapshot_action = QAction("Take Snapshot", self)
        snapshot_action.triggered.connect(lambda: self._take_snapshot(camera_id))
        menu.addAction(snapshot_action)

        # Show menu
        menu.exec_(QCursor.pos())

    def _toggle_detections(self, camera_id: str, show: bool):
        """Toggle detection display for a camera"""
        view = self.camera_views.get(camera_id)
        if view:
            view.show_detections = show

    def _toggle_rois(self, camera_id: str, show: bool):
        """Toggle ROI display for a camera"""
        view = self.camera_views.get(camera_id)
        if view:
            view.show_rois = show

    def _take_snapshot(self, camera_id: str):
        """Take a snapshot from a camera"""
        camera = self.camera_manager.get_camera(camera_id)
        if camera and camera.connection_ok:
            from core.alert_manager import AlertManager
            # This is just a placeholder implementation - in real code
            # you would likely want to dispatch this to a dedicated snapshot manager
            alert_manager = AlertManager()
            frame = camera.get_frame()
            if frame is not None:
                snapshot_path = alert_manager.save_snapshot(frame)
                # Provide feedback
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Snapshot Taken", f"Snapshot saved to {snapshot_path}")