import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QSizePolicy, QDialog,
                             QFormLayout, QLineEdit, QSpinBox, QApplication, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from core.video_source import VideoSource
from core.roi_manager import ROIManager
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger("FOD.CameraView")

class CameraConnectDialog(QDialog):
    """Dialog for connecting to a camera with transport protocol options"""

    def __init__(self, video_source=None, current_url="", parent=None):
        super().__init__(parent)
        self.video_source = video_source
        self.setWindowTitle("Connect to Camera")
        self.setMinimumWidth(500)

        # Create layout
        layout = QFormLayout(self)

        # Add URL field
        self.url_input = QLineEdit(current_url)
        layout.addRow("RTSP URL or File Path:", self.url_input)

        # Transport protocol selection
        self.transport_combo = QComboBox()
        self.transport_combo.addItem("TCP (More reliable)", "tcp")
        self.transport_combo.addItem("UDP (Lower latency)", "udp")
        self.transport_combo.addItem("Auto-detect (Recommended)", "auto")

        # Set default based on video source
        if video_source and hasattr(video_source, 'rtsp_transport'):
            if video_source.rtsp_transport == "tcp":
                self.transport_combo.setCurrentIndex(0)
            elif video_source.rtsp_transport == "udp":
                self.transport_combo.setCurrentIndex(1)
            else:
                self.transport_combo.setCurrentIndex(2)
        else:
            self.transport_combo.setCurrentIndex(2)  # Default to auto-detect

        layout.addRow("RTSP Transport:", self.transport_combo)

        # Add buffer size option
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(5, 100)
        self.buffer_size.setSingleStep(5)
        self.buffer_size.setValue(30)  # Default
        self.buffer_size.setSuffix(" frames")
        layout.addRow("Buffer Size:", self.buffer_size)

        # Add example label
        examples = QLabel("Examples:\n" +
                          "- RTSP: rtsp://admin:password@192.168.1.100:554/Stream/Channels/101\n" +
                          "- Local file: video.mp4\n" +
                          "- Webcam: 0 (use a number for local webcams)")
        examples.setStyleSheet("color: gray;")
        layout.addRow("", examples)

        # Add buttons
        button_layout = QHBoxLayout()

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.accept)
        button_layout.addWidget(self.connect_button)

        self.test_button = QPushButton("Test Connection")
        self.test_button.clicked.connect(self.test_connection)
        button_layout.addWidget(self.test_button)

        self.recommended_button = QPushButton("Find Best Settings")
        self.recommended_button.clicked.connect(self.find_recommended_settings)
        button_layout.addWidget(self.recommended_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addRow("", button_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        layout.addRow("", self.status_label)

    def get_url(self):
        """Get the entered URL"""
        return self.url_input.text().strip()

    def get_transport(self):
        """Get the selected transport protocol"""
        return self.transport_combo.currentData()

    def get_buffer_size(self):
        """Get the selected buffer size"""
        return self.buffer_size.value()

    def test_connection(self):
        """Test the connection with current settings"""
        url = self.get_url()
        if not url:
            QMessageBox.warning(self, "Empty URL", "Please enter a URL.")
            return

        transport = self.get_transport()
        self.status_label.setText(f"Testing connection with {transport} transport...")
        self.status_label.setStyleSheet("color: blue;")
        QApplication.processEvents()  # Update UI

        # Create a temporary VideoSource to test
        try:
            from core.video_source import VideoSource
            # If we have an existing VideoSource, use its class for testing
            if self.video_source:
                temp_video_source = VideoSource(
                    url,
                    resize_width=self.video_source.resize_width,
                    resize_height=self.video_source.resize_height,
                    rtsp_transport=transport if transport != "auto" else "tcp"
                )
            else:
                temp_video_source = VideoSource(url, rtsp_transport=transport if transport != "auto" else "tcp")

            success, message = temp_video_source.test_connection()

            if success:
                self.status_label.setText(f"Connection successful with {transport}!")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                QMessageBox.information(self, "Connection Test", "Connection successful!")
            else:
                self.status_label.setText(f"Connection failed: {message}")
                self.status_label.setStyleSheet("color: red;")
                QMessageBox.warning(self, "Connection Test", f"Connection failed: {message}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Connection Test Error", f"Error testing connection: {e}")

    def find_recommended_settings(self):
        """Find the recommended transport protocol by testing both TCP and UDP"""
        url = self.get_url()
        if not url:
            QMessageBox.warning(self, "Empty URL", "Please enter a URL.")
            return

        if not url.lower().startswith("rtsp://"):
            QMessageBox.information(self, "Transport Settings", "Transport settings only apply to RTSP streams.")
            return

        self.status_label.setText("Testing TCP and UDP to find optimal settings...")
        self.status_label.setStyleSheet("color: blue;")
        QApplication.processEvents()  # Update UI

        # First test TCP
        try:
            from core.video_source import VideoSource
            tcp_video_source = VideoSource(url, rtsp_transport="tcp")
            tcp_success, tcp_message = tcp_video_source.test_connection()

            # Then test UDP
            udp_video_source = VideoSource(url, rtsp_transport="udp")
            udp_success, udp_message = udp_video_source.test_connection()

            # Determine recommendation
            if tcp_success and udp_success:
                msg = "Both TCP and UDP work! UDP often has lower latency but TCP can be more reliable."
                recommended = "udp"  # Generally prefer UDP if both work
            elif tcp_success:
                msg = "TCP works well, but UDP failed."
                recommended = "tcp"
            elif udp_success:
                msg = "UDP works well, but TCP failed."
                recommended = "udp"
            else:
                msg = "Both TCP and UDP failed. Check your URL and network settings."
                recommended = "tcp"  # Default to TCP as fallback

            # Set recommendation in combo box
            if recommended == "tcp":
                self.transport_combo.setCurrentIndex(0)
            else:
                self.transport_combo.setCurrentIndex(1)

            # Update status
            self.status_label.setText(f"Recommendation: {recommended.upper()} - {msg}")
            if tcp_success or udp_success:
                self.status_label.setStyleSheet("color: green;")
                QMessageBox.information(self, "Transport Settings",
                                        f"Recommendation: Use {recommended.upper()}\n\n{msg}")
            else:
                self.status_label.setStyleSheet("color: red;")
                QMessageBox.warning(self, "Transport Settings", msg)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", f"Error testing connection: {e}")


class CameraViewWidget(QWidget):
    """
    Widget to display the camera feed with overlaid information
    """

    # Signal for when a frame is clicked
    frame_clicked = pyqtSignal(int, int)

    def __init__(self, video_source: VideoSource, roi_manager: ROIManager, parent=None):
        super().__init__(parent)

        self.video_source = video_source
        self.roi_manager = roi_manager

        # Image display variables
        self.current_frame = None
        self.zoom_factor = 1.0
        self.zoom_center = None

        # Display options
        self.show_detections = True
        self.show_rois = True
        self.show_info = True

        # Setup UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create image display with size policy to allow resizing
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(640, 480)  # Minimum size, will expand with window
        self.image_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.image_label)

        # Make QLabel use smooth scaling
        self.image_label.setScaledContents(False)  # We handle scaling manually with better quality

        # Create options layout
        options_layout = QHBoxLayout()

        # Add zoom controls
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedSize(30, 30)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        options_layout.addWidget(self.zoom_out_button)

        self.zoom_reset_button = QPushButton("100%")
        self.zoom_reset_button.setFixedSize(60, 30)
        self.zoom_reset_button.clicked.connect(self.zoom_reset)
        options_layout.addWidget(self.zoom_reset_button)

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setFixedSize(30, 30)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        options_layout.addWidget(self.zoom_in_button)

        options_layout.addStretch()

        # Add display options
        self.detection_check = QPushButton("Show Detections")
        self.detection_check.setCheckable(True)
        self.detection_check.setChecked(self.show_detections)
        self.detection_check.clicked.connect(self.toggle_detections)
        options_layout.addWidget(self.detection_check)

        self.roi_check = QPushButton("Show ROIs")
        self.roi_check.setCheckable(True)
        self.roi_check.setChecked(self.show_rois)
        self.roi_check.clicked.connect(self.toggle_rois)
        options_layout.addWidget(self.roi_check)

        self.info_check = QPushButton("Show Info")
        self.info_check.setCheckable(True)
        self.info_check.setChecked(self.show_info)
        self.info_check.clicked.connect(self.toggle_info)
        options_layout.addWidget(self.info_check)

        # Add layout to main layout
        main_layout.addLayout(options_layout)

        # Make the widget accept mouse events
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_mouse_press
        self.image_label.mouseMoveEvent = self.on_mouse_move
        self.image_label.wheelEvent = self.on_wheel

    def update_frame(self, frame):
        """
        Update the displayed frame

        Args:
            frame: The new frame to display (numpy array)
        """
        if frame is None:
            return

        # Check if widget is still valid
        if not hasattr(self, 'image_label') or self.image_label is None:
            return  # Skip update if label has been deleted

        try:
            self.current_frame = frame.copy()

            # Apply zoom if needed
            display_frame = self.apply_zoom(frame)

            # Add info overlay if enabled
            if self.show_info:
                display_frame = self.add_info_overlay(display_frame)

            # Convert frame to QImage
            height, width, channels = display_frame.shape
            bytes_per_line = channels * width
            q_image = QImage(display_frame.data, width, height,
                             bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Scale the image to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)

            # Get the size of the image label
            label_size = self.image_label.size()

            # Scale the pixmap to fit the label while preserving aspect ratio
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Set the scaled image to the label
            self.image_label.setPixmap(scaled_pixmap)
        except (RuntimeError, AttributeError) as e:
            # Handle the case where the label has been deleted
            pass

    def apply_zoom(self, frame):
        """
        Apply zoom to the frame

        Args:
            frame: Original frame

        Returns:
            Zoomed frame
        """
        if self.zoom_factor == 1.0:
            return frame

        height, width = frame.shape[:2]

        # Calculate zoom center if not set
        if self.zoom_center is None:
            self.zoom_center = (width // 2, height // 2)

        # Calculate the region to crop
        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)

        # Ensure zoom center is within frame bounds
        x_center = min(max(self.zoom_center[0], new_width // 2), width - new_width // 2)
        y_center = min(max(self.zoom_center[1], new_height // 2), height - new_height // 2)

        # Calculate crop coordinates
        x1 = max(0, x_center - new_width // 2)
        y1 = max(0, y_center - new_height // 2)
        x2 = min(width, x1 + new_width)
        y2 = min(height, y1 + new_height)

        # Crop the frame
        cropped = frame[y1:y2, x1:x2]

        # Resize back to original size
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def add_info_overlay(self, frame):
        """
        Add information overlay to the frame

        Args:
            frame: Original frame

        Returns:
            Frame with information overlay
        """
        overlay = frame.copy()

        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(overlay, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        # Add zoom info
        zoom_text = f"Zoom: {int(self.zoom_factor * 100)}%"
        cv2.putText(overlay, zoom_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        # Add connection info
        connection_text = "Connected" if self.video_source.connection_ok else "Disconnected"
        cv2.putText(overlay, connection_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0) if self.video_source.connection_ok else (0, 0, 255), 2)

        # Add FPS info
        fps_text = f"FPS: {self.video_source.fps:.1f}"
        cv2.putText(overlay, fps_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        return overlay

    def zoom_in(self):
        """Increase zoom factor"""
        self.zoom_factor = min(5.0, self.zoom_factor + 0.2)
        self.zoom_reset_button.setText(f"{int(self.zoom_factor * 100)}%")

    def zoom_out(self):
        """Decrease zoom factor"""
        self.zoom_factor = max(1.0, self.zoom_factor - 0.2)
        self.zoom_reset_button.setText(f"{int(self.zoom_factor * 100)}%")

    def zoom_reset(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.zoom_center = None
        self.zoom_reset_button.setText("100%")

    def toggle_detections(self, checked):
        """Toggle display of detections"""
        self.show_detections = checked

    def toggle_rois(self, checked):
        """Toggle display of ROIs"""
        self.show_rois = checked

    def toggle_info(self, checked):
        """Toggle display of information overlay"""
        self.show_info = checked

    def on_mouse_press(self, event):
        """
        Handle mouse press events

        Args:
            event: Mouse event
        """
        if self.current_frame is None:
            return

        # Get image coordinates considering zoom factor
        height, width = self.current_frame.shape[:2]

        # Calculate relative position in the displayed image
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # Calculate scaling factor between original frame and displayed size
        scale_x = width / label_width
        scale_y = height / label_height

        # Get relative position in the image
        rel_x = event.x() * scale_x
        rel_y = event.y() * scale_y

        # Adjust for zoom
        if self.zoom_factor > 1.0:
            # Calculate the region that is currently visible
            new_width = int(width / self.zoom_factor)
            new_height = int(height / self.zoom_factor)

            if self.zoom_center is None:
                self.zoom_center = (width // 2, height // 2)

            # Calculate crop coordinates
            x1 = max(0, self.zoom_center[0] - new_width // 2)
            y1 = max(0, self.zoom_center[1] - new_height // 2)

            # Adjust click coordinates
            rel_x = x1 + rel_x / self.zoom_factor
            rel_y = y1 + rel_y / self.zoom_factor

        # Emit signal with image coordinates
        self.frame_clicked.emit(int(rel_x), int(rel_y))

    def on_mouse_move(self, event):
        """
        Handle mouse move events

        Args:
            event: Mouse event
        """
        pass  # Placeholder for future functionality

    def on_wheel(self, event):
        """
        Handle mouse wheel events for zooming

        Args:
            event: Wheel event
        """
        delta = event.angleDelta().y()

        if delta > 0:
            # Zoom in
            self.zoom_factor = min(5.0, self.zoom_factor + 0.1)
        else:
            # Zoom out
            self.zoom_factor = max(1.0, self.zoom_factor - 0.1)

        # Update zoom center based on mouse position
        if self.current_frame is not None and self.zoom_factor > 1.0:
            height, width = self.current_frame.shape[:2]

            # Calculate relative position in the displayed image
            label_width = self.image_label.width()
            label_height = self.image_label.height()

            # Calculate scaling factor between original frame and displayed size
            scale_x = width / label_width
            scale_y = height / label_height

            # Get relative position in the image
            rel_x = event.x() * scale_x
            rel_y = event.y() * scale_y

            self.zoom_center = (int(rel_x), int(rel_y))

        self.zoom_reset_button.setText(f"{int(self.zoom_factor * 100)}%")