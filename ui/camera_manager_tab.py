import logging
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QPushButton, QLineEdit, QSpinBox,
                             QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QGroupBox, QTabWidget, QMessageBox,
                             QDialog, QDialogButtonBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSettings

from core.camera_manager import CameraManager
from ui.camera_view import CameraConnectDialog
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger("FOD.CameraManagerTab")


class CameraInfoDialog(QDialog):
    """Dialog for adding or editing a camera"""

    def __init__(self, camera_info: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)

        self.camera_info = camera_info or {}
        self.setWindowTitle("Camera Information")
        self.setMinimumWidth(500)

        self.init_ui()
        self.load_camera_info()

    def init_ui(self):
        """Initialize UI components"""
        layout = QFormLayout(self)

        # Camera ID
        self.camera_id_edit = QLineEdit()
        layout.addRow("Camera ID:", self.camera_id_edit)

        # Camera Name
        self.camera_name_edit = QLineEdit()
        layout.addRow("Camera Name:", self.camera_name_edit)

        # Camera URL
        self.camera_url_edit = QLineEdit()
        layout.addRow("RTSP URL or File:", self.camera_url_edit)

        # Resolution
        resolution_layout = QHBoxLayout()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setSingleStep(16)
        self.width_spin.setValue(640)
        resolution_layout.addWidget(self.width_spin)

        resolution_layout.addWidget(QLabel("×"))

        self.height_spin = QSpinBox()
        self.height_spin.setRange(180, 2160)
        self.height_spin.setSingleStep(16)
        self.height_spin.setValue(480)
        resolution_layout.addWidget(self.height_spin)

        layout.addRow("Resolution:", resolution_layout)

        # RTSP Transport
        self.transport_combo = QComboBox()
        self.transport_combo.addItem("TCP (More reliable)", "tcp")
        self.transport_combo.addItem("UDP (Lower latency)", "udp")
        self.transport_combo.addItem("Auto-detect", "auto")
        layout.addRow("RTSP Transport:", self.transport_combo)

        # Buffer Size
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(5, 100)
        self.buffer_spin.setSingleStep(5)
        self.buffer_spin.setValue(30)
        self.buffer_spin.setSuffix(" frames")
        layout.addRow("Buffer Size:", self.buffer_spin)

        # Auto-connect
        self.auto_connect_check = QCheckBox("Automatically connect on startup")
        layout.addRow("", self.auto_connect_check)

        # Extra settings group
        extra_group = QGroupBox("Advanced Settings")
        extra_layout = QFormLayout(extra_group)

        # Connection timeout
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 60)
        self.timeout_spin.setValue(15)
        self.timeout_spin.setSuffix(" seconds")
        extra_layout.addRow("Connection Timeout:", self.timeout_spin)

        # Auto-switch transport
        self.auto_switch_check = QCheckBox("Automatically switch transport on connection issues")
        self.auto_switch_check.setChecked(True)
        extra_layout.addRow("", self.auto_switch_check)

        # Adaptive buffer
        self.adaptive_buffer_check = QCheckBox("Dynamically adjust buffer based on network quality")
        self.adaptive_buffer_check.setChecked(True)
        extra_layout.addRow("", self.adaptive_buffer_check)

        layout.addRow("", extra_group)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow("", buttons)

    def load_camera_info(self):
        """Load camera info into form fields"""
        if not self.camera_info:
            return

        # Set field values from camera info
        self.camera_id_edit.setText(self.camera_info.get("camera_id", ""))
        self.camera_name_edit.setText(self.camera_info.get("name", ""))
        self.camera_url_edit.setText(self.camera_info.get("url", ""))

        self.width_spin.setValue(self.camera_info.get("resize_width", 640))
        self.height_spin.setValue(self.camera_info.get("resize_height", 480))

        transport = self.camera_info.get("rtsp_transport", "tcp")
        index = self.transport_combo.findData(transport)
        if index >= 0:
            self.transport_combo.setCurrentIndex(index)

        self.buffer_spin.setValue(self.camera_info.get("buffer_size", 30))
        self.auto_connect_check.setChecked(self.camera_info.get("auto_connect", False))

        # Advanced settings
        self.timeout_spin.setValue(self.camera_info.get("connection_timeout", 15))
        self.auto_switch_check.setChecked(self.camera_info.get("rtsp_auto_switch", True))
        self.adaptive_buffer_check.setChecked(self.camera_info.get("enable_adaptive_buffer", True))

        # If editing existing camera, disable ID field
        if "camera_id" in self.camera_info:
            self.camera_id_edit.setEnabled(False)

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera info from form fields"""
        info = {
            "camera_id": self.camera_id_edit.text().strip(),
            "name": self.camera_name_edit.text().strip(),
            "url": self.camera_url_edit.text().strip(),
            "resize_width": self.width_spin.value(),
            "resize_height": self.height_spin.value(),
            "rtsp_transport": self.transport_combo.currentData(),
            "buffer_size": self.buffer_spin.value(),
            "auto_connect": self.auto_connect_check.isChecked(),
            "connection_timeout": self.timeout_spin.value(),
            "rtsp_auto_switch": self.auto_switch_check.isChecked(),
            "enable_adaptive_buffer": self.adaptive_buffer_check.isChecked(),
            "enabled": True
        }
        return info

    def validate(self) -> Tuple[bool, str]:
        """Validate form fields"""
        # Check camera ID
        camera_id = self.camera_id_edit.text().strip()
        if not camera_id:
            return False, "Camera ID is required"

        # Check camera name
        camera_name = self.camera_name_edit.text().strip()
        if not camera_name:
            return False, "Camera Name is required"

        # Check URL
        url = self.camera_url_edit.text().strip()
        if not url:
            return False, "Camera URL is required"

        return True, ""

    def accept(self):
        """Handle dialog acceptance"""
        valid, message = self.validate()
        if valid:
            super().accept()
        else:
            QMessageBox.warning(self, "Validation Error", message)


class CameraManagerTab(QWidget):
    """
    Tab for managing multiple cameras
    """

    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)

        self.camera_manager = camera_manager

        # Initialize UI
        self.init_ui()

        # Update camera list
        self.refresh_camera_list()

    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)

        # Camera list
        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(7)
        self.camera_table.setHorizontalHeaderLabels([
            "ID", "Name", "URL", "Resolution", "Status", "FPS", "Transport"
        ])
        self.camera_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.camera_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.camera_table.setSelectionMode(QTableWidget.SingleSelection)
        self.camera_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.camera_table.doubleClicked.connect(self.on_camera_double_clicked)
        layout.addWidget(self.camera_table)

        # Control buttons
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Add Camera")
        self.add_button.clicked.connect(self.add_camera)
        button_layout.addWidget(self.add_button)

        self.edit_button = QPushButton("Edit Camera")
        self.edit_button.clicked.connect(self.edit_selected_camera)
        button_layout.addWidget(self.edit_button)

        self.remove_button = QPushButton("Remove Camera")
        self.remove_button.clicked.connect(self.remove_selected_camera)
        button_layout.addWidget(self.remove_button)

        button_layout.addStretch()

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_selected_camera)
        button_layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_selected_camera)
        button_layout.addWidget(self.disconnect_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_camera_list)
        button_layout.addWidget(refresh_button)

        layout.addLayout(button_layout)

        # Camera test group
        test_group = QGroupBox("Camera Connection Test")
        test_layout = QVBoxLayout(test_group)

        test_button = QPushButton("Test Connection Settings")
        test_button.clicked.connect(self.show_connection_test)
        test_layout.addWidget(test_button)

        layout.addWidget(test_group)

    def refresh_camera_list(self):
        """Update the camera list display"""
        self.camera_table.setRowCount(0)

        cameras = self.camera_manager.get_all_cameras()
        if not cameras:
            return

        # Add cameras to table
        self.camera_table.setRowCount(len(cameras))

        for i, (camera_id, info) in enumerate(cameras.items()):
            # ID
            self.camera_table.setItem(i, 0, QTableWidgetItem(camera_id))

            # Name
            name_item = QTableWidgetItem(info["name"])
            if info["is_active"]:
                font = name_item.font()
                font.setBold(True)
                name_item.setFont(font)
            self.camera_table.setItem(i, 1, name_item)

            # URL
            self.camera_table.setItem(i, 2, QTableWidgetItem(info["url"]))

            # Resolution
            self.camera_table.setItem(i, 3, QTableWidgetItem(info["resolution"]))

            # Status
            status_item = QTableWidgetItem("Connected" if info["connected"] else "Disconnected")
            status_item.setForeground(QBrush(QColor("green" if info["connected"] else "red")))
            self.camera_table.setItem(i, 4, status_item)

            # FPS
            self.camera_table.setItem(i, 5, QTableWidgetItem(f"{info['fps']:.1f}" if info["connected"] else "N/A"))

            # Transport
            self.camera_table.setItem(i, 6, QTableWidgetItem(info["transport"].upper()))

    def on_camera_double_clicked(self, index):
        """Handle double-click on camera list"""
        row = index.row()
        if row < 0:
            return

        # Get camera ID from first column
        camera_id = self.camera_table.item(row, 0).text()

        # Toggle connection status
        camera = self.camera_manager.get_camera(camera_id)
        if camera:
            if camera.connection_ok:
                self.camera_manager.disconnect_camera(camera_id)
            else:
                self.camera_manager.connect_camera(camera_id)

        # Update the list
        self.refresh_camera_list()

    def add_camera(self):
        """Add a new camera"""
        dialog = CameraInfoDialog(parent=self)

        if dialog.exec_() == QDialog.Accepted:
            camera_info = dialog.get_camera_info()

            # Check if camera ID already exists
            if camera_info["camera_id"] in self.camera_manager.cameras:
                QMessageBox.warning(self, "Duplicate ID",
                                    f"Camera ID '{camera_info['camera_id']}' already exists")
                return

            # Add the camera
            self.camera_manager.add_camera(
                camera_id=camera_info["camera_id"],
                name=camera_info["name"],
                url=camera_info["url"],
                resize_width=camera_info["resize_width"],
                resize_height=camera_info["resize_height"],
                buffer_size=camera_info["buffer_size"],
                rtsp_transport=camera_info["rtsp_transport"],
                auto_connect=camera_info["auto_connect"]
            )

            # Update the list
            self.refresh_camera_list()

    def edit_selected_camera(self):
        """Edit the selected camera"""
        selected_items = self.camera_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a camera to edit")
            return

        # Get camera ID from first column of selected row
        row = selected_items[0].row()
        camera_id = self.camera_table.item(row, 0).text()

        # Get camera info
        cameras = self.camera_manager.get_all_cameras()
        if camera_id not in cameras:
            return

        camera_info = cameras[camera_id]

        # Get full camera config
        camera_config = self.camera_manager.config_manager.get("cameras", {}).get(camera_id, {})
        camera_info.update(camera_config)

        # Show edit dialog
        dialog = CameraInfoDialog(camera_info, parent=self)

        if dialog.exec_() == QDialog.Accepted:
            updated_info = dialog.get_camera_info()

            # Remove the camera and add it back with new settings
            was_connected = False
            camera = self.camera_manager.get_camera(camera_id)
            if camera:
                was_connected = camera.connection_ok
                if was_connected:
                    camera.stop()

            # Update camera config
            self.camera_manager._update_camera_config(camera_id, updated_info)

            # Create a new video source with updated settings
            new_camera = self.camera_manager.add_camera(
                camera_id=camera_id,
                name=updated_info["name"],
                url=updated_info["url"],
                resize_width=updated_info["resize_width"],
                resize_height=updated_info["resize_height"],
                buffer_size=updated_info["buffer_size"],
                rtsp_transport=updated_info["rtsp_transport"],
                auto_connect=False  # Don't auto-connect now
            )

            # Reconnect if it was connected before
            if was_connected:
                new_camera.start()

            # Update the list
            self.refresh_camera_list()

    def remove_selected_camera(self):
        """Remove the selected camera"""
        selected_items = self.camera_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a camera to remove")
            return

        # Get camera ID from first column of selected row
        row = selected_items[0].row()
        camera_id = self.camera_table.item(row, 0).text()

        # Confirm removal
        reply = QMessageBox.question(
            self,
            "Remove Camera",
            f"Are you sure you want to remove camera '{camera_id}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Remove the camera
            self.camera_manager.remove_camera(camera_id)

            # Update the list
            self.refresh_camera_list()

    def connect_selected_camera(self):
        """Connect the selected camera"""
        selected_items = self.camera_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a camera to connect")
            return

        # Get camera ID from first column of selected row
        row = selected_items[0].row()
        camera_id = self.camera_table.item(row, 0).text()

        # Check if already connected
        cameras = self.camera_manager.get_all_cameras()
        if camera_id in cameras and cameras[camera_id]["connected"]:
            QMessageBox.information(self, "Already Connected",
                                    f"Camera '{camera_id}' is already connected")
            return

        # Connect the camera
        self.camera_manager.connect_camera(camera_id)

        # Update the list
        QTimer.singleShot(500, self.refresh_camera_list)

    def disconnect_selected_camera(self):
        """Disconnect the selected camera"""
        selected_items = self.camera_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a camera to disconnect")
            return

        # Get camera ID from first column of selected row
        row = selected_items[0].row()
        camera_id = self.camera_table.item(row, 0).text()

        # Check if already disconnected
        cameras = self.camera_manager.get_all_cameras()
        if camera_id in cameras and not cameras[camera_id]["connected"]:
            QMessageBox.information(self, "Already Disconnected",
                                    f"Camera '{camera_id}' is already disconnected")
            return

        # Disconnect the camera
        self.camera_manager.disconnect_camera(camera_id)

        # Update the list
        QTimer.singleShot(500, self.refresh_camera_list)

    def show_connection_test(self):
        """Show connection test dialog"""
        dialog = CameraConnectDialog(parent=self)
        dialog.exec_()