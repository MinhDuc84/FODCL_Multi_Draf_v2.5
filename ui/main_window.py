import sys
import os
import time
import logging
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QGridLayout, QStatusBar, QAction, QToolBar,
                             QFileDialog, QMessageBox, QSplitter, QFrame,
                             QDialog, QLineEdit, QFormLayout, QComboBox,
                             QSpinBox, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSettings
from PyQt5.QtGui import QIcon, QPixmap, QImage
from typing import Dict, List, Any, Optional, Tuple, Callable

from ui.camera_view import CameraViewWidget, CameraConnectDialog
from ui.roi_editor import ROIEditorWidget
from ui.settings_panel import SettingsPanel
from ui.alerts_view import AlertsViewWidget
from ui.statistics_view import StatisticsViewWidget

from core.video_source import VideoSource
from core.detector import YOLODetector
from core.roi_manager import ROIManager
from core.alert_manager import AlertManager, Alert

from utils.config import ConfigManager
from utils.system_info import SystemInfo

# Import notifiers
from notifications.telegram import TelegramNotifier
from notifications.email import EmailNotifier
from notifications.sound import SoundNotifier

from ui.class_editor import ClassEditorWidget
from storage.class_manager import ClassManager

from core.model_transition_manager import ModelTransitionManager
from ui.class_priority_panel import ClassPriorityPanel
from ui.class_mapping_dialog import ClassMappingDialog

#from ui.multi_camera_view import MultiCameraView
from ui.multi_camera_view import CameraGridWidget as MultiCameraView
from ui.camera_manager_tab import CameraManagerTab
from core.camera_manager import CameraManager

logger = logging.getLogger("FOD.MainWindow")


class MainWindow(QMainWindow):
    """
    Main application window with multiple tabs for FOD detection system
    """

    def __init__(self):
        super().__init__()

        # Load configuration
        self.config = ConfigManager("config.yaml")

        # Initialize system components
        self.init_components()

        # Setup UI
        self.init_ui()

        # Connect signals and slots
        self.connect_signals()

        # Start system timers
        self.init_timers()

        # Restore window state
        self.restore_settings()

        # Auto-connect camera if configured
        if self.config.get("auto_connect_camera", False):
            QTimer.singleShot(1000, self.connect_camera)

        logger.info("Main window initialized")

    # Replace the video_source initialization with camera_manager initialization
    def init_components(self):
        """Initialize core system components"""
        # Processing state
        self.detection_active = False
        self.recording = False
        self.edit_mode = False

        # Initialize system info monitor
        from utils.system_info import SystemInfo
        self.system_info = SystemInfo()

        # Initialize class manager
        self.class_manager = ClassManager()

        # Initialize camera manager instead of a single video source
        self.camera_manager = CameraManager(self.config)

        # Get active camera - for backwards compatibility
        self.video_source = self.camera_manager.get_active_camera()
        if self.video_source is None and self.camera_manager.cameras:
            # Use the first camera as a fallback
            self.video_source = next(iter(self.camera_manager.cameras.values()))

        # Add connection listener for UI updates
        self.camera_manager.add_connection_listener(self.on_camera_connection_changed)

        # Initialize detector
        try:
            # Make sure the model path exists before initializing
            model_path = self.config.get("yolo_model_path", "FOD-AAA.pt")
            if not os.path.exists(model_path):
                logger.warning(f"YOLO model path {model_path} not found. Create an empty detector.")
                self.detector = None
            else:
                self.detector = YOLODetector(
                    model_path=model_path,
                    confidence=self.config.get("yolo_confidence_threshold", 0.25),
                    use_gpu=self.config.get("use_gpu", True),
                    classes_of_interest=self.config.get("classes_of_interest"),
                    class_manager=self.class_manager  # Pass class manager here directly
                )
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            self.detector = None

        # Initialize ROI manager with class manager
        self.roi_manager = ROIManager("rois_config.json", self.class_manager)

        # Initialize alert manager with class manager
        self.alert_manager = AlertManager(
            snapshot_dir="Snapshots",
            video_dir="EventVideos",
            db_path="alerts.db",
            class_manager=self.class_manager
        )

        self.model_transition_manager = ModelTransitionManager(
            self.config,
            self.class_manager,
            self.roi_manager,
            self.detector
        )

        # Connect ROI manager with class manager if not already connected
        if not hasattr(self.roi_manager, 'class_manager') or not self.roi_manager.class_manager:
            self.roi_manager.set_class_manager(self.class_manager)

        # Add notification channels
        if self.config.get("telegram_bot_token") and self.config.get("telegram_chat_id"):
            telegram_notifier = TelegramNotifier(
                bot_token=self.config.get("telegram_bot_token"),
                chat_id=self.config.get("telegram_chat_id"),
                thread_id=self.config.get("telegram_message_thread_id")
            )
            self.alert_manager.add_notifier(telegram_notifier)

        # Add email notifier if configured
        if self.config.get("email_enabled", False):
            email_notifier = EmailNotifier(
                smtp_server=self.config.get("email_smtp_server"),
                smtp_port=self.config.get("email_smtp_port"),
                username=self.config.get("email_username"),
                password=self.config.get("email_password"),
                from_addr=self.config.get("email_from"),
                to_addrs=self.config.get("email_to"),
                use_ssl=self.config.get("email_use_ssl", True)
            )
            self.alert_manager.add_notifier(email_notifier)

        # Add sound notifier if configured
        sound_file = self.config.get("sound_alert_file")
        if self.config.get("enable_sound_alert", False) and sound_file:
            if os.path.exists(sound_file):
                sound_notifier = SoundNotifier(
                    sound_file=sound_file,
                    min_severity=1  # Always notify on sound
                )
                self.alert_manager.add_notifier(sound_notifier)
            else:
                logger.warning(f"Sound file not found: {sound_file}")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("FOD Detection System")
        self.setMinimumSize(1280, 800)  # Keep minimum size
        self.resize(1600, 900)  # Set a larger default size for higher resolution feeds

        # Create main central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Create main monitoring tab
        self.monitoring_tab = QWidget()
        self.tabs.addTab(self.monitoring_tab, "Monitoring")

        # Create ROI configuration tab
        self.roi_tab = QWidget()
        self.tabs.addTab(self.roi_tab, "ROI Configuration")

        # Create alerts tab
        self.alerts_tab = QWidget()
        self.tabs.addTab(self.alerts_tab, "Alerts")

        # Create statistics tab
        self.statistics_tab = QWidget()
        self.tabs.addTab(self.statistics_tab, "Statistics")

        # Create settings tab
        self.settings_tab = QWidget()
        self.tabs.addTab(self.settings_tab, "Settings")

        self.camera_tab = QWidget()
        self.tabs.addTab(self.camera_tab, "Camera Management")

        # Setup the contents of each tab
        self.setup_monitoring_tab()
        self.setup_camera_management_tab()
        self.setup_roi_tab()
        self.setup_alerts_tab()
        self.setup_statistics_tab()
        self.setup_settings_tab()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add status bar widgets
        self.status_fps = QLabel("FPS: 0")
        self.status_bar.addWidget(self.status_fps, 1)

        self.status_detector = QLabel("Detector: Inactive")
        self.status_bar.addWidget(self.status_detector, 1)

        self.status_connection = QLabel("Camera: Disconnected")
        self.status_connection.setStyleSheet("color: red;")
        self.status_bar.addWidget(self.status_connection, 1)

        self.status_system = QLabel("CPU: 0% | RAM: 0% | GPU: 0%")
        self.status_bar.addWidget(self.status_system, 2)

        # Create toolbar
        self.create_toolbar()

        # Create menu
        self.create_menu()

    def setup_monitoring_tab(self):
        """Setup the monitoring tab contents with multi-camera support"""
        layout = QVBoxLayout(self.monitoring_tab)

        # Create camera manager (if not already created)
        if not hasattr(self, 'camera_manager'):
            self.camera_manager = CameraManager(self.config)

        # Create multi-camera view
        self.multi_camera_view = MultiCameraView(self.camera_manager)
        self.multi_camera_view.camera_selected.connect(self.on_camera_selected)
        self.multi_camera_view.frame_clicked.connect(self.on_frame_clicked)
        layout.addWidget(self.multi_camera_view)

        # Detection control buttons (same as before)
        control_layout = QHBoxLayout()

        self.btn_start_detection = QPushButton("Start Detection")
        self.btn_start_detection.clicked.connect(self.toggle_detection)
        control_layout.addWidget(self.btn_start_detection)

        self.btn_edit_roi = QPushButton("Edit ROIs")
        self.btn_edit_roi.clicked.connect(self.toggle_edit_mode)
        control_layout.addWidget(self.btn_edit_roi)

        self.btn_save_snapshot = QPushButton("Save Snapshot")
        self.btn_save_snapshot.clicked.connect(self.save_current_snapshot)
        control_layout.addWidget(self.btn_save_snapshot)

        self.btn_record_video = QPushButton("Record Video")
        self.btn_record_video.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.btn_record_video)

        layout.addLayout(control_layout)

    # Add a new setup_camera_management_tab method
    def setup_camera_management_tab(self):
        """Setup the camera management tab"""
        layout = QVBoxLayout(self.camera_tab)

        # Create camera manager tab
        self.camera_manager_tab = CameraManagerTab(self.camera_manager)
        layout.addWidget(self.camera_manager_tab)

    # Add these methods to handle camera selection and frame clicks
    def on_camera_selected(self, camera_id):
        """Handle camera selection in the multi-camera view"""
        # Update the active camera in detector, roi_manager, etc.
        self.camera_manager.set_active_camera(camera_id)

        # Update status bar
        camera_info = self.camera_manager.get_all_cameras().get(camera_id, {})
        if camera_info:
            self.status_connection.setText(f"Camera: {camera_info['name']} ({camera_id})")
            self.status_connection.setStyleSheet("color: green;" if camera_info["connected"] else "color: red;")

    def on_frame_clicked(self, camera_id, x, y):
        """Handle frame clicks in multi-camera view"""
        # Only handle clicks for the active camera
        if camera_id == self.camera_manager.active_camera_id:
            # Update ROI coordinates for the active camera
            if hasattr(self, 'roi_manager'):
                self.roi_manager.on_frame_clicked((x, y))

    def setup_roi_tab(self):
        """Setup the ROI configuration tab contents"""
        layout = QVBoxLayout(self.roi_tab)

        # Create ROI editor widget
        self.roi_editor = ROIEditorWidget(self.video_source, self.roi_manager)
        layout.addWidget(self.roi_editor)

        # Control buttons
        control_layout = QHBoxLayout()

        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_add_roi.clicked.connect(self.roi_editor.start_roi_creation)
        control_layout.addWidget(self.btn_add_roi)

        self.btn_delete_roi = QPushButton("Delete Selected ROI")
        self.btn_delete_roi.clicked.connect(self.roi_editor.delete_selected_roi)
        control_layout.addWidget(self.btn_delete_roi)

        self.btn_clear_rois = QPushButton("Clear All ROIs")
        self.btn_clear_rois.clicked.connect(self.roi_editor.clear_all_rois)
        control_layout.addWidget(self.btn_clear_rois)

        self.btn_save_rois = QPushButton("Save ROI Config")
        self.btn_save_rois.clicked.connect(self.save_roi_config)
        control_layout.addWidget(self.btn_save_rois)

        self.btn_load_rois = QPushButton("Load ROI Config")
        self.btn_load_rois.clicked.connect(self.load_roi_config)
        control_layout.addWidget(self.btn_load_rois)

        layout.addLayout(control_layout)

    def setup_alerts_tab(self):
        """Setup the alerts tab contents"""
        layout = QVBoxLayout(self.alerts_tab)

        # Create alerts view widget
        self.alerts_view = AlertsViewWidget(self.alert_manager.db)
        layout.addWidget(self.alerts_view)

        # Control buttons
        control_layout = QHBoxLayout()

        self.btn_refresh_alerts = QPushButton("Refresh")
        self.btn_refresh_alerts.clicked.connect(self.alerts_view.refresh)
        control_layout.addWidget(self.btn_refresh_alerts)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self.export_alerts_csv)
        control_layout.addWidget(self.btn_export_csv)

        self.btn_clear_alerts = QPushButton("Clear Alerts")
        self.btn_clear_alerts.clicked.connect(self.clear_alerts)
        control_layout.addWidget(self.btn_clear_alerts)

        layout.addLayout(control_layout)

    def setup_statistics_tab(self):
        """Setup the statistics tab contents"""
        layout = QVBoxLayout(self.statistics_tab)

        # Create statistics view widget
        self.statistics_view = StatisticsViewWidget(self.alert_manager)
        layout.addWidget(self.statistics_view)

        # Control buttons
        control_layout = QHBoxLayout()

        self.btn_refresh_stats = QPushButton("Refresh Statistics")
        self.btn_refresh_stats.clicked.connect(self.statistics_view.refresh)
        control_layout.addWidget(self.btn_refresh_stats)

        self.btn_export_stats = QPushButton("Export Report")
        self.btn_export_stats.clicked.connect(self.statistics_view.export_report)
        control_layout.addWidget(self.btn_export_stats)

        layout.addLayout(control_layout)

    def setup_settings_tab(self):
        """Setup the settings tab contents"""
        # Create a tab widget for settings
        settings_tabs = QTabWidget()
        layout = QVBoxLayout(self.settings_tab)
        layout.addWidget(settings_tabs)

        # General settings tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Create settings panel - pass class_manager here
        self.settings_panel = SettingsPanel(self.config, self.class_manager)
        general_layout.addWidget(self.settings_panel)

        # Control buttons
        control_layout = QHBoxLayout()

        self.btn_save_settings = QPushButton("Save Settings")
        self.btn_save_settings.clicked.connect(self.save_settings)
        control_layout.addWidget(self.btn_save_settings)

        self.btn_reload_settings = QPushButton("Reload Settings")
        self.btn_reload_settings.clicked.connect(self.reload_settings)
        control_layout.addWidget(self.btn_reload_settings)

        general_layout.addLayout(control_layout)

        # Class management tab
        class_tab = QWidget()
        class_layout = QVBoxLayout(class_tab)

        # Create class editor widget
        self.class_editor = ClassEditorWidget(self.class_manager)
        # Connect the classes_changed signal
        self.class_editor.classes_changed.connect(self.on_classes_changed)
        class_layout.addWidget(self.class_editor)

        # Class priority tab (new)
        priority_tab = QWidget()
        priority_layout = QVBoxLayout(priority_tab)

        # Create class priority panel
        from ui.class_priority_panel import ClassPriorityPanel
        self.class_priority_panel = ClassPriorityPanel(self.class_manager, self.config)
        self.class_priority_panel.priorities_changed.connect(self.on_priorities_changed)
        priority_layout.addWidget(self.class_priority_panel)

        # Model transition tab (new)
        transition_tab = QWidget()
        transition_layout = QVBoxLayout(transition_tab)

        # Create model transition settings
        transition_group = QGroupBox("Model Transition Settings")
        transition_form = QFormLayout(transition_group)

        self.auto_mapping_check = QCheckBox("Enable automatic class mapping")
        self.auto_mapping_check.setChecked(self.config.get("enable_auto_class_mapping", True))
        self.auto_mapping_check.stateChanged.connect(lambda state:
                                                     self.config.set("enable_auto_class_mapping", bool(state)))
        transition_form.addRow("", self.auto_mapping_check)

        self.prompt_mapping_check = QCheckBox("Show mapping dialog when switching models")
        self.prompt_mapping_check.setChecked(self.config.get("prompt_for_class_mapping", True))
        self.prompt_mapping_check.stateChanged.connect(lambda state:
                                                       self.config.set("prompt_for_class_mapping", bool(state)))
        transition_form.addRow("", self.prompt_mapping_check)

        self.preserve_custom_check = QCheckBox("Preserve custom class definitions")
        self.preserve_custom_check.setChecked(self.config.get("preserve_custom_classes", True))
        self.preserve_custom_check.stateChanged.connect(lambda state:
                                                        self.config.set("preserve_custom_classes", bool(state)))
        transition_form.addRow("", self.preserve_custom_check)

        self.sync_components_check = QCheckBox("Synchronize components automatically")
        self.sync_components_check.setChecked(self.config.get("synchronize_components", True))
        self.sync_components_check.stateChanged.connect(lambda state:
                                                        self.config.set("synchronize_components", bool(state)))
        transition_form.addRow("", self.sync_components_check)

        transition_layout.addWidget(transition_group)

        # Mapping management section
        mapping_group = QGroupBox("Class Mapping Management")
        mapping_layout = QVBoxLayout(mapping_group)

        # Buttons for managing mappings
        mapping_buttons = QHBoxLayout()

        self.edit_mappings_button = QPushButton("Edit Class Mappings")
        self.edit_mappings_button.clicked.connect(self.show_mapping_editor)
        mapping_buttons.addWidget(self.edit_mappings_button)

        self.clear_mappings_button = QPushButton("Clear All Mappings")
        self.clear_mappings_button.clicked.connect(self.clear_all_mappings)
        mapping_buttons.addWidget(self.clear_mappings_button)

        mapping_layout.addLayout(mapping_buttons)
        transition_layout.addWidget(mapping_group)
        transition_layout.addStretch()

        # Add tabs to settings tab widget
        settings_tabs.addTab(general_tab, "General Settings")
        settings_tabs.addTab(class_tab, "Class Management")
        settings_tabs.addTab(priority_tab, "Class Priorities")
        settings_tabs.addTab(transition_tab, "Model Transitions")

    def on_classes_changed(self):
        """Handle changes to class definitions"""
        # If detector is initialized, update its class names cache
        if self.detector and hasattr(self.detector, '_class_cache'):
            self.detector._class_cache = {}

        # ROI manager will be updated through its event listener

        # Update class priorities panel
        if hasattr(self, 'class_priority_panel'):
            self.class_priority_panel.load_classes()

        logger.info("Class definitions updated")

    def on_priorities_changed(self):
        """Handle changes to class priorities"""
        # Update Alert class priorities
        from core.alert_manager import Alert
        Alert._class_manager = None  # Reset to force reload

        logger.info("Class priorities updated")

    # Update the model loading logic to automatically scan for classes
    # Add this to the existing method for loading YOLO models
    def load_yolo_model(self, model_path):
        """Load a new YOLO model and update class definitions"""
        try:
            # First check if path exists
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Model Not Found", f"Model file not found: {model_path}")
                return False

            # Store current model for mapping
            previous_model = self.config.get("yolo_model_path")

            # Check if this is a model change or initial load
            is_model_change = (previous_model and
                               os.path.exists(previous_model) and
                               os.path.abspath(previous_model) != os.path.abspath(model_path))

            # Update the detector
            if self.detector is None:
                self.detector = YOLODetector(
                    model_path=model_path,
                    confidence=self.config.get("yolo_confidence_threshold", 0.25),
                    use_gpu=self.config.get("use_gpu", True),
                    classes_of_interest=self.config.get("classes_of_interest"),
                    class_manager=self.class_manager  # Pass class_manager here
                )
            else:
                # Stop detection first
                was_active = self.detection_active
                if was_active:
                    self.stop_detection()

                # Update the model
                self.detector.model_path = model_path
                self.detector.load_model()

                # Restart detection if it was active
                if was_active:
                    self.start_detection()

            # Save the model path to config
            self.config.set("yolo_model_path", model_path)
            self.config.save()

            # Update UI
            self.status_detector.setText(f"Detector: Model loaded ({os.path.basename(model_path)})")

            # If this is a model change, handle through transition manager
            if is_model_change:
                if self.config.get("prompt_for_class_mapping", True):
                    self.handle_model_transition(previous_model, model_path)

            # Notify user of success
            QMessageBox.information(
                self,
                "Model Loaded",
                f"YOLO model loaded successfully: {os.path.basename(model_path)}"
            )

            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load model: {str(e)}")
            return False

    def handle_model_transition(self, old_model, new_model):
        """
        Handle transition between models with improved ROI class mapping

        Args:
            old_model: Previous model path
            new_model: New model path
        """
        try:
            logger.info(f"Handling model transition from {old_model} to {new_model}")

            # First let model transition manager handle the basic mapping
            self.model_transition_manager.handle_model_transition(old_model, new_model)

            # Then show mapping dialog for manual adjustments if configured
            if self.config.get("prompt_for_class_mapping", True):
                from ui.class_mapping_dialog import ClassMappingDialog

                QMessageBox.information(
                    self,
                    "Model Changed",
                    f"The model has changed from {os.path.basename(old_model)} to {os.path.basename(new_model)}.\n\n" +
                    "You will now be able to review and adjust class mappings between the models."
                )

                # Show the dialog and apply mappings
                if ClassMappingDialog.show_mapping_dialog(old_model, new_model, self.class_manager, self):
                    # After manual mapping, update ROIs again to use new mappings
                    self.model_transition_manager.update_rois(old_model, new_model)

                    # Force refresh of class lists in ROI editor
                    if hasattr(self, 'roi_editor'):
                        self.roi_editor.refresh_roi_list()

                    QMessageBox.information(
                        self,
                        "ROI Classes Updated",
                        "ROI class mappings have been updated to match the new model."
                    )
        except Exception as e:
            logger.error(f"Error during model transition: {e}")
            QMessageBox.warning(
                self,
                "Model Transition Error",
                f"An error occurred during model transition: {str(e)}\n\n" +
                "Some class mappings may not have been applied correctly."
            )

    def show_mapping_editor(self):
        """Show dialog for editing class mappings"""
        try:
            # Get list of models
            models = []

            # Check standard model locations
            model_dirs = ['.', 'models']
            model_extensions = ['.pt', '.pth', '.weights']

            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if any(file.endswith(ext) for ext in model_extensions):
                            models.append(os.path.join(model_dir, file))

            if not models:
                QMessageBox.warning(
                    self,
                    "No Models Found",
                    "No model files found in the current directory or 'models' folder."
                )
                return

            # Show dialog to select models
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QLabel

            dialog = QDialog(self)
            dialog.setWindowTitle("Select Models for Mapping")
            dialog.setMinimumWidth(400)

            layout = QVBoxLayout(dialog)

            # Source model
            source_layout = QHBoxLayout()
            source_layout.addWidget(QLabel("Source Model:"))
            source_combo = QComboBox()
            for model in models:
                source_combo.addItem(os.path.basename(model), model)
            source_layout.addWidget(source_combo)
            layout.addLayout(source_layout)

            # Target model
            target_layout = QHBoxLayout()
            target_layout.addWidget(QLabel("Target Model:"))
            target_combo = QComboBox()
            for model in models:
                target_combo.addItem(os.path.basename(model), model)
            target_layout.addWidget(target_combo)
            layout.addLayout(target_layout)

            # Set current model as target by default
            current_model = self.config.get("yolo_model_path")
            if current_model in models:
                index = target_combo.findData(current_model)
                if index >= 0:
                    target_combo.setCurrentIndex(index)

            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Edit Mappings")
            ok_button.clicked.connect(dialog.accept)
            button_layout.addWidget(ok_button)

            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_button)

            layout.addLayout(button_layout)

            # Show dialog
            if dialog.exec_() == QDialog.Accepted:
                source_model = source_combo.currentData()
                target_model = target_combo.currentData()

                if source_model == target_model:
                    QMessageBox.warning(
                        self,
                        "Same Models Selected",
                        "Source and target models must be different."
                    )
                    return

                # Show mapping dialog
                from ui.class_mapping_dialog import ClassMappingDialog
                ClassMappingDialog.show_mapping_dialog(source_model, target_model, self.class_manager, self)

        except Exception as e:
            logger.error(f"Error showing mapping editor: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open mapping editor: {str(e)}"
            )

    def clear_all_mappings(self):
        """Clear all class mappings"""
        reply = QMessageBox.question(
            self,
            "Clear Mappings",
            "Are you sure you want to clear all class mappings?\n\n" +
            "This will remove all mappings between models.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Clear mappings in class manager
                self.class_manager.mapper.mappings = {}
                self.class_manager.mapper.save_mappings()

                QMessageBox.information(
                    self,
                    "Mappings Cleared",
                    "All class mappings have been cleared."
                )
            except Exception as e:
                logger.error(f"Error clearing mappings: {e}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to clear mappings: {str(e)}"
                )

    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        self.addToolBar(toolbar)

        # Add camera actions
        connect_action = QAction("Connect Camera", self)
        connect_action.triggered.connect(self.show_connect_dialog)
        toolbar.addAction(connect_action)

        disconnect_action = QAction("Disconnect Camera", self)
        disconnect_action.triggered.connect(self.disconnect_camera)
        toolbar.addAction(disconnect_action)

        toolbar.addSeparator()

        # Add detection actions
        start_action = QAction("Start/Stop Detection", self)
        start_action.triggered.connect(self.toggle_detection)
        toolbar.addAction(start_action)

        snapshot_action = QAction("Take Snapshot", self)
        snapshot_action.triggered.connect(self.save_current_snapshot)
        toolbar.addAction(snapshot_action)

        record_action = QAction("Record Video", self)
        record_action.triggered.connect(self.toggle_recording)
        toolbar.addAction(record_action)

        toolbar.addSeparator()

        roi_action = QAction("ROI Editor", self)
        roi_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        toolbar.addAction(roi_action)

        alerts_action = QAction("View Alerts", self)
        alerts_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        toolbar.addAction(alerts_action)

        stats_action = QAction("Statistics", self)
        stats_action.triggered.connect(lambda: self.tabs.setCurrentIndex(3))
        toolbar.addAction(stats_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(lambda: self.tabs.setCurrentIndex(4))
        toolbar.addAction(settings_action)

    def create_menu(self):
        """Create the application menu"""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # Add camera menu items
        connect_action = QAction("Connect Camera", self)
        connect_action.triggered.connect(self.show_connect_dialog)
        file_menu.addAction(connect_action)

        disconnect_action = QAction("Disconnect Camera", self)
        disconnect_action.triggered.connect(self.disconnect_camera)
        file_menu.addAction(disconnect_action)

        file_menu.addSeparator()

        save_snapshot_action = QAction("Save Snapshot", self)
        save_snapshot_action.triggered.connect(self.save_current_snapshot)
        file_menu.addAction(save_snapshot_action)

        export_csv_action = QAction("Export Alerts to CSV", self)
        export_csv_action.triggered.connect(self.export_alerts_csv)
        file_menu.addAction(export_csv_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ROI menu
        roi_menu = menu_bar.addMenu("ROI")

        add_roi_action = QAction("Add ROI", self)
        add_roi_action.triggered.connect(lambda: (self.tabs.setCurrentIndex(1), self.roi_editor.start_roi_creation()))
        roi_menu.addAction(add_roi_action)

        save_roi_action = QAction("Save ROI Configuration", self)
        save_roi_action.triggered.connect(self.save_roi_config)
        roi_menu.addAction(save_roi_action)

        load_roi_action = QAction("Load ROI Configuration", self)
        load_roi_action.triggered.connect(self.load_roi_config)
        roi_menu.addAction(load_roi_action)

        # Detection menu
        detection_menu = menu_bar.addMenu("Detection")

        start_detection_action = QAction("Start Detection", self)
        start_detection_action.triggered.connect(self.start_detection)
        detection_menu.addAction(start_detection_action)

        stop_detection_action = QAction("Stop Detection", self)
        stop_detection_action.triggered.connect(self.stop_detection)
        detection_menu.addAction(stop_detection_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def connect_signals(self):
        """Connect signals and slots"""
        # Connect camera manager signals
        self.camera_manager.camera_connected_signal.connect(self.on_camera_connection_changed)

        # Connect camera view signals

        # Connect ROI editor signals

        # Connect alert manager signals
        pass

    def init_timers(self):
        """Initialize periodic timers"""
        # Status update timer (every 1 second)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

        # Processing timer (30fps target)
        self.processing_timer = QTimer(self)
        self.processing_timer.timeout.connect(self.process_frame)
        self.processing_timer.start(50)  # ~30fps

    def on_camera_connection_changed(self, camera_id, is_connected):
        """Handle camera connection status changes"""
        transport = "Unknown"
        camera = self.camera_manager.get_camera(camera_id)
        if camera:
            transport = camera.rtsp_transport.upper() if hasattr(camera, 'rtsp_transport') else "TCP"

        camera_info = self.camera_manager.get_all_cameras().get(camera_id, {})
        camera_name = camera_info.get("name", camera_id)

        if is_connected:
            if camera and camera.is_local_file:
                status_text = f"Camera: {camera_name} - Connected (Local File)"
            else:
                status_text = f"Camera: {camera_name} - Connected ({transport})"

            self.status_connection.setText(status_text)
            self.status_connection.setStyleSheet("color: green;")

            # Update camera status label if it exists
            if hasattr(self, 'camera_status_label'):
                self.camera_status_label.setText(f"Camera Status: {camera_name} - Connected")
                self.camera_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_connection.setText(f"Camera: {camera_name} - Disconnected")
            self.status_connection.setStyleSheet("color: red;")

            # Update camera status label if it exists
            if hasattr(self, 'camera_status_label'):
                self.camera_status_label.setText(f"Camera Status: {camera_name} - Disconnected")
                self.camera_status_label.setStyleSheet("color: red; font-weight: bold;")

        # Stop detection if it was running on this camera
        if camera_id == self.camera_manager.active_camera_id and not is_connected:
            if self.detection_active:
                self.stop_detection()

    def show_connect_dialog(self):
        """Show the camera connection dialog"""
        current_url = self.video_source.source_url
        dialog = CameraConnectDialog(self.video_source, current_url, self)

        if dialog.exec_() == QDialog.Accepted:
            url = dialog.get_url()
            transport = dialog.get_transport()
            buffer_size = dialog.get_buffer_size()

            if url:
                # Handle auto-detect transport
                if transport == "auto" and url.lower().startswith("rtsp://"):
                    # Set video source temporarily to run the test
                    self.video_source.set_source_url(url)
                    recommended = self.video_source.get_recommended_transport()
                    transport = recommended
                    logger.info(f"Auto-detected optimal transport: {transport}")

                self.connect_camera(url, transport, buffer_size)

    def connect_camera(self, url=None, transport=None, buffer_size=None):
        """
        Connect to active camera with the given URL and settings

        Args:
            url: Camera URL (if None, use from config)
            transport: RTSP transport protocol (if None, use from config)
            buffer_size: Frame buffer size (if None, use from config)
        """
        active_camera_id = self.camera_manager.active_camera_id
        if not active_camera_id:
            self.show_connect_dialog()
            return

        # Connect using camera manager
        self.camera_manager.connect_camera(active_camera_id, url, transport)

        # Update buffer size if provided
        if buffer_size is not None:
            active_camera = self.camera_manager.get_active_camera()
            if active_camera:
                active_camera.buffer_size = buffer_size
                active_camera.initial_buffer_size = buffer_size

        # Save to config
        if url:
            self.config.set("rtsp_url", url)
        if transport:
            self.config.set("rtsp_transport", transport)
        if buffer_size:
            self.config.set("buffer_size", buffer_size)
        self.config.save()

    # Update disconnect_camera method
    def disconnect_camera(self):
        """Disconnect active camera"""
        active_camera_id = self.camera_manager.active_camera_id
        if active_camera_id:
            self.camera_manager.disconnect_camera(active_camera_id)

    def process_frame(self):
        """Process frames from all cameras"""
        # Process the active camera for detection
        active_camera = self.camera_manager.get_active_camera()
        if active_camera:
            frame = active_camera.get_frame()
            if frame is not None:
                # Create a copy for processing
                processing_frame = frame.copy()
                display_frame = frame.copy()

                # Perform detection if active
                detections = []
                detections_in_roi = []
                rois_with_detections = []

                if self.detection_active and self.detector is not None and active_camera.connection_ok:
                    # Run YOLO detection
                    detections = self.detector.detect(processing_frame)

                    # Process detections against ROIs
                    detections_in_roi, rois_with_detections = self.roi_manager.process_detections(detections)

                    # Draw detections on frame
                    display_frame = self.detector.draw_detections(display_frame, detections, detections_in_roi)

                    # Draw ROIs on frame
                    display_frame = self.roi_manager.draw_rois(display_frame)

                    # Process alerts
                    current_time = time.time()
                    for roi_idx in rois_with_detections:
                        roi = self.roi_manager.rois[roi_idx]
                        if roi.should_alert(current_time):
                            # Create alert with current camera ID
                            alert = self.alert_manager.create_alert(
                                roi_id=roi_idx,
                                roi_name=roi.name,
                                class_counts=roi.class_counts.copy(),
                                camera_id=active_camera.camera_id,
                                frame=display_frame,
                                save_snapshot=True,
                                start_recording=self.start_recording if not self.recording else None
                            )
                            roi.last_alert_time = current_time
                else:
                    # Draw ROIs on frame even if detection is not active
                    display_frame = self.roi_manager.draw_rois(display_frame)

                # Send the processed frame to multi-camera view for the active camera
                self.multi_camera_view.update_frame(active_camera.camera_id, display_frame)

        # Update all camera views (including non-active ones without detection)
        self.multi_camera_view.update_all_frames()

        # If in ROI edit mode, also update ROI editor
        if self.edit_mode and hasattr(self, 'roi_editor'):
            active_camera = self.camera_manager.get_active_camera()
            if active_camera:
                frame = active_camera.get_frame()
                if frame is not None:
                    self.roi_editor.update_frame(frame)

    def get_class_priorities_from_config(self) -> Dict[int, int]:
        """Get class priorities from settings panel if available"""
        # First try to get from class manager
        try:
            priorities = self.class_manager.get_class_priorities()
            if priorities:
                return priorities
        except Exception as e:
            logger.warning(f"Error getting priorities from class manager: {e}")

        # If that fails, try to get from settings panel
        if hasattr(self, 'settings_panel') and hasattr(self.settings_panel, 'priority_combos'):
            priorities = {}
            for class_id, combo in self.settings_panel.priority_combos.items():
                # Get current index and convert to priority level (1-4)
                priorities[class_id] = combo.currentIndex() + 1
            return priorities
        else:
            # Return default priorities from Alert class as a last resort
            from core.alert_manager import Alert
            return Alert.DEFAULT_CLASS_PRIORITIES

    def update_status(self):
        """Update status bar information"""
        # Update FPS
        self.status_fps.setText(f"FPS: {self.video_source.fps:.1f}")

        # Update detector status
        self.status_detector.setText(f"Detector: {'Active' if self.detection_active else 'Inactive'}")

        # Update connection status with transport info
        if self.video_source.connection_ok:
            transport = self.video_source.rtsp_transport.upper()
            if not self.video_source.is_local_file:
                # Get network quality assessment
                quality = self.video_source._check_network_quality() if hasattr(self.video_source,
                                                                                '_check_network_quality') else 0.5
                quality_text = "Excellent" if quality > 0.8 else "Good" if quality > 0.6 else "Fair" if quality > 0.4 else "Poor"
                drop_rate = self.video_source.frame_drop_rate if hasattr(self.video_source, 'frame_drop_rate') else 0

                self.status_connection.setText(
                    f"Camera: Connected ({transport}) | Quality: {quality_text} | Drop Rate: {drop_rate:.1%}")

                # Color based on quality
                if quality > 0.8:
                    self.status_connection.setStyleSheet("color: green; font-weight: bold;")
                elif quality > 0.5:
                    self.status_connection.setStyleSheet("color: green;")
                elif quality > 0.3:
                    self.status_connection.setStyleSheet("color: orange;")
                else:
                    self.status_connection.setStyleSheet("color: red;")
            else:
                # Local file
                self.status_connection.setText(f"Camera: Connected (Local File)")
                self.status_connection.setStyleSheet("color: green;")
        else:
            self.status_connection.setText("Camera: Disconnected")
            self.status_connection.setStyleSheet("color: red;")

        # Update system info
        try:
            sys_info = self.system_info.get_system_info()

            # Get resource usage percentages
            cpu_percent = sys_info.get('cpu_percent', 0)
            memory_percent = sys_info.get('memory_percent', 0)
            gpu_percent = sys_info.get('gpu_percent', 0)

            # Format system status text with consistent formatting
            system_text = (
                f"CPU: {cpu_percent:.1f}% | "
                f"RAM: {memory_percent:.1f}% | "
                f"GPU: {gpu_percent:.1f}%"
            )

            self.status_system.setText(system_text)

            # Add visual indication of high resource usage with color
            if any(x > 90 for x in [cpu_percent, memory_percent, gpu_percent]):
                self.status_system.setStyleSheet("color: red; font-weight: bold;")
            elif any(x > 70 for x in [cpu_percent, memory_percent, gpu_percent]):
                self.status_system.setStyleSheet("color: orange;")
            else:
                self.status_system.setStyleSheet("color: black;")
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
            self.status_system.setText("System info unavailable")
            self.status_system.setStyleSheet("color: red;")

        # Update statistics view if visible
        if self.tabs.currentIndex() == 3:
            self.statistics_view.refresh()

    def toggle_detection(self):
        """Toggle detection mode on/off"""
        if self.detection_active:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self):
        """Start detection mode"""
        if self.detector is None:
            QMessageBox.warning(self, "YOLO Model Missing",
                                "No YOLO model available. Please set a valid model path in Settings tab.")
            return

        if not self.video_source.connection_ok:
            QMessageBox.warning(self, "No Camera", "No camera connected. Please connect a camera first.")
            return

        if len(self.roi_manager.rois) == 0:
            QMessageBox.warning(self, "No ROIs", "Please define at least one ROI before starting detection.")
            return

        self.detection_active = True
        self.btn_start_detection.setText("Stop Detection")
        self.status_detector.setText("Detector: Active")
        logger.info("Detection mode activated")

    def stop_detection(self):
        """Stop detection mode"""
        self.detection_active = False
        self.btn_start_detection.setText("Start Detection")
        self.status_detector.setText("Detector: Inactive")
        logger.info("Detection mode deactivated")

    def toggle_edit_mode(self):
        """Toggle ROI edit mode on/off"""
        self.edit_mode = not self.edit_mode
        self.btn_edit_roi.setText("Exit Edit Mode" if self.edit_mode else "Edit ROIs")

        if self.edit_mode:
            # Switch to ROI tab
            self.tabs.setCurrentIndex(1)
            self.roi_editor.start_edit_mode()
        else:
            self.roi_editor.stop_edit_mode()

        logger.info(f"ROI edit mode {'activated' if self.edit_mode else 'deactivated'}")

    def save_current_snapshot(self):
        """Save a snapshot of the current frame from active camera"""
        active_camera = self.camera_manager.get_active_camera()
        if active_camera and active_camera.connection_ok:
            frame = active_camera.get_frame()
            if frame is None:
                QMessageBox.warning(self, "No Frame", "No frame available to save.")
                return

            snapshot_path = self.alert_manager.save_snapshot(frame)
            QMessageBox.information(self, "Snapshot Saved", f"Snapshot saved to {snapshot_path}")
        else:
            QMessageBox.warning(self, "No Camera", "No active camera connected.")

    def toggle_recording(self):
        """Toggle video recording on/off"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording(self.video_source.get_frame())

    def start_recording(self, frame):
        """Start video recording"""
        import cv2

        if self.recording:
            return

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(self.alert_manager.video_dir, f"recording_{timestamp}.mp4")

        # Ensure directory exists
        os.makedirs(self.alert_manager.video_dir, exist_ok=True)

        self.video_writer = cv2.VideoWriter(
            video_filename,
            fourcc,
            20.0,  # FPS
            (self.config.get("resize_width", 640), self.config.get("resize_height", 480))
        )

        self.recording = True
        self.recording_start_time = time.time()
        self.btn_record_video.setText("Stop Recording")
        logger.info(f"Started video recording: {video_filename}")

        return video_filename

    def stop_recording(self):
        """Stop video recording"""
        if not self.recording:
            return

        self.video_writer.release()
        self.recording = False
        self.btn_record_video.setText("Record Video")
        logger.info("Video recording stopped")

    def save_roi_config(self):
        """Save ROI configuration to file"""
        success = self.roi_manager.save_config()
        if success:
            QMessageBox.information(self, "ROI Configuration", "ROI configuration saved successfully.")
        else:
            QMessageBox.warning(self, "ROI Configuration", "Failed to save ROI configuration.")

    def load_roi_config(self):
        """Load ROI configuration from file"""
        success = self.roi_manager.load_config()
        if success:
            QMessageBox.information(self, "ROI Configuration", "ROI configuration loaded successfully.")
            # Refresh ROI editor
            self.roi_editor.refresh_roi_list()
        else:
            QMessageBox.warning(self, "ROI Configuration", "Failed to load ROI configuration or file not found.")

    def export_alerts_csv(self):
        """Export alerts to CSV file"""
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Alerts", "alerts_export.csv", "CSV Files (*.csv)")
        if file_name:
            success = self.alert_manager.db.export_to_csv(file_name)
            if success:
                QMessageBox.information(self, "Export Alerts", f"Alerts exported to {file_name}")
            else:
                QMessageBox.warning(self, "Export Alerts", "Failed to export alerts.")

    def clear_alerts(self):
        """Clear all alerts from database"""
        reply = QMessageBox.question(self, "Clear Alerts",
                                     "Are you sure you want to clear all alerts? This cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            success = self.alert_manager.db.clear_all_alerts()
            if success:
                QMessageBox.information(self, "Clear Alerts", "All alerts have been cleared.")
                # Refresh alerts view
                self.alerts_view.refresh()
            else:
                QMessageBox.warning(self, "Clear Alerts", "Failed to clear alerts.")

    def save_settings(self):
        """Save application settings"""
        # Get settings from panel
        new_settings = self.settings_panel.get_settings()

        # Update config
        for key, value in new_settings.items():
            self.config.set(key, value)

        # Save config to file
        success = self.config.save()

        if success:
            QMessageBox.information(self, "Settings",
                                    "Settings saved successfully. Some changes may require a restart.")
        else:
            QMessageBox.warning(self, "Settings", "Failed to save settings.")

    def reload_settings(self):
        """Reload settings from file"""
        success = self.config.load()
        if success:
            # Update settings panel
            self.settings_panel.update_settings(self.config.get_all())
            QMessageBox.information(self, "Settings", "Settings reloaded successfully.")
        else:
            QMessageBox.warning(self, "Settings", "Failed to reload settings.")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About FOD Detection System",
                          "FOD Detection System v2.0\n\n"
                          "A modern application for Foreign Object Detection using YOLOv8.\n\n"
                          "© 2023\n")

    def restore_settings(self):
        """Restore window position and size"""
        settings = QSettings("FODDetection", "MainWindow")
        if settings.contains("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        if settings.contains("windowState"):
            self.restoreState(settings.value("windowState"))

    def save_window_settings(self):
        """Save window position and size"""
        settings = QSettings("FODDetection", "MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        """Handle close event"""
        # Stop all background processes
        self.processing_timer.stop()
        self.status_timer.stop()

        # Stop video source
        self.video_source.stop()

        # Stop alert manager
        self.alert_manager.stop_worker()

        # Stop recording if active
        if self.recording:
            self.stop_recording()

        # Save window settings
        self.save_window_settings()

        # Save ROI configuration
        self.roi_manager.save_config()

        event.accept()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("fod_detection.log"),
            logging.StreamHandler()
        ]
    )

    # Create application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look and feel

    # Create main window
    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())