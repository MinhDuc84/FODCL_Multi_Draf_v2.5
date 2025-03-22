import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable
from core.video_source import VideoSource
from PyQt5.QtCore import pyqtSignal, QObject

logger = logging.getLogger("FOD.CameraManager")


class CameraManager(QObject):
    """
    Manages multiple camera connections and synchronization
    """
    # Define signals
    camera_connected_signal = pyqtSignal(str, bool)

    def __init__(self, config_manager):
        super().__init__()  # Initialize QObject
        """
        Initialize the camera manager

        Args:
            config_manager: Configuration manager instance
        """
        from core.sync_manager import SyncManager
        self.sync_manager = SyncManager(self)

        self.config_manager = config_manager
        self.cameras = {}  # Dictionary of camera_id -> VideoSource
        self.active_camera_id = None
        self.sync_mode = False
        self.connection_listeners = []

        # Load camera configurations
        self._load_camera_configs()

    def enable_sync(self, enabled=True):
        """Enable synchronization between cameras"""
        self.sync_manager.enable_sync(enabled)
        self.sync_mode = enabled

    def add_sync_process_callback(self, callback):
        """Add callback for synchronized frame processing"""
        self.sync_manager.set_process_callback(callback)

    def remove_sync_process_callback(self, callback):
        """Remove callback for synchronized frame processing"""
        self.sync_manager.remove_process_callback(callback)

    def _load_camera_configs(self):
        """Load camera configurations from settings"""
        camera_configs = self.config_manager.get("cameras", {})

        # If no cameras configured, add the main camera from legacy config
        if not camera_configs:
            main_camera = {
                "camera_id": "main",
                "name": "Main Camera",
                "url": self.config_manager.get("rtsp_url", ""),
                "resize_width": self.config_manager.get("resize_width", 640),
                "resize_height": self.config_manager.get("resize_height", 480),
                "buffer_size": self.config_manager.get("buffer_size", 30),
                "rtsp_transport": self.config_manager.get("rtsp_transport", "tcp"),
                "enabled": True
            }
            camera_configs = {"main": main_camera}
            # Save to config
            self.config_manager.set("cameras", camera_configs)

        # Create VideoSource instances for each camera
        for camera_id, config in camera_configs.items():
            if config.get("enabled", True):
                self.add_camera(
                    camera_id=camera_id,
                    name=config.get("name", f"Camera {camera_id}"),
                    url=config.get("url", ""),
                    resize_width=config.get("resize_width", 640),
                    resize_height=config.get("resize_height", 480),
                    buffer_size=config.get("buffer_size", 30),
                    rtsp_transport=config.get("rtsp_transport", "tcp"),
                    auto_connect=config.get("auto_connect", False)
                )

    def add_camera(self, camera_id: str, name: str, url: str,
                   resize_width: int = 640, resize_height: int = 480,
                   buffer_size: int = 30, rtsp_transport: str = "tcp",
                   auto_connect: bool = False) -> VideoSource:
        """
        Add a new camera

        Args:
            camera_id: Unique identifier for the camera
            name: Human-readable name
            url: RTSP URL or file path
            resize_width: Width to resize frames to
            resize_height: Height to resize frames to
            buffer_size: Frame buffer size
            rtsp_transport: RTSP transport protocol ('tcp' or 'udp')
            auto_connect: Whether to connect automatically

        Returns:
            The created VideoSource instance
        """
        # Create VideoSource
        video_source = VideoSource(
            source_url=url,
            camera_id=camera_id,
            resize_width=resize_width,
            resize_height=resize_height,
            buffer_size=buffer_size,
            auto_connect=auto_connect,
            rtsp_transport=rtsp_transport
        )

        # Add connection callback
        video_source.set_connection_callback(lambda connected: self._on_camera_connection_changed(camera_id, connected))

        # Add to cameras dictionary
        self.cameras[camera_id] = video_source

        # Set as active camera if it's the first one
        if self.active_camera_id is None:
            self.active_camera_id = camera_id

        # Update config
        self._update_camera_config(camera_id, {
            "camera_id": camera_id,
            "name": name,
            "url": url,
            "resize_width": resize_width,
            "resize_height": resize_height,
            "buffer_size": buffer_size,
            "rtsp_transport": rtsp_transport,
            "auto_connect": auto_connect,
            "enabled": True
        })

        logger.info(f"Added camera '{camera_id}' ({name})")
        return video_source

    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera

        Args:
            camera_id: ID of camera to remove

        Returns:
            True if successful, False otherwise
        """
        if camera_id not in self.cameras:
            return False

        # Disconnect camera
        self.cameras[camera_id].stop()

        # Remove from dictionary
        del self.cameras[camera_id]

        # Update active camera if this was the active one
        if self.active_camera_id == camera_id:
            if self.cameras:
                self.active_camera_id = next(iter(self.cameras))
            else:
                self.active_camera_id = None

        # Update config (mark as disabled)
        camera_configs = self.config_manager.get("cameras", {})
        if camera_id in camera_configs:
            camera_configs[camera_id]["enabled"] = False
            self.config_manager.set("cameras", camera_configs)

        logger.info(f"Removed camera '{camera_id}'")
        return True

    def get_camera(self, camera_id: str) -> Optional[VideoSource]:
        """Get camera by ID"""
        return self.cameras.get(camera_id)

    def get_active_camera(self) -> Optional[VideoSource]:
        """Get currently active camera"""
        if self.active_camera_id is None:
            return None
        return self.cameras.get(self.active_camera_id)

    def set_active_camera(self, camera_id: str) -> bool:
        """
        Set the active camera

        Args:
            camera_id: ID of camera to make active

        Returns:
            True if successful, False otherwise
        """
        if camera_id not in self.cameras:
            return False

        self.active_camera_id = camera_id
        logger.info(f"Set active camera to '{camera_id}'")
        return True

    def connect_camera(self, camera_id: str, url: Optional[str] = None,
                       transport: Optional[str] = None) -> bool:
        """
        Connect to a camera

        Args:
            camera_id: ID of camera to connect
            url: Optional URL to override the stored one
            transport: Optional transport protocol to use

        Returns:
            True if connection initiated, False otherwise
        """
        if camera_id not in self.cameras:
            return False

        camera = self.cameras[camera_id]

        # Update URL if provided
        if url:
            camera.set_source_url(url)
            # Update in config
            self._update_camera_config(camera_id, {"url": url})

        # Update transport if provided
        if transport:
            camera.rtsp_transport = transport
            # Update in config
            self._update_camera_config(camera_id, {"rtsp_transport": transport})

        # Start the camera
        return camera.start()

    def disconnect_camera(self, camera_id: str) -> bool:
        """
        Disconnect a camera

        Args:
            camera_id: ID of camera to disconnect

        Returns:
            True if successful, False otherwise
        """
        if camera_id not in self.cameras:
            return False

        self.cameras[camera_id].stop()
        return True

    def disconnect_all(self):
        """Disconnect all cameras"""
        for camera in self.cameras.values():
            camera.stop()

    def connect_all(self):
        """Connect all cameras"""
        for camera_id in self.cameras:
            self.connect_camera(camera_id)

    def enable_sync_mode(self, enabled: bool = True):
        """
        Enable or disable synchronized frame processing

        Args:
            enabled: Whether sync mode is enabled
        """
        self.sync_mode = enabled
        logger.info(f"Camera synchronization mode {'enabled' if enabled else 'disabled'}")

    def _update_camera_config(self, camera_id: str, config_updates: Dict[str, Any]):
        """Update camera configuration in settings"""
        camera_configs = self.config_manager.get("cameras", {})

        if camera_id not in camera_configs:
            camera_configs[camera_id] = {}

        camera_configs[camera_id].update(config_updates)
        self.config_manager.set("cameras", camera_configs)

    def _on_camera_connection_changed(self, camera_id, is_connected):
        # Emit signal instead of calling listeners directly
        self.camera_connected_signal.emit(camera_id, is_connected)

    def add_connection_listener(self, listener: Callable[[str, bool], None]):
        """
        Add a listener for camera connection events

        Args:
            listener: Function accepting (camera_id, is_connected)
        """
        self.connection_listeners.append(listener)

    def remove_connection_listener(self, listener: Callable):
        """Remove a connection listener"""
        if listener in self.connection_listeners:
            self.connection_listeners.remove(listener)

    def get_all_cameras(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all cameras

        Returns:
            Dictionary of camera_id -> camera info dictionaries
        """
        result = {}

        for camera_id, camera in self.cameras.items():
            result[camera_id] = {
                "id": camera_id,
                "name": self.config_manager.get("cameras", {}).get(camera_id, {}).get("name", f"Camera {camera_id}"),
                "url": camera.source_url,
                "connected": camera.connection_ok,
                "resolution": f"{camera.resize_width}x{camera.resize_height}",
                "fps": camera.fps,
                "transport": camera.rtsp_transport,
                "is_active": (camera_id == self.active_camera_id),
                "is_local_file": camera.is_local_file
            }

        return result