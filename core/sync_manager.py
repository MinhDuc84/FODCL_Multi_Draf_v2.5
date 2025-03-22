import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("FOD.SyncManager")


class SyncManager:
    """
    Manages synchronization between multiple video sources and processing
    """

    def __init__(self, camera_manager):
        """
        Initialize the synchronization manager

        Args:
            camera_manager: Camera manager instance
        """
        self.camera_manager = camera_manager
        self.sync_enabled = False
        self.sync_processing = False
        self.process_callbacks = []  # List of callbacks for frame processing

        # Frame synchronization
        self.sync_thread = None
        self.stop_event = threading.Event()
        self.sync_interval = 1.0 / 30.0  # 30 FPS target
        self.adaptive_interval = True

        # Frame timestamps and buffers
        self.frame_timestamps = {}  # camera_id -> timestamp
        self.frame_buffers = {}  # camera_id -> [frames]
        self.max_buffer_size = 30

    def enable_sync(self, enabled: bool = True):
        """
        Enable or disable synchronization

        Args:
            enabled: Whether sync is enabled
        """
        if enabled == self.sync_enabled:
            return

        self.sync_enabled = enabled

        if enabled:
            self.start_sync_thread()
        else:
            self.stop_sync_thread()

        logger.info(f"Synchronization {'enabled' if enabled else 'disabled'}")

    def set_process_callback(self, callback: Callable[[str, Any], None]):
        """
        Set callback for synchronized frame processing

        Args:
            callback: Function accepting (camera_id, frame)
        """
        self.process_callbacks.append(callback)

    def remove_process_callback(self, callback: Callable):
        """Remove a processing callback"""
        if callback in self.process_callbacks:
            self.process_callbacks.remove(callback)

    def start_sync_thread(self):
        """Start the synchronization thread"""
        if self.sync_thread is not None and self.sync_thread.is_alive():
            return

        self.stop_event.clear()
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("Started synchronization thread")

    def stop_sync_thread(self):
        """Stop the synchronization thread"""
        if self.sync_thread is None or not self.sync_thread.is_alive():
            return

        self.stop_event.set()
        self.sync_thread.join(timeout=3)
        self.sync_thread = None
        logger.info("Stopped synchronization thread")

    def _sync_loop(self):
        """Main synchronization loop"""
        sync_counter = 0

        while not self.stop_event.is_set():
            start_time = time.time()

            try:
                # Capture frames from all cameras
                self._capture_all_frames()

                # Process frames (every Nth frame to reduce processing load)
                sync_counter += 1
                if sync_counter >= 1:  # Process every frame by default
                    sync_counter = 0
                    self._process_frames()

                # Calculate time to sleep
                elapsed = time.time() - start_time
                sleep_time = max(0, self.sync_interval - elapsed)

                # Adjust interval if adaptive
                if self.adaptive_interval:
                    self._adjust_sync_interval()

                # Sleep with check for stop event
                if sleep_time > 0 and not self.stop_event.is_set():
                    # Use shorter sleep durations to check for stop event more frequently
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")
                time.sleep(0.1)  # Brief delay on error

    def _capture_all_frames(self):
        """Capture frames from all connected cameras"""
        # Get all connected cameras
        for camera_id, camera in self.camera_manager.cameras.items():
            if camera.connection_ok:
                # Get frame
                frame = camera.get_frame()
                if frame is not None:
                    # Record timestamp
                    self.frame_timestamps[camera_id] = time.time()

                    # Add to buffer
                    if camera_id not in self.frame_buffers:
                        self.frame_buffers[camera_id] = []

                    buffer = self.frame_buffers[camera_id]
                    buffer.append((self.frame_timestamps[camera_id], frame))

                    # Trim buffer if needed
                    if len(buffer) > self.max_buffer_size:
                        buffer.pop(0)

    def _process_frames(self):
        """Process the synchronized frames"""
        if not self.frame_buffers or not self.process_callbacks:
            return

        # Set processing flag
        self.sync_processing = True

        # Call all callbacks with the latest frame from each camera
        for callback in self.process_callbacks:
            for camera_id, buffer in self.frame_buffers.items():
                if buffer:
                    # Use the newest frame
                    timestamp, frame = buffer[-1]
                    try:
                        callback(camera_id, frame)
                    except Exception as e:
                        logger.error(f"Error in process callback for camera {camera_id}: {e}")

        # Clear processing flag
        self.sync_processing = False

    def _adjust_sync_interval(self):
        """Adaptively adjust the sync interval based on camera FPS and processing load"""
        # Get average FPS of all connected cameras
        fps_values = []

        for camera_id, camera in self.camera_manager.cameras.items():
            if camera.connection_ok and camera.fps > 0:
                fps_values.append(camera.fps)

        if not fps_values:
            return

        # Use minimum FPS for sync to ensure all cameras can keep up
        min_fps = min(fps_values)
        target_interval = 1.0 / min_fps

        # Apply a small damping factor to avoid rapid changes
        damping = 0.2
        self.sync_interval = (1.0 - damping) * self.sync_interval + damping * target_interval

        # Clamp to reasonable range (5-60 FPS)
        self.sync_interval = max(1.0 / 60.0, min(1.0 / 5.0, self.sync_interval))