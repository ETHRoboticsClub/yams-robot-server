import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
import pyzed.sl as sl

from lerobot.cameras import Camera, ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras.utils import get_cv2_rotation

from .zed_config import ZEDCameraConfig

logger = logging.getLogger(__name__)


class ZEDCamera(Camera):
    def __init__(self, config: ZEDCameraConfig):
        super().__init__(config)

        self.config = config
        self.camera_id = config.camera_id
        self.color_mode = config.color_mode
        self.depth_mode = config.depth_mode

        self.zed: sl.Camera | None = None
        self.runtime_params: sl.RuntimeParameters | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.camera_id})"

    @property
    def is_connected(self) -> bool:
        return self.zed is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        found_cameras = []
        cameras = sl.Camera.get_device_list()

        for idx, cam_info in enumerate(cameras):
            camera_dict = {
                "type": "ZED",
                "id": idx,
                "serial_number": cam_info.serial_number,
                "model": str(cam_info.camera_model),
            }
            found_cameras.append(camera_dict)

        return found_cameras

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.set_from_camera_id(self.camera_id)
        init_params.camera_resolution = self._get_resolution()
        init_params.camera_fps = self.fps if self.fps else 30
        init_params.depth_mode = self._get_depth_mode()
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.zed = None
            raise ConnectionError(
                f"Failed to open {self}: {status}. Run `lerobot-find-cameras` to find available cameras."
            )

        self.runtime_params = sl.RuntimeParameters()

        cam_info = self.zed.get_camera_information()
        if self.fps is None:
            self.fps = int(cam_info.camera_configuration.fps)

        if self.width is None or self.height is None:
            resolution = cam_info.camera_configuration.resolution
            actual_width = resolution.width
            actual_height = resolution.height

            if self.rotation in [
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

        if warmup:
            for _ in range(10):
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _get_resolution(self) -> sl.RESOLUTION:
        if self.width is None or self.height is None:
            return sl.RESOLUTION.HD720

        if self.width == 2208 and self.height == 1242:
            return sl.RESOLUTION.HD2K
        elif self.width == 1920 and self.height == 1080:
            return sl.RESOLUTION.HD1080
        elif self.width == 1280 and self.height == 720:
            return sl.RESOLUTION.HD720
        elif self.width == 672 and self.height == 376:
            return sl.RESOLUTION.VGA
        else:
            logger.warning(
                f"Resolution {self.width}x{self.height} not standard, using HD720"
            )
            return sl.RESOLUTION.HD720

    def _get_depth_mode(self) -> sl.DEPTH_MODE:
        depth_modes = {
            "NONE": sl.DEPTH_MODE.NONE,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
        }
        return depth_modes.get(self.depth_mode.upper(), sl.DEPTH_MODE.PERFORMANCE)

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.zed is None or self.runtime_params is None:
            raise RuntimeError(f"{self}: zed camera not initialized.")

        start_time = time.perf_counter()

        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"{self} failed to grab frame.")

        image_zed = sl.Mat()
        self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)

        frame = image_zed.get_data()[:, :, :3]

        frame_processed = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame_processed

    def _postprocess_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None
    ) -> NDArray[Any]:
        target_mode = color_mode if color_mode else self.color_mode

        processed = image
        if target_mode == ColorMode.RGB:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]:
            processed = cv2.rotate(processed, self.rotation)

        return processed

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialized.")

        while not self.stop_event.is_set():
            try:
                frame = self.read()

                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(
                    f"Error reading frame in background thread for {self}: {e}"
                )

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms}ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(
                f"Internal error: Event set but no frame available for {self}."
            )

        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.zed is not None:
            self.zed.close()
            self.zed = None
            self.runtime_params = None

        logger.info(f"{self} disconnected.")
