from .opencv import find_opencv_cameras
from .zed_config import ZEDCameraConfig

__all__ = ["ZEDCameraConfig", "ZEDCamera", "find_opencv_cameras"]


def __getattr__(name):
    if name != "ZEDCamera":
        raise AttributeError(name)
    from .zed_camera import ZEDCamera

    return ZEDCamera
