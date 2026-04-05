from .cached_config import OpenCVCameraCachedConfig

try:
    from .realsense_cached_config import RealSenseCameraCachedConfig
except Exception:
    RealSenseCameraCachedConfig = None

__all__ = [
    "OpenCVCameraCached",
    "OpenCVCameraCachedConfig",
    "RealSenseCameraCached",
    "RealSenseCameraCachedConfig",
]


def __getattr__(name):
    if name == "OpenCVCameraCached":
        from .camera_opencv_cached import OpenCVCameraCached

        return OpenCVCameraCached
    if name == "OpenCVCameraCachedConfig":
        from .cached_config import OpenCVCameraCachedConfig

        return OpenCVCameraCachedConfig
    if name == "RealSenseCameraCached":
        from .camera_realsense_cached import RealSenseCameraCached

        return RealSenseCameraCached
    if name == "RealSenseCameraCachedConfig":
        from .realsense_cached_config import RealSenseCameraCachedConfig

        return RealSenseCameraCachedConfig
    raise AttributeError(name)
