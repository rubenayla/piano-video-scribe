"""Video I/O wrappers.

ResizingCapture: drop-in cv2.VideoCapture wrapper that downscales every
frame to a maximum height. Source videos shot at 4K cost ~4x more per
frame than 1080p; for keyboard detection the extra resolution is wasted
(black-key columns at 1080p are still ~6 px wide). Calibration runs on
the resized frames so all downstream pixel coordinates stay consistent.
"""

import cv2


class ResizingCapture:
    """Wraps cv2.VideoCapture, returning frames downscaled to <= max_height."""

    def __init__(self, cap: cv2.VideoCapture, max_height: int):
        self._cap = cap
        self._max_h = int(max_height)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if src_h > self._max_h:
            self._scale = self._max_h / src_h
            self._dst_w = int(round(src_w * self._scale))
            self._dst_h = self._max_h
        else:
            self._scale = 1.0
            self._dst_w = src_w
            self._dst_h = src_h
        self._src_h = src_h
        self._src_w = src_w

    @property
    def scale(self) -> float:
        return self._scale

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._dst_w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._dst_h)
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        return self._cap.set(prop_id, value)

    def grab(self):
        return self._cap.grab()

    def read(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return ret, frame
        if self._scale != 1.0:
            frame = cv2.resize(frame, (self._dst_w, self._dst_h),
                               interpolation=cv2.INTER_AREA)
        return ret, frame

    def release(self):
        return self._cap.release()
