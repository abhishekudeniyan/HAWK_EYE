"""YOLO and ByteTrack helpers for CrowdWatch."""

import time

from ultralytics import YOLO


def load_person_model(model_name):
    """Load a fresh model so tracker state cannot leak between sessions."""
    return YOLO(model_name)


def track_people(frame, model, confidence):
    """Detect people in a frame and return their ByteTrack IDs when available."""
    start = time.perf_counter()
    result = model.track(
        frame,
        conf=confidence,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )[0]
    elapsed = max(time.perf_counter() - start, 1e-9)
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return result.plot(), 0, set(), True, elapsed

    track_tensor = boxes.id
    if track_tensor is None:
        return result.plot(), len(boxes), set(), False, elapsed

    track_ids = {int(track_id) for track_id in track_tensor.cpu().tolist()}
    return result.plot(), len(track_ids), track_ids, True, elapsed
