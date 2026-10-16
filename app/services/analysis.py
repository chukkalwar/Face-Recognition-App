"""
Single-frame face analysis: the one gate sequence every endpoint shares.

Gates run in this order, and the first failure decides the response:

    1. image size        short side >= [detection] min_frame_px
    2. face found        at least one face
    3. one face          no second sizeable face (often a phone held up)
    4. distance          face width / shorter image side inside
                         [detection] min_face_ratio .. max_face_ratio.
                         The shorter side is the width on a portrait phone
                         and the height on a landscape webcam, so the same
                         numbers work for both.
                         Too close -> "Please keep distance from the camera".
                         Scores are NOT reported until the distance is right.
    5. centred           the liveness crop (2.7x the face) is mostly in frame
    6. detection score   det_score >= min_det_score
    7. sharpness         Laplacian variance >= min_blur_var
    8. pose              frontal, when the caller requires it
    9. liveness          score < [antispoof] threshold -> spoof
                         score < min_liveness          -> liveness not confirmed
   10. recognition       cosine similarity >= [detection] rec_thresh

Why distance comes before liveness: MiniFASNet looks at crops 2.7x and 4.0x
the size of the face. Close to the camera those crops are clamped to the image
border, the model never sees a screen bezel or paper edge, and its score means
little. So a close face gets only the distance message.

`stage` stops the sequence early: "face" after gate 7, "liveness" after
gate 9, "recognize" runs everything.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.core.config import config
from app.ml.antispoof import MODELS as SPOOF_MODELS
from app.services.pose import estimate_yaw, pose_of

DET = config.detection
SPOOF_FLOOR = config.antispoof.threshold
CONTEXT_SCALE = min(scale for _, scale, _ in SPOOF_MODELS)   # 2.7

MESSAGES = {
    "ok": "Live face verified",
    "image_too_small": "Image too small - send the full camera frame",
    "no_face": "No face detected - look at the camera",
    "multiple_faces": "Only one person should be in view",
    "too_close": "Please keep distance from the camera",
    "too_far": "Please move a little closer to the camera",
    "off_centre": "Keep your face in the centre of the frame",
    "too_dark": "Too dark - face a light or raise the screen brightness",
    "low_detection": "Face not clear - improve lighting and hold still",
    "blurry": "Image is blurry - hold the phone still",
    "not_frontal": "Look straight at the camera",
    "spoof": "Spoofing detected - use your live face, not a photo or screen",
    "liveness_low": "Liveness not confirmed - face the light and hold still",
    "liveness_unavailable": "Liveness model unavailable",
    "not_recognized": "Face not recognized - please register first",
    "attendance_marked": "Attendance marked",
   
}
STAGES = ("face", "liveness", "recognize")


@dataclass
class Analysis:
    status: str
    face_ratio: Optional[float] = None
    detection_score: Optional[float] = None
    liveness_score: Optional[float] = None
    similarity: Optional[float] = None
    person: Optional[dict] = None
    bbox: Optional[list] = None
    yaw: Optional[float] = None
    face: object = field(default=None, repr=False)          # insightface Face
    embedding: object = field(default=None, repr=False)

    @property
    def ok(self):
        return self.status in ("ok", "attendance_marked")

    @property
    def spoof(self):
        return self.status == "spoof"

    @property
    def message(self):
        return MESSAGES[self.status]

    def to_dict(self):
        person = self.person or {}
        return {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "spoof": self.spoof,
            "liveness_score": _round(self.liveness_score),
            "detection_score": _round(self.detection_score),
            "face_ratio": _round(self.face_ratio),
            "similarity": _round(self.similarity),
            "name": person.get("name"),
            "employee_id": person.get("employee_id"),
            "employee_number": person.get("employee_number"),
            "bbox": self.bbox,
        }


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def context_coverage(frame_shape, bbox, scale=CONTEXT_SCALE):
    """Share of the `scale`x crop around `bbox` that lies inside the frame."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0
    cx, cy, cw, ch = (x1 + x2) / 2, (y1 + y2) / 2, bw * scale, bh * scale
    ix = max(0.0, min(w, cx + cw / 2) - max(0.0, cx - cw / 2))
    iy = max(0.0, min(h, cy + ch / 2) - max(0.0, cy - ch / 2))
    return (ix * iy) / (cw * ch)


def sharpness(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

# i am adding brightness function so it can work in the low light conditions too

def brightness(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    return 0.0 if crop.size == 0 else float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())

def face_crop_jpeg(frame, bbox, side, quality):
    """Square-resized face crop as JPEG bytes, for stored reference images."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.resize(crop, (side, side)),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


class Gates:
    """Distance and centring limits. Attendance and detection use [detection];
    registration and group attendance pass their own."""

    def __init__(self, min_face_ratio=None, max_face_ratio=None, min_context=None,
                 min_face_px=0):
        self.min_face_ratio = DET.min_face_ratio if min_face_ratio is None else min_face_ratio
        self.max_face_ratio = DET.max_face_ratio if max_face_ratio is None else max_face_ratio
        self.min_context = DET.min_context if min_context is None else min_context
        self.min_face_px = min_face_px


def analyze(engine, antispoof, frame, stage="recognize", min_det_score=None,
            min_liveness=None, require_frontal=False, gates=None):
    """Run the gates on the one person in a BGR frame. See the module docstring."""
    h, w = frame.shape[:2]
    if min(h, w) < DET.min_frame_px:
        return Analysis("image_too_small")
    faces = engine.detect(frame)
    if not faces:
        return Analysis("no_face")
    face = max(faces, key=_area)
    width = float(face.bbox[2] - face.bbox[0])
    rivals = [f for f in faces if f is not face and f.det_score >= 0.6
              and (f.bbox[2] - f.bbox[0]) >= 0.4 * width]
    if rivals:
        return Analysis("multiple_faces")
    return analyze_face(engine, antispoof, frame, face, stage, min_det_score,
                        min_liveness, require_frontal, gates)

#------------------------------------------------------------------------------------------------------------
#changed the new analyze_group function to include brightness check and max_faces check for group attendance
#------------------------------------------------------------------------------------------------------------
# def analyze_group(engine, antispoof, frame, stage="recognize", min_det_score=None,
#                   min_liveness=None, gates=None, max_faces=10):
#     """Run the gates on every face in the frame, independently.

#     Returns (frame_status, [Analysis, ...]) with faces ordered left to right.
#     frame_status is "ok", "image_too_small" or "no_face". Faces the detector
#     is unsure of (det_score < 0.5) are dropped as noise, not reported.
#     """
#     h, w = frame.shape[:2]
#     if min(h, w) < DET.min_frame_px:
#         return "image_too_small", []
#     faces = [f for f in engine.detect(frame) if f.det_score >= 0.5]
#     if not faces:
#         return "no_face", []
#     faces = sorted(faces, key=_area, reverse=True)[:max_faces]
#     faces.sort(key=lambda f: f.bbox[0])
#     return "ok", [analyze_face(engine, antispoof, frame, f, stage, min_det_score,
#                                min_liveness, False, gates) for f in faces]

def analyze_group(engine, antispoof, frame, stage="recognize", min_det_score=None,
                  min_liveness=None, gates=None, max_faces=10):
    """Run the gates on every face in the frame, independently.

    Returns (frame_status, [Analysis, ...]) with faces ordered left to right.
    frame_status is "ok", "image_too_small", "no_face", "too_dark" or
    "too_many_faces". Faces the detector is unsure of (det_score < 0.5) are
    dropped as noise, not reported.
    """
    h, w = frame.shape[:2]
    if min(h, w) < DET.min_frame_px:
        return "image_too_small", []
    faces = [f for f in engine.detect(frame) if f.det_score >= 0.5]
    if not faces:
        small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
        return ("too_dark" if small.mean() < DET.min_brightness else "no_face"), []
    # more people than allowed: nobody is marked on this frame. Only faces big
    # enough to be checked count, so tiny faces far behind do not block the group.
    min_px = gates.min_face_px if gates else 0
    if sum(1 for f in faces if (f.bbox[2] - f.bbox[0]) >= min_px) > max_faces:
        return "too_many_faces", []
    faces = sorted(faces, key=_area, reverse=True)[:max_faces]
    faces.sort(key=lambda f: f.bbox[0])
    return "ok", [analyze_face(engine, antispoof, frame, f, stage, min_det_score,
                               min_liveness, False, gates) for f in faces]

def _area(face):
    return (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])


def analyze_face(engine, antispoof, frame, face, stage="recognize", min_det_score=None,
                 min_liveness=None, require_frontal=False, gates=None):
    """Gates 4-10 for one detected face."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    min_det_score = DET.min_det_score if min_det_score is None else min_det_score
    min_liveness = config.antispoof.min_liveness if min_liveness is None else min_liveness
    gates = gates or Gates()
    h, w = frame.shape[:2]

    # against the shorter side: the liveness crop is 2.7x the face on BOTH
    # axes, so it is the short side that runs out of room first
    width = float(face.bbox[2] - face.bbox[0])
    ratio = width / min(w, h)
    bbox = [int(v) for v in face.bbox]
    if ratio > gates.max_face_ratio:
        return Analysis("too_close", face_ratio=ratio, bbox=bbox, face=face)
    if ratio < gates.min_face_ratio or width < gates.min_face_px:
        return Analysis("too_far", face_ratio=ratio, bbox=bbox, face=face)

    result = Analysis("ok", face_ratio=ratio, bbox=bbox, face=face,
                      detection_score=float(face.det_score),
                      yaw=estimate_yaw(face.kps))
    if gates.min_context and context_coverage(frame.shape, face.bbox) < gates.min_context:
        result.status = "off_centre"
        return result
    if brightness(frame, face.bbox) < DET.min_brightness:
        result.status = "too_dark"
        return result
    if face.det_score < min_det_score:
        result.status = "low_detection"
        return result
    if sharpness(frame, face.bbox) < DET.min_blur_var:
        result.status = "blurry"
        return result
    if require_frontal and pose_of(result.yaw) != "center":
        result.status = "not_frontal"
        return result
    if stage == "face":
        return result

    if antispoof is None or not antispoof.available:
        result.status = "liveness_unavailable"     # never fail open
        return result
    result.liveness_score = antispoof.score(frame, face.bbox.astype(int))
    if result.liveness_score is None:
        result.status = "liveness_low"
        return result
    if result.liveness_score < SPOOF_FLOOR:
        result.status = "spoof"
        return result
    if result.liveness_score < min_liveness:
        result.status = "liveness_low"
        return result
    if stage == "liveness":
        return result

    result.embedding = np.asarray(engine.embed(frame, face), dtype=np.float32)
    result.person, result.similarity = engine.match(result.embedding)
    if result.person is None:
        result.status = "not_recognized"
    return result
