"""Head pose from the detector's five keypoints. No landmark model needed.

kps rows: [left_eye, right_eye, nose, left_mouth, right_mouth], image coords.
"""
import cv2
import numpy as np

# Yaw-ratio bins. The gap between CENTER_MAX and SIDE_MIN is a dead zone so no
# pose is assigned while the head is mid-turn.
CENTER_MAX = 0.18
SIDE_MIN = 0.25

# Generic 3D face in mm, ordered like the keypoints. Only the shape matters.
_FACE_3D = np.array([
    [-30.0, 32.0, -30.0], [30.0, 32.0, -30.0], [0.0, 0.0, 0.0],
    [-28.0, -28.0, -25.0], [28.0, -28.0, -25.0],
], dtype=np.float64)


def estimate_yaw(kps):
    """Nose offset from the eye midpoint over the inter-eye distance. About 0
    when frontal, positive when the person turns to THEIR left, negative to
    their right. 0.25 or more is roughly a 25-35 degree turn."""
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    if inter_eye < 1e-6:
        return 0.0
    return float((nose[0] - (left_eye[0] + right_eye[0]) / 2.0) / inter_eye)


def pose_of(yaw, side_min=SIDE_MIN):
    """"center", "left", "right", or None while mid-turn."""
    if abs(yaw) <= CENTER_MAX:
        return "center"
    if yaw >= side_min:
        return "left"
    if yaw <= -side_min:
        return "right"
    return None


def head_pose(kps, frame_w, frame_h):
    """(yaw, pitch) in degrees, for UI feedback only. Positive yaw is the
    person's own left, positive pitch is chin up. None if the solve fails."""
    focal = float(frame_w)
    camera = np.array([[focal, 0, frame_w / 2.0], [0, focal, frame_h / 2.0], [0, 0, 1]],
                      dtype=np.float64)
    points = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    # SQPNP: the default ITERATIVE solver needs 6 points and we have 5
    ok, rvec, _ = cv2.solvePnP(_FACE_3D, points, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    forward = -(rotation @ np.array([0.0, 0.0, 1.0]))
    yaw = -float(np.degrees(np.arctan2(forward[0], forward[2])))
    pitch = float(np.degrees(np.arcsin(np.clip(forward[1], -1.0, 1.0))))
    return round(yaw, 1), round(pitch, 1)
