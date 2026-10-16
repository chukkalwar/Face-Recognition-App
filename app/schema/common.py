"""Response fields shared by every endpoint that analyses a frame."""
from typing import List, Optional

from pydantic import BaseModel, Field


class FrameResult(BaseModel):
    ok: bool = Field(description="True only when every gate passed")
    status: str = Field(description="Machine-readable outcome, e.g. ok, spoof, "
                                    "too_close, too_far, no_face, liveness_low")
    message: str = Field(description="Text to show the user as-is")
    spoof: bool = Field(False, description="True when spoofing was detected")
    liveness_score: Optional[float] = Field(None, description="P(live) 0-1. Null "
                                            "until the distance is right")
    detection_score: Optional[float] = Field(None, description="Detector confidence 0-1. "
                                             "Null until the distance is right")
    face_ratio: Optional[float] = Field(None, description="Face width / shorter image "
                                        "side, the distance measure")
    similarity: Optional[float] = None
    name: Optional[str] = None
    employee_id: Optional[str] = None
    employee_number: Optional[str] = None
    bbox: Optional[List[int]] = Field(None, description="[x1, y1, x2, y2] in image pixels")


class Message(BaseModel):
    detail: str
