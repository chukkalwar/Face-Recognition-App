from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import FrameResult


class VerifyResult(FrameResult):
    """FrameResult plus the field names the first version of the app used."""
    reason: Optional[str] = Field(None, description="Same as message when ok is false")
    label: Optional[str] = Field(None, description="Matched name, 'unknown', or null")
    score: Optional[float] = Field(None, description="Same as similarity")
    liveness: Optional[float] = Field(None, description="Same as liveness_score")


class GroupVerifyResult(BaseModel):
    status: str = Field(description="ok, no_face or image_too_small")
    message: str
    count: int = Field(description="Faces found")
    recognized: int = Field(description="Faces that passed every gate and matched")
    faces: List[FrameResult] = Field(description="Every face, left to right")
