from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import FrameResult


class RegistrationStart(BaseModel):
    name: str = Field(min_length=1, examples=["Jane Doe"])
    employee_id: str = Field(min_length=1, examples=["E-1042"])
    employee_number: str = Field(min_length=1, examples=["1042"])


class SessionRef(BaseModel):
    session_id: str


class FrameFeedback(FrameResult):
    accepted: bool = Field(False, description="This frame was kept as a sample")
    problem: Optional[str] = Field(None, description="Why the frame was not used")
    pose: Optional[str] = Field(None, description="Pose the head is in: center/left/right")
    yaw: Optional[float] = Field(None, description="Head yaw in degrees, for UI")
    pitch: Optional[float] = Field(None, description="Head pitch in degrees, for UI")


class ExistingPerson(BaseModel):
    name: str
    employee_id: str
    employee_number: str
    created_at: Optional[str] = Field(None, description="When they were registered")
    samples: Optional[int] = None
    has_image: Optional[bool] = Field(None, description="GET /api/people/{employee_id}/image")
    similarity: float = Field(description="How closely this face matches them")


class RegisteredCheck(BaseModel):
    already_registered: Optional[bool] = Field(
        description="Null when the image could not be checked; see frame.message")
    message: str
    person: Optional[ExistingPerson] = None
    frame: FrameFeedback


class RegistrationStatus(BaseModel):
    session_id: str
    person: str
    employee_id: str
    employee_number: str
    state: str = Field(description="running, complete, already_registered, timeout, "
                                   "cancelled or failed")
    instruction: str
    counts: Dict[str, int]
    samples_per_pose: int
    poses: List[str]
    remaining_poses: List[str]
    warning: Optional[str] = None
    existing_person: Optional[ExistingPerson] = Field(
        None, description="Set when state is already_registered")
    message: str = ""
    saved: int = 0
    frame: Optional[FrameFeedback] = Field(None, description="Feedback for the last frame")
