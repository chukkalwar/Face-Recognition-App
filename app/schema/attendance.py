from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import FrameResult


class AttendanceStart(BaseModel):
    employee_id: Optional[str] = Field(
        None, description="Optional 1:1 check: the face must be this employee")


class AttendanceRecord(BaseModel):
    id: int
    name: str
    employee_id: str
    employee_number: str
    similarity: float
    liveness: float
    liveness_min: float
    detection_score: float
    face_ratio: float
    frames: int
    challenge: Optional[str] = None
    mode: str = Field("single", description="single, or group for group attendance")
    created_at: str


class AttendanceFrame(FrameResult):
    accepted: bool = False


class VerifiedEmployee(BaseModel):
    name: str
    employee_id: str
    employee_number: str


class AttendanceStatus(BaseModel):
    verified: bool = Field(False, description="True once the face check passed (state complete)")
    employee: Optional[VerifiedEmployee] = Field(
        None, description="Who was verified; send this to the HRMS backend. Null until verified")
    session_id: str
    state: str = Field(description="running, complete, rejected, timeout or cancelled")
    phase: str = Field(description="passive (face front) or challenge (slight turn left or right)")
    instruction: Optional[str] = None
    progress: float
    frames_counted: int
    frames_required: int
    spoof_frames: int
    seconds_left: float
    reason: Optional[str] = None
    already_marked: bool = False
    attendance: Optional[AttendanceRecord] = None
    frame: Optional[AttendanceFrame] = Field(None, description="Feedback for the last frame")


class AttendanceList(BaseModel):
    records: List[AttendanceRecord]


# -- group attendance ------------------------------------------------------------

class GroupFace(FrameResult):
    counted: bool = Field(False, description="This face counted towards its person")
    marked: bool = Field(False, description="This person's attendance is recorded")
    frames_counted: int = 0
    frames_required: int = 0


class GroupFrame(BaseModel):
    status: str = Field(description="ok, no_face, image_too_small or duplicate_frame")
    message: str
    faces: List[GroupFace] = Field(description="Every face, left to right")


class GroupPerson(BaseModel):
    name: str
    employee_id: str
    employee_number: str
    frames_counted: int
    frames_required: int
    marked: bool
    already_marked: bool = False
    attendance: Optional[AttendanceRecord] = None


class GroupStatus(BaseModel):
    session_id: str
    state: str = Field(description="running, finished, timeout or cancelled")
    seconds_left: float
    spoof_faces: int = Field(description="Faces judged spoof so far, all frames")
    marked: List[GroupPerson] = Field(description="People whose attendance is recorded")
    people: List[GroupPerson] = Field(description="Everyone recognized so far")
    frame: Optional[GroupFrame] = Field(None, description="Feedback for the last frame")
