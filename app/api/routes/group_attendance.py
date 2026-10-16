"""Step 3b: attendance for several people in one camera frame.

Start a session, send frames to /frame while the group faces the camera, and
call /finish when done. Each person is marked as soon as their own face has
passed. See app/services/group_attendance.py.
"""
import time

from fastapi import APIRouter, Form, HTTPException

from app.api.deps import ImageDep, ServicesDep
from app.schemas.attendance import GroupStatus
from app.schemas.registration import SessionRef
from app.services.group_attendance import GroupAttendanceSession
from app.services.sessions import SessionLimitError

router = APIRouter(prefix="/api/attendance/group", tags=["3. Attendance"])


def _status(session):
    return {"session_id": session.id, "state": session.state,
            "seconds_left": max(0.0, round(session.deadline - time.time(), 1)),
            "spoof_faces": session.spoof_faces, "marked": session.marked(),
            "people": session.people(), "frame": session.last}


def _session(services, session_id):
    session = services.groups.get(session_id)
    if session is None:
        raise HTTPException(404, "No such group attendance session")
    return session


@router.post("/start", response_model=GroupStatus, summary="Start group attendance")
def start(services=ServicesDep):
    if services.antispoof is None or not services.antispoof.available:
        raise HTTPException(503, "Liveness model unavailable - attendance disabled")
    services.engine.maybe_reload()
    try:
        session = services.groups.add(GroupAttendanceSession(services.engine,
                                                             services.antispoof))
    except SessionLimitError as e:
        raise HTTPException(429, str(e))
    return _status(session)


@router.post("/frame", response_model=GroupStatus, summary="Send one camera frame")
def frame(session_id: str = Form(...), image=ImageDep, services=ServicesDep):
    session = _session(services, session_id)
    with session.lock:
        session.step(image.frame, image.raw)
    return _status(session)


@router.get("/status/{session_id}", response_model=GroupStatus,
            summary="Current status without sending a frame")
def status(session_id: str, services=ServicesDep):
    return _status(_session(services, session_id))


@router.post("/finish", response_model=GroupStatus,
             summary="End the session; returns everyone marked")
def finish(req: SessionRef, services=ServicesDep):
    session = _session(services, req.session_id)
    with session.lock:
        session.finish()
    return _status(session)


@router.post("/cancel", response_model=GroupStatus,
             summary="Stop; people already marked stay marked")
def cancel(req: SessionRef, services=ServicesDep):
    session = _session(services, req.session_id)
    with session.lock:
        session.cancel()
    return _status(session)
