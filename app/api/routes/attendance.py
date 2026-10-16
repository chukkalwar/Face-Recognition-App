"""Step 3 of the flow: mark attendance for a live, recognized person.

Start a session, then send frames to /frame until `state` is no longer
"running". Show `instruction` and `frame.message` to the user. See
app/services/attendance.py for what a session requires.
"""
import logging
import time
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import Response

from app.api.deps import ImageDep, ServicesDep
from app.core.config import config
from app.schemas.attendance import (AttendanceList, AttendanceRecord, AttendanceStart,
                                    AttendanceStatus)
from app.schemas.registration import SessionRef
from app.services.attendance import AttendanceSession
from app.services.sessions import SessionLimitError

logger = logging.getLogger("app.attendance")

router = APIRouter(prefix="/api/attendance", tags=["3. Attendance"])


def _status(session):
    person = session.person if session.state == "complete" else None
    return {
        # `verified` / `employee`: a stable shortcut to who was verified.
        "verified": session.state == "complete",
        "employee": ({"name": person["name"], "employee_id": person["employee_id"],
                      "employee_number": person["employee_number"]} if person else None),
        "session_id": session.id, "state": session.state, "phase": session.phase,
        "instruction": session.instruction(), "progress": session.progress(),
        "frames_counted": len(session.counted),
        "frames_required": config.attendance.frames_required,
        "spoof_frames": session.spoof_frames,
        "seconds_left": max(0.0, round(session.deadline - time.time(), 1)),
        "reason": session.reason, "already_marked": session.already_marked,
        "attendance": session.attendance, "frame": session.last or None,
    }


def _log_response(status):
    """One line per frame: what the app is told. The identified user is logged
    once, when the session ends."""
    f = status["frame"] or {}
    logger.info("frame session=%s state=%s phase=%s status=%s accepted=%s yaw=%s "
                "liveness=%s sim=%s progress=%s msg=%r",
                status["session_id"], status["state"], status["phase"], f.get("status"),
                f.get("accepted"), f.get("yaw"), f.get("liveness_score"),
                f.get("similarity"), status["progress"], f.get("message"))
    if status["state"] != "running":
        logger.info("session=%s ended state=%s reason=%r name=%r employee_id=%r "
                    "employee_number=%r", status["session_id"], status["state"],
                    status["reason"], f.get("name"), f.get("employee_id"),
                    f.get("employee_number"))


def _session(services, session_id):
    session = services.attendance.get(session_id)
    if session is None:
        raise HTTPException(404, "No such attendance session")
    return session


@router.post("/start", response_model=AttendanceStatus, summary="Start an attendance check")
def start(req: Optional[AttendanceStart] = None, services=ServicesDep):
    if services.antispoof is None or not services.antispoof.available:
        raise HTTPException(503, "Liveness model unavailable - attendance disabled")
    employee_id = ((req.employee_id if req else None) or "").strip() or None
    if employee_id and services.engine.store.get_person(employee_id) is None:
        raise HTTPException(404, f"No registered employee '{employee_id}'")
    services.engine.maybe_reload()
    try:
        session = services.attendance.add(
            AttendanceSession(services.engine, services.antispoof, employee_id))
    except SessionLimitError as e:
        raise HTTPException(429, str(e))
    return _status(session)


@router.post("/frame", response_model=AttendanceStatus, summary="Send one camera frame")
def frame(session_id: str = Form(...), image=ImageDep, services=ServicesDep):
    session = _session(services, session_id)
    with session.lock:
        session.step(image.frame, image.raw)
    status = _status(session)
    _log_response(status)
    return status


@router.get("/status/{session_id}", response_model=AttendanceStatus,
            summary="Current status without sending a frame")
def status(session_id: str, services=ServicesDep):
    return _status(_session(services, session_id))


@router.post("/cancel", response_model=AttendanceStatus, summary="Stop the check")
def cancel(req: SessionRef, services=ServicesDep):
    session = _session(services, req.session_id)
    with session.lock:
        session.cancel()
    return _status(session)


@router.get("", response_model=AttendanceList, summary="Attendance records")
def records(day: Optional[str] = Query(None, description="YYYY-MM-DD, server local time"),
            employee_id: Optional[str] = None,
            limit: int = Query(100, ge=1, le=1000), services=ServicesDep):
    return {"records": services.engine.store.list_attendance(limit, day, employee_id)}


@router.get("/{attendance_id}", response_model=AttendanceRecord, summary="One record")
def record(attendance_id: int, services=ServicesDep):
    row = services.engine.store.get_attendance(attendance_id)
    if row is None:
        raise HTTPException(404, "No such attendance record")
    return row


@router.get("/{attendance_id}/image", summary="Face photo taken with the record",
            response_class=Response)
def record_image(attendance_id: int, services=ServicesDep):
    image = services.engine.store.get_attendance_image(attendance_id)
    if image is None:
        raise HTTPException(404, "No image for this attendance record")
    return Response(content=image, media_type="image/jpeg")
