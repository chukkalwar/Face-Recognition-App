"""Step 1 of the flow: register a person with guided head poses.

Call start, then send frames to /frame about 3-4 times a second while the
camera preview is open, until `state` is no longer "running".
"""
from fastapi import APIRouter, Form, HTTPException

from app.api.deps import ImageDep, ServicesDep
from app.schemas.registration import (RegisteredCheck, RegistrationStart,
                                      RegistrationStatus, SessionRef)
from app.services.registration import (POSES, SAMPLES_PER_POSE, Registration,
                                       already_registered_message, check_registered)
from app.services.sessions import SessionLimitError

router = APIRouter(prefix="/api/enroll", tags=["1. Registration"])


def _status(session):
    return {
        "session_id": session.id, "person": session.name,
        "employee_id": session.employee_id, "employee_number": session.employee_number,
        "state": session.state, "instruction": session.instruction(),
        "counts": session.counts(), "samples_per_pose": SAMPLES_PER_POSE,
        "poses": [p for p, _ in POSES], "remaining_poses": session.pending(),
        "warning": session.warning, "existing_person": session.existing_person,
        "message": session.message, "saved": session.saved,
        "frame": session.last or None,
    }


def _session(services, session_id):
    session = services.registrations.get(session_id)
    if session is None:
        raise HTTPException(404, "No such registration session")
    return session


@router.post("/start", response_model=RegistrationStatus, summary="Start a registration")
def start(req: RegistrationStart, services=ServicesDep):
    name, eid, enum = req.name.strip(), req.employee_id.strip(), req.employee_number.strip()
    if not (name and eid and enum):
        raise HTTPException(400, "name, employee_id and employee_number are required")
    conflict = services.engine.store.find_conflict(name, eid, enum)
    if conflict:
        person = next(p for p in services.engine.store.list_people() if p["name"] == conflict)
        raise HTTPException(409, {"status": "already_registered",
                                  "message": f"employee_id or employee_number already "
                                             f"registered to '{conflict}'",
                                  "person": {k: v for k, v in person.items() if k != "id"}})
    if services.antispoof is None or not services.antispoof.available:
        raise HTTPException(503, "Liveness model unavailable - registration disabled")
    try:
        session = services.registrations.add(
            Registration(services.engine, services.antispoof, name, eid, enum))
    except SessionLimitError as e:
        raise HTTPException(429, str(e))
    return _status(session)


@router.post("/check", response_model=RegisteredCheck,
             summary="Is this face already registered? One image, before starting")
def check(image=ImageDep, services=ServicesDep):
    registered, person, result = check_registered(services.engine, services.antispoof,
                                                  image.frame)
    frame = {**result.to_dict(), "accepted": False,
             "problem": None if registered is not None else result.message}
    if registered:
        message = already_registered_message(person)
    elif registered is None:
        message = result.message
    else:
        message = "Not registered - you can start registration"
    return {"already_registered": registered, "message": message,
            "person": person, "frame": frame}


@router.post("/frame", response_model=RegistrationStatus, summary="Send one camera frame")
def frame(session_id: str = Form(...), image=ImageDep, services=ServicesDep):
    session = _session(services, session_id)
    with session.lock:
        session.step(image.frame)
    return _status(session)


@router.get("/status/{session_id}", response_model=RegistrationStatus,
            summary="Current status without sending a frame")
def status(session_id: str, services=ServicesDep):
    return _status(_session(services, session_id))


@router.post("/cancel", response_model=RegistrationStatus, summary="Stop; nothing is saved")
def cancel(req: SessionRef, services=ServicesDep):
    session = _session(services, req.session_id)
    with session.lock:
        session.cancel()
    return _status(session)
