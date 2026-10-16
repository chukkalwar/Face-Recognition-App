"""Step 2 of the flow: single-frame checks, one endpoint per function.

Each takes one image and runs the shared gates up to its stage:

    /api/detect/face      face found, distance, centring, detection score, sharpness
    /api/detect/liveness  the above, plus liveness: spoof or live
    /api/verify           the above, plus recognition against registered people
    /api/verify/group     every face in the frame, each checked on its own

When the face is too close or too far, only the distance message comes back
and the scores stay null: the liveness score is not trustworthy at the wrong
distance.
"""
from fastapi import APIRouter

from app.api.deps import ImageDep, ServicesDep
from app.schemas.common import FrameResult
from app.schemas.detection import GroupVerifyResult, VerifyResult
from app.services.analysis import analyze, analyze_group
from app.services.group_attendance import CFG as GROUP_CFG, GATES as GROUP_GATES

router = APIRouter(tags=["2. Detection"])


@router.post("/api/detect/face", response_model=FrameResult,
             summary="Face detection, distance and quality")
def detect_face(image=ImageDep, services=ServicesDep):
    return analyze(services.engine, services.antispoof, image.frame, stage="face").to_dict()


@router.post("/api/detect/liveness", response_model=FrameResult,
             summary="Liveness: is this a live face or a spoof")
def detect_liveness(image=ImageDep, services=ServicesDep):
    return analyze(services.engine, services.antispoof, image.frame,
                   stage="liveness").to_dict()


@router.post("/api/verify", response_model=VerifyResult,
             summary="Liveness + recognition: who is this")
def verify(image=ImageDep, services=ServicesDep):
    services.engine.maybe_reload()
    result = analyze(services.engine, services.antispoof, image.frame, stage="recognize")
    out = result.to_dict()
    recognized = result.person is not None
    out.update(reason=None if result.ok else result.message,
               label=(out["name"] if recognized else
                      "unknown" if result.status == "not_recognized" else None),
               score=out["similarity"], liveness=out["liveness_score"])
    return out


@router.post("/api/verify/group", response_model=GroupVerifyResult,
             summary="Every face in the frame: liveness + recognition for each")
def verify_group(image=ImageDep, services=ServicesDep):
    services.engine.maybe_reload()
    status, results = analyze_group(services.engine, services.antispoof, image.frame,
                                    stage="recognize", gates=GROUP_GATES,
                                    max_faces=GROUP_CFG.max_faces)
    faces = [r.to_dict() for r in results]
    recognized = sum(1 for r in results if r.ok)
    message = {"no_face": "No face detected - look at the camera",
               "image_too_small": "Image too small - send the full camera frame"}.get(
        status, f"{len(faces)} face(s) found, {recognized} recognized")
    return {"status": status, "message": message, "count": len(faces),
            "recognized": recognized, "faces": faces}
