"""
Group attendance: several people in one camera frame, each marked on their own.

Every face in each frame goes through the shared gates at the strict bars
(detection score, liveness >= [antispoof] min_liveness, recognized with
similarity >= [attendance] min_similarity), using the [group_attendance]
distance limits. A person is marked once THEIR face has passed on
`frames_required` frames spread over `min_span_s`. Other people in the frame
do not hold them up, and a spoof beside them does not block them: the spoof
face simply never passes, so it is never marked.

Two faces matching the same employee in one frame means one of them is a
photo or screen. That frame counts for nobody with that identity.

There is no head-turn challenge in a group. Records are stored with
mode = "group" so they can be told apart from single attendance.

The session runs until the app calls finish, or `max_seconds` passes.
"""
import hashlib
import time
from collections import Counter
from statistics import median

# from scipy.datasets import face #commented out because it was not used in the code

from app.core.config import config
from app.services.analysis import Gates, analyze_group, face_crop_jpeg

CFG = config.group_attendance
ATT = config.attendance
GATES = Gates(CFG.min_face_ratio, CFG.max_face_ratio, CFG.min_context, CFG.min_face_px)


class _Candidate:
    def __init__(self, person):
        self.person = person
        self.frames = []            # metrics of passing frames
        self.best = None            # (quality, jpeg)
        self.attendance = None
        self.already_marked = False

    def to_dict(self):
        return {
            "name": self.person["name"],
            "employee_id": self.person["employee_id"],
            "employee_number": self.person["employee_number"],
            "frames_counted": len(self.frames),
            "frames_required": CFG.frames_required,
            "marked": self.attendance is not None,
            "already_marked": self.already_marked,
            "attendance": self.attendance,
        }


class GroupAttendanceSession:
    def __init__(self, engine, antispoof):
        self.engine = engine
        self.antispoof = antispoof
        self.id = None
        self.lock = None
        self.deadline = time.time() + CFG.max_seconds
        self.finished_at = None
        self.state = "running"          # running | finished | timeout | cancelled
        self.candidates = {}            # employee_id -> _Candidate
        self.spoof_faces = 0            # faces judged spoof, across all frames
        self.hashes = set()
        self.last = None                # last frame's feedback

    @property
    def final(self):
        return self.state != "running"

    def marked(self):
        return [c.to_dict() for c in self.candidates.values() if c.attendance is not None]

    def people(self):
        return [c.to_dict() for c in self.candidates.values()]

    def _end(self, state):
        self.state = state
        self.finished_at = time.time()

    def finish(self):
        if not self.final:
            self._end("finished")

    def cancel(self):
        if not self.final:
            self._end("cancelled")

    def step(self, frame, raw_bytes):
        """Process one frame. Returns the per-frame feedback dict."""
        if self.final:
            return self.last
        if time.time() > self.deadline:
            self._end("timeout")
            return self.last

        digest = hashlib.sha256(raw_bytes).hexdigest()
        if digest in self.hashes:
            self.last = {"status": "duplicate_frame", "faces": [],
                         "message": "Same image received twice - use the live camera"}
            return self.last
        self.hashes.add(digest)

        frame_status, results = analyze_group(self.engine, self.antispoof, frame,
                                              stage="recognize", gates=GATES,
                                              max_faces=CFG.max_faces)
        ids = Counter(r.person["employee_id"] for r in results if r.person)
        faces = []
        now = time.time()
        for r in results:
            face = {**r.to_dict(), "counted": False, "marked": False,
                    "frames_counted": 0, "frames_required": CFG.frames_required}
            if r.spoof:
                self.spoof_faces += 1
            if r.ok and ids[r.person["employee_id"]] > 1:
                face.update(ok=False, status="duplicate_identity",
                            message="Same person appears twice - one is a photo or screen")
            #----------------------------------------------------------------------------------------------- 
            # changed this line to use CFG.min_similarity instead of ATT.min_similarity for group attendance
            #-----------------------------------------------------------------------------------------------
            # elif r.ok and r.similarity < ATT.min_similarity:
            #     face.update(ok=False, status="not_recognized",
            #                 message="Face not recognized - please register first")
            elif r.ok and r.similarity < CFG.min_similarity:
                face.update(ok=False, status="not_recognized",
                            message="Face not recognized - please register first")
            elif r.ok and _margin(self.engine, r.embedding,
                                  r.person["employee_id"]) < CFG.min_margin:
                face.update(ok=False, status="not_recognized",
                            message="Face could not be told apart from another person - look at the camera")
            #----------------------------- Till here is changed to use CFG.min_margin instead of ATT.min_margin for group attendance-------------
            elif r.ok:
                self._count(r, frame, now, face)
            # elif not r.spoof:
                # self._show_if_marked(r, frame, face)   
            #replaced above elif with below elif to include "low_detection" status
            elif r.status in ("too_close", "too_far", "off_centre", "too_dark", "blurry",
                  "low_detection"):
                self._show_if_marked(r, frame, face)
            faces.append(face)
        self.last = {"status": frame_status, "faces": faces,
                     "message": _frame_message(frame_status, faces)}
        return self.last

    def _count(self, r, frame, now, face):
        eid = r.person["employee_id"]
        cand = self.candidates.setdefault(eid, _Candidate(r.person))
        if cand.attendance is None:
            cand.frames.append({"t": now, "liveness": r.liveness_score,
                                "similarity": r.similarity,
                                "detection_score": r.detection_score,
                                "face_ratio": r.face_ratio})
            quality = r.liveness_score * r.similarity
            if cand.best is None or quality > cand.best[0]:
                cand.best = (quality, face_crop_jpeg(frame, r.bbox, 160, 80))
            span = cand.frames[-1]["t"] - cand.frames[0]["t"]
            if len(cand.frames) >= CFG.frames_required and span >= CFG.min_span_s:
                self._mark(cand)
        face.update(counted=True, frames_counted=len(cand.frames),
                    marked=cand.attendance is not None)
        if cand.attendance is not None:
            _mark_face(face, cand)

    def _show_if_marked(self, r, frame, face):
        """A face that failed a gate this frame (too close, liveness low...)
        but belongs to someone already marked in this session shows as marked
        rather than as a problem. Spoofs never get here. Only runs once
        someone is marked, and only on faces big enough to recognize."""
        marked = {eid: c for eid, c in self.candidates.items() if c.attendance}
        if not marked or r.face is None or (r.face.bbox[2] - r.face.bbox[0]) < 40:
            return
        embedding = r.embedding
        if embedding is None:
            embedding = self.engine.embed(frame, r.face)
        person, similarity = self.engine.match(embedding)
        # if person is None or similarity < ATT.min_similarity:
        if person is None or similarity < CFG.min_similarity:  # changed this line to use CFG.min_similarity instead of ATT.min_similarity for group attendance
            return
        cand = marked.get(person["employee_id"])
        if cand is None:
            return
        face.update(name=person["name"], employee_id=person["employee_id"],
                    employee_number=person["employee_number"],
                    similarity=round(float(similarity), 4),
                    frames_counted=len(cand.frames))
        _mark_face(face, cand)

    def _mark(self, cand):
        store = self.engine.store
        previous = store.last_attendance(cand.person["employee_id"], ATT.cooldown_s)
        if previous is not None:
            cand.attendance, cand.already_marked = previous, True
        else:
            lv = [m["liveness"] for m in cand.frames]
            cand.attendance = store.add_attendance(
                cand.person,
                similarity=median(m["similarity"] for m in cand.frames),
                liveness=median(lv), liveness_min=min(lv),
                detection_score=min(m["detection_score"] for m in cand.frames),
                face_ratio=median(m["face_ratio"] for m in cand.frames),
                frames=len(cand.frames), image=cand.best[1] if cand.best else None,
                mode="group")
        cand.best = None


def _mark_face(face, cand):
    name = cand.person["name"]
    face.update(ok=True, status="attendance_marked", marked=True, spoof=False,
                message=(f"Attendance already marked for {name}" if cand.already_marked
                         else f"Attendance marked for {name}"))

#-------------------------------- The below function is commented out and replaced with a new one that includes a check for too many faces in the frame. --------------------------------

# def _frame_message(status, faces):
#     if status == "image_too_small":
#         return "Image too small - send the full camera frame"
#     if status == "no_face":
#         return "No face detected - look at the camera"
#     marked = sum(1 for f in faces if f["marked"])
#     spoofs = sum(1 for f in faces if f["spoof"])
#     text = f"{len(faces)} face(s) in view, {marked} marked"
#     if spoofs:
#         text += f", spoofing detected on {spoofs}"
#     return text

def _frame_message(status, faces):
    if status == "image_too_small":
        return "Image too small - send the full camera frame"
    if status == "too_many_faces":
        return (f"More members in the frame than expected - please keep only "
                f"{CFG.max_faces} (fewer is fine)")
    if status == "too_dark":
        return "Too dark - turn on the light or the torch"
    if status == "no_face":
        return "No face detected - look at the camera"
    marked = sum(1 for f in faces if f["marked"])
    spoofs = sum(1 for f in faces if f["spoof"])
    text = f"{len(faces)} face(s) in view, {marked} marked"
    if spoofs:
        text += f", spoofing detected on {spoofs}"
    return text


def _margin(engine, embedding, employee_id):
    """How far the best match beats the closest DIFFERENT employee."""
    with engine._index_lock:
        people, embeddings = engine.people, engine.embeddings
    sims = embeddings @ embedding
    own = [float(s) for s, p in zip(sims, people) if p["employee_id"] == employee_id]
    other = [float(s) for s, p in zip(sims, people) if p["employee_id"] != employee_id]
    if not own or not other:
        return 1.0
    return max(own) - max(other)

#---------------------------------------------------------------------------------------------------------------------- 