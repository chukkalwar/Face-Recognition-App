"""
Attendance: accepted only for a live, recognized person, over several frames.

A frame counts only when app/services/analysis.py returns "ok" for it at the
strict settings: frontal, detection score >= [detection] min_det_score,
liveness >= [antispoof] min_liveness, recognized with similarity >=
[attendance] min_similarity. Then:

  1. `frames_required` counted frames, spread over `min_span_s`
  2. if `challenge`: a slight head turn to EITHER side (a photo cannot turn).
     `challenge_sides_required` = 2 asks for both sides, in either order.
     Identity is checked on every turned frame too, so there is no
     return-to-centre step.
  3. the record is written, unless the same employee was marked within
     `cooldown_s`, in which case the earlier record is returned.

`max_spoof_frames` frames scored below [antispoof] threshold reject the whole
session. A different person appearing mid-session also rejects it.
"""
import hashlib
import random
import time
from statistics import median

from app.core.config import config
from app.services.analysis import MESSAGES, SPOOF_FLOOR, Gates, analyze, face_crop_jpeg

CFG = config.attendance
INSTRUCTIONS = {
    "passive": "Hold the phone at arm's length and look straight at the camera",
    "left": "Slowly turn your head to your LEFT",
    "right": "Slowly turn your head to your RIGHT",
    "turn": "Slowly turn your head to your left or right",
    "done": "Attendance complete",
}


class AttendanceSession:
    def __init__(self, engine, antispoof, employee_id=None, rng=random):
        self.engine = engine
        self.antispoof = antispoof
        self.claimed = employee_id            # optional 1:1 check
        self.id = None
        self.lock = None
        self.deadline = time.time() + CFG.max_seconds
        self.finished_at = None
        self.state = "running"                # running | complete | rejected | timeout | cancelled
        self.phase = "passive"                # passive | challenge
        self.challenge = None                 # side actually turned, set when the challenge passes
        self.sides_required = min(2, max(1, int(CFG.challenge_sides_required)))
        self.turns_done = set()               # sides already shown in the challenge
        self.person = None
        self.counted = []                     # metrics of counted frames
        self.best = None                      # (quality, jpeg)
        self.spoof_frames = 0
        self.hashes = set()
        self.reason = None
        self.attendance = None
        self.already_marked = False
        self.last = {}

    @property
    def final(self):
        return self.state != "running"

    def instruction(self):
        if self.final:
            return INSTRUCTIONS["done"] if self.state == "complete" else self.reason
        return INSTRUCTIONS[self._next_turn() if self.phase == "challenge" else self.phase]

    def _next_turn(self):
        """Either side is fine for the first turn; then only the missing one."""
        if len(self.turns_done) == 1 and self.sides_required == 2:
            return "right" if "left" in self.turns_done else "left"
        return "turn"

    def progress(self):
        if self.state == "complete":
            return 1.0
        need = CFG.frames_required + (self.sides_required if CFG.challenge else 0)
        done = min(len(self.counted), CFG.frames_required) + len(self.turns_done)
        return round(done / need, 2)

    def _end(self, state, reason=None):
        self.state, self.reason = state, reason
        self.finished_at = time.time()

    def step(self, frame, raw_bytes):
        """Process one frame. Returns the per-frame feedback dict."""
        if self.final:
            return self.last
        if time.time() > self.deadline:
            self._end("timeout", "Time limit reached - please try again")
            return self.last

        digest = hashlib.sha256(raw_bytes).hexdigest()
        if digest in self.hashes:
            self.last = {"ok": False, "status": "duplicate_frame", "accepted": False,
                         "message": "Same image received twice - use the live camera"}
            return self.last
        self.hashes.add(digest)

        turning = self.phase == "challenge"
        result = analyze(self.engine, self.antispoof, frame, stage="recognize",
                         # a turned head scores lower on both models; the strict
                         # bars apply to the frontal frames
                         min_det_score=0.6 if turning else None,
                         min_liveness=SPOOF_FLOOR if turning else None,
                         require_frontal=not turning,
                         # a turning head swells the face box and jitters it, so
                         # the "too close" bar is looser than for the frontal frame
                         gates=Gates(max_face_ratio=CFG.turn_max_face_ratio) if turning else None)
        feedback = {**result.to_dict(), "accepted": False}
        self.last = feedback

        if result.spoof:
            self.spoof_frames += 1
            if self.spoof_frames >= CFG.max_spoof_frames:
                self._end("rejected", MESSAGES["spoof"])
            return feedback
        if not result.ok:
            return feedback
        person = result.person
        if result.similarity < CFG.min_similarity:
            feedback.update(ok=False, status="not_recognized",
                            message=MESSAGES["not_recognized"])
            return feedback
        if self.claimed and person["employee_id"] != self.claimed:
            self._end("rejected", "Face does not match the selected employee")
            feedback.update(ok=False, status="mismatch", message=self.reason)
            return feedback
        if self.person and person["employee_id"] != self.person["employee_id"]:
            self._end("rejected", "A different person appeared during the check")
            feedback.update(ok=False, status="mismatch", message=self.reason)
            return feedback
        self.person = person

        metrics = {"t": time.time(), "liveness": result.liveness_score,
                   "similarity": result.similarity,
                   "detection_score": result.detection_score,
                   "face_ratio": result.face_ratio}
        if self.phase == "passive":
            feedback["accepted"] = True
            self.counted.append(metrics)
            quality = result.liveness_score * result.similarity
            if self.best is None or quality > self.best[0]:
                jpeg = face_crop_jpeg(frame, result.bbox, 160, 80)
                self.best = (quality, jpeg)
            span = self.counted[-1]["t"] - self.counted[0]["t"]
        # changed the "if" function below to allow skipping challenge if the liveness and similarity are strong enough
            # if len(self.counted) >= CFG.frames_required and span >= CFG.min_span_s:
            #     if CFG.challenge:
            #         self.phase = "challenge"
            #     else:
            #         self._complete()
            if len(self.counted) >= CFG.frames_required and span >= CFG.min_span_s:
                strong = (result.liveness_score >= CFG.skip_challenge_liveness and result.similarity >= CFG.skip_challenge_similarity)
                if CFG.challenge and not strong:
                    self.phase = "challenge"
                else:
                    self._complete()
        else:  # challenge: a slight turn to either side
            yaw = result.yaw
            direction = ("left" if yaw >= CFG.challenge_min_yaw else
                         "right" if yaw <= -CFG.challenge_min_yaw else None)
            if direction is None:
                feedback.update(ok=False, status="turn_head",
                                message=INSTRUCTIONS[self._next_turn()])
            elif direction in self.turns_done:
                feedback.update(ok=False, status="turn_head",
                                message=INSTRUCTIONS[self._next_turn()])
            else:
                feedback["accepted"] = True
                self.turns_done.add(direction)
                self.counted.append(metrics)
                if self.challenge is None:
                    self.challenge = direction
                if len(self.turns_done) >= self.sides_required:
                    self._complete()
        if self.state == "complete":
            # the final word for this session: show the mark, not a frame problem
            name = self.person["name"]
            feedback.update(ok=True, status="attendance_marked",
                            message=(f"Attendance already marked for {name}"
                                     if self.already_marked else f"Attendance marked for {name}"),
                            name=name, employee_id=self.person["employee_id"],
                            employee_number=self.person["employee_number"])
        return feedback

    def _complete(self):
        store = self.engine.store
        previous = store.last_attendance(self.person["employee_id"], CFG.cooldown_s)
        if previous is not None:
            self.attendance, self.already_marked = previous, True
        else:
            lv = [m["liveness"] for m in self.counted]
            self.attendance = store.add_attendance(
                self.person,
                similarity=median(m["similarity"] for m in self.counted),
                liveness=median(lv), liveness_min=min(lv),
                detection_score=min(m["detection_score"] for m in self.counted),
                face_ratio=median(m["face_ratio"] for m in self.counted),
                frames=len(self.counted), challenge=self.challenge,
                image=self.best[1] if self.best else None)
        self.best = None
        self._end("complete")

    def cancel(self):
        if not self.final:
            self._end("cancelled", "Attendance cancelled")
