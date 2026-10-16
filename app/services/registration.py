"""
Guided registration: walk the person through center, left and right head
poses and collect `samples_per_pose` good samples for each.

Every frame goes through the shared gates in app/services/analysis.py, so the
phone gets the same distance, spoof and score feedback as detection. Head
turns double as active liveness, because a flat photo cannot turn.

As soon as `duplicate_check_samples` frontal samples are in, the face is
compared with everyone registered. A match at `duplicate_sim` or above ends
the session as "already_registered" with the existing person's details, and
nothing is saved. check_registered() does the same check on a single image,
so the app can ask before starting.

Nothing is stored until every pose is filled. Then all embeddings and one
reference face image are written in a single transaction.
"""
import time

import numpy as np

from app.core.config import config
from app.services.analysis import Gates, analyze, face_crop_jpeg, sharpness
from app.services.pose import head_pose, pose_of

CFG = config.enrollment
SAMPLES_PER_POSE = CFG.samples_per_pose
# closer than attendance: see [enrollment] in config.toml
GATES = Gates(CFG.min_face_ratio, CFG.max_face_ratio, CFG.min_context)
POSES = [
    ("center", "Look straight at the camera"),
    ("left", "Turn your head slowly to your LEFT"),
    ("right", "Turn your head slowly to your RIGHT"),
]
INSTRUCTIONS = dict(POSES)


class Registration:
    def __init__(self, engine, antispoof, name, employee_id, employee_number):
        self.engine = engine
        self.antispoof = antispoof
        self.name = name
        self.employee_id = employee_id
        self.employee_number = employee_number
        self.id = None
        self.lock = None
        self.captured = {pose: [] for pose, _ in POSES}
        self.deadline = time.time() + CFG.max_seconds
        self.finished_at = None
        self.last_sample = 0.0
        # running | complete | already_registered | timeout | cancelled | failed
        self.state = "running"
        self.warning = None             # set when block_duplicates is off
        self.existing_person = None     # details of the matching registered person
        self.saved = 0
        self.message = ""
        self.last = {}                  # last frame's feedback, for status polls
        self._best = None               # (sharpness, jpeg) of the best center sample
        self._checked_duplicate = False

    @property
    def final(self):
        return self.state != "running"

    def counts(self):
        return {pose: len(v) for pose, v in self.captured.items()}

    def pending(self):
        return [p for p, _ in POSES if len(self.captured[p]) < SAMPLES_PER_POSE]

    def instruction(self):
        pending = self.pending()
        return INSTRUCTIONS[pending[0]] if pending else "Registration complete"

    def step(self, frame):
        """Process one frame. Returns the per-frame feedback dict."""
        if self.final:
            return self.last
        if time.time() > self.deadline:
            self._end("timeout", "Timed out before all poses were captured - nothing saved. "
                                 "Try better lighting.")
            return self.last

        pending = self.pending()
        result = analyze(self.engine, self.antispoof, frame, stage="liveness",
                         min_det_score=CFG.min_det_score, min_liveness=CFG.min_liveness,
                         gates=GATES)
        feedback = {**result.to_dict(), "accepted": False, "problem": None,
                    "pose": None, "yaw": None, "pitch": None}
        if result.face is not None:
            feedback["pose"] = pose_of(result.yaw)
            aim = head_pose(result.face.kps, frame.shape[1], frame.shape[0])
            if aim:
                feedback["yaw"], feedback["pitch"] = aim

        if not result.ok:
            feedback["problem"] = result.message
        else:
            pose = feedback["pose"]
            now = time.time()
            if pose in pending and now - self.last_sample >= CFG.sample_cooldown_s:
                emb = np.asarray(self.engine.embed(frame, result.face), dtype=np.float32)
                taken = [e for v in self.captured.values() for e in v]
                if taken and float(np.max(np.stack(taken) @ emb)) > CFG.dedup_sim:
                    feedback["problem"] = "Too similar to a previous sample - move slightly"
                else:
                    self.captured[pose].append(emb)
                    self.last_sample = now
                    feedback["accepted"] = True
                    if pose == "center":
                        self._keep_best(frame, result.face.bbox)
            elif pose not in pending:
                feedback["problem"] = self.instruction()

        self._check_duplicate(feedback)
        if not self.final and not self.pending():
            self._commit()
        self.last = feedback
        return feedback

    def _keep_best(self, frame, bbox):
        score = sharpness(frame, bbox)
        if self._best is None or score > self._best[0]:
            jpeg = face_crop_jpeg(frame, bbox, CFG.image_max_side, CFG.image_jpeg_quality)
            if jpeg:
                self._best = (score, jpeg)

    def _check_duplicate(self, feedback):
        """Compare the first frontal samples with everyone registered."""
        center = self.captured["center"]
        if self._checked_duplicate or len(center) < CFG.duplicate_check_samples:
            return
        self._checked_duplicate = True
        mean = np.mean(center, axis=0)
        match = find_registered(self.engine, mean / np.linalg.norm(mean))
        if match is None:
            return
        self.existing_person = match
        text = already_registered_message(match)
        if CFG.block_duplicates:
            self._end("already_registered", text + " - registration stopped, nothing saved.")
            feedback.update(ok=False, status="already_registered", message=text,
                            accepted=False, problem=text, name=match["name"],
                            employee_id=match["employee_id"],
                            employee_number=match["employee_number"],
                            similarity=match["similarity"])
        else:
            self.warning = text

    def _commit(self):
        vectors = [e for pose, _ in POSES for e in self.captured[pose]]
        try:
            self.engine.store.save_person(self.name, self.employee_id, self.employee_number,
                                          vectors, image=self._best[1] if self._best else None)
        except ValueError as e:
            self._end("failed", str(e))
            return
        self.engine.reload()
        self.saved = len(vectors)
        self._end("complete", f"Registered '{self.name}' with {self.saved} samples.")

    def cancel(self):
        if not self.final:
            self._end("cancelled", "Registration cancelled - nothing saved.")

    def _end(self, state, message):
        self.state, self.message = state, message
        self.finished_at = time.time()
        self._best = None


def find_registered(engine, embedding):
    """Full details of the registered person this embedding matches at
    `duplicate_sim` or above, with the similarity, or None."""
    person, similarity = engine.match(embedding)
    if person is None or similarity < CFG.duplicate_sim:
        return None
    details = engine.store.get_person(person["employee_id"]) or dict(person)
    details.pop("id", None)
    return {**details, "similarity": round(float(similarity), 4)}


def already_registered_message(person):
    return (f"Already registered as {person['name']} "
            f"(employee ID {person['employee_id']}, number {person['employee_number']})")


def check_registered(engine, antispoof, frame):
    """Single-image check: is this live face already registered?

    Returns (already_registered, person, analysis). already_registered is None
    when the frame could not be checked (too close, spoof, blurry...); the
    analysis status and message say why.
    """
    engine.maybe_reload()
    result = analyze(engine, antispoof, frame, stage="recognize",
                     min_det_score=CFG.min_det_score, min_liveness=CFG.min_liveness,
                     require_frontal=True, gates=GATES)
    if result.status not in ("ok", "not_recognized"):
        return None, None, result
    match = find_registered(engine, result.embedding)
    return match is not None, match, result
