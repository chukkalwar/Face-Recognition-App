"""
Face detection and recognition with insightface on ONNX Runtime.

The `buffalo_sc` bundle: SCRFD-500M detector and a MobileFaceNet recognizer.
Detection and embedding are separate calls, so a frame that fails a cheap
check (distance, blur) never pays for the recognizer.

Changing MODEL_BUNDLE changes the recognizer, which invalidates every stored
embedding: re-register everyone after a change.
"""
import threading

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from app.core.config import config
from app.core.runtime import insightface_session_options
from app.db.store import FaceStore

MODEL_BUNDLE = "buffalo_sc"


class FaceEngine:
    def __init__(self, store=None, providers=None):
        providers = providers or config.providers()
        self.rec_thresh = config.detection.rec_thresh
        self.store = store or FaceStore()
        with insightface_session_options():
            self.app = FaceAnalysis(name=MODEL_BUNDLE, providers=providers,
                                    allowed_modules=["detection", "recognition"])
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        self.app.prepare(ctx_id=ctx_id, det_size=config.det_size_wh())
        self._index_lock = threading.Lock()
        self._version = -1
        self.reload()

    # -- gallery -------------------------------------------------------------

    def reload(self):
        people, embeddings = self.store.load_all()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        with self._index_lock:
            self.people = people
            self.embeddings = embeddings / norms
            self._version = self.store.version()

    def maybe_reload(self):
        """Reload if the store changed since the last load. Cheap to call."""
        if self.store.version() != self._version:
            self.reload()
            return True
        return False

    @property
    def size(self):
        return len(self.people)

    # -- inference -------------------------------------------------------------

    def detect(self, frame_bgr):
        """Faces with bbox, 5-point kps and det_score. No embedding yet."""
        bboxes, kpss = self.app.det_model.detect(frame_bgr, max_num=0, metric="default")
        return [Face(bbox=bboxes[i, 0:4],
                     kps=kpss[i] if kpss is not None else None,
                     det_score=float(bboxes[i, 4]))
                for i in range(bboxes.shape[0])]

    def embed(self, frame_bgr, face):
        """L2-normalised 512-d embedding, cut from the full-resolution frame."""
        self.app.models["recognition"].get(frame_bgr, face)
        return face.normed_embedding

    def match(self, embedding):
        """(person dict or None, cosine similarity). None below rec_thresh."""
        with self._index_lock:
            people, embeddings = self.people, self.embeddings
        if not people:
            return None, 0.0
        sims = embeddings @ embedding
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (people[best] if score >= self.rec_thresh else None), score
