"""
Face recognition API for the Flutter attendance app.

Flow:  1. registration  ->  2. detection  ->  3. attendance (single or group)

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Interactive docs at /docs. Run ONE worker: sessions live in memory.
"""
import logging
import os
import time
import numpy as np
from insightface.app.common import Face
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import Services, require_api_key
from app.api.routes import (attendance, detection, group_attendance, health, people,
                            registration)
from app.core.config import config
from app.core.runtime import configure_opencv
from app.ml.antispoof import load_antispoof
from app.ml.face_engine import FaceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

def _warm_up(engine, antispoof, rounds=2):
    t0 = time.perf_counter()
    try:
        frame = np.full((720, 1280, 3), 128, np.uint8)
        face = Face(bbox=np.array([500.0, 200.0, 780.0, 520.0]),
                    kps=np.array([[590, 320], [690, 320], [640, 390], [605, 450], [675, 450]],
                                 dtype=np.float32),
                    det_score=1.0)
        for _ in range(rounds):
            engine.detect(frame)
            engine.embed(frame, face)
            if antispoof.available:
                antispoof.score(frame, face.bbox.astype(int))
        logger.info("Warm-up done in %.0f ms", (time.perf_counter() - t0) * 1000)
    except Exception:
        logger.exception("Warm-up failed (continuing)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_opencv()
    engine = FaceEngine()
    antispoof = load_antispoof(providers=config.providers())
    app.state.services = Services(engine=engine, antispoof=antispoof)
    logger.info(f"Config: {config.source}")
    logger.info(f"Gallery: {engine.size} embeddings, "
                f"liveness model {'on' if antispoof.available else 'OFF'}")
    if not os.environ.get("API_KEY"):
        logger.warning("API_KEY is not set - the API is open to anyone who can reach it")
    _warm_up(engine, antispoof)
    yield


app = FastAPI(title="Face Recognition Attendance API", version="2.0.0",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(health.router)
protected = [Depends(require_api_key)]
# group before attendance, so /api/attendance/group/... is matched first
for module in (registration, detection, group_attendance, attendance, people):
    app.include_router(module.router, dependencies=protected)
