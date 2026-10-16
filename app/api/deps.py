"""Shared request dependencies: the loaded models, image decoding, API key."""
import hmac
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi import Depends, File, Header, HTTPException, Request, UploadFile

from app.core.config import config
from app.services.sessions import SessionRegistry


@dataclass
class Services:
    engine: object
    antispoof: object
    registrations: SessionRegistry = field(default_factory=SessionRegistry)
    attendance: SessionRegistry = field(default_factory=SessionRegistry)
    groups: SessionRegistry = field(default_factory=SessionRegistry)


def get_services(request: Request) -> Services:
    return request.app.state.services


def require_api_key(x_api_key: str = Header(None, description="Needed when the "
                                            "server sets API_KEY")):
    """When the API_KEY environment variable is set, every /api call must send
    it in the X-API-Key header. Unset, the API is open: only for a trusted
    network or local testing."""
    expected = os.environ.get("API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(401, "Missing or invalid X-API-Key")


class DecodedImage:
    def __init__(self, raw, frame):
        self.raw, self.frame = raw, frame


def read_image(image: UploadFile = File(..., description="JPEG or PNG of the full "
                                        "camera frame")) -> DecodedImage:
    raw = image.file.read(config.api.max_image_mb * 1024 * 1024 + 1)
    if not raw:
        raise HTTPException(400, "Empty image")
    if len(raw) > config.api.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {config.api.max_image_mb} MB")
    # IMREAD_COLOR applies EXIF rotation, so phone photos arrive upright
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image - send a JPEG or PNG")
    return DecodedImage(raw, frame)


ServicesDep = Depends(get_services)
ImageDep = Depends(read_image)
