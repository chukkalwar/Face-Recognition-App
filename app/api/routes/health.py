from fastapi import APIRouter

from app.api.deps import ServicesDep

router = APIRouter(tags=["Health"])


@router.get("/", summary="Service health")
def health(services=ServicesDep):
    return {"service": "face-recognition", "status": "ok",
            "liveness_model": bool(services.antispoof and services.antispoof.available),
            "registered_samples": services.engine.size}
