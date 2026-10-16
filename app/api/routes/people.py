"""Registered people."""
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.api.deps import ServicesDep
from app.schemas.people import Deleted, Person

router = APIRouter(prefix="/api/people", tags=["People"])


@router.get("", response_model=List[Person], summary="All registered people")
def people(services=ServicesDep):
    return services.engine.store.list_people()


@router.get("/{employee_id}", response_model=Person, summary="One person")
def person(employee_id: str, services=ServicesDep):
    row = services.engine.store.get_person(employee_id)
    if row is None:
        raise HTTPException(404, f"No registered employee '{employee_id}'")
    return row


@router.get("/{employee_id}/image", summary="Reference face image (JPEG)",
            response_class=Response)
def person_image(employee_id: str, services=ServicesDep):
    image = services.engine.store.get_face_image(employee_id)
    if image is None:
        raise HTTPException(404, f"No stored image for employee_id '{employee_id}'")
    return Response(content=image, media_type="image/jpeg")


@router.delete("/{name}", response_model=Deleted, summary="Delete a person and their data")
def delete_person(name: str, services=ServicesDep):
    if not services.engine.store.delete_person(name):
        raise HTTPException(404, f"No person named '{name}'")
    services.engine.reload()
    return {"deleted": name}
