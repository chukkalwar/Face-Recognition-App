from pydantic import BaseModel


class Person(BaseModel):
    id: int
    name: str
    employee_id: str
    employee_number: str
    created_at: str
    samples: int
    has_image: bool


class Deleted(BaseModel):
    deleted: str
