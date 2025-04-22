from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class UnitBase(BaseModel):
    title: str
    description: Optional[str] = None
    order: int
    module_id: int


class UnitCreate(UnitBase):
    pass


class UnitUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = None


class UnitResponse(UnitBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
