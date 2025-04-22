from typing import List, Optional
from pydantic import BaseModel


class TOCMetadata(BaseModel):
    generated_at: str
    source_file: str


class TOCUnit(BaseModel):
    id: int
    title: str
    description: Optional[str] = ""
    order: int
    content_generated: bool


class TOCModule(BaseModel):
    id: int
    title: str
    description: Optional[str] = ""
    order: int
    units: List[TOCUnit]


class TOCResponse(BaseModel):
    course_id: int
    course_title: str
    modules: List[TOCModule]
    metadata: TOCMetadata
