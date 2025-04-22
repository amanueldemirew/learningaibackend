from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.schemas.course import ContentGenerationOptions


class SummaryGenerationRequest(BaseModel):
    unit_id: int = Field(..., description="Unit ID to generate summary for")
    length: str = Field(
        "medium", description="Length of the summary (short, medium, long)"
    )
    style: str = Field(
        "academic",
        description="Style of the summary (academic, conversational, technical)",
    )
    difficulty_level: str = Field(
        "intermediate",
        description="Difficulty level (beginner, intermediate, advanced)",
    )
    target_audience: str = Field(
        "students", description="Target audience (students, professionals, general)"
    )
    options: Optional[ContentGenerationOptions] = None


class ExampleGenerationRequest(BaseModel):
    unit_id: int = Field(..., description="Unit ID to generate examples for")
    count: int = Field(3, description="Number of examples to generate")
    difficulty_level: str = Field(
        "intermediate",
        description="Difficulty level (beginner, intermediate, advanced)",
    )
    include_solutions: bool = Field(True, description="Whether to include solutions")
    format: str = Field(
        "markdown", description="Format of the examples (markdown, latex)"
    )
    options: Optional[ContentGenerationOptions] = None


class ExerciseGenerationRequest(BaseModel):
    unit_id: int = Field(..., description="Unit ID to generate exercises for")
    count: int = Field(5, description="Number of exercises to generate")
    difficulty_level: str = Field(
        "intermediate",
        description="Difficulty level (beginner, intermediate, advanced)",
    )
    include_answer_key: bool = Field(True, description="Whether to include answer key")
    format: str = Field(
        "markdown", description="Format of the exercises (markdown, latex)"
    )
    options: Optional[ContentGenerationOptions] = None


class GenerationResponse(BaseModel):
    data: Dict[str, Any]
    meta: Dict[str, Any]


class BatchGenerationResponse(BaseModel):
    data: Dict[str, Any]
    meta: Dict[str, Any]


class WebSocketMessage(BaseModel):
    batch_id: str
    status: str
    progress: int
    message: str
    timestamp: str
