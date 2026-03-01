from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class RegistrationParams(BaseModel):
    """Parameters for model registration."""
    model_name: str
    checkpoint_path: str
    registry_url: str
    version: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    promote_to: Optional[str] = Field(default=None, pattern=r"^(staging|production)$")


class RegistrationResult(BaseModel):
    """Result of a model registration operation."""
    model_name: str
    version: str
    registry_url: str
    registered_at: datetime
    promoted_to: Optional[str] = None
    artifact_uri: str
