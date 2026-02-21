from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from datetime import datetime
from uuid import UUID


class UserResponse(BaseModel):
    id: UUID = Field(..., description="The unique identifier of the user")
    email: EmailStr = Field(..., description="The email address of the user")
    full_name: Optional[str] = Field(None, description="The full name of the user")
    created_at: datetime = Field(..., description="The creation timestamp of the user")

    model_config = ConfigDict(from_attributes=True)


class UserRegister(BaseModel):
    email: EmailStr = Field(..., description="The email address of the user")
    full_name: Optional[str] = Field(None, description="The full name of the user")
    password: str = Field(..., min_length=8, description="The password for the user")
