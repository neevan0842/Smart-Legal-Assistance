from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.db.models.user import User
from app.schema.auth import UserRegister, UserResponse
from app.service.auth import get_password_hash, get_user_by_email


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(user: UserRegister, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    if await get_user_by_email(email=user.email, db=db) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists"
        )
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email, full_name=user.full_name, hashed_password=hashed_password
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
