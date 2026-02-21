from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models.user import User
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()


async def get_user_by_email(email: str, db: AsyncSession):
    """Get a user by email"""
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    return result.scalars().first()


def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password):
    return password_hash.hash(password)
