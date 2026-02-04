"""
Authentication API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional

router = APIRouter()


class UserRegister(BaseModel):
    """User registration"""
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    """User login"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """JWT token"""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


@router.post("/register", response_model=Token)
async def register_user(user: UserRegister):
    """Register a new user"""
    # TODO: Implement actual registration with database
    return Token(
        access_token="mock_token_" + user.email,
        user_id="user_123",
        email=user.email
    )


@router.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    """Login existing user"""
    # TODO: Implement actual authentication
    return Token(
        access_token="mock_token_" + user.email,
        user_id="user_123",
        email=user.email
    )


@router.get("/me")
async def get_current_user():
    """Get current user info"""
    # TODO: Implement JWT verification
    return {
        "user_id": "user_123",
        "email": "user@example.com",
        "full_name": "Demo User",
        "plan": "free",
        "credits": {
            "training_hours": 10,
            "compressions": 5,
            "api_requests": 1000
        }
    }
