"""
Authentication API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uuid

from database import db

router = APIRouter()

# Security configuration
SECRET_KEY = "igqk_secret_key_change_in_production_2024"  # TODO: Move to environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer for token authentication
security = HTTPBearer()


# ==================== MODELS ====================

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
    full_name: str


class UserInfo(BaseModel):
    """User information"""
    user_id: str
    email: str
    full_name: str
    tier: str
    quota_jobs_remaining: int


# ==================== UTILITIES ====================

def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode JWT access token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = decode_access_token(token)

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


# ==================== ENDPOINTS ====================

@router.post("/register", response_model=Token)
async def register_user(user: UserRegister):
    """Register a new user"""
    # Check if user already exists
    existing_user = db.get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Validate password strength
    if len(user.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )

    # Create user
    user_id = str(uuid.uuid4())
    password_hash = hash_password(user.password)
    api_key = f"igqk_{uuid.uuid4().hex[:32]}"

    success = db.create_user(
        user_id=user_id,
        email=user.email,
        password_hash=password_hash,
        full_name=user.full_name,
        api_key=api_key
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user_id, "email": user.email}
    )

    return Token(
        access_token=access_token,
        user_id=user_id,
        email=user.email,
        full_name=user.full_name
    )


@router.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    """Login existing user"""
    # Get user from database
    db_user = db.get_user_by_email(user.email)

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verify password
    if not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check if user is active
    if not db_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": db_user["user_id"], "email": db_user["email"]}
    )

    return Token(
        access_token=access_token,
        user_id=db_user["user_id"],
        email=db_user["email"],
        full_name=db_user["full_name"]
    )


@router.get("/me", response_model=UserInfo)
async def get_current_user(current_user: dict = Depends(get_current_user_from_token)):
    """Get current user info"""
    return UserInfo(
        user_id=current_user["user_id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        tier=current_user.get("tier", "free"),
        quota_jobs_remaining=current_user.get("quota_jobs_remaining", 10)
    )
