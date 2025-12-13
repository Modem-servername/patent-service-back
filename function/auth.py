"""
Authentication Module - SHA-256 Hash-based Login System

Frontend: Hashes with SHA-256 → Sends to server
Backend: Hashes plain password with SHA-256 and compares
"""

import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
from jose import JWTError, jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Read configuration from environment variables
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # Plain text password
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# HTTP Bearer authentication scheme
security = HTTPBearer()


class LoginRequest(BaseModel):
    """Login request schema"""
    password_hash: str  # SHA-256 hashed value from frontend


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # In seconds


def hash_password(plain_password: str) -> str:
    """
    Hash plain password with SHA-256

    Args:
        plain_password: Plain text password

    Returns:
        SHA-256 hash (lowercase hex)
    """
    return hashlib.sha256(plain_password.encode('utf-8')).hexdigest()


def authenticate(client_hash: str) -> bool:
    """
    Authenticate using SHA-256 hash from client

    Args:
        client_hash: Password hashed with SHA-256 from frontend

    Returns:
        Authentication success status
    """
    if not ADMIN_PASSWORD:
        print("[Auth] Warning: ADMIN_PASSWORD not set in environment variables")
        return False

    # Hash server's plain password with SHA-256
    server_hash = hash_password(ADMIN_PASSWORD)
    return client_hash.lower() == server_hash.lower()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate JWT access token

    Args:
        data: Data to include in token
        expires_delta: Expiration time (default: 24 hours)

    Returns:
        JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verify and decode JWT token

    Args:
        token: JWT token

    Returns:
        Token payload

    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        print(f"[Auth] Token verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """
    Get current authenticated user information (dependency)

    Use in protected endpoints:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            ...

    Args:
        credentials: Bearer token from HTTP Authorization header

    Returns:
        User information dictionary

    Raises:
        HTTPException: On authentication failure
    """
    token = credentials.credentials
    payload = verify_token(token)

    # Extract user information from token
    user_type = payload.get("type")
    if user_type != "admin":
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )

    return payload


def login(password_hash: str) -> TokenResponse:
    """
    Process login

    Args:
        password_hash: Password hashed with SHA-256

    Returns:
        JWT token response

    Raises:
        HTTPException: On authentication failure
    """
    if not authenticate(password_hash):
        print(f"[Auth] Login failed: invalid password hash")
        raise HTTPException(
            status_code=401,
            detail="Incorrect password"
        )

    # Generate JWT token
    access_token = create_access_token(
        data={"type": "admin", "authenticated": True}
    )

    print(f"[Auth] Login successful")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    )


def generate_client_password_hash(plain_password: str) -> str:
    """
    Hash plain password with SHA-256 (for client simulation)

    Frontend generates the same hash in JavaScript:
    const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(password));

    Args:
        plain_password: Plain text password

    Returns:
        SHA-256 hash (lowercase hex)
    """
    return hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
