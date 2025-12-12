"""
인증 모듈 - SHA-256 해시 기반 로그인 시스템

프론트엔드: SHA-256으로 해시 → 서버 전송
백엔드: 평문 비밀번호를 SHA-256으로 해시하여 비교
"""

import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# 환경 변수에서 설정 읽기
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # 평문 비밀번호
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24시간

# HTTP Bearer 인증 스키마
security = HTTPBearer()


class LoginRequest(BaseModel):
    """로그인 요청 스키마"""
    password_hash: str  # 프론트엔드에서 SHA-256 해시된 값


class TokenResponse(BaseModel):
    """토큰 응답 스키마"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 초 단위


def hash_password(plain_password: str) -> str:
    """
    평문 비밀번호를 SHA-256으로 해시

    Args:
        plain_password: 평문 비밀번호

    Returns:
        SHA-256 해시 (소문자 hex)
    """
    return hashlib.sha256(plain_password.encode('utf-8')).hexdigest()


def authenticate(client_hash: str) -> bool:
    """
    클라이언트에서 전송한 SHA-256 해시로 인증

    Args:
        client_hash: 프론트엔드에서 SHA-256 해시한 비밀번호

    Returns:
        인증 성공 여부
    """
    if not ADMIN_PASSWORD:
        print("[Auth] Warning: ADMIN_PASSWORD not set in environment variables")
        return False

    # 서버의 평문 비밀번호를 SHA-256으로 해시
    server_hash = hash_password(ADMIN_PASSWORD)

    # 클라이언트 해시와 비교
    return client_hash == server_hash


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    JWT 액세스 토큰 생성

    Args:
        data: 토큰에 포함할 데이터
        expires_delta: 만료 시간 (기본값: 24시간)

    Returns:
        JWT 토큰 문자열
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
    JWT 토큰 검증 및 디코딩

    Args:
        token: JWT 토큰

    Returns:
        토큰 페이로드

    Raises:
        HTTPException: 토큰이 유효하지 않은 경우
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
    현재 인증된 사용자 정보 가져오기 (의존성)

    보호된 엔드포인트에서 사용:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            ...

    Args:
        credentials: HTTP Authorization 헤더의 Bearer 토큰

    Returns:
        사용자 정보 딕셔너리

    Raises:
        HTTPException: 인증 실패 시
    """
    token = credentials.credentials
    payload = verify_token(token)

    # 토큰에서 사용자 정보 추출
    user_type = payload.get("type")
    if user_type != "admin":
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )

    return payload


def login(password_hash: str) -> TokenResponse:
    """
    로그인 처리

    Args:
        password_hash: SHA-256으로 해시된 비밀번호

    Returns:
        JWT 토큰 응답

    Raises:
        HTTPException: 인증 실패 시
    """
    if not authenticate(password_hash):
        print(f"[Auth] Login failed: invalid password hash")
        raise HTTPException(
            status_code=401,
            detail="Incorrect password"
        )

    # JWT 토큰 생성
    access_token = create_access_token(
        data={"type": "admin", "authenticated": True}
    )

    print(f"[Auth] Login successful")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # 초 단위로 변환
    )


def generate_client_password_hash(plain_password: str) -> str:
    """
    평문 비밀번호를 SHA-256으로 해시 (클라이언트 시뮬레이션용)

    프론트엔드에서는 JavaScript로 동일한 해시를 생성:
    const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(password));

    Args:
        plain_password: 평문 비밀번호

    Returns:
        SHA-256 해시 (소문자 hex)
    """
    return hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
