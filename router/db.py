import sqlite3
from fastapi import APIRouter
import hashlib
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("data.db")

router = APIRouter(
    prefix="/db",
    tags=["db"]
)

@contextmanager
def get_db_connection():
    """DB 연결 컨텍스트 매니저"""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """데이터베이스 초기화"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # prompt 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # password 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS password (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                password TEXT NOT NULL UNIQUE
            )
        """)

        conn.commit()
        print("[DB] data.db initialized")


@router.get("/insertPrompt")
async def insertPrompt(input: str):
    """프롬프트 저장 (SQL Injection 방지)"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Parameterized query로 SQL Injection 방지
        cursor.execute("INSERT INTO prompt (prompt) VALUES (?)", (input,))
        conn.commit()
    return {"success": True}


@router.get("/login")
async def login(input: str):
    """로그인 확인 (SQL Injection 방지)"""
    target = input.encode()
    sha256_hash = hashlib.sha256()
    sha256_hash.update(target)
    result = sha256_hash.hexdigest()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Parameterized query로 SQL Injection 방지
        cursor.execute("SELECT * FROM password WHERE password = ?", (result,))
        ans = cursor.fetchall()

    return {"authenticated": len(ans) > 0}


# 앱 시작 시 DB 초기화
init_db()