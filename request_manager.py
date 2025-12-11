"""
분석 요청 관리 모듈
UUID 기반 요청 추적, 상태 관리, 취소 기능
"""

import sqlite3
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("analysis_requests.db")


@contextmanager
def get_db_connection():
    """DB 연결 컨텍스트 매니저"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # dict-like access
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """데이터베이스 초기화 및 테이블 생성"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 분석 요청 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_requests (
                request_id TEXT PRIMARY KEY,
                patent_number TEXT,
                input_type TEXT,  -- 'patent_number' or 'pdf_upload'
                filename TEXT,  -- PDF 업로드 시 파일명

                -- 요청 파라미터
                max_candidates INTEGER,
                create_detailed_chart BOOLEAN,
                model TEXT,
                follow_up_questions TEXT,  -- JSON array

                -- 상태
                status TEXT,  -- 'pending', 'processing', 'completed', 'failed', 'cancelled'

                -- 결과
                result_json TEXT,  -- 전체 분석 결과 JSON
                markdown_report TEXT,
                pdf_file_path TEXT,  -- 원본 특허 PDF 파일 경로

                -- 메타데이터
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                cancelled_at TEXT,

                -- 통계
                processing_time_seconds REAL,
                total_cost REAL,
                total_tokens INTEGER,

                -- 에러
                error_message TEXT,
                error_traceback TEXT
            )
        """)

        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON analysis_requests(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON analysis_requests(created_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_patent_number
            ON analysis_requests(patent_number)
        """)

        conn.commit()
        print("[DB] Analysis requests database initialized")


def create_request(
    patent_number: Optional[str] = None,
    filename: Optional[str] = None,
    max_candidates: int = 10,
    create_detailed_chart: bool = True,
    model: str = "gpt-5",
    follow_up_questions: Optional[str] = None  # Changed from list to str
) -> str:
    """
    새 분석 요청 생성 및 DB 저장

    Returns:
        request_id (str): 생성된 UUID
    """
    request_id = str(uuid.uuid4())

    input_type = "pdf_upload" if filename else "patent_number"

    # Store follow_up_questions as-is (string)
    # No need to json.dumps since it's already a string
    follow_up_questions_str = follow_up_questions if follow_up_questions else None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_requests (
                request_id, patent_number, input_type, filename,
                max_candidates, create_detailed_chart, model, follow_up_questions,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            patent_number,
            input_type,
            filename,
            max_candidates,
            create_detailed_chart,
            model,
            follow_up_questions_str,
            "pending",
            datetime.now().isoformat()
        ))
        conn.commit()

    print(f"[Request Manager] Created request: {request_id}")
    return request_id


def update_status(request_id: str, status: str):
    """요청 상태 업데이트"""
    timestamp_field = None
    if status == "processing":
        timestamp_field = "started_at"
    elif status == "completed":
        timestamp_field = "completed_at"
    elif status == "cancelled":
        timestamp_field = "cancelled_at"

    with get_db_connection() as conn:
        cursor = conn.cursor()

        if timestamp_field:
            cursor.execute(f"""
                UPDATE analysis_requests
                SET status = ?, {timestamp_field} = ?
                WHERE request_id = ?
            """, (status, datetime.now().isoformat(), request_id))
        else:
            cursor.execute("""
                UPDATE analysis_requests
                SET status = ?
                WHERE request_id = ?
            """, (status, request_id))

        conn.commit()

    print(f"[Request Manager] Updated {request_id} status to: {status}")


def save_result(
    request_id: str,
    result_json: Dict[Any, Any],
    markdown_report: str,
    processing_time: float,
    total_cost: float,
    total_tokens: int,
    pdf_file_path: Optional[str] = None
):
    """분석 결과 저장"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analysis_requests
            SET status = ?,
                result_json = ?,
                markdown_report = ?,
                pdf_file_path = ?,
                completed_at = ?,
                processing_time_seconds = ?,
                total_cost = ?,
                total_tokens = ?
            WHERE request_id = ?
        """, (
            "completed",
            json.dumps(result_json, ensure_ascii=False),
            markdown_report,
            pdf_file_path,
            datetime.now().isoformat(),
            processing_time,
            total_cost,
            total_tokens,
            request_id
        ))
        conn.commit()

    print(f"[Request Manager] Saved result for: {request_id}")


def save_error(request_id: str, error_message: str, traceback_str: Optional[str] = None):
    """에러 정보 저장"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analysis_requests
            SET status = ?,
                error_message = ?,
                error_traceback = ?,
                completed_at = ?
            WHERE request_id = ?
        """, (
            "failed",
            error_message,
            traceback_str,
            datetime.now().isoformat(),
            request_id
        ))
        conn.commit()

    print(f"[Request Manager] Saved error for: {request_id}")


def get_request(request_id: str) -> Optional[Dict]:
    """요청 정보 조회"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM analysis_requests
            WHERE request_id = ?
        """, (request_id,))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def is_cancelled(request_id: str) -> bool:
    """
    요청이 취소되었는지 확인

    Args:
        request_id: 요청 ID

    Returns:
        bool: 취소되었으면 True, 아니면 False
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT status FROM analysis_requests
            WHERE request_id = ?
        """, (request_id,))

        row = cursor.fetchone()
        if not row:
            return False

        return row[0] == "cancelled"


def cancel_request(request_id: str) -> bool:
    """
    요청 취소

    Returns:
        bool: 취소 성공 여부
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 현재 상태 확인
        cursor.execute("""
            SELECT status FROM analysis_requests
            WHERE request_id = ?
        """, (request_id,))

        row = cursor.fetchone()
        if not row:
            return False

        current_status = row[0]

        # 이미 완료되었거나 실패한 요청은 취소 불가
        if current_status in ["completed", "failed", "cancelled"]:
            print(f"[Request Manager] Cannot cancel {request_id}: already {current_status}")
            return False

        # 취소 처리
        cursor.execute("""
            UPDATE analysis_requests
            SET status = ?, cancelled_at = ?
            WHERE request_id = ?
        """, ("cancelled", datetime.now().isoformat(), request_id))

        conn.commit()
        print(f"[Request Manager] Cancelled request: {request_id}")
        return True


def get_all_requests(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> list:
    """
    모든 요청 조회 (페이징)

    Args:
        status: 특정 상태만 필터링 (선택)
        limit: 최대 개수
        offset: 시작 위치
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM analysis_requests
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (status, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM analysis_requests
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_request(request_id: str) -> bool:
    """요청 삭제 (관리자용)"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM analysis_requests
            WHERE request_id = ?
        """, (request_id,))
        conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            print(f"[Request Manager] Deleted request: {request_id}")
        return deleted


def get_statistics() -> Dict:
    """통계 정보 조회"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 전체 요청 수
        cursor.execute("SELECT COUNT(*) FROM analysis_requests")
        total = cursor.fetchone()[0]

        # 상태별 개수
        cursor.execute("""
            SELECT status, COUNT(*)
            FROM analysis_requests
            GROUP BY status
        """)
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # 총 비용
        cursor.execute("""
            SELECT SUM(total_cost)
            FROM analysis_requests
            WHERE status = 'completed'
        """)
        total_cost = cursor.fetchone()[0] or 0.0

        # 평균 처리 시간
        cursor.execute("""
            SELECT AVG(processing_time_seconds)
            FROM analysis_requests
            WHERE status = 'completed'
        """)
        avg_time = cursor.fetchone()[0] or 0.0

        return {
            "total_requests": total,
            "status_counts": status_counts,
            "total_cost": round(total_cost, 4),
            "average_processing_time": round(avg_time, 2)
        }


def cleanup_stale_requests(timeout_minutes: int = 60):
    """
    서버 시작 시 오래된 pending/processing 요청들을 정리

    Args:
        timeout_minutes: 이 시간(분) 이상 지난 요청을 stale로 간주
    """
    from datetime import datetime, timedelta

    cutoff_time = (datetime.now() - timedelta(minutes=timeout_minutes)).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Stale 요청 찾기 (pending 또는 processing 상태이면서 오래된 것들)
        cursor.execute("""
            SELECT request_id, status, created_at, started_at
            FROM analysis_requests
            WHERE status IN ('pending', 'processing')
            AND (
                (status = 'pending' AND created_at < ?)
                OR (status = 'processing' AND COALESCE(started_at, created_at) < ?)
            )
        """, (cutoff_time, cutoff_time))

        stale_requests = cursor.fetchall()

        if not stale_requests:
            print("[Request Manager] No stale requests found")
            return

        print(f"[Request Manager] Found {len(stale_requests)} stale request(s), marking as failed...")

        # Stale 요청들을 failed로 변경
        for row in stale_requests:
            request_id = row[0]
            status = row[1]

            cursor.execute("""
                UPDATE analysis_requests
                SET status = ?,
                    error_message = ?,
                    completed_at = ?
                WHERE request_id = ?
            """, (
                "failed",
                f"Request timed out or server was restarted while in '{status}' state",
                datetime.now().isoformat(),
                request_id
            ))

            print(f"  - {request_id}: {status} -> failed")

        conn.commit()
        print(f"[Request Manager] Cleanup complete: {len(stale_requests)} request(s) marked as failed")


# 앱 시작 시 DB 초기화
init_database()

# 서버 시작 시 stale 요청 정리 (60분 이상 pending/processing 상태인 것들)
cleanup_stale_requests(timeout_minutes=60)
