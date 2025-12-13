"""
Analysis Request Management Module
UUID-based request tracking, status management, cancellation functionality
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
    """Database connection context manager"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # dict-like access
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize database and create tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Analysis request table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_requests (
                request_id TEXT PRIMARY KEY,
                patent_number TEXT,
                input_type TEXT,  -- 'patent_number' or 'pdf_upload'
                filename TEXT,  -- Filename for PDF uploads

                -- Request parameters
                max_candidates INTEGER,
                create_detailed_chart BOOLEAN,
                model TEXT,
                follow_up_questions TEXT,  -- JSON array

                -- Status
                status TEXT,  -- 'pending', 'processing', 'completed', 'failed', 'cancelled'

                -- Results
                result_json TEXT,  -- Full analysis result JSON
                markdown_report TEXT,
                pdf_file_path TEXT,  -- Original patent PDF file path

                -- Metadata
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                cancelled_at TEXT,

                -- Statistics
                processing_time_seconds REAL,
                total_cost REAL,
                total_tokens INTEGER,

                -- Errors
                error_message TEXT,
                error_traceback TEXT
            )
        """)

        # Create indexes
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
    Create new analysis request and save to DB

    Returns:
        request_id (str): Generated UUID
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
    """Update request status"""
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
    """Save analysis results (automatically calculates accurate processing time)"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Calculate accurate processing time: difference between started_at and current time
        cursor.execute("""
            SELECT started_at FROM analysis_requests
            WHERE request_id = ?
        """, (request_id,))
        row = cursor.fetchone()

        accurate_processing_time = processing_time  # Default value
        if row and row[0]:
            started_at_str = row[0]
            try:
                started_at = datetime.fromisoformat(started_at_str)
                completed_at = datetime.now()
                accurate_processing_time = (completed_at - started_at).total_seconds()
                print(f"[Request Manager] Accurate processing time: {accurate_processing_time:.2f}s (provided: {processing_time:.2f}s)")
            except Exception as e:
                print(f"[Request Manager] Warning: Could not calculate accurate processing time: {e}")

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
            accurate_processing_time,
            total_cost,
            total_tokens,
            request_id
        ))
        conn.commit()

    print(f"[Request Manager] Saved result for: {request_id}")


def save_error(request_id: str, error_message: str, traceback_str: Optional[str] = None):
    """Save error information"""
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
    """Query request information"""
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
    Check if request has been cancelled

    Args:
        request_id: Request ID

    Returns:
        bool: True if cancelled, False otherwise
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
    Cancel request

    Returns:
        bool: Cancellation success status
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check current status
        cursor.execute("""
            SELECT status FROM analysis_requests
            WHERE request_id = ?
        """, (request_id,))

        row = cursor.fetchone()
        if not row:
            return False

        current_status = row[0]

        # Cannot cancel requests that are already completed or failed
        if current_status in ["completed", "failed", "cancelled"]:
            print(f"[Request Manager] Cannot cancel {request_id}: already {current_status}")
            return False

        # Process cancellation
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
    Query all requests (with pagination)

    Args:
        status: Filter by specific status (optional)
        limit: Maximum number of results
        offset: Starting position
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
    """Delete request (admin only)"""
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
    """Query statistics information"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total request count
        cursor.execute("SELECT COUNT(*) FROM analysis_requests")
        total = cursor.fetchone()[0]

        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*)
            FROM analysis_requests
            GROUP BY status
        """)
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Total cost
        cursor.execute("""
            SELECT SUM(total_cost)
            FROM analysis_requests
            WHERE status = 'completed'
        """)
        total_cost = cursor.fetchone()[0] or 0.0

        # Average processing time
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
    Clean up old pending/processing requests at server startup

    Args:
        timeout_minutes: Consider requests older than this as stale (in minutes)
    """
    from datetime import datetime, timedelta

    cutoff_time = (datetime.now() - timedelta(minutes=timeout_minutes)).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Find stale requests (pending or processing and older than timeout)
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

        # Mark stale requests as failed
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


# Initialize DB at app startup
init_database()

# Clean up stale requests at server startup (pending/processing for 60+ minutes)
cleanup_stale_requests(timeout_minutes=60)
