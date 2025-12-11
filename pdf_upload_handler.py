"""
PDF Upload Handler for Patent Analysis
사용자가 직접 PDF를 업로드하여 특허 침해 분석을 수행
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from pathlib import Path
import time
import asyncio

# 기존 모듈에서 분석 로직 import
from infringement_search_v2 import (
    SimplifiedInfringementAnalyzer,
    format_analysis_report
)

# Request manager import
import request_manager

# main.py에서 PDF 분석 함수 import
import sys
sys.path.append(str(Path(__file__).parent))

router = APIRouter()

# 업로드된 파일 저장 디렉토리 (임시)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 영구 보관용 PDF 디렉토리
PATENT_PDF_DIR = Path("patent_pdfs")
PATENT_PDF_DIR.mkdir(exist_ok=True)

# Analyzer 인스턴스 (전역)
analyzer_instance = None


def init_analyzer(openai_api_key: str, tavily_api_key: Optional[str] = None):
    """Analyzer 초기화"""
    global analyzer_instance
    analyzer_instance = SimplifiedInfringementAnalyzer(
        api_key=openai_api_key,
        tavily_api_key=tavily_api_key
    )


def check_cancellation(request_id: str) -> bool:
    """
    요청이 취소되었는지 확인

    Returns:
        True if cancelled, False otherwise
    """
    if request_manager.is_cancelled(request_id):
        print(f"[PDF Upload] Analysis cancelled: {request_id}")
        return True
    return False


class CancelledException(Exception):
    """분석이 취소되었을 때 발생하는 예외"""
    pass


async def perform_pdf_analysis_background(
    request_id: str,
    temp_filepath: Path,
    temp_permanent_filepath: Path,
    max_candidates: int,
    create_detailed_chart: bool,
    model: str,
    questions_list: Optional[List[str]],
    file_filename: str
):
    """백그라운드에서 PDF 분석 수행"""
    import time as time_module
    import traceback
    import json

    start_time = time_module.time()
    permanent_filepath = None

    try:
        # Update status to processing
        request_manager.update_status(request_id, "processing")

        # 취소 확인
        if check_cancellation(request_id):
            raise CancelledException("Analysis cancelled by user")

        # PDF 분석
        print(f"[PDF Upload] Starting PDF analysis...")
        from main import analyze_pdf_file
        patent_data = await analyze_pdf_file(str(temp_filepath))

        if not patent_data:
            raise Exception("Failed to analyze PDF - could not extract patent data")

        print(f"[PDF Upload] PDF analysis completed")
        print(f"[PDF Upload] Title: {patent_data.get('title', 'N/A')}")
        print(f"[PDF Upload] Claims count: {len(patent_data.get('claims', []))}")

        # 취소 확인
        if check_cancellation(request_id):
            raise CancelledException("Analysis cancelled by user")

        # 침해 분석 수행
        print(f"[PDF Upload] Performing infringement analysis with model {model}...")
        analysis_result = await analyzer_instance.analyze_infringement(
            patent_data=patent_data,
            max_candidates=max_candidates,
            create_detailed_chart=create_detailed_chart,
            model=model,
            follow_up_questions=questions_list
        )

        # 취소 확인
        if check_cancellation(request_id):
            raise CancelledException("Analysis cancelled by user")

        # 보고서 생성
        print(f"[PDF Upload] Generating report...")
        markdown_report = format_analysis_report(analysis_result)

        print(f"[PDF Upload] Analysis completed successfully")

        # 특허번호로 파일명 변경
        patent_number = analysis_result.issued_number or "UNKNOWN"
        safe_patent_number = "".join(c for c in patent_number if c.isalnum() or c in ('-', '_'))
        permanent_filename = f"{safe_patent_number}.pdf"
        permanent_filepath = PATENT_PDF_DIR / permanent_filename

        # 기존 파일이 있으면 삭제
        if permanent_filepath.exists():
            os.remove(permanent_filepath)
            print(f"[PDF Upload] Removed existing file: {permanent_filepath}")

        # 임시 파일을 특허번호 파일명으로 변경
        shutil.move(str(temp_permanent_filepath), str(permanent_filepath))
        print(f"[PDF Upload] Renamed to patent number: {permanent_filepath}")

        # Calculate processing time
        processing_time = time_module.time() - start_time

        # 비용 및 토큰 집계
        # PDF 분석 비용 (analyze_pdf_file에서 발생)
        pdf_input_tokens = patent_data.get('_input_tokens', 0)
        pdf_output_tokens = patent_data.get('_output_tokens', 0)
        pdf_cost = patent_data.get('_cost', 0.0)
        pdf_total_tokens = pdf_input_tokens + pdf_output_tokens

        # 침해 분석 비용 (analyzer_instance에서 발생)
        infringement_input_tokens = analyzer_instance.total_input_tokens
        infringement_output_tokens = analyzer_instance.total_output_tokens
        infringement_cost = analyzer_instance.total_cost
        infringement_tokens = infringement_input_tokens + infringement_output_tokens

        # 전체 합산
        total_cost = pdf_cost + infringement_cost
        total_tokens = pdf_total_tokens + infringement_tokens

        print(f"[PDF Upload] PDF Analysis: ${pdf_cost:.4f} ({pdf_total_tokens:,} tokens)")
        print(f"[PDF Upload] Infringement Analysis: ${infringement_cost:.4f} ({infringement_tokens:,} tokens)")
        print(f"[PDF Upload] Total: ${total_cost:.4f} ({total_tokens:,} tokens)")

        # Prepare result JSON
        result_json = {
            "success": True,
            "request_id": request_id,
            "filename": file_filename,
            "analysis": {
                "title": analysis_result.title,
                "applicant": analysis_result.applicant,
                "issued_number": analysis_result.issued_number,
                "application_date": analysis_result.application_date,
                "independent_claims": analysis_result.independent_claims,
                "technology_summary": analysis_result.technology_summary,
                "potentially_matching_companies": [
                    {
                        "company": c.company,
                        "product_service": c.product_service,
                        "matching_feature": c.matching_feature,
                        "relevance_score": c.relevance_score
                    }
                    for c in analysis_result.potentially_matching_companies
                ],
                "independent_claim_analyses": [
                    {
                        "claim_number": ca.claim_number,
                        "claim_original": ca.claim_original,
                        "claim_korean": ca.claim_korean,
                        "claim_chart": [
                            {
                                "claim_element": elem.claim_element,
                                "product_feature": elem.product_feature,
                                "comment": elem.comment,
                                "infringement_likelihood": elem.infringement_likelihood
                            }
                            for elem in ca.claim_chart
                        ] if ca.claim_chart else []
                    }
                    for ca in analysis_result.independent_claim_analyses
                ],
                "follow_up_responses": [
                    {
                        "question": fr.question,
                        "answer": fr.answer
                    }
                    for fr in analysis_result.follow_up_responses
                ],
                "analysis_notes": analysis_result.analysis_notes
            },
            "markdown_report": markdown_report
        }

        # Save results to DB
        request_manager.save_result(
            request_id=request_id,
            result_json=result_json,
            markdown_report=markdown_report,
            processing_time=processing_time,
            total_cost=total_cost,
            total_tokens=total_tokens,
            pdf_file_path=str(permanent_filepath)
        )

    except CancelledException as e:
        # 취소 처리 (이미 cancelled 상태이므로 추가 작업 불필요)
        print(f"[PDF Upload] Analysis cancelled: {request_id}")
        # 임시 파일 정리
        try:
            if temp_permanent_filepath.exists():
                os.remove(temp_permanent_filepath)
                print(f"[PDF Upload] Cleaned up temporary file after cancellation")
        except Exception as cleanup_e:
            print(f"[PDF Upload] Warning: Could not clean up after cancellation: {cleanup_e}")

    except Exception as e:
        # Save error to DB
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        request_manager.save_error(request_id, error_msg, error_traceback)

        print(f"[PDF Upload] Error: {e}")
        traceback.print_exc()

        # 에러 시 파일 정리
        try:
            if permanent_filepath and permanent_filepath.exists():
                os.remove(permanent_filepath)
            elif temp_permanent_filepath.exists():
                os.remove(temp_permanent_filepath)
        except Exception as cleanup_e:
            print(f"[PDF Upload] Warning: Could not clean up after error: {cleanup_e}")

    finally:
        # 임시 파일 삭제
        try:
            if temp_filepath.exists():
                os.remove(temp_filepath)
                print(f"[PDF Upload] Temporary file deleted: {temp_filepath}")
        except Exception as e:
            print(f"[PDF Upload] Warning: Could not delete temporary file: {e}")


class PDFUploadAnalysisRequest(BaseModel):
    """PDF 업로드 분석 요청 스키마"""
    max_candidates: int = 10
    create_detailed_chart: bool = True
    model: str = "gpt-5"
    follow_up_questions: Optional[str] = None  # Changed from List[str] to str


@router.post("/upload-pdf-analysis")
async def upload_pdf_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_candidates: int = Form(10),
    create_detailed_chart: bool = Form(True),
    model: str = Form("gpt-5"),
    follow_up_questions: Optional[str] = Form(None)
):
    """
    PDF 파일을 업로드하여 특허 침해 분석 수행 (백그라운드)

    분석은 백그라운드에서 실행되며, 즉시 request_id를 반환합니다.
    /analysis-status/{request_id}로 상태를 확인하고,
    /cancel-analysis/{request_id}로 취소할 수 있습니다.

    Args:
        file: PDF 파일 (multipart/form-data)
        max_candidates: 최대 침해 후보 수
        create_detailed_chart: 상세 Claim Chart 생성 여부
        model: 사용할 AI 모델 (gpt-5, gpt-4o, etc.)
        follow_up_questions: 추가 질문 (JSON 문자열 또는 쉼표로 구분)

    Returns:
        request_id 및 상태 조회 URL
    """
    import time as time_module
    import traceback
    import json

    print(f"\n[PDF Upload] ===== PDF Upload Analysis Request =====")
    print(f"[PDF Upload] File: {file.filename}")
    print(f"[PDF Upload] Max Candidates: {max_candidates}")
    print(f"[PDF Upload] Create Detailed Chart: {create_detailed_chart}")
    print(f"[PDF Upload] AI Model: {model}")
    print(f"[PDF Upload] Follow-up Questions: {follow_up_questions}")

    # Analyzer 확인
    if not analyzer_instance:
        raise HTTPException(500, "Analyzer not initialized - check API keys")

    # 파일 확장자 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")

    # Follow-up questions 파싱 (convert string to list by splitting on newlines)
    questions_list = None
    if follow_up_questions:
        # Split by newlines and filter out empty lines
        questions_list = [q.strip() for q in follow_up_questions.split('\n') if q.strip()]
        print(f"[PDF Upload] Parsed {len(questions_list)} follow-up question(s)")

    # Create request ID and save to DB
    # Store original follow_up_questions string (not the parsed list)
    request_id = request_manager.create_request(
        filename=file.filename,
        max_candidates=max_candidates,
        create_detailed_chart=create_detailed_chart,
        model=model,
        follow_up_questions=follow_up_questions  # Store original string
    )

    print(f"[PDF Upload] Request ID: {request_id}")

    # 임시 파일 저장
    timestamp = int(time.time())
    temp_filename = f"upload_{timestamp}_{file.filename}"
    temp_filepath = UPLOAD_DIR / temp_filename

    # 영구 보관용 파일명 (임시로 request_id 기반, 나중에 특허번호로 변경)
    temp_permanent_filename = f"{request_id}_{file.filename}"
    temp_permanent_filepath = PATENT_PDF_DIR / temp_permanent_filename
    permanent_filepath = None  # Will be set after getting patent number

    # 파일 저장
    print(f"[PDF Upload] Saving uploaded file to: {temp_filepath}")

    with open(temp_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(temp_filepath)
    print(f"[PDF Upload] File saved successfully ({file_size} bytes)")

    # 임시 영구 보관용 복사본 생성
    shutil.copy2(temp_filepath, temp_permanent_filepath)
    print(f"[PDF Upload] Temporary permanent copy saved to: {temp_permanent_filepath}")

    # 백그라운드 태스크로 분석 시작
    background_tasks.add_task(
        perform_pdf_analysis_background,
        request_id=request_id,
        temp_filepath=temp_filepath,
        temp_permanent_filepath=temp_permanent_filepath,
        max_candidates=max_candidates,
        create_detailed_chart=create_detailed_chart,
        model=model,
        questions_list=questions_list,
        file_filename=file.filename
    )

    print(f"[PDF Upload] Background analysis task started for request: {request_id}")

    # 즉시 request_id 반환
    return JSONResponse(content={
        "success": True,
        "request_id": request_id,
        "message": "Analysis started in background",
        "status_url": f"/analysis-status/{request_id}",
        "cancel_url": f"/cancel-analysis/{request_id}"
    })


@router.post("/analyze-pdf-only")
async def analyze_pdf_only(file: UploadFile = File(...)):
    """
    PDF 파일만 분석 (침해 분석 없이 특허 정보만 추출)

    Args:
        file: PDF 파일

    Returns:
        특허 정보 (제목, 청구항, 설명 등)
    """
    import time as time_module
    import traceback
    import json

    print(f"\n[PDF Upload] ===== PDF Only Analysis Request =====")
    print(f"[PDF Upload] File: {file.filename}")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")

    # Create request ID for PDF-only analysis
    request_id = request_manager.create_request(
        filename=file.filename,
        max_candidates=0,  # PDF-only, no infringement search
        create_detailed_chart=False,
        model="N/A",
        follow_up_questions=None
    )

    print(f"[PDF Upload] Request ID: {request_id}")

    timestamp = int(time.time())
    temp_filename = f"upload_{timestamp}_{file.filename}"
    temp_filepath = UPLOAD_DIR / temp_filename

    # 영구 보관용 파일명 (임시로 request_id 기반, 나중에 특허번호로 변경)
    temp_permanent_filename = f"{request_id}_{file.filename}"
    temp_permanent_filepath = PATENT_PDF_DIR / temp_permanent_filename
    permanent_filepath = None  # Will be set after getting patent number

    start_time = time_module.time()

    try:
        # 파일 저장
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[PDF Upload] File saved: {temp_filepath}")

        # 임시 영구 보관용 복사본 생성
        shutil.copy2(temp_filepath, temp_permanent_filepath)
        print(f"[PDF Upload] Temporary permanent copy saved to: {temp_permanent_filepath}")

        # Update status to processing
        request_manager.update_status(request_id, "processing")

        # PDF 분석
        from main import analyze_pdf_file
        patent_data = await analyze_pdf_file(str(temp_filepath))

        if not patent_data:
            raise HTTPException(400, "Failed to analyze PDF")

        print(f"[PDF Upload] Analysis completed")

        # 특허번호로 파일명 변경
        patent_number = patent_data.get('patent_number') or patent_data.get('issued_number') or "UNKNOWN"
        # 파일명에 사용할 수 없는 문자 제거
        safe_patent_number = "".join(c for c in patent_number if c.isalnum() or c in ('-', '_'))
        permanent_filename = f"{safe_patent_number}.pdf"
        permanent_filepath = PATENT_PDF_DIR / permanent_filename

        # 기존 파일이 있으면 삭제
        if permanent_filepath.exists():
            os.remove(permanent_filepath)
            print(f"[PDF Upload] Removed existing file: {permanent_filepath}")

        # 임시 파일을 특허번호 파일명으로 변경
        shutil.move(str(temp_permanent_filepath), str(permanent_filepath))
        print(f"[PDF Upload] Renamed to patent number: {permanent_filepath}")

        # Calculate processing time
        processing_time = time_module.time() - start_time

        # 비용 및 토큰 집계 (PDF 분석만)
        pdf_input_tokens = patent_data.get('_input_tokens', 0)
        pdf_output_tokens = patent_data.get('_output_tokens', 0)
        pdf_cost = patent_data.get('_cost', 0.0)
        pdf_total_tokens = pdf_input_tokens + pdf_output_tokens

        print(f"[PDF Upload] PDF Analysis: ${pdf_cost:.4f} ({pdf_total_tokens:,} tokens)")

        # Prepare result
        result_json = {
            "success": True,
            "request_id": request_id,
            "filename": file.filename,
            "patent_data": patent_data
        }

        # Save results to DB
        request_manager.save_result(
            request_id=request_id,
            result_json=result_json,
            markdown_report="",  # No markdown report for PDF-only analysis
            processing_time=processing_time,
            total_cost=pdf_cost,
            total_tokens=pdf_total_tokens,
            pdf_file_path=str(permanent_filepath)
        )

        return JSONResponse(content=result_json)

    except HTTPException as http_exc:
        # Save error to DB
        error_msg = str(http_exc.detail)
        request_manager.save_error(request_id, error_msg, None)
        raise
    except Exception as e:
        # Save error to DB
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        request_manager.save_error(request_id, error_msg, error_traceback)

        print(f"[PDF Upload] Error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    finally:
        # 임시 파일만 삭제 (영구 보관용은 유지)
        try:
            if temp_filepath.exists():
                os.remove(temp_filepath)
                print(f"[PDF Upload] Temporary file deleted: {temp_filepath}")
        except Exception as e:
            print(f"[PDF Upload] Warning: Could not delete temporary file: {e}")

        # 에러 발생 시 영구 파일도 삭제
        try:
            req_data = request_manager.get_request(request_id)
            if req_data and req_data.get('status') in ['failed', 'cancelled']:
                # permanent_filepath가 설정되었으면 삭제
                if permanent_filepath and permanent_filepath.exists():
                    os.remove(permanent_filepath)
                    print(f"[PDF Upload] Permanent file deleted due to failure: {permanent_filepath}")
                # 아직 특허번호로 변경되지 않았으면 temp_permanent_filepath 삭제
                elif temp_permanent_filepath.exists():
                    os.remove(temp_permanent_filepath)
                    print(f"[PDF Upload] Temporary permanent file deleted due to failure: {temp_permanent_filepath}")
        except Exception as e:
            print(f"[PDF Upload] Warning: Could not clean up permanent file: {e}")


@router.post("/cancel-analysis/{request_id}")
async def cancel_analysis(request_id: str):
    """
    진행 중인 분석 취소

    Args:
        request_id: 취소할 요청 ID

    Returns:
        취소 성공 여부
    """
    success = request_manager.cancel_request(request_id)

    if success:
        return JSONResponse(content={
            "success": True,
            "message": f"Request {request_id} cancelled successfully"
        })
    else:
        # 요청이 없거나 이미 완료/취소된 경우
        req_data = request_manager.get_request(request_id)
        if not req_data:
            raise HTTPException(404, f"Request {request_id} not found")

        current_status = req_data.get('status')
        raise HTTPException(400, f"Cannot cancel request: already {current_status}")


@router.get("/analysis-status/{request_id}")
async def get_analysis_status(request_id: str):
    """
    분석 요청 상태 조회

    Args:
        request_id: 조회할 요청 ID

    Returns:
        요청 상태 정보
    """
    req_data = request_manager.get_request(request_id)

    if not req_data:
        raise HTTPException(404, f"Request {request_id} not found")

    return JSONResponse(content={
        "request_id": req_data['request_id'],
        "status": req_data['status'],
        "filename": req_data.get('filename'),
        "patent_number": req_data.get('patent_number'),
        "created_at": req_data.get('created_at'),
        "started_at": req_data.get('started_at'),
        "completed_at": req_data.get('completed_at'),
        "cancelled_at": req_data.get('cancelled_at'),
        "processing_time_seconds": req_data.get('processing_time_seconds'),
        "error_message": req_data.get('error_message')
    })


@router.get("/upload-status")
async def upload_status():
    """업로드 기능 상태 확인"""
    return {
        "status": "ready",
        "upload_dir": str(UPLOAD_DIR),
        "analyzer_ready": analyzer_instance is not None,
        "supported_formats": [".pdf"],
        "max_file_size_mb": 50  # 예시
    }
