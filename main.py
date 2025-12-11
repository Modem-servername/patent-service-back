"""
Patent PDF analysis API with Infringement Analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from typing import Optional, List, Dict
import re
import pdfplumber
import io
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import json
import fitz
from PIL import Image
import time
import base64
# from router import db  # Commented out - router module does not exist
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import request manager
import request_manager

# Import simplified infringement analysis module
from infringement_search_v2 import (
    SimplifiedInfringementAnalyzer,
    PatentInfringementAnalysis,
    format_analysis_report
)

load_dotenv()

app = FastAPI(title="Complete Patent Analysis API with Infringement Detection")

# PDF Upload router import (나중에 초기화)
pdf_upload_router = None


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if OPENAI_API_KEY and OPENAI_API_KEY != "your_api_key_here":
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    infringement_analyzer = SimplifiedInfringementAnalyzer(
        api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY
    )
    print("api key loaded")
    if TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here":
        print("tavily api key loaded - web search enabled")
    else:
        print("tavily api key not found - web search disabled")

    # PDF Upload router 초기화
    try:
        from pdf_upload_handler import router as upload_router, init_analyzer
        init_analyzer(OPENAI_API_KEY, TAVILY_API_KEY)
        app.include_router(upload_router, prefix="/api", tags=["PDF Upload"])
        print("pdf upload handler loaded")
    except Exception as e:
        print(f"warning: pdf upload handler failed to load: {e}")
else:
    openai_client = None
    infringement_analyzer = None

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768")) 
MAX_PAGES_PER_BATCH = int(os.getenv("MAX_PAGES_PER_BATCH", "30")) 
print(f"Configuration: MAX_IMAGE_SIZE={MAX_IMAGE_SIZE}px, MAX_PAGES_PER_BATCH={MAX_PAGES_PER_BATCH}")

# Model pricing (per million tokens)
MODEL_PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-4o": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate API cost based on model and token usage"""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-5"])  # Default to GPT-5
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

# PDF URL Cache (in-memory with TTL)
_pdf_url_cache: Dict[str, tuple[str, float]] = {}  # {patent_number: (url, timestamp)}
PDF_URL_CACHE_TTL = 86400  # 24 hours in seconds

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PDF_DIR = Path("downloaded_patents")
PDF_DIR.mkdir(exist_ok=True)

IMAGE_CACHE_DIR = Path("image_cache")
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

class PatentRequest(BaseModel):
    patent_number: str
    model: str = "gpt-5"  # AI model to use (e.g., "gpt-5", "gpt-4o", "gpt-4-turbo")

class InfringementAnalysisRequest(BaseModel):
    patent_number: str
    max_candidates: int = 10
    create_detailed_chart: bool = True
    model: str = "gpt-5"  # AI model to use (e.g., "gpt-5", "gpt-4o", "gpt-4-turbo")
    custom_prompt: Optional[str] = None  # User-provided analysis instructions (legacy)
    follow_up_questions: Optional[str] = None # User-provided follow-up questions (string, will be split by newlines)

class PatentPDFResponse(BaseModel):
    patent_number: str
    success: bool
    error: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_downloaded: bool = False
    pdf_path: Optional[str] = None
    pdf_pages: Optional[int] = None
    pdf_size_kb: Optional[float] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    claims: Optional[List[str]] = None
    claims_count: Optional[int] = None
    description: Optional[str] = None
    inventors: Optional[List[str]] = None
    assignee: Optional[str] = None
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    classifications: Optional[Dict] = None
    model_used: Optional[str] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    batches_processed: Optional[int] = None

def _cache_pdf_url(patent_number: str, pdf_url: Optional[str]) -> Optional[str]:
    """Cache PDF URL and return it"""
    if pdf_url:
        _pdf_url_cache[patent_number] = (pdf_url, time.time())
    return pdf_url

async def get_pdf_url_from_page(patent_number: str) -> Optional[str]:
    # Check cache first
    if patent_number in _pdf_url_cache:
        cached_url, timestamp = _pdf_url_cache[patent_number]
        if time.time() - timestamp < PDF_URL_CACHE_TTL:
            print(f"[Patent] Using cached URL for {patent_number}")
            return cached_url
        else:
            # Cache expired, remove it
            del _pdf_url_cache[patent_number]

    try:
        url = f"https://patents.google.com/patent/{patent_number}"
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            pdf_link = soup.find('a', string=re.compile(r'Download PDF', re.IGNORECASE))
            if pdf_link and pdf_link.get('href'):
                return _cache_pdf_url(patent_number, pdf_link['href'])

            # 정규표현식 개선: kind code가 없는 경우도 처리 (예: KR10055894381)
            match = re.match(r'([A-Z]{2})(\d+)([A-Z]\d*)?', patent_number)
            if match:
                country, number, kind = match.groups()
                last_two = number[-2:] if len(number) >= 2 else number

                # kind code가 없는 경우 처리
                if not kind:
                    print(f"[Patent] Warning: No kind code found for {patent_number}, trying without it")
                    # 일반적인 kind code 시도 (B1, B2 등)
                    for default_kind in ['B1', 'B2', 'A1', 'A']:
                        test_patent_number = f"{country}{number}{default_kind}"
                        test_url = f"https://patentimages.storage.googleapis.com/{last_two}/{number}/{test_patent_number}.pdf"
                        print(f"[Patent] Trying: {test_url}")
                        try:
                            test_response = await client.head(test_url, timeout=10.0)
                            if test_response.status_code == 200:
                                print(f"[Patent] Found PDF with kind code: {default_kind}")
                                return _cache_pdf_url(patent_number, test_url)
                        except:
                            continue

                    # kind code 없이 시도
                    fallback_url = f"https://patentimages.storage.googleapis.com/{last_two}/{number}/{patent_number}.pdf"
                    return _cache_pdf_url(patent_number, fallback_url)

                constructed_url = f"https://patentimages.storage.googleapis.com/{last_two}/{number}/{patent_number}.pdf"
                return _cache_pdf_url(patent_number, constructed_url)

            return None
    except:
        return None

async def download_pdf(pdf_url: str, patent_number: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=60.0, headers={'User-Agent': 'Mozilla/5.0'}) as client:
            response = await client.get(pdf_url, follow_redirects=True)
            if response.status_code != 200:
                return None
            
            pdf_path = PDF_DIR / f"{patent_number}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            return str(pdf_path)
    except:
        return None

def resize_image(img: Image.Image, max_size: int = 1536) -> Image.Image:
    # Force load image data to avoid lazy loading issues
    img.load()

    w, h = img.size
    if w <= max_size and h <= max_size:
        # Return a copy to avoid mutation issues
        return img.copy()
    if w > h:
        return img.resize((max_size, int(h * max_size / w)), Image.Resampling.LANCZOS)
    else:
        return img.resize((int(w * max_size / h), max_size), Image.Resampling.LANCZOS)

def _convert_single_page(pdf_path: str, page_num: int, max_size: int) -> tuple:
    """단일 페이지 변환 (병렬 처리용 헬퍼 함수)"""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
        # Convert to bytes and create PIL Image
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        # Force load and convert to RGB to avoid format issues
        img = img.convert("RGB")
        # Resize
        resized = resize_image(img, max_size)
        return (page_num, resized, None)
    except Exception as e:
        return (page_num, None, str(e))
    finally:
        if doc is not None:
            doc.close()

def convert_pages(pdf_path: str, max_size: int = 1024, patent_number: Optional[str] = None) -> List[Image.Image]:
    """convert all pdf pages to images (병렬 처리, 캐싱 지원)"""
    try:
        # Check cache first if patent_number is provided
        if patent_number:
            cache_key = f"{patent_number}_{max_size}"
            cache_subdir = IMAGE_CACHE_DIR / cache_key

            if cache_subdir.exists():
                # Try to load from cache
                cached_images = []
                cache_files = sorted(cache_subdir.glob("page_*.png"))

                if cache_files:
                    print(f"  loading {len(cache_files)} cached images for {patent_number}...")
                    try:
                        for cache_file in cache_files:
                            img = Image.open(cache_file)
                            img.load()  # Force load into memory
                            cached_images.append(img)

                        print(f"  ✓ loaded {len(cached_images)} images from cache")
                        return cached_images
                    except Exception as e:
                        print(f"  cache load failed: {e}, regenerating...")
                        cached_images = []  # Clear partial results

        # 총 페이지 수 확인
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        print(f"  converting {total_pages} pages in parallel...")

        # 병렬로 페이지 변환
        results = {}
        max_workers = min(8, total_pages)  # 최대 8개 워커

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_convert_single_page, pdf_path, i, max_size): i
                for i in range(total_pages)
            }

            completed = 0
            for future in as_completed(futures):
                page_num, img, error = future.result()
                completed += 1

                if completed % 10 == 0 or completed == total_pages:
                    print(f"  progress: {completed}/{total_pages} pages")

                if error:
                    print(f"  error converting page {page_num+1}: {error}")
                    continue

                results[page_num] = img

        # 순서대로 정렬
        images = [results[i] for i in range(total_pages) if i in results]

        print(f"  ✓ conversion complete: {len(images)}/{total_pages} pages")

        # Save to cache if patent_number is provided
        if patent_number and images:
            try:
                cache_key = f"{patent_number}_{max_size}"
                cache_subdir = IMAGE_CACHE_DIR / cache_key
                cache_subdir.mkdir(parents=True, exist_ok=True)

                print(f"  caching {len(images)} images...")
                for idx, img in enumerate(images):
                    cache_file = cache_subdir / f"page_{idx:04d}.png"
                    img.save(cache_file, "PNG", optimize=False)

                print(f"  ✓ cached {len(images)} images to {cache_key}")
            except Exception as e:
                print(f"  warning: cache save failed: {e}")

        return images
    except Exception as e:
        print(f"  conversion error: {str(e)}")
        return []

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    # Ensure image is fully loaded before encoding
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def extract_json(text: str) -> Optional[str]:
    """extract json from various response formats"""
    if not text:
        return None

    text = text.strip()

    if text.startswith('{') and text.endswith('}'):
        return text

    if "```json" in text.lower():
        try:
            pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    if '{' in match and '}' in match:
                        return match.strip()
        except:
            pass

    if "```" in text:
        try:
            parts = text.split("```")
            for i, part in enumerate(parts):
                if i == 0:
                    continue
                part = part.strip()
                if part.lower().startswith(('json', 'javascript', 'js')):
                    part = part[part.find('\n')+1:] if '\n' in part else part[4:]
                if part.startswith('{') and '}' in part:
                    return part
        except:
            pass

    try:
        patterns = [
            r'(?:here\'?s?|here is|json:?)\s*(\{.*\})',
            r'(?:result|output|response):?\s*(\{.*\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
    except:
        pass

    try:
        json_candidates = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            if len(candidate) > 10 and '"' in candidate:
                                json_candidates.append(candidate)
                            i = j
                            break
            i += 1

        if json_candidates:
            return max(json_candidates, key=len)
    except:
        pass

    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return text[start:end]
    except:
        pass
    
    return None

def parse_json_safely(json_text: str) -> Optional[Dict]:
    """try parsing json with multiple fallback strategies"""
    if not json_text:
        return None

    try:
        return json.loads(json_text)
    except:
        pass

    try:
        cleaned = json_text
        cleaned = cleaned.replace("'", '"')
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        return json.loads(cleaned)
    except:
        pass

    try:
        cleaned = json_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        cleaned = cleaned.replace("'", '"')
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return json.loads(cleaned)
    except:
        pass

    try:
        import codecs
        decoded = codecs.decode(json_text, 'unicode_escape')
        cleaned = decoded.replace("'", '"')
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        return json.loads(cleaned)
    except:
        pass

    try:
        cleaned = ''.join(char if ord(char) >= 32 or char in '\t\n\r' else ' ' for char in json_text)
        cleaned = cleaned.replace("'", '"')
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        return json.loads(cleaned)
    except:
        pass
    
    return None

def _merge_batch_results(batch_results: List[Dict]) -> Dict:
    """
    여러 배치의 분석 결과를 하나로 병합

    Args:
        batch_results: 각 배치의 분석 결과 리스트

    Returns:
        병합된 분석 결과
    """
    if not batch_results:
        return {'success': False, 'error': 'No batch results to merge'}

    if len(batch_results) == 1:
        return batch_results[0]

    print(f"  [Merge] Combining {len(batch_results)} batch results...")

    # 첫 번째 배치를 기본으로 사용 (보통 title, abstract 등이 있음)
    merged = batch_results[0].copy()
    merged_data = merged.get('data', {}).copy()

    # Claims: 모든 배치에서 수집하여 병합 (중복 제거)
    all_claims = []
    seen_claim_nums = set()

    for batch_idx, batch_result in enumerate(batch_results, 1):
        batch_data = batch_result.get('data', {})
        batch_claims = batch_data.get('claims', [])

        for claim in batch_claims:
            # 청구항 번호 추출
            claim_num = None
            match = re.match(r'^(\d+)[\.\)]\s', claim)
            if match:
                claim_num = match.group(1)

            # 중복이 아니면 추가
            if claim_num and claim_num not in seen_claim_nums:
                all_claims.append(claim)
                seen_claim_nums.add(claim_num)
            elif not claim_num:
                # 번호를 찾을 수 없는 경우 일단 추가
                all_claims.append(claim)

        print(f"  [Merge]   Batch {batch_idx}: {len(batch_claims)} claims")

    merged_data['claims'] = all_claims
    print(f"  [Merge]   Total unique claims: {len(all_claims)}")

    # Description: 모든 배치의 description 합치기
    all_descriptions = []
    for batch_result in batch_results:
        batch_data = batch_result.get('data', {})
        desc = batch_data.get('description', '')
        if desc and desc.strip():
            all_descriptions.append(desc.strip())

    if all_descriptions:
        # 중복 제거 (같은 내용이 여러 번 나오는 경우)
        merged_data['description'] = '\n\n'.join(all_descriptions)
        print(f"  [Merge]   Combined descriptions from {len(all_descriptions)} batches")

    # Title, Abstract: 첫 번째 배치 사용 (이미 merged_data에 있음)
    # 만약 첫 번째 배치에 없으면 다른 배치에서 찾기
    if not merged_data.get('title'):
        for batch_result in batch_results[1:]:
            title = batch_result.get('data', {}).get('title')
            if title:
                merged_data['title'] = title
                print(f"  [Merge]   Found title in later batch")
                break

    if not merged_data.get('abstract'):
        for batch_result in batch_results[1:]:
            abstract = batch_result.get('data', {}).get('abstract')
            if abstract:
                merged_data['abstract'] = abstract
                print(f"  [Merge]   Found abstract in later batch")
                break

    # Tokens and cost: 모두 합산
    total_input_tokens = sum(br.get('input_tokens', 0) for br in batch_results)
    total_output_tokens = sum(br.get('output_tokens', 0) for br in batch_results)
    total_cost = sum(br.get('cost', 0) for br in batch_results)

    print(f"  [Merge]   Total cost: ${total_cost:.4f}")
    print(f"  [Merge]   Total tokens: {total_input_tokens} input + {total_output_tokens} output")

    return {
        'success': True,
        'data': merged_data,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'cost': total_cost,
        'num_batches': len(batch_results)
    }


async def analyze_patent(images: List[Image.Image], max_pages_per_batch: int = None, model: str = "gpt-5") -> Dict:
    """
    Analyze patent from images, with batch processing for large documents

    Args:
        images: List of page images
        max_pages_per_batch: Maximum pages to process in one API call (default: from env or 100)
        model: AI model to use (default: gpt-5)
    """
    import asyncio

    if not images or not openai_client:
        return {'success': False, 'error': 'No images or API client'}

    # Use global setting if not specified
    if max_pages_per_batch is None:
        max_pages_per_batch = MAX_PAGES_PER_BATCH

    # If pages exceed limit, process in batches
    total_pages = len(images)
    if total_pages > max_pages_per_batch:
        print(f"  large document detected ({total_pages} pages), processing in limited parallel batches (max 2 concurrent)...")

        # Prepare all batches
        batches = []
        for start_idx in range(0, total_pages, max_pages_per_batch):
            end_idx = min(start_idx + max_pages_per_batch, total_pages)
            batch_images = images[start_idx:end_idx]
            batches.append((start_idx, end_idx, batch_images))

        # Semaphore to limit concurrent batches to 2
        semaphore = asyncio.Semaphore(2)

        async def process_batch_limited(batch_num: int, start_idx: int, end_idx: int, batch_images: List[Image.Image]):
            async with semaphore:
                print(f"  batch {batch_num}: pages {start_idx+1}-{end_idx}")
                # Run sync function in thread pool
                batch_result = await asyncio.to_thread(_analyze_patent_batch, batch_images, MAX_IMAGE_SIZE, model)

                if not batch_result.get('success'):
                    print(f"  batch {batch_num} failed: {batch_result.get('error', 'unknown error')}")
                    return None

                print(f"  batch {batch_num} completed: ${batch_result.get('cost', 0):.4f}")
                return batch_result

        # Process all batches with limited parallelism
        tasks = [
            process_batch_limited(idx + 1, start_idx, end_idx, batch_images)
            for idx, (start_idx, end_idx, batch_images) in enumerate(batches)
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed batches and exceptions
        all_batch_results = []
        first_batch_failed = False

        for idx, result in enumerate(all_results):
            if isinstance(result, Exception):
                print(f"  batch {idx+1} raised exception: {result}")
                if idx == 0:
                    first_batch_failed = True
                continue

            if result is None:
                if idx == 0:
                    first_batch_failed = True
                continue

            all_batch_results.append(result)

        # If first batch fails, return error
        if first_batch_failed:
            return {'success': False, 'error': 'First batch failed'}

        if not all_batch_results:
            return {'success': False, 'error': 'All batches failed'}

        # Merge all batch results
        print(f"  merging {len(all_batch_results)} batch results...")
        merged_result = _merge_batch_results(all_batch_results)

        return merged_result
    else:
        # Process all pages at once
        return await asyncio.to_thread(_analyze_patent_batch, images, MAX_IMAGE_SIZE, model)

def _analyze_patent_batch(images: List[Image.Image], base_max_size: int, model: str = "gpt-5") -> Dict:
    """Process a batch of patent images"""
    if not images or not openai_client:
        return {'success': False, 'error': 'No images or API client'}

    current_max_size = base_max_size

    for attempt in range(3):
        try:
            if attempt > 0:
                current_max_size = int(base_max_size * (0.7 ** attempt))
                print(f"  retry {attempt}/2, resizing to {current_max_size}px")

            prompt = """You are a precise patent document extractor. Extract data EXACTLY as written in the images. DO NOT make up, infer, or hallucinate any information.

CRITICAL RULES:
- ONLY extract text that is CLEARLY VISIBLE in the provided images
- If you cannot read something clearly, leave it empty or null
- DO NOT invent words, phrases, or technical terms that don't exist in the document
- DO NOT paraphrase - copy the exact text as written
- DO NOT fill in gaps with your own knowledge
- If a section is not visible, return empty string or empty array for that field

CLAIMS SECTION - CRITICAL INSTRUCTIONS:
- Find section titled "Claims", "What is claimed is:", "청구범위", or "청구항"
- Extract EVERY numbered claim IN ORDER (1., 2., 3., etc.) EXACTLY as written
- Include cancelled claims marked as "(cancelled)"
- PRESERVE THE ENTIRE CLAIM TEXT including:
  * Claim references like "claim 1", "제1항", "제1항 또는 제2항"
  * All technical terms and phrases
  * Complete sentences from start to end
- DO NOT:
  * Skip any words or phrases, especially claim references
  * Mix up claim numbers with wrong text
  * Cut off text at page breaks
- Claims may span multiple pages - read ALL pages carefully
- Format: ["1. [COMPLETE exact claim text]", "2. (cancelled)", "3. [COMPLETE exact claim text]", ...]
- VERIFY: Each claim number matches its correct text
- If you cannot find claims section, return empty array []

DESCRIPTION SECTION:
- ONLY extract these specific sections if they exist:
  1. Technical Field / Field of the Invention
  2. Background of the Invention / Background Art
  3. Summary of the Invention / Brief Summary
- Extract the EXACT text from these sections only
- DO NOT include: Detailed Description, Brief Description of Drawings, Examples, Embodiments, or any other sections
- If you cannot find these sections clearly, return empty string ""

OTHER FIELDS:
- title: Extract from first page header (exact text)
- abstract: Copy the abstract section exactly as written
- inventors: List names exactly as they appear
- assignee: Copy company/person name exactly
- filing_date: Use format YYYY-MM-DD only if clearly visible
- publication_date: Use format YYYY-MM-DD only if clearly visible
- patent_number: Copy exactly as shown
- classifications: Extract CPC/IPC codes only if clearly visible

Return this exact JSON structure:
{
  "title": "exact title or empty string",
  "abstract": "exact abstract text or empty string",
  "claims": ["1. exact claim", "2. (cancelled)", "3. exact claim"] or [],
  "description": "exact text from specified sections only or empty string",
  "inventors": ["exact names"] or [],
  "assignee": "exact name or empty string",
  "filing_date": "YYYY-MM-DD or empty string",
  "publication_date": "YYYY-MM-DD or empty string",
  "patent_number": "exact number or empty string",
  "classifications": {"cpc": ["exact codes"] or [], "ipc": ["exact codes"] or []}
}

REMEMBER: Accuracy over completeness. Empty fields are better than made-up information."""
            
            content = [{"type": "text", "text": prompt}]

            print(f"  encoding {len(images)} images in parallel...")

            # 병렬로 이미지 인코딩
            def encode_single_image(idx_img_tuple):
                idx, img = idx_img_tuple
                resized = resize_image(img, current_max_size)
                b64 = img_to_b64(resized)
                return (idx, b64, len(b64))

            b64_results = {}
            total_size = 0
            max_workers = min(8, len(images))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(encode_single_image, (idx, img)): idx
                    for idx, img in enumerate(images)
                }

                completed = 0
                for future in as_completed(futures):
                    idx, b64, size = future.result()
                    b64_results[idx] = b64
                    total_size += size
                    completed += 1

                    if completed % 10 == 0 or completed == len(images):
                        print(f"    progress: {completed}/{len(images)} images")

            # 순서대로 정렬
            b64_images = [b64_results[i] for i in range(len(images))]

            print(f"  ✓ encoding complete: {len(b64_images)} images, {total_size/1024:.1f}KB total")

            if total_size > 15_000_000:
                print(f"  warning: large payload {total_size/1024/1024:.1f}MB, may take longer")
            
            for b64 in b64_images:
                content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}", 
                        "detail": "high"
                    }
                })
            
            if attempt == 0:
                print(f"  preparing api request with {len(content)-1} images")

            print(f"  sending to openai api... (this may take 30-60s)")
            t = time.time()

            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=16000,
                response_format={"type": "json_object"}
            )

            elapsed = time.time() - t
            print(f"  api response received in {elapsed:.1f}s")

            inp = resp.usage.prompt_tokens
            out = resp.usage.completion_tokens
            cost = calculate_cost(model, inp, out)

            print(f"  ${cost:.4f} (tokens: {inp:,} + {out:,})")
            
            txt = resp.choices[0].message.content

            if txt is None:
                print(f"  response is None (attempt {attempt + 1}/3)")
                print(f"  full response: {resp}")
                if attempt < 2:
                    time.sleep(3)
                    continue
                else:
                    return {'success': False, 'error': 'Response is None',
                            'input_tokens': inp, 'output_tokens': out, 'cost': cost}

            txt = txt.strip()

            if not txt or len(txt) < 10:
                print(f"  empty or too short: {len(txt)} chars (attempt {attempt + 1}/3)")
                print(f"  content: '{txt}'")
                print(f"  type: {type(txt)}")
                print(f"  full response:")
                print(f"     {resp}")
                if attempt < 2:
                    time.sleep(3)
                    continue
                else:
                    return {'success': False, 'error': f'Empty or too short response: {len(txt)} chars',
                            'input_tokens': inp, 'output_tokens': out, 'cost': cost,
                            'raw_response': txt}

            print(f"  response length: {len(txt)} chars")
            print(f"  preview: {txt[:100]}")
            
            try:
                data = json.loads(txt)
                print(f"  json parsed ok")

                required_fields = ['title', 'abstract', 'claims', 'description']
                has_data = any(data.get(field) for field in required_fields)

                if not has_data:
                    print(f"  all fields empty")
                    print(f"  data: {data}")
                    if attempt < 2:
                        time.sleep(3)
                        continue

                return {'success': True, 'data': data, 'input_tokens': inp, 'output_tokens': out, 'cost': cost}

            except json.JSONDecodeError as e:
                print(f"  json parse failed: {str(e)}")
                print(f"  full response ({len(txt)} chars):")
                print(f"     {txt[:1000]}")
                if len(txt) > 1000:
                    print(f"     ... truncated ...")
                    print(f"     {txt[-500:]}")

                print(f"  trying extraction method...")
                jtxt = extract_json(txt)
                if jtxt:
                    print(f"  extracted json ({len(jtxt)} chars)")
                    data = parse_json_safely(jtxt)
                    if data:
                        print(f"  parsed after extraction")
                        return {'success': True, 'data': data, 'input_tokens': inp, 'output_tokens': out, 'cost': cost}
                    else:
                        print(f"  extraction parse failed")
                        print(f"     {jtxt[:500]}")
                else:
                    print(f"  extraction failed")
                
                if attempt < 2:
                    time.sleep(3)
                    continue
                else:
                    return {'success': False, 'error': f'JSON decode error: {str(e)}',
                            'input_tokens': inp, 'output_tokens': out, 'cost': cost,
                            'raw_response': txt[:1000]}
        
        except Exception as e:
            error_msg = str(e)
            print(f"  api error (attempt {attempt + 1}/3):")
            print(f"     {error_msg}")

            import traceback
            print(f"  stack trace:")
            traceback.print_exc()

            if "content_policy_violation" in error_msg.lower():
                return {'success': False, 'error': 'Content policy violation - possible issue with images'}

            if "maximum context length" in error_msg.lower():
                print(f"  context too long, reducing size")
                if attempt < 2:
                    time.sleep(2)
                    continue

            if "does not support" in error_msg.lower() or "json_object" in error_msg.lower():
                print(f"  json mode not supported, switching to regular mode")
                try:
                    resp = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=16000
                    )

                    inp = resp.usage.prompt_tokens
                    out = resp.usage.completion_tokens
                    cost = calculate_cost(model, inp, out)

                    txt = resp.choices[0].message.content
                    if txt:
                        txt = txt.strip()
                        print(f"  regular mode response: {len(txt)} chars")
                        jtxt = extract_json(txt)
                        if jtxt:
                            data = parse_json_safely(jtxt)
                            if data:
                                print(f"  regular mode success")
                                return {'success': True, 'data': data, 'input_tokens': inp, 'output_tokens': out, 'cost': cost}
                except Exception as fallback_error:
                    print(f"  fallback failed: {str(fallback_error)}")
                    traceback.print_exc()
            
            if attempt < 2:
                time.sleep(3)
                continue
            else:
                return {'success': False, 'error': f'API error: {error_msg}'}
    
    return {'success': False, 'error': 'All retry attempts failed'}


async def analyze_patent_robust(images: List[Image.Image], num_attempts: int = 1, model: str = "gpt-5") -> Dict:
    """
    특허 분석 (최적화 버전 - 단일 실행)

    Args:
        images: 분석할 이미지 리스트
        num_attempts: 실행 횟수 (기본값: 1, 비용 최적화)
        model: AI model to use (default: gpt-5)

    Returns:
        분석 결과
    """
    print(f"\n[Patent Analysis] Starting optimized analysis...")

    # 단일 실행으로 최적화 (비용 절감)
    try:
        result = await analyze_patent(images, model=model)
        if result.get('success'):
            print(f"[Patent Analysis] Analysis succeeded")
            print(f"[Patent Analysis] Cost: ${result.get('cost', 0):.4f}")
            return result
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"[Patent Analysis] Analysis failed: {error_msg}")
            return {'success': False, 'error': error_msg}
    except Exception as e:
        print(f"[Patent Analysis] Analysis error: {e}")
        return {'success': False, 'error': str(e)}


async def analyze_pdf_file(pdf_path: str) -> Optional[Dict]:
    """
    PDF 파일을 분석하여 특허 데이터 추출 (업로드된 파일용)

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        특허 데이터 딕셔너리 또는 None
    """
    print(f"\n[PDF Analysis] Analyzing uploaded PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"[PDF Analysis] File not found: {pdf_path}")
        return None

    try:
        # PDF 페이지 수 확인
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)

        if pages == 0:
            print("[PDF Analysis] No pages found")
            return None

        print(f"[PDF Analysis] {pages} pages found")

        # PDF를 이미지로 변환
        print("[PDF Analysis] Converting PDF to images...")
        imgs = convert_pages(pdf_path, MAX_IMAGE_SIZE)

        if not imgs:
            print("[PDF Analysis] Image conversion failed")
            return None

        # 분석 수행
        print("[PDF Analysis] Starting analysis...")
        result = await analyze_patent_robust(imgs)

        if not result or not result.get('success'):
            error = result.get('error', 'unknown') if result else 'no response'
            print(f"[PDF Analysis] Analysis failed: {error}")
            return None

        data = result.get('data', {})

        # 데이터 검증
        if not data.get('title') and not data.get('claims') and not data.get('abstract'):
            print("[PDF Analysis] Failed to extract valid data")
            return None

        print(f"[PDF Analysis] Analysis successful - Cost: ${result.get('cost', 0):.4f}")
        print(f"[PDF Analysis] Title: {data.get('title', 'N/A')}")
        print(f"[PDF Analysis] Claims: {len(data.get('claims', []))}")

        # 토큰 정보를 data에 추가 (기존 호환성 유지)
        data['_input_tokens'] = result.get('input_tokens', 0)
        data['_output_tokens'] = result.get('output_tokens', 0)
        data['_cost'] = result.get('cost', 0.0)

        return data

    except Exception as e:
        print(f"[PDF Analysis] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def process_patent(patent_number: str, model: str = "gpt-5") -> PatentPDFResponse:
    print(f"\n{'='*60}\nprocessing {patent_number}\n{'='*60}\n")
    print(f"Using AI model: {model}")

    pdf_url = await get_pdf_url_from_page(patent_number)
    if not pdf_url:
        return PatentPDFResponse(patent_number=patent_number, success=False, error="url not found")

    pdf_path = await download_pdf(pdf_url, patent_number)
    if not pdf_path:
        return PatentPDFResponse(patent_number=patent_number, success=False, error="download failed", pdf_url=pdf_url)

    size_kb = os.path.getsize(pdf_path) / 1024

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)
    except:
        pages = 0

    if pages == 0:
        return PatentPDFResponse(patent_number=patent_number, success=False, error="no pages",
                                pdf_url=pdf_url, pdf_downloaded=True, pdf_path=pdf_path)

    print(f"{pages} pages found\n")

    print("converting pdf to images...")
    imgs = convert_pages(pdf_path, MAX_IMAGE_SIZE, patent_number)
    if not imgs:
        error_msg = "image conversion failed"
        print(f"\n{error_msg}\n")
        return PatentPDFResponse(
            patent_number=patent_number, success=False, error=error_msg,
            pdf_url=pdf_url, pdf_downloaded=True, pdf_path=pdf_path,
            pdf_pages=pages, pdf_size_kb=round(size_kb, 2)
        )

    print("\nstarting analysis...")
    result = await analyze_patent_robust(imgs, model=model)

    if not result or not result.get('success'):
        error_detail = result.get('error', 'unknown error') if result else 'no response'
        print(f"\nanalysis failed: {error_detail}\n")
        return PatentPDFResponse(
            patent_number=patent_number, success=False,
            error=f"analysis failed: {error_detail}",
            pdf_url=pdf_url, pdf_downloaded=True, pdf_path=pdf_path,
            pdf_pages=pages, pdf_size_kb=round(size_kb, 2)
        )

    data = result.get('data', {})

    if not data.get('title') and not data.get('claims') and not data.get('abstract'):
        return PatentPDFResponse(
            patent_number=patent_number, success=False,
            error="failed to extract valid data",
            pdf_url=pdf_url, pdf_downloaded=True, pdf_path=pdf_path,
            pdf_pages=pages, pdf_size_kb=round(size_kb, 2),
            total_input_tokens=result.get('input_tokens', 0),
            total_output_tokens=result.get('output_tokens', 0),
            estimated_cost=result.get('cost', 0.0)
        )

    print(f"done: ${result.get('cost', 0.0):.4f}\n{'='*60}\n")

    claims_list = data.get('claims', [])
    valid_claims_count = sum(1 for claim in claims_list if claim and '(cancelled)' not in claim.lower())

    # Handle classifications - convert list to dict if needed
    classifications_raw = data.get('classifications')
    classifications = None
    if classifications_raw:
        if isinstance(classifications_raw, dict):
            classifications = classifications_raw
        elif isinstance(classifications_raw, list):
            # Convert list to dict format
            classifications = {"codes": classifications_raw}
        else:
            classifications = None

    return PatentPDFResponse(
        patent_number=patent_number, success=True, pdf_url=pdf_url, pdf_downloaded=True,
        pdf_path=pdf_path, pdf_pages=pages, pdf_size_kb=round(size_kb, 2),
        title=data.get('title'), abstract=data.get('abstract'),
        claims=claims_list if claims_list else None,
        claims_count=valid_claims_count, description=data.get('description'),
        inventors=data.get('inventors') if data.get('inventors') else None,
        assignee=data.get('assignee'), filing_date=data.get('filing_date'),
        publication_date=data.get('publication_date'),
        classifications=classifications,
        model_used=model,
        total_input_tokens=result.get('input_tokens', 0),
        total_output_tokens=result.get('output_tokens', 0),
        estimated_cost=result.get('cost', 0.0),
        batches_processed=result.get('num_batches', 1)
    )

@app.get("/")
async def root():
    return {
        "message": "Patent Analysis API with Infringement Detection", 
        "version": "8.0", 
        "model": "gpt-5", 
        "json_mode": True, 
        "features": ["patent_extraction", "infringement_analysis"]
    }

@app.post("/analyze", response_model=PatentPDFResponse)
async def analyze(req: PatentRequest):
    if not req.patent_number:
        raise HTTPException(400, "patent number required")
    # Normalize patent number to uppercase
    patent_number = req.patent_number.strip().upper()
    # Security: Validate patent number format to prevent path traversal
    if not re.match(r'^[A-Z]{2}\d+[A-Z\d]*$', patent_number):
        raise HTTPException(400, "Invalid patent number format")
    return await process_patent(patent_number, model=req.model)

@app.post("/analyze-infringement")
async def analyze_infringement(req: InfringementAnalysisRequest):
    """
    특허 침해 분석 엔드포인트

    Args:
        req: 특허 번호, 최대 후보 수, 상세 차트 생성 여부

    Returns:
        침해 분석 결과 (JSON 및 마크다운 보고서) + request_id
    """
    # 실행 시간 측정 시작 (요청 받은 시점부터)
    start_time = time.time()

    # 1. Request ID 생성 및 DB 저장
    request_id = request_manager.create_request(
        patent_number=req.patent_number,
        max_candidates=req.max_candidates,
        create_detailed_chart=req.create_detailed_chart,
        model=req.model,
        follow_up_questions=req.follow_up_questions
    )

    print(f"\n[API] ===== Infringement Analysis Request =====")
    print(f"[API] Request ID: {request_id}")
    print(f"[API] Patent Number: {req.patent_number}")
    print(f"[API] Max Candidates: {req.max_candidates}")
    print(f"[API] Create Detailed Chart: {req.create_detailed_chart}")
    print(f"[API] AI Model: {req.model}")
    print(f"[API] Follow-up Questions: {req.follow_up_questions}")

    if not req.patent_number:
        print("[API] Error: No patent number provided")
        request_manager.save_error(request_id, "patent number required")
        raise HTTPException(400, "patent number required")

    # Normalize patent number to uppercase
    patent_number = req.patent_number.strip().upper()

    # Security: Validate patent number format to prevent path traversal
    if not re.match(r'^[A-Z]{2}\d+[A-Z\d]*$', patent_number):
        print("[API] Error: Invalid patent number format")
        request_manager.save_error(request_id, "Invalid patent number format")
        raise HTTPException(400, "Invalid patent number format")

    if not infringement_analyzer:
        print("[API] Error: Infringement analyzer not initialized")
        request_manager.save_error(request_id, "Infringement analyzer not initialized - check API key")
        raise HTTPException(500, "Infringement analyzer not initialized - check API key")

    # 2. 상태를 processing으로 업데이트
    request_manager.update_status(request_id, "processing")

    try:
        print(f"\n[API] Starting infringement analysis for {patent_number}")

        # Check cancellation before Step 1
        if request_manager.is_cancelled(request_id):
            print(f"[API] Request {request_id} was cancelled before Step 1")
            return JSONResponse(content={
                "success": False,
                "request_id": request_id,
                "error": "Request was cancelled by user"
            })

        # Step 1: 특허 분석 (기존 기능)
        print(f"[API] Step 1: Analyzing patent {patent_number}...")
        patent_response = await process_patent(patent_number, model=req.model)

        if not patent_response.success:
            print(f"[API] Patent analysis failed: {patent_response.error}")
            raise HTTPException(400, f"Patent analysis failed: {patent_response.error}")

        print(f"[API] Patent analysis successful")
        print(f"[API] Title: {patent_response.title}")
        print(f"[API] Claims count: {patent_response.claims_count}")

        # Check cancellation before Step 2
        if request_manager.is_cancelled(request_id):
            print(f"[API] Request {request_id} was cancelled after Step 1")
            return JSONResponse(content={
                "success": False,
                "request_id": request_id,
                "error": "Request was cancelled by user"
            })

        # Step 2: 특허 데이터를 Dict로 변환
        print(f"[API] Step 2: Converting patent data to dict...")
        patent_data = {
            "patent_number": patent_response.patent_number,
            "title": patent_response.title,
            "abstract": patent_response.abstract,
            "claims": patent_response.claims,
            "description": patent_response.description,
            "inventors": patent_response.inventors,
            "assignee": patent_response.assignee,
            "filing_date": patent_response.filing_date,
            "publication_date": patent_response.publication_date,
            "classifications": patent_response.classifications
        }

        # Check cancellation before Step 3
        if request_manager.is_cancelled(request_id):
            print(f"[API] Request {request_id} was cancelled before Step 3")
            return JSONResponse(content={
                "success": False,
                "request_id": request_id,
                "error": "Request was cancelled by user"
            })

        # Step 3: 침해 분석 수행
        print(f"[API] Step 3: Performing infringement analysis with model {req.model}...")
        if req.custom_prompt:
            print(f"[API] Custom prompt provided (Note: simplified analyzer doesn't use custom prompts): {req.custom_prompt[:100]}...")

        # Convert follow_up_questions from string to list
        follow_up_questions_list = None
        if req.follow_up_questions:
            # Split by newlines and filter out empty lines
            follow_up_questions_list = [
                q.strip() for q in req.follow_up_questions.split('\n')
                if q.strip()
            ]
            print(f"[API] Parsed {len(follow_up_questions_list)} follow-up question(s)")

        analysis_result = await infringement_analyzer.analyze_infringement(
            patent_data=patent_data,
            max_candidates=req.max_candidates,
            create_detailed_chart=req.create_detailed_chart,
            model=req.model,
            follow_up_questions=follow_up_questions_list
        )

        # Check cancellation after Step 3
        if request_manager.is_cancelled(request_id):
            print(f"[API] Request {request_id} was cancelled after Step 3")
            return JSONResponse(content={
                "success": False,
                "request_id": request_id,
                "error": "Request was cancelled by user"
            })

        # Step 4: 보고서 생성
        print(f"[API] Step 4: Generating report...")
        markdown_report = format_analysis_report(analysis_result)

        # Step 5: 결과 저장
        processing_time = time.time() - start_time

        # 특허 분석 비용 및 토큰 (Step 1에서 발생한 비용)
        patent_cost = patent_response.estimated_cost or 0.0
        patent_input_tokens = patent_response.total_input_tokens or 0
        patent_output_tokens = patent_response.total_output_tokens or 0
        patent_total_tokens = patent_input_tokens + patent_output_tokens

        # 침해 분석 비용 및 토큰 (Step 3에서 발생한 비용)
        # analyzer 인스턴스에서 직접 읽어와 정확성 보장
        infringement_input_tokens = infringement_analyzer.total_input_tokens
        infringement_output_tokens = infringement_analyzer.total_output_tokens
        infringement_tokens = infringement_input_tokens + infringement_output_tokens
        infringement_cost = infringement_analyzer.total_cost

        # 전체 합산 (특허 분석 + 침해 분석)
        total_cost = patent_cost + infringement_cost
        total_tokens = patent_total_tokens + infringement_tokens

        result_json = {
            "success": True,
            "request_id": request_id,
            "patent_number": patent_number,
            "analysis": analysis_result.model_dump(),
            "markdown_report": markdown_report
        }

        request_manager.save_result(
            request_id=request_id,
            result_json=result_json,
            markdown_report=markdown_report,
            processing_time=processing_time,
            total_cost=total_cost,
            total_tokens=total_tokens
        )

        print(f"[API] ===== Analysis Complete =====")
        print(f"[API] Request ID: {request_id}")
        print(f"[API] Patent Number: {patent_number}")
        print(f"[API] Total Processing Time: {processing_time:.2f}s")
        print(f"[API] ")
        print(f"[API] Patent Analysis:")
        print(f"[API]   Cost: ${patent_cost:.4f}")
        print(f"[API]   Tokens: {patent_total_tokens:,} (Input: {patent_input_tokens:,}, Output: {patent_output_tokens:,})")
        print(f"[API] ")
        print(f"[API] Infringement Analysis:")
        print(f"[API]   Cost: ${infringement_cost:.4f}")
        print(f"[API]   Tokens: {infringement_tokens:,} (Input: {infringement_input_tokens:,}, Output: {infringement_output_tokens:,})")
        print(f"[API] ")
        print(f"[API] Total Cost: ${total_cost:.4f}")
        print(f"[API] Total Tokens: {total_tokens:,}")
        print(f"[API] ===============================")

        return JSONResponse(content=result_json)

    except HTTPException as he:
        # HTTP exceptions에 대해 에러 저장
        request_manager.save_error(request_id, str(he.detail))
        raise he
    except Exception as e:
        print(f"[API] Error in infringement analysis: {str(e)}")
        print(f"[API] Error type: {type(e).__name__}")
        import traceback
        tb_str = traceback.format_exc()
        traceback.print_exc()

        # 에러 저장
        request_manager.save_error(request_id, str(e), tb_str)

        raise HTTPException(500, f"Infringement analysis failed: {str(e)}")

@app.get("/download/{patent_number}")
async def download(patent_number: str):
    # Normalize patent number to uppercase
    patent_number = patent_number.strip().upper()
    # Security: Validate patent number format to prevent path traversal
    if not re.match(r'^[A-Z]{2}\d+[A-Z\d]*$', patent_number):
        raise HTTPException(400, "Invalid patent number format")

    p = PDF_DIR / f"{patent_number}.pdf"
    if p.exists():
        return FileResponse(p, filename=f"{patent_number}.pdf", media_type="application/pdf")

    url = await get_pdf_url_from_page(patent_number)
    if not url:
        raise HTTPException(404, "url not found")

    path = await download_pdf(url, patent_number)
    if not path:
        raise HTTPException(404, "download failed")

    return FileResponse(path, filename=f"{patent_number}.pdf", media_type="application/pdf")


# ===== 요청 관리 API =====

def parse_request_json_fields(request_dict: Dict) -> Dict:
    """Parse JSON string fields in request dict"""
    if request_dict.get('result_json') and isinstance(request_dict['result_json'], str):
        try:
            request_dict['result_json'] = json.loads(request_dict['result_json'])
        except:
            request_dict['result_json'] = None

    # follow_up_questions is now stored as a plain string (not JSON), so no parsing needed
    # Keep it as-is

    return request_dict

@app.get("/request-status/{request_id}")
async def get_request_status(request_id: str):
    """
    요청 상태 조회

    Args:
        request_id: 요청 ID (UUID)

    Returns:
        요청 상태 및 결과 (완료된 경우)
    """
    result = request_manager.get_request(request_id)

    if not result:
        raise HTTPException(404, f"Request not found: {request_id}")

    # Parse JSON fields
    result = parse_request_json_fields(result)

    return JSONResponse(content=result)


@app.post("/cancel-request/{request_id}")
async def cancel_request(request_id: str):
    """
    요청 취소

    Args:
        request_id: 취소할 요청 ID

    Returns:
        취소 성공 여부
    """
    success = request_manager.cancel_request(request_id)

    if not success:
        # 이미 완료되었거나 존재하지 않는 요청
        request = request_manager.get_request(request_id)

        if not request:
            raise HTTPException(404, f"Request not found: {request_id}")

        status = request.get('status')
        if status in ["completed", "failed"]:
            raise HTTPException(400, f"Cannot cancel request: already {status}")
        elif status == "cancelled":
            return {"success": True, "message": "Request already cancelled"}

    return {
        "success": True,
        "request_id": request_id,
        "message": "Request cancelled successfully"
    }


@app.get("/requests")
async def get_requests(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    요청 목록 조회 (페이징)

    Args:
        status: 상태 필터 (pending, processing, completed, failed, cancelled)
        limit: 최대 개수 (기본 100)
        offset: 시작 위치 (기본 0)

    Returns:
        요청 목록
    """
    requests = request_manager.get_all_requests(status, limit, offset)

    # Parse JSON fields for all requests
    requests = [parse_request_json_fields(req) for req in requests]

    return {
        "total": len(requests),
        "limit": limit,
        "offset": offset,
        "requests": requests
    }


@app.get("/statistics")
async def get_statistics():
    """
    분석 통계 조회

    Returns:
        전체 요청 수, 상태별 개수, 총 비용, 평균 처리 시간
    """
    stats = request_manager.get_statistics()
    return stats


@app.delete("/request/{request_id}")
async def delete_request(request_id: str):
    """
    요청 삭제 (관리자용)

    Args:
        request_id: 삭제할 요청 ID

    Returns:
        삭제 성공 여부
    """
    success = request_manager.delete_request(request_id)

    if not success:
        raise HTTPException(404, f"Request not found: {request_id}")

    return {
        "success": True,
        "request_id": request_id,
        "message": "Request deleted successfully"
    }


@app.get("/download-patent-pdf/{request_id}")
async def download_patent_pdf(request_id: str):
    """
    원본 특허 PDF 파일 다운로드

    Args:
        request_id: 요청 ID (UUID)

    Returns:
        원본 특허 PDF 파일
    """
    # DB에서 요청 조회
    request = request_manager.get_request(request_id)

    if not request:
        raise HTTPException(404, f"Request not found: {request_id}")

    # PDF 파일 경로 확인
    pdf_file_path = request.get('pdf_file_path')

    if not pdf_file_path:
        raise HTTPException(404, f"No PDF file associated with this request")

    # 파일 존재 여부 확인
    pdf_path = Path(pdf_file_path)
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF file not found: {pdf_file_path}")

    # 원본 파일명 추출 (request_id_ 제거)
    original_filename = request.get('filename')
    if not original_filename:
        # 파일명에서 request_id_ 부분 제거
        original_filename = pdf_path.name.replace(f"{request_id}_", "")

    print(f"[API] Serving patent PDF for request {request_id}: {pdf_file_path}")

    return FileResponse(
        path=str(pdf_path),
        filename=original_filename,
        media_type="application/pdf"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)