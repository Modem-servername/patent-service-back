"""
Simplified Patent Infringement Analysis Module
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
import asyncio
import json
import time
import re


class InfringementCandidate(BaseModel):
    """Product/service candidate with potential infringement"""
    company: str
    product_service: str
    matching_feature: str
    relevance_score: float
    market_size: Optional[str] = None
    source_url: Optional[str] = None
    evidence_urls: Optional[List[str]] = []
    evidence_description: Optional[str] = None


class ClaimChartElement(BaseModel):
    """Claim Chart element model"""
    claim_element: str
    product_feature: str
    comment: str
    infringement_likelihood: str  # "High", "Medium", "Low"


class FollowUpResponse(BaseModel):
    """Response model for user follow-up questions"""
    question: str
    answer: str


class IndependentClaimAnalysis(BaseModel):
    """Individual independent claim analysis result model"""
    claim_number: str
    claim_original: str
    claim_english: str
    claim_korean: str
    claim_chart: List[ClaimChartElement]


class PatentInfringementAnalysis(BaseModel):
    """Patent infringement analysis result model"""
    title: Optional[str] = ""
    applicant: Optional[str] = ""
    issued_number: Optional[str] = ""
    application_number: Optional[str] = "N/A"
    priority_date: Optional[str] = "N/A"
    application_date: Optional[str] = "N/A"
    independent_claims: List[str] = []
    technology_summary: Optional[str] = ""
    potentially_matching_companies: List[InfringementCandidate] = []
    independent_claim_analyses: List[IndependentClaimAnalysis] = []
    follow_up_responses: List[FollowUpResponse] = []
    analysis_notes: Optional[str] = None


class SimplifiedInfringementAnalyzer:
    """Simplified patent infringement analyzer"""

    def __init__(self, api_key: str, tavily_api_key: Optional[str] = None, max_concurrent_requests: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)  # Add async client
        self.model = "gpt-5"

        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        # Limit concurrent requests (prevent rate limit)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Tavily web search (optional)
        self.tavily_client = None
        if tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                print("[Init] ✓ Web search enabled")
            except:
                print("[Init] ✗ Web search disabled")

    def _track_tokens(self, response) -> None:
        """Track token usage from API response"""
        if hasattr(response, 'usage'):
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # GPT-5 pricing: Input $1.25/M, Output $10.00/M
            cost = input_tokens * 1.25 / 1_000_000 + output_tokens * 10.00 / 1_000_000
            self.total_cost += cost

    def _reset_token_tracking(self) -> None:
        """Reset token tracking"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def _extract_claim_text(self, claim_full: str, claim_num: str) -> str:
        """
        Extract actual text from claim, excluding the number

        Args:
            claim_full: Full claim text (e.g., "1. A method..." or "청구항 1. ...")
            claim_num: Claim number (e.g., "1")

        Returns:
            Claim text (without number)
        """
        # Handle various formats: "1.", "1)", "청구항 1.", "Claim 1.", etc.
        patterns = [
            rf'^{claim_num}[\.\)]\s*',  # "1. " or "1) "
            rf'^청구항\s*{claim_num}\s*[:\.]\s*',  # "청구항 1: " or "청구항 1. "
            rf'^제\s*{claim_num}\s*항\s*[:\.]\s*',  # "제1항: " or "제1항. "
            rf'^Claim\s+{claim_num}\s*[:\.]\s*',  # "Claim 1: " or "Claim 1. "
        ]

        claim_text = claim_full.strip()
        for pattern in patterns:
            claim_text = re.sub(pattern, '', claim_text, flags=re.IGNORECASE)
            claim_text = claim_text.strip()

        return claim_text

    async def identify_independent_claims(self, claims: List[str]) -> List[Dict[str, Any]]:
        """
        Identify independent claims (optimized 2-step verification)

        Strategy:
        1. Rule-based pre-filtering (enhanced pattern matching)
        2. AI single verification (clear prompt)

        Args:
            claims: List of claims

        Returns:
            Independent claims list [{"number": "1", "text": "...", "full_text": "1. ..."}]
        """
        print("[Claims] Identifying independent claims...")

        if not claims:
            return []

        # Step 1: Enhanced rule-based filtering
        print("[Claims] Rule-based filtering...")

        definitely_dependent = set()
        possibly_independent = {}  # number -> full claim text

        for claim in claims:
            # Exclude cancelled/deleted claims
            claim_lower = claim.lower()
            if len(claim) < 50:
                continue

            # Check for various cancelled/deleted expressions (handle both British/American spelling, with/without parentheses)
            cancelled_keywords = [
                "cancelled", "canceled",  # Cancelled
                "deleted", "withdrawn",    # Deleted/withdrawn
                "void", "removed"          # Void/removed
            ]
            if any(keyword in claim_lower for keyword in cancelled_keywords):
                continue

            # Extract claim number (support various formats)
            claim_num = None

            # Format 1: "1. ..." or "1) ..."
            match = re.match(r'^(\d+)[\.\)]\s', claim)
            if match:
                claim_num = match.group(1)

            # Format 2: "청구항 1. ..." or "청구항 1: ..."
            if not claim_num:
                match = re.match(r'^청구항\s*(\d+)\s*[:\.]\s', claim)
                if match:
                    claim_num = match.group(1)

            # Format 3: "제1항. ..." or "제 1 항. ..."
            if not claim_num:
                match = re.match(r'^제\s*(\d+)\s*항\s*[:\.]\s', claim)
                if match:
                    claim_num = match.group(1)

            # Format 4: "Claim 1. ..." or "Claim 1: ..."
            if not claim_num:
                match = re.match(r'^Claim\s+(\d+)\s*[:\.]\s', claim, re.IGNORECASE)
                if match:
                    claim_num = match.group(1)

            if not claim_num:
                continue

            # Enhanced dependent claim pattern check
            is_dependent = False

            # Remove claim number to extract pure text
            claim_text_only = claim.strip()
            for prefix_pattern in [
                rf'^{claim_num}[\.\)]\s*',
                rf'^Claim\s+{claim_num}\s*[:\.]\s*',
                rf'^청구항\s*{claim_num}\s*[:\.]\s*',
                rf'^제\s*{claim_num}\s*항\s*[:\.]\s*',
            ]:
                claim_text_only = re.sub(prefix_pattern, '', claim_text_only, flags=re.IGNORECASE).strip()

            claim_text_lower = claim_text_only.lower()

            # English dependent claim patterns - check sentence start first
            english_strong_patterns = [
                r'^the\s+\w+\s+(of|in|according\s+to|as\s+defined\s+in)\s+claim\s+\d+',
                r'^according\s+to\s+(any\s+one\s+of\s+)?claims?\s+\d+',
                r'^the\s+\w+\s+of\s+claims?\s+\d+',
            ]

            for pattern in english_strong_patterns:
                if re.search(pattern, claim_text_lower):
                    is_dependent = True
                    break

            # English dependent claim patterns - anywhere in sentence
            if not is_dependent:
                english_patterns = [
                    r'of\s+claims?\s+\d+',
                    r'in\s+claims?\s+\d+',
                    r'according\s+to\s+claims?\s+\d+',
                    r'as\s+(claimed|defined|described)\s+in\s+claims?\s+\d+',
                    r'claims?\s+\d+\s*[-,]\s*\d+',
                    r'any\s+(one\s+)?of\s+claims?\s+\d+',
                ]

                for pattern in english_patterns:
                    if re.search(pattern, claim_text_lower):
                        is_dependent = True
                        break

            # Korean dependent claim patterns
            if not is_dependent:
                korean_patterns = [
                    r'^제\s*\d+\s*항에',
                    r'^제\s*\d+\s*항의',
                    r'^제\s*\d+\s*항\s+내지',
                    r'^제\s*\d+\s*항\s+또는',
                    r'^청구항\s*\d+에',
                    r'제\s*\d+\s*항에\s*있어서',
                    r'제\s*\d+\s*항에\s*따른',
                    r'제\s*\d+\s*항\s*내지\s*제\s*\d+\s*항',
                ]

                for pattern in korean_patterns:
                    if re.search(pattern, claim_text_only):
                        is_dependent = True
                        break

            if is_dependent:
                definitely_dependent.add(claim_num)
            else:
                possibly_independent[claim_num] = claim

        if not possibly_independent:
            # Fallback: 첫 번째 청구항 사용
            if claims:
                first = claims[0]
                num_str = first.split('.')[0].strip()
                num = num_str if num_str.isdigit() else "1"
                return [{"number": num,
                         "text": self._extract_claim_text(first, num),
                         "full_text": first}]
            return []

        print(f"[Claims] Found {len(possibly_independent)} potentially independent claim(s)")
        print("[Claims] AI verification...")

        # 가능성 있는 청구항들만 AI로 검증
        claims_to_verify = []
        for num in sorted(possibly_independent.keys(), key=int):
            claim_text = possibly_independent[num]
            claims_to_verify.append(f"[Claim {num}]\n{claim_text}")

        claims_text = "\n\n".join(claims_to_verify)

        prompt = f"""You are a patent claim expert. Identify which claims are INDEPENDENT (do not reference other claims).

**INDEPENDENT CLAIM**:
- Stands alone completely
- Starts with "A method...", "An apparatus...", "A system...", etc.
- NO references to other claim numbers
- Example: "1. A method comprising: step A; step B."

**DEPENDENT CLAIM** (mark as is_independent: false):
- References another claim number
- English: "claim 1", "according to claim", "the method of claim", "as in claim"
- Korean: "제1항", "제1항에", "제1항의", "청구항 1"
- Example: "2. The method of claim 1, further comprising..."

**YOUR TASK**: For EACH claim below, determine if it is independent or dependent.

CLAIMS:
{claims_text}

Return JSON:
{{
  "claims": [
    {{"number": "1", "is_independent": true, "reason": "No claim references"}},
    {{"number": "2", "is_independent": false, "reason": "References claim 1"}}
  ]
}}

**CRITICAL**: Look carefully for claim number references. If you find ANY reference to another claim, mark as dependent.

Return ONLY valid JSON."""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a patent claim analyzer. Be precise and thorough. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=4000
            )

            self._track_tokens(response)  # 토큰 추적

            response_text = response.choices[0].message.content
            if response_text and response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(response_text)

            # AI 결과 파싱 및 최종 검증
            verified_claims = []
            for item in result.get("claims", []):
                if item.get("is_independent"):
                    claim_num = str(item["number"])
                    if claim_num in possibly_independent:
                        full_claim = possibly_independent[claim_num]

                        # 추가 안전 체크: 다른 청구항 번호 참조가 있는지 확인
                        has_reference = False
                        all_other_nums = [n for n in possibly_independent.keys() if n != claim_num]

                        for other_num in all_other_nums:
                            # 영어 참조 체크
                            if re.search(rf'\bclaim\s+{other_num}\b', full_claim, re.IGNORECASE):
                                has_reference = True
                                break
                            # 한국어 참조 체크
                            if re.search(rf'제\s*{other_num}\s*항', full_claim):
                                has_reference = True
                                break

                        if not has_reference:
                            claim_text = self._extract_claim_text(full_claim, claim_num)
                            verified_claims.append({
                                "number": claim_num,
                                "text": claim_text,
                                "full_text": full_claim
                            })

            if not verified_claims:
                # Fallback: 첫 번째 후보 사용
                first_num = min(possibly_independent.keys(), key=int)
                first_claim = possibly_independent[first_num]
                return [{"number": first_num,
                         "text": self._extract_claim_text(first_claim, first_num),
                         "full_text": first_claim}]

            print(f"[Claims] ✓ Identified {len(verified_claims)} independent claim(s)")
            return verified_claims

        except Exception as e:
            print(f"[Claims] ✗ AI error: {e}")
            # Fallback: 룰 기반 결과 사용
            independent_claims = []
            for num in sorted(possibly_independent.keys(), key=int):
                full_claim = possibly_independent[num]
                independent_claims.append({
                    "number": num,
                    "text": self._extract_claim_text(full_claim, num),
                    "full_text": full_claim
                })
            print(f"[Claims] ✓ Using rule-based results: {len(independent_claims)} claim(s)")
            return independent_claims

    async def summarize_technology(self, patent_data: Dict) -> str:
        """Generate technology summary"""
        print("[Tech Summary] Generating summary...")

        title = patent_data.get("title", "")
        abstract = patent_data.get("abstract", "")
        description = patent_data.get("description", "")

        prompt = f"""Summarize the technology disclosed in this patent in 2-3 sentences IN KOREAN.

Title: {title}
Abstract: {abstract[:500]}
Description: {description[:500]}

**IMPORTANT**: Respond in Korean language only."""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a patent technology summarizer. Always respond in Korean."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=4000
            )

            self._track_tokens(response)

            summary = response.choices[0].message.content.strip()
            print(f"[Tech Summary] ✓ Generated")
            return summary

        except Exception as e:
            print(f"[Tech Summary] ✗ Error: {e}")
            return abstract[:200] if abstract else title

    async def find_potential_infringers(
        self,
        patent_data: Dict,
        claim_text: str,
        tech_summary: str,
        max_candidates: int = 5
    ) -> List[InfringementCandidate]:
        """Search for potential infringers"""
        print(f"[Infringers] Searching for potential infringers...")

        title = patent_data.get("title", "")

        # Use web search if available
        if self.tavily_client:
            print("[Infringers] Using web search...")
            try:
                query = f"{title} companies products similar technology"
                search_result = self.tavily_client.search(query=query, max_results=5)

                web_context = "\n\n".join([
                    f"- {r.get('title', '')}: {r.get('content', '')[:200]}"
                    for r in search_result.get('results', [])
                ])
            except Exception as e:
                print(f"[Infringers] Web search failed: {e}")
                web_context = ""
        else:
            web_context = ""

        # Extract applicant information
        applicant = patent_data.get("assignee", "") or patent_data.get("applicant", "")

        # Concise and clear prompt (Korean response)
        prompt = f"""Analyze patent for potential infringement.

PATENT:
- Title: {title}
- Technology: {tech_summary}
- Main Claim: {claim_text}
- Patent Applicant/Assignee: {applicant}

TASK: Find {max_candidates} real companies using similar technology.

CRITICAL EXCLUSION RULE:
DO NOT include the patent applicant or any companies from the same corporate group.
- If applicant is "삼성디스플레이 주식회사" or "Samsung Display", exclude ALL Samsung group companies (Samsung Electronics, Samsung SDI, etc.)
- If applicant is "LG전자" or "LG Electronics", exclude ALL LG group companies (LG Display, LG Innotek, LG Energy Solution, etc.)
- If applicant is "SK Hynix", exclude ALL SK group companies (SK Telecom, SK Innovation, etc.)
- If applicant is "현대자동차" or "Hyundai Motor", exclude ALL Hyundai/Kia group companies
- Apply this logic to ALL corporate groups worldwide (Apple/subsidiaries, Google/Alphabet, Microsoft/GitHub/LinkedIn, Meta/Facebook/Instagram, etc.)

OUTPUT (valid JSON only, Korean descriptions):
{{
  "candidates": [
    {{
      "company": "CompanyName Inc.",
      "product_service": "제품명 (한국어)",
      "matching_feature": "기술적 유사성 설명 (한국어로 100단어 이상)",
      "relevance_score": 0.80,
      "source_url": "https://company.com/product",
      "evidence_urls": ["https://news.com/article1", "https://techblog.com/article2"],
      "evidence_description": "근거 자료 설명 (한국어): 공식 발표, 기술 블로그, 뉴스 기사, 특허 출원 등의 구체적인 출처와 내용"
    }}
  ]
}}

Companies to consider: Apple, Google, Microsoft, Amazon, Samsung, IBM, Intel, NVIDIA, Oracle, Adobe, Cisco, Dell, HP, Sony, LG, Qualcomm, Tesla, etc.

RULES:
- Return ONLY valid JSON (start with {{ end with }})
- {max_candidates} candidates exactly
- Company names in English, descriptions in Korean
- Scores between 0.70-0.95
- Each matching_feature: 100+ words in Korean
- MUST provide source_url (company website, product page, or official tech documentation)
- MUST provide evidence_urls (news articles, technical blogs, patents, press releases - real URLs)
- MUST provide evidence_description explaining why these sources prove the technology match
- NEVER include applicant's corporate group"""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a patent infringement analyst. You MUST respond with valid JSON only. Write product names and descriptions in Korean."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=8000  # 증가: 침해사례 생성에 충분한 토큰
            )

            # 토큰 사용량 추적
            self._track_tokens(response)

            # 응답 검증 강화
            message = response.choices[0].message

            # 디버깅: 응답 객체 전체 확인
            print(f"[Infringers] DEBUG - Response object: {response}")
            print(f"[Infringers] DEBUG - Message: {message}")
            print(f"[Infringers] DEBUG - Has refusal attr: {hasattr(message, 'refusal')}")

            # Refusal 체크
            if hasattr(message, 'refusal') and message.refusal:
                print(f"[Infringers] ✗ API refused the request: {message.refusal}")
                return []

            response_text = message.content
            print(f"[Infringers] DEBUG - Response text type: {type(response_text)}")
            print(f"[Infringers] DEBUG - Response text length: {len(response_text) if response_text else 0}")
            if response_text:
                print(f"[Infringers] DEBUG - First 200 chars: {response_text[:200]}")

            if response_text is None or len(response_text.strip()) == 0:
                print(f"[Infringers] ✗ Empty response, trying fallback...")
                # Fallback: 간단한 프롬프트로 재시도
                fallback_prompt = f"""Based on this patent technology: {tech_summary[:300]}

Generate {max_candidates} major companies that might use similar technology.
Return ONLY valid JSON in this exact format (all text in Korean except company names):
{{
  "candidates": [
    {{"company": "CompanyName", "product_service": "제품명 (한국어)", "matching_feature": "기술 설명 (한국어로 100단어)", "relevance_score": 0.75, "source_url": "https://company.com/product", "evidence_urls": ["https://news.com"], "evidence_description": "근거 설명 (한국어)"}}
  ]
}}"""

                try:
                    print(f"[Infringers] Retrying with simplified prompt...")
                    retry_response = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a patent analyst. Respond with valid JSON only. Write in Korean except for company names."},
                            {"role": "user", "content": fallback_prompt}
                        ],
                        max_completion_tokens=4000
                    )
                    self._track_tokens(retry_response)
                    retry_message = retry_response.choices[0].message
                    retry_text = retry_message.content

                    print(f"[Infringers] DEBUG - Fallback response: {retry_response}")
                    print(f"[Infringers] DEBUG - Fallback message: {retry_message}")
                    print(f"[Infringers] DEBUG - Fallback text type: {type(retry_text)}")
                    print(f"[Infringers] DEBUG - Fallback text length: {len(retry_text) if retry_text else 0}")
                    if retry_text:
                        print(f"[Infringers] DEBUG - Fallback first 200 chars: {retry_text[:200]}")

                    if retry_text and len(retry_text.strip()) > 0:
                        print(f"[Infringers] ✓ Fallback succeeded")
                        response_text = retry_text
                    else:
                        print(f"[Infringers] ✗ Fallback failed - empty response")
                        return []
                except Exception as fallback_error:
                    print(f"[Infringers] ✗ Fallback error: {fallback_error}")
                    return []

            # JSON 추출 시도
            json_text = response_text.strip()

            # 코드 블록 제거
            if json_text.startswith("```"):
                lines = json_text.split('\n')
                json_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_text
                json_text = json_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(json_text)

            # 출원인(applicant) 정보 추출
            applicant = patent_data.get("assignee", "") or patent_data.get("applicant", "")
            applicant_lower = applicant.lower().strip()

            candidates = []
            excluded_count = 0
            for item in result.get("candidates", [])[:max_candidates]:
                company_name = item.get("company", "")
                company_lower = company_name.lower().strip()

                # 안전장치: 출원인과 정확하게 동일한 회사명은 제외 (AI가 놓칠 수 있으므로)
                if applicant_lower and company_lower:
                    # 회사명이 출원인에 포함되거나 출원인이 회사명에 포함되는 경우 제외
                    if company_lower in applicant_lower or applicant_lower in company_lower:
                        excluded_count += 1
                        continue

                candidates.append(InfringementCandidate(
                    company=company_name,
                    product_service=item.get("product_service", ""),
                    matching_feature=item.get("matching_feature", ""),
                    relevance_score=item.get("relevance_score", 0.0),
                    source_url=item.get("source_url"),
                    evidence_urls=item.get("evidence_urls", []),
                    evidence_description=item.get("evidence_description")
                ))

            status_msg = f"[Infringers] ✓ Found {len(candidates)} candidates"
            if excluded_count > 0:
                status_msg += f" (excluded {excluded_count} applicant-related)"
            print(status_msg)
            return candidates

        except Exception as e:
            print(f"[Infringers] ✗ Error: {e}")
            return []

    async def translate_claim_async(self, claim_text: str) -> str:
        """Translate claim to Korean (async version - for parallel processing)"""
        # Check if Korean
        if re.search(r'[가-힣]', claim_text):
            return claim_text

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Translate the patent claim to Korean accurately."},
                    {"role": "user", "content": claim_text}
                ],
                max_completion_tokens=4000
            )
            self._track_tokens(response)  # 토큰 추적
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Translation] Error: {e}")
            return claim_text  # 실패 시 원문 반환

    async def translate_claim_to_english_async(self, claim_text: str) -> str:
        """Translate claim to English (async version - for parallel processing)"""
        # Check if already English (no Korean characters)
        if not re.search(r'[가-힣]', claim_text):
            return claim_text

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Translate the patent claim to English accurately."},
                    {"role": "user", "content": claim_text}
                ],
                max_completion_tokens=4000
            )
            self._track_tokens(response)  # 토큰 추적
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Translation to English] Error: {e}")
            return claim_text  # 실패 시 원문 반환

    async def translate_claims_batch(self, claims_texts: List[str], batch_size: int = 10) -> List[str]:
        """Batch translate claims to Korean (parallel processing)"""
        all_translations = []
        total_batches = (len(claims_texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(claims_texts), batch_size):
            batch = claims_texts[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            print(f"[Translation to Korean] Batch {batch_num}/{total_batches}: Processing {len(batch)} claim(s)...")

            # 배치 내에서 병렬 처리
            batch_translations = await asyncio.gather(*[
                self.translate_claim_async(text) for text in batch
            ])

            all_translations.extend(batch_translations)
            print(f"[Translation to Korean] ✓ Batch {batch_num}/{total_batches} completed")

        return all_translations

    async def translate_claims_to_english_batch(self, claims_texts: List[str], batch_size: int = 10) -> List[str]:
        """Batch translate claims to English (parallel processing)"""
        all_translations = []
        total_batches = (len(claims_texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(claims_texts), batch_size):
            batch = claims_texts[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            print(f"[Translation to English] Batch {batch_num}/{total_batches}: Processing {len(batch)} claim(s)...")

            # 배치 내에서 병렬 처리
            batch_translations = await asyncio.gather(*[
                self.translate_claim_to_english_async(text) for text in batch
            ])

            all_translations.extend(batch_translations)
            print(f"[Translation to English] ✓ Batch {batch_num}/{total_batches} completed")

        return all_translations

    def translate_claim(self, claim_text: str) -> str:
        """Translate claim (English → Korean, Korean → keep) - sync version (backward compatibility)"""
        print("[Translation] Translating claim...")

        # Check if Korean
        if re.search(r'[가-힣]', claim_text):
            print("[Translation] Already in Korean, skipping")
            return claim_text

        try: # This synchronous method is not used in the main async flow, so it's fine.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Translate the patent claim to Korean accurately."},
                    {"role": "user", "content": claim_text}
                ],
                max_completion_tokens=4000
            )

            self._track_tokens(response)  # 토큰 추적

            translation = response.choices[0].message.content.strip()
            print(f"[Translation] OK Translated: {translation[:100]}...")
            return translation

        except Exception as e:
            print(f"[Translation] Error: {e}")
            return claim_text

    async def create_claim_chart_with_limit(
        self,
        claim_num: str,
        claim_text: str,
        company: str,
        product: str,
        patent_data: Dict
    ) -> tuple:
        """Generate Claim Chart (with rate limit control)"""
        async with self.semaphore:
            print(f"[Claim Chart] Creating chart for Claim {claim_num}...")
            result = await self.create_claim_chart(claim_text, company, product, patent_data)
            print(f"[Claim Chart] ✓ Claim {claim_num} completed ({len(result)} elements)")
            return (claim_num, result)

    async def create_claim_chart(
        self,
        claim_text: str,
        company: str,
        product: str,
        patent_data: Dict
    ) -> List[ClaimChartElement]:
        """Generate Claim Chart (simplified)"""
        prompt = f"""Create a claim chart comparing the patent claim with the product IN KOREAN.

Patent Claim (Korean):
{claim_text}

Target Product: {company} - {product}

Break down the claim into key elements and compare with product features.

Return JSON IN KOREAN:
{{
  "elements": [
    {{
      "claim_element": "청구항의 구성요소 (한국어)",
      "product_feature": "제품의 대응하는 기능 (한국어)",
      "comment": "분석 및 비교 설명 (한국어)",
      "infringement_likelihood": "High/Medium/Low"
    }}
  ]
}}

**IMPORTANT**: Write claim_element, product_feature, and comment in KOREAN language.
Return ONLY valid JSON."""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a patent claim chart expert. Respond ONLY with valid JSON. Write all descriptions in Korean."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=4000
            )

            self._track_tokens(response)  # 토큰 추적

            response_text = response.choices[0].message.content
            if response_text and response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(response_text)

            chart_elements = []
            for item in result.get("elements", []):
                chart_elements.append(ClaimChartElement(
                    claim_element=item.get("claim_element", ""),
                    product_feature=item.get("product_feature", ""),
                    comment=item.get("comment", ""),
                    infringement_likelihood=item.get("infringement_likelihood", "Low")
                ))

            return chart_elements

        except Exception as e:
            print(f"[Claim Chart] ✗ Error: {e}")
            return []

    async def answer_follow_up_question_with_limit(self, question: str, patent_data: Dict, analysis_context: str, index: int, total: int) -> FollowUpResponse:
        """Answer follow-up question (with rate limit control)"""
        async with self.semaphore:
            print(f"[Follow-up] Processing question {index}/{total}...")
            answer = await self.answer_follow_up_question(question, patent_data, analysis_context)
            print(f"[Follow-up] ✓ Question {index}/{total} completed")
            return FollowUpResponse(question=question, answer=answer)

    async def answer_follow_up_question(self, question: str, patent_data: Dict, analysis_context: str) -> str:
        """Answer user's follow-up question"""
        prompt = f"""You are a world-class patent and business analyst. You have already performed an initial patent analysis. Now, answer the user's follow-up question based on the provided context.

**INITIAL ANALYSIS CONTEXT:**
{analysis_context}

**FULL PATENT DATA:**
- Title: {patent_data.get('title', 'N/A')}
- Abstract: {patent_data.get('abstract', 'N/A')}
- First 5 Claims: {patent_data.get('claims', [])[:5]}

**USER'S FOLLOW-UP QUESTION:**
"{question}"

**YOUR TASK:**
1.  Thoroughly understand the user's question in the context of the patent analysis.
2.  If the question is about technology or infringement, use the provided analysis context and patent data.
3.  If the question requires external, real-time, or market data (e.g., sales figures, market share for '삼성전자 갤럭시'), use your general knowledge. If you have access to web search results, use them and cite the source URL.
4.  Provide a clear, concise, and helpful answer in KOREAN.

**Answer (in Korean):**
"""
        if self.tavily_client and any(keyword in question for keyword in ["매출", "시장", "얼마", "조사", "판매량"]):
            print(f"[Follow-up] Using web search for question: {question}")
            try:
                query = f"{patent_data.get('title', '')} {question}"
                search_result = self.tavily_client.search(query=query, search_depth="advanced", max_results=3)
                if search_result and search_result.get('results'):
                    web_context = "\n\n**ADDITIONAL INFORMATION FROM WEB SEARCH:**\n" + "\n\n".join([
                        f"- Source: {r.get('url', 'N/A')}\n- Content: {r.get('content', '')}"
                        for r in search_result.get('results', [])
                    ])
                    prompt += web_context
            except Exception as e:
                print(f"[Follow-up] Web search failed: {e}")

        try:
            response = await self.async_client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": "You are a helpful patent and business analyst assistant. Always respond in Korean. Be thorough and cite sources if you use web search results."}, {"role": "user", "content": prompt}], max_completion_tokens=4000)
            self._track_tokens(response)
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            print(f"[Follow-up] ✗ Error: {e}")
            return "추가 질문에 대한 답변을 생성하는 중 오류가 발생했습니다."

    async def analyze_infringement(
        self,
        patent_data: Dict,
        max_candidates: int = 10,
        create_detailed_chart: bool = True,
        model: str = "gpt-5",
        follow_up_questions: Optional[List[str]] = None
    ) -> PatentInfringementAnalysis:
        """Perform complete infringement analysis (optimized version - async)"""
        # Model configuration (can use different models per request)
        original_model = self.model
        self.model = model

        print("\n" + "="*60)
        print("Patent Infringement Analysis")
        print("="*60)
        print(f"AI Model: {self.model}")
        print(f"Max Candidates: {max_candidates}")
        print("="*60)

        start_time = time.time()
        self._reset_token_tracking()  # Reset token tracking

        try:
            # 1. Extract metadata
            metadata = {
                "title": patent_data.get("title", ""),
                "applicant": patent_data.get("assignee", ""),
                "issued_number": patent_data.get("patent_number", ""),
                "application_date": patent_data.get("filing_date", "N/A")
            }

            # 2. Identify independent claims
            claims = patent_data.get("claims", [])

            # Pre-filter cancelled/deleted claims
            cancelled_keywords = ["cancelled", "canceled", "deleted", "withdrawn", "void", "removed"]
            valid_claims = []
            for claim in claims:
                claim_lower = claim.lower()
                if any(keyword in claim_lower for keyword in cancelled_keywords):
                    continue
                valid_claims.append(claim)

            print(f"\n[Analysis] Processing {len(valid_claims)} valid claim(s) (filtered {len(claims) - len(valid_claims)} cancelled)")

            # 2 & 3. Identify independent claims + Summarize technology (parallel processing)
            print(f"[Analysis] Step 1: Identifying claims & summarizing tech (parallel)...")
            independent_claims_data, tech_summary = await asyncio.gather(
                self.identify_independent_claims(valid_claims),
                self.summarize_technology(patent_data)
            )

            if not independent_claims_data:
                independent_claims_data = [{"number": "1", "text": valid_claims[0] if valid_claims else ""}]

            print(f"[Analysis] ✓ Step 1 completed: {len(independent_claims_data)} independent claims, tech summary ready")

            # 4. Search for infringers
            print(f"\n[Analysis] Step 2: Searching for potential infringers...")
            first_claim = independent_claims_data[0]["text"]
            candidates = await self.find_potential_infringers(
                patent_data, first_claim, tech_summary, max_candidates
            )

            # 5. Analyze each independent claim (optimized parallel processing)
            print(f"\n[Analysis] Step 3: Analyzing {len(independent_claims_data)} independent claim(s)...")

            # Check if Korean patent (check only once)
            is_korean_patent = bool(re.search(r'[가-힣]', first_claim))
            print(f"[Analysis] Patent language: {'Korean' if is_korean_patent else 'English/Other'}")

            # Batch translation processing (10 at a time in parallel - speed improvement)
            claim_texts = [claim["text"] for claim in independent_claims_data]

            if is_korean_patent:
                print("[Analysis] Translation to Korean: Skipped (already Korean)")
                translations_korean = claim_texts
                print(f"[Analysis] Translation to English: Processing {len(independent_claims_data)} claim(s)...")
                translations_english = await self.translate_claims_to_english_batch(claim_texts, batch_size=10)
                print(f"[Analysis] ✓ English translation completed")
            else:
                print(f"[Analysis] Translation to Korean: Processing {len(independent_claims_data)} claim(s)...")
                translations_korean = await self.translate_claims_batch(claim_texts, batch_size=10)
                print(f"[Analysis] ✓ Korean translation completed")
                print("[Analysis] Translation to English: Skipped (already English)")
                translations_english = claim_texts

            # Generate Claim Charts in parallel
            independent_claim_analyses = []
            if create_detailed_chart and candidates:
                print(f"\n[Analysis] Step 4: Creating claim charts for {len(independent_claims_data)} claim(s) in parallel...")
                top_candidate = candidates[0]

                # Generate all claim charts in parallel
                chart_tasks = [
                    self.create_claim_chart_with_limit(
                        ind_claim["number"],
                        claim_korean,
                        top_candidate.company,
                        top_candidate.product_service,
                        patent_data
                    )
                    for ind_claim, claim_korean in zip(independent_claims_data, translations_korean)
                ]

                chart_results = await asyncio.gather(*chart_tasks)

                # Combine results
                for ind_claim, claim_korean, claim_english, (_, claim_chart) in zip(independent_claims_data, translations_korean, translations_english, chart_results):
                    independent_claim_analyses.append(IndependentClaimAnalysis(
                        claim_number=ind_claim["number"],
                        claim_original=ind_claim.get("full_text", ind_claim["text"]),
                        claim_english=claim_english,
                        claim_korean=claim_korean,
                        claim_chart=claim_chart
                    ))

                print(f"[Analysis] ✓ Step 4 completed: All claim charts created")
            else:
                # Generate analysis results without claim charts
                for ind_claim, claim_korean, claim_english in zip(independent_claims_data, translations_korean, translations_english):
                    independent_claim_analyses.append(IndependentClaimAnalysis(
                        claim_number=ind_claim["number"],
                        claim_original=ind_claim.get("full_text", ind_claim["text"]),
                        claim_english=claim_english,
                        claim_korean=claim_korean,
                        claim_chart=[]
                    ))
                print(f"[Analysis] Skipping claim charts (no candidates or disabled)")

            # 6. Answer follow-up questions (parallel processing)
            follow_up_responses = []
            if follow_up_questions:
                print(f"\n[Analysis] Step 5: Answering {len(follow_up_questions)} follow-up question(s) in parallel...")
                # Build context for follow-up questions
                analysis_context = f"""
- Patent Title: {metadata.get('title', 'N/A')}
- Technology Summary: {tech_summary}
- Independent Claims Identified: {', '.join([f'Claim {ic["number"]}' for ic in independent_claims_data])}
- Potentially Matching Companies Found: {', '.join([c.company for c in candidates]) if candidates else 'None'}
- Top Candidate: {candidates[0].company if candidates else 'N/A'} (Product: {candidates[0].product_service if candidates else 'N/A'})"""

                # Process all questions in parallel
                followup_tasks = [
                    self.answer_follow_up_question_with_limit(
                        question, patent_data, analysis_context, idx + 1, len(follow_up_questions)
                    )
                    for idx, question in enumerate(follow_up_questions)
                ]

                follow_up_responses = await asyncio.gather(*followup_tasks)
                print(f"[Analysis] ✓ Step 5 completed: All questions answered")

            elapsed = time.time() - start_time

            print(f"\n{'='*60}")
            print(f"✓ Analysis Completed Successfully")
            print(f"{'='*60}")
            print(f"Time elapsed:        {elapsed:.1f}s")
            print(f"Independent claims:  {len(independent_claims_data)}")
            print(f"Candidates found:    {len(candidates)}")
            print(f"{'='*60}")
            print(f"Token Usage:")
            print(f"  Input:  {self.total_input_tokens:,} tokens")
            print(f"  Output: {self.total_output_tokens:,} tokens")
            print(f"  Total:  {self.total_input_tokens + self.total_output_tokens:,} tokens")
            print(f"  Cost:   ${self.total_cost:.4f}")
            print(f"{'='*60}\n")

            return PatentInfringementAnalysis(
                title=metadata["title"],
                applicant=metadata["applicant"],
                issued_number=metadata["issued_number"],
                application_date=metadata["application_date"],
                independent_claims=[f"Claim {ic['number']}" for ic in independent_claims_data],
                technology_summary=tech_summary,
                potentially_matching_companies=candidates,
                independent_claim_analyses=independent_claim_analyses,
                follow_up_responses=follow_up_responses,
                analysis_notes=f"Analysis time: {elapsed:.1f}s, Independent claims: {len(independent_claims_data)}, Total tokens: {self.total_input_tokens + self.total_output_tokens:,} (Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,}), Cost: ${self.total_cost:.6f}"
            )

        except Exception as e:
            print(f"\n[Error] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original model
            self.model = original_model


def format_analysis_report(analysis: PatentInfringementAnalysis) -> str:
    """Format analysis results as markdown report"""
    report = f"""# Patent Infringement Analysis Report

## 1. Patent Information

**Title:** {analysis.title}
**Applicant:** {analysis.applicant}
**Patent Number:** {analysis.issued_number}
**Application Date:** {analysis.application_date}

---

## 2. Technology Summary

{analysis.technology_summary}

---

## 3. Independent Claims

"""
    for ic in analysis.independent_claims:
        report += f"- {ic}\n"

    report += "\n---\n\n## 4. Potential Infringers\n\n"

    for idx, candidate in enumerate(analysis.potentially_matching_companies, 1):
        report += f"### {idx}. {candidate.company}\n\n"
        report += f"**제품/서비스:** {candidate.product_service}\n\n"
        report += f"**유사도 점수:** {candidate.relevance_score:.2f}\n\n"
        report += f"**기술적 유사성:**\n{candidate.matching_feature}\n\n"

        if candidate.source_url:
            report += f"**출처:** [{candidate.source_url}]({candidate.source_url})\n\n"

        if candidate.evidence_urls:
            report += f"**근거 자료:**\n"
            for evidence_url in candidate.evidence_urls:
                report += f"- [{evidence_url}]({evidence_url})\n"
            report += "\n"

        if candidate.evidence_description:
            report += f"**근거 설명:**\n{candidate.evidence_description}\n\n"

        report += "---\n\n"

    # 각 독립항 분석
    for idx, claim_analysis in enumerate(analysis.independent_claim_analyses, 1):
        report += f"## {4 + idx}. Claim {claim_analysis.claim_number} Analysis\n\n"
        report += f"### Original\n\n```\n{claim_analysis.claim_original}\n```\n\n"
        report += f"### English Translation\n\n{claim_analysis.claim_english}\n\n"
        report += f"### Korean Translation\n\n{claim_analysis.claim_korean}\n\n"

        if claim_analysis.claim_chart:
            report += f"### Claim Chart\n\n"
            report += "| Claim Element | Product Feature | Comment | Likelihood |\n"
            report += "|---------------|-----------------|---------|------------|\n"

            for element in claim_analysis.claim_chart:
                report += f"| {element.claim_element[:30]}... | {element.product_feature[:30]}... | {element.comment[:30]}... | **{element.infringement_likelihood}** |\n"

        report += "\n---\n\n"

    # 추가 질문 답변 섹션
    if analysis.follow_up_responses:
        follow_up_start_index = 4 + len(analysis.independent_claim_analyses) + 1
        report += f"## {follow_up_start_index}. Additional User Inquiries\n\n"
        for i, followup in enumerate(analysis.follow_up_responses, 1):
            report += f"### Inquiry {i}: {followup.question}\n\n"
            report += f"**Answer:**\n{followup.answer}\n\n"
        report += "---\n\n"

    if analysis.analysis_notes:
        report += f"## Analysis Info\n\n{analysis.analysis_notes}\n"

    return report
