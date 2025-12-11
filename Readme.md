# 분석 요청 취소 API 문서

## 개요

특허 침해 분석은 시간이 오래 걸리는 작업입니다. 사용자가 분석 중에 요청을 취소할 수 있도록 취소 API를 제공합니다.

---

## 1. 요청 취소 API

### Endpoint
```
POST /cancel-request/{request_id}
```

### Parameters
- `request_id` (path parameter): 취소하려는 분석 요청의 UUID

### Response

#### 성공 (200 OK)
```json
{
  "success": true,
  "message": "Request cancelled successfully",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 이미 완료된 요청 (400 Bad Request)
```json
{
  "success": false,
  "message": "Cannot cancel: request already completed"
}
```

#### 존재하지 않는 요청 (404 Not Found)
```json
{
  "detail": "Request not found"
}
```

### 사용 예시
```javascript
// JavaScript/TypeScript
async function cancelAnalysis(requestId) {
  const response = await fetch(`/cancel-request/${requestId}`, {
    method: 'POST'
  });

  const result = await response.json();

  if (result.success) {
    console.log('분석이 취소되었습니다');
  } else {
    console.log('취소 실패:', result.message);
  }
}
```

---

## 2. 요청 상태 확인 API

### Endpoint
```
GET /request-status/{request_id}
```

### Response
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "created_at": "2024-01-15T10:30:00",
  "started_at": "2024-01-15T10:30:05",
  "cancelled_at": "2024-01-15T10:32:30",
  "patent_number": "US1234567A",
  "error_message": null
}
```

### 상태 값
- `pending`: 대기 중
- `processing`: 분석 진행 중
- `completed`: 완료
- `failed`: 실패
- `cancelled`: 취소됨

---

## 3. 취소 가능 여부

| 상태 | 취소 가능 여부 |
|------|---------------|
| `pending` | ✅ 가능 |
| `processing` | ✅ 가능 |
| `completed` | ❌ 불가능 |
| `failed` | ❌ 불가능 |
| `cancelled` | ❌ 불가능 (이미 취소됨) |

---

## 4. 프론트엔드 구현 가이드

### 4.1 기본 구현 (React 예시)

```jsx
import { useState, useEffect } from 'react';

function AnalysisComponent() {
  const [requestId, setRequestId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [isCancelling, setIsCancelling] = useState(false);

  // 분석 시작
  const startAnalysis = async (patentNumber) => {
    const response = await fetch('/analyze-infringement', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patent_number: patentNumber,
        max_candidates: 10,
        create_detailed_chart: true,
        model: 'gpt-4o-mini'
      })
    });

    const data = await response.json();

    if (data.success) {
      setRequestId(data.request_id);
      setStatus('processing');
      startPolling(data.request_id);
    }
  };

  // 폴링으로 상태 확인
  const startPolling = (reqId) => {
    const intervalId = setInterval(async () => {
      const response = await fetch(`/request-status/${reqId}`);
      const data = await response.json();

      setStatus(data.status);

      // 완료, 실패, 취소 시 폴링 중단
      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        clearInterval(intervalId);
      }
    }, 2000); // 2초마다 확인
  };

  // 취소 버튼 클릭
  const handleCancel = async () => {
    if (!requestId) return;

    setIsCancelling(true);

    try {
      const response = await fetch(`/cancel-request/${requestId}`, {
        method: 'POST'
      });

      const result = await response.json();

      if (result.success) {
        setStatus('cancelled');
      } else {
        alert(result.message);
      }
    } catch (error) {
      console.error('취소 실패:', error);
    } finally {
      setIsCancelling(false);
    }
  };

  return (
    <div>
      <h2>특허 침해 분석</h2>

      {/* 상태 표시 */}
      <div>상태: {status}</div>

      {/* 취소 버튼 (processing 중에만 표시) */}
      {status === 'processing' && (
        <button
          onClick={handleCancel}
          disabled={isCancelling}
        >
          {isCancelling ? '취소 중...' : '분석 취소'}
        </button>
      )}

      {/* 결과 표시 */}
      {status === 'completed' && <div>분석 완료!</div>}
      {status === 'cancelled' && <div>분석이 취소되었습니다</div>}
      {status === 'failed' && <div>분석 실패</div>}
    </div>
  );
}
```

### 4.2 Vanilla JavaScript 예시

```javascript
class AnalysisManager {
  constructor(requestId) {
    this.requestId = requestId;
    this.pollingInterval = null;
  }

  // 상태 폴링 시작
  startPolling(onStatusChange) {
    this.pollingInterval = setInterval(async () => {
      const response = await fetch(`/request-status/${this.requestId}`);
      const data = await response.json();

      onStatusChange(data.status);

      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        this.stopPolling();
      }
    }, 2000);
  }

  // 폴링 중단
  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  }

  // 요청 취소
  async cancel() {
    const response = await fetch(`/cancel-request/${this.requestId}`, {
      method: 'POST'
    });
    return await response.json();
  }
}

// 사용 예시
const manager = new AnalysisManager('550e8400-e29b-41d4-a716-446655440000');

manager.startPolling((status) => {
  console.log('현재 상태:', status);
  document.getElementById('status').textContent = status;
});

document.getElementById('cancelBtn').addEventListener('click', async () => {
  const result = await manager.cancel();
  if (result.success) {
    alert('취소되었습니다');
  }
});
```

---

## 5. 중요 사항

### 5.1 취소 타이밍
- 취소 요청은 **즉시 반영되지 않습니다**
- 백엔드는 분석의 각 단계 사이에서 취소 여부를 확인합니다
- 현재 진행 중인 단계가 완료된 후 취소가 적용됩니다
- 최대 지연 시간은 현재 단계의 처리 시간입니다

### 5.2 취소 체크포인트
분석은 다음 단계들 사이에서 취소를 확인합니다:
1. Step 1 (특허 문서 로드) 전
2. Step 2 (선행기술 검색) 전
3. Step 3 (침해 분석) 전
4. Step 4 (추가 질문 처리) 전

### 5.3 서버 재시작 시 동작
- 서버가 재시작되면 60분 이상 `pending` 또는 `processing` 상태인 요청들이 자동으로 `failed` 처리됩니다
- 에러 메시지: `"Request timed out or server was restarted while in '{status}' state"`
- 프론트엔드는 이러한 경우를 감지하여 사용자에게 재시도를 안내해야 합니다

### 5.4 폴링 권장사항
- **폴링 간격**: 2-5초 권장
  - 너무 짧으면 서버 부하 증가
  - 너무 길면 UX 저하
- **타임아웃**: 프론트엔드에서 최대 대기 시간 설정 권장 (예: 30분)
- **에러 처리**: 네트워크 에러 시 재시도 로직 구현

### 5.5 UX 권장사항
1. 취소 버튼에 확인 다이얼로그 추가
2. 취소 중 로딩 인디케이터 표시
3. 취소 완료 후 명확한 피드백 제공
4. 페이지 이탈 시 경고 (분석 진행 중일 때)

---

## 6. 전체 워크플로우

```
[사용자]
   │
   ├─ POST /analyze-infringement
   │     └─ response: { request_id: "xxx" }
   │
   ├─ 폴링 시작 (2초마다)
   │     └─ GET /request-status/{request_id}
   │           └─ { status: "processing" }
   │
   ├─ [사용자가 취소 버튼 클릭]
   │     └─ POST /cancel-request/{request_id}
   │           └─ { success: true }
   │
   └─ 폴링으로 상태 확인
         └─ GET /request-status/{request_id}
               └─ { status: "cancelled" }
```

---

## 7. 참고사항

### 데이터베이스 스키마
취소 관련 필드는 `analysis_requests` 테이블에 저장됩니다:
- `status`: 요청 상태
- `cancelled_at`: 취소된 시각 (ISO 8601 형식)

### 로그 메시지
서버 로그에서 다음과 같은 메시지를 확인할 수 있습니다:
```
[Request Manager] Updated {request_id} status to: cancelled
[API] Request {request_id} was cancelled before Step 2
```

---

## 8. 추가 개발 시 고려사항

### 향후 개선 가능한 부분
1. **WebSocket 지원**: 폴링 대신 실시간 상태 업데이트
2. **부분 결과 저장**: 취소 시점까지의 부분 결과 제공
3. **취소 이유 수집**: 사용자가 취소 이유를 입력할 수 있도록
4. **우선순위 큐**: 취소된 요청의 리소스를 다른 요청에 재할당

---

**문의사항이나 버그 리포트는 백엔드 팀에게 전달해주세요.**
