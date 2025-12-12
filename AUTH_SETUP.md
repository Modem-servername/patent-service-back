# 로그인 시스템 설정 가이드

bcrypt + JWT 기반 공통 비밀번호 로그인 시스템

---

## 1. 비밀번호 해시 생성

### 단계 1: 해시 생성 스크립트 실행
```bash
python generate_password_hash.py
```

### 단계 2: 비밀번호 입력
- 원하는 비밀번호 입력 (최소 8자)
- 확인을 위해 동일한 비밀번호 재입력

### 단계 3: 출력된 해시 복사
```
✅ 해시 생성 완료!
==================================================
다음 내용을 .env 파일에 추가하세요:

ADMIN_PASSWORD_HASH=$2b$12$abcdefghijklmnopqrstuvwxyz1234567890...
```

---

## 2. .env 파일 설정

`.env` 파일에 다음 두 줄을 추가하세요:

```env
# 관리자 비밀번호 해시 (generate_password_hash.py로 생성)
ADMIN_PASSWORD_HASH=$2b$12$abcdefghijklmnopqrstuvwxyz1234567890...

# JWT 시크릿 키 (랜덤 문자열, 최소 32자 권장)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-to-random-string
```

### JWT 시크릿 키 생성 (선택사항)
```bash
# Python으로 랜덤 키 생성
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 3. 서버 재시작

설정 변경 후 서버를 재시작하세요:
```bash
# 서버 종료 (Ctrl+C)
# 서버 재시작
python main.py
```

---

## 4. API 사용법

### 로그인
```bash
POST http://localhost:8000/login
Content-Type: application/json

{
  "password": "your-password-here"
}
```

**응답:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### 보호된 엔드포인트 접근
```bash
GET http://localhost:8000/protected-example
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**응답:**
```json
{
  "message": "This is a protected route",
  "user": {
    "type": "admin",
    "authenticated": true,
    "exp": 1234567890
  }
}
```

---

## 5. 프론트엔드 구현 예시

### JavaScript/TypeScript

```javascript
// 로그인
async function login(password) {
  const response = await fetch('http://localhost:8000/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ password })
  });

  if (!response.ok) {
    throw new Error('Login failed');
  }

  const data = await response.json();

  // 토큰 저장 (localStorage 또는 sessionStorage)
  localStorage.setItem('access_token', data.access_token);

  return data;
}

// 보호된 API 호출
async function callProtectedAPI(url) {
  const token = localStorage.getItem('access_token');

  const response = await fetch(url, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });

  if (response.status === 401) {
    // 토큰 만료 또는 유효하지 않음 → 로그인 페이지로 리다이렉트
    window.location.href = '/login';
    return;
  }

  return await response.json();
}

// 로그아웃
function logout() {
  localStorage.removeItem('access_token');
  window.location.href = '/login';
}
```

### React 예시

```jsx
import { useState } from 'react';

function LoginPage() {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      });

      if (!response.ok) {
        throw new Error('비밀번호가 올바르지 않습니다');
      }

      const data = await response.json();
      localStorage.setItem('access_token', data.access_token);

      // 로그인 성공 → 메인 페이지로 이동
      window.location.href = '/';
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div>
      <h1>로그인</h1>
      <form onSubmit={handleLogin}>
        <input
          type="password"
          placeholder="비밀번호"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit">로그인</button>
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </form>
    </div>
  );
}
```

---

## 6. 보호된 엔드포인트 만들기

기존 엔드포인트에 인증을 추가하려면:

```python
from fastapi import Depends
import auth

@app.get("/protected-route")
async def protected_route(user: dict = Depends(auth.get_current_user)):
    """
    인증이 필요한 엔드포인트
    """
    return {
        "message": "Authenticated!",
        "user": user
    }
```

### 인증이 필요한 엔드포인트 예시

다음 엔드포인트들을 보호할 수 있습니다:
- `DELETE /request/{request_id}` - 요청 삭제
- `GET /all-requests` - 전체 요청 목록
- `GET /statistics` - 통계 정보
- 기타 관리자 전용 기능

---

## 7. 보안 권장사항

### ✅ 해야 할 것
1. **강력한 비밀번호 사용**: 최소 12자, 대소문자+숫자+특수문자 조합
2. **JWT 시크릿 키 변경**: 기본값 사용 금지, 랜덤 문자열 사용
3. **HTTPS 사용**: 프로덕션 환경에서는 반드시 HTTPS
4. **토큰 만료 시간 설정**: 필요에 따라 조정 (현재 24시간)
5. **.env 파일 보안**: Git에 커밋하지 않기 (.gitignore 확인)

### ❌ 하지 말아야 할 것
1. 비밀번호를 코드에 하드코딩
2. JWT 시크릿 키를 공개 저장소에 업로드
3. HTTP로 비밀번호 전송 (HTTPS 필수)
4. 토큰을 URL 파라미터로 전송
5. 브라우저 콘솔에 토큰 로그 출력

---

## 8. 문제 해결

### 로그인 실패: "ADMIN_PASSWORD_HASH not set"
- `.env` 파일에 `ADMIN_PASSWORD_HASH`가 설정되었는지 확인
- 서버를 재시작했는지 확인

### 로그인 실패: "Incorrect password"
- 입력한 비밀번호가 올바른지 확인
- `generate_password_hash.py`로 새 해시를 생성하여 `.env` 업데이트

### 401 Unauthorized
- 토큰이 만료되었을 수 있음 (24시간 후)
- 토큰이 올바르게 전달되었는지 확인 (`Authorization: Bearer <token>`)
- 다시 로그인하여 새 토큰 발급

---

**설치 완료!** 🎉

이제 공통 비밀번호 기반 로그인 시스템을 사용할 수 있습니다.
