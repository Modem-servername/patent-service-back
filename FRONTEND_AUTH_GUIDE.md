# 프론트엔드 인증 구현 가이드

SHA-256 해싱 방식 (간소화)

---

## 🔐 보안 방식

### 1단계: 프론트엔드 (SHA-256)
```
사용자 입력: "servername2006**"
         ↓ SHA-256 해싱
해시값: "8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52"
         ↓ 서버로 전송
```

### 2단계: 백엔드 (SHA-256 비교)
```
받은 해시: "8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52"
         ↓
.env 평문 비밀번호를 SHA-256으로 해시
         ↓
두 해시값 비교 → JWT 토큰 발급
```

---

## 💡 JavaScript/TypeScript 구현

### SHA-256 해시 함수
```javascript
/**
 * 비밀번호를 SHA-256으로 해시
 * @param {string} password - 평문 비밀번호
 * @returns {Promise<string>} SHA-256 해시 (소문자 hex)
 */
async function hashPassword(password) {
  const encoder = new TextEncoder();
  const data = encoder.encode(password);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return hashHex;
}
```

### 로그인 함수
```javascript
/**
 * 로그인
 * @param {string} password - 평문 비밀번호
 * @returns {Promise<Object>} 토큰 정보
 */
async function login(password) {
  // 1. 비밀번호를 SHA-256으로 해시
  const passwordHash = await hashPassword(password);

  console.log('SHA-256 해시:', passwordHash);
  // 예상: 8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52

  // 2. 해시를 서버로 전송
  const response = await fetch('https://patent-service-back.vercel.app//login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      password_hash: passwordHash
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Login failed');
  }

  const data = await response.json();

  // 3. 토큰 저장
  localStorage.setItem('access_token', data.access_token);

  return data;
}
```

### 보호된 API 호출
```javascript
/**
 * 인증이 필요한 API 호출
 * @param {string} url - API URL
 * @param {Object} options - fetch 옵션
 * @returns {Promise<Object>} 응답 데이터
 */
async function authenticatedFetch(url, options = {}) {
  const token = localStorage.getItem('access_token');

  if (!token) {
    throw new Error('Not authenticated');
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });

  if (response.status === 401) {
    // 토큰 만료 또는 유효하지 않음
    localStorage.removeItem('access_token');
    window.location.href = '/login';
    throw new Error('Token expired');
  }

  return await response.json();
}
```

---

## ⚛️ React 구현 예시

### LoginPage.jsx
```jsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// SHA-256 해시 함수
async function hashPassword(password) {
  const encoder = new TextEncoder();
  const data = encoder.encode(password);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

function LoginPage() {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // 1. 비밀번호를 SHA-256으로 해시
      const passwordHash = await hashPassword(password);

      // 2. 서버로 전송
      const response = await fetch('https://patent-service-back.vercel.app//login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password_hash: passwordHash })
      });

      if (!response.ok) {
        throw new Error('비밀번호가 올바르지 않습니다');
      }

      const data = await response.json();

      // 3. 토큰 저장
      localStorage.setItem('access_token', data.access_token);

      // 4. 메인 페이지로 이동
      navigate('/');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <h1>로그인</h1>
      <form onSubmit={handleLogin}>
        <input
          type="password"
          placeholder="비밀번호"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={loading}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? '로그인 중...' : '로그인'}
        </button>
        {error && <p className="error">{error}</p>}
      </form>
    </div>
  );
}

export default LoginPage;
```

### useAuth Hook
```jsx
import { useState, useEffect } from 'react';

export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    setIsAuthenticated(!!token);
    setLoading(false);
  }, []);

  const logout = () => {
    localStorage.removeItem('access_token');
    setIsAuthenticated(false);
    window.location.href = '/login';
  };

  return { isAuthenticated, loading, logout };
}
```

### ProtectedRoute 컴포넌트
```jsx
import { Navigate } from 'react-router-dom';
import { useAuth } from './useAuth';

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

export default ProtectedRoute;
```

---

## 🧪 테스트

### 비밀번호 해시 검증
```javascript
// 브라우저 콘솔에서 실행
async function testHash() {
  const password = "servername2006**";
  const hash = await hashPassword(password);
  console.log("비밀번호:", password);
  console.log("SHA-256 해시:", hash);
  console.log("예상 해시:", "8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52");
  console.log("일치:", hash === "8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52");
}

testHash();
```

**예상 결과:**
```
비밀번호: servername2006**
SHA-256 해시: 8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52
예상 해시: 8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52
일치: true
```

---

## 🔒 보안 고려사항

### ✅ 장점
- 평문 비밀번호가 네트워크를 통해 전송되지 않음
- 구현이 간단하고 추가 의존성 없음 (bcrypt 불필요)
- 프론트엔드에서 즉시 해시하여 전송

### ⚠️ 주의사항
1. **HTTPS 필수**: 프로덕션 환경에서는 반드시 HTTPS 사용
2. **해시 재사용 불가**: SHA-256 해시를 다른 곳에 재사용하지 말 것
3. **토큰 보안**: localStorage는 XSS에 취약 → httpOnly 쿠키 고려
4. **Rate Limiting**: 무차별 대입 공격 방지를 위한 속도 제한 필요
5. **.env 보안**: 서버의 .env 파일에 평문 비밀번호가 저장되므로 파일 권한 관리 필수

### 📊 보안 레벨 비교
```
평문 전송 (HTTP)                    ⚠️ 매우 위험
평문 전송 (HTTPS)                   ⚠️ 중간
SHA-256 전송 (HTTP)                 ⚠️ 위험
SHA-256 전송 (HTTPS)                ✅ 양호 ← 현재 방식
```

---

## 🚀 배포 체크리스트

- [ ] .env 파일에 JWT_SECRET_KEY 설정
- [ ] HTTPS 인증서 설정 (Let's Encrypt 등)
- [ ] CORS 정책 확인
- [ ] Rate Limiting 설정
- [ ] 로그 모니터링 설정
- [ ] 토큰 만료 시간 조정 (필요 시)

---

**구현 완료!** 🎉

프론트엔드에서 SHA-256 해시를 사용한 안전한 로그인이 가능합니다.
