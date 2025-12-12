"""
인증 설정 스크립트

SHA-256 해싱 방식 (간소화)
비밀번호: servername2006**
"""

import hashlib
import secrets

# 비밀번호 설정
plain_password = "servername2006**"

# 클라이언트 측 SHA-256 해싱 (프론트엔드에서 수행될 해시)
client_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()

print("=" * 70)
print("인증 설정")
print("=" * 70)
print()
print("비밀번호:", plain_password)
print("SHA-256 해시 (프론트엔드 전송값):", client_hash)
print()

# JWT 시크릿 키 생성
jwt_secret = secrets.token_urlsafe(32)

# .env 파일 생성 또는 업데이트
env_content = f"""# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Tavily API Key (optional, for web search)
TAVILY_API_KEY=your_tavily_api_key_here

# Admin Password (평문)
# 프론트엔드에서 SHA-256 해시로 전송 → 백엔드에서 평문을 해시하여 비교
ADMIN_PASSWORD={plain_password}

# JWT Secret Key
JWT_SECRET_KEY={jwt_secret}
"""

# .env 파일 작성
try:
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    print("✅ .env 파일이 생성되었습니다!")
except Exception as e:
    print(f"❌ .env 파일 생성 실패: {e}")
    print()
    print("수동으로 .env 파일을 만들고 다음 내용을 추가하세요:")
    print()
    print(env_content)

print()
print("=" * 70)
print("설정 완료!")
print("=" * 70)
print()
print("다음 단계:")
print("  1. .env 파일에서 OPENAI_API_KEY를 설정하세요")
print("  2. 서버를 재시작하세요: python main.py")
print()
print("프론트엔드 구현:")
print("  - JavaScript/TypeScript에서 비밀번호를 SHA-256으로 해시")
print(f"  - 예상 해시값: {client_hash}")
print("  - 이 해시를 서버로 전송")
print("  - 백엔드에서 평문 비밀번호를 해시하여 비교")
print()
print("=" * 70)
