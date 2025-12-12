"""
비밀번호 해시 생성 유틸리티

사용법:
    python generate_password_hash.py

입력한 비밀번호의 bcrypt 해시를 생성합니다.
생성된 해시를 .env 파일의 ADMIN_PASSWORD_HASH에 저장하세요.
"""

import bcrypt
import getpass

def generate_password_hash():
    """비밀번호를 입력받아 bcrypt 해시 생성"""
    print("=" * 60)
    print("비밀번호 해시 생성기")
    print("=" * 60)
    print()

    # 비밀번호 입력 (화면에 표시되지 않음)
    password = getpass.getpass("비밀번호를 입력하세요: ")
    confirm = getpass.getpass("비밀번호를 다시 입력하세요: ")

    if password != confirm:
        print("\n❌ 비밀번호가 일치하지 않습니다.")
        return

    if len(password) < 8:
        print("\n❌ 비밀번호는 최소 8자 이상이어야 합니다.")
        return

    # bcrypt 해시 생성
    salt = bcrypt.gensalt(rounds=12)  # rounds=12 (보안 수준)
    password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
    hash_str = password_hash.decode('utf-8')

    print()
    print("=" * 60)
    print("✅ 해시 생성 완료!")
    print("=" * 60)
    print()
    print("다음 내용을 .env 파일에 추가하세요:")
    print()
    print(f"ADMIN_PASSWORD_HASH={hash_str}")
    print()
    print("=" * 60)

if __name__ == "__main__":
    generate_password_hash()
