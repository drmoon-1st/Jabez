# front_realsense

Windows 데스크탑용 RealSense 녹화 + S3 업로드 데모 스캐폴드

요약:
- Intel RealSense로 RGB(depth 포함) 프레임을 녹화
- 녹화 세션을 ZIP으로 패키징
- AWS Cognito Authorization Code + PKCE로 인증(데스크탑 권장)
- 백엔드 `/api/upload`에 `Authorization: Bearer <access_token>`로 요청하여 presigned PUT URL 수신
- presigned URL에 ZIP을 업로드

주의 사항:
- `pyrealsense2`는 바이너리 의존성이 있어 PyInstaller로 빌드 시 추가 설정 필요합니다.
- Cognito에서 데스크탑용 앱으로 PKCE를 허용하고 loopback redirect URI(예: http://127.0.0.1:5000/callback)를 등록해야 합니다.

빠른 시작:
1. `config.template.json`을 복사하여 `config.json`로 저장하고 값 채움.
2. PowerShell로 아래 실행:

```powershell
# 가상환경 생성
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# 테스트: GUI 실행
python ui.py
```

3. exe 빌드(Windows, pyrealsense2 네이티브 파일 확인 필요):

```powershell
.\\build_windows.ps1
```

문제 및 제약:
- pyrealsense2 설치/배포는 환경에 따라 다릅니다. 빌드시 pyrealsense2의 DLL과 의존 파일을 PyInstaller spec에 포함해야 합니다.
