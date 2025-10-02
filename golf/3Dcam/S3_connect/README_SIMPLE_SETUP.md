간단 실행 안내 (Windows)
=========================

목표: 사용자가 복잡한 설치 없이, 가상환경에서 스크립트를 실행해 RealSense 데이터를 S3로 업로드할 수 있도록 최소화된 절차입니다.

사전 준비
- Python 3.8+ 설치
- Intel RealSense 드라이버(타깃 PC에 설치되어 있어야 함)

실행 방법
1. PowerShell 열기
2. 프로젝트 폴더로 이동 (예: d:\Jabez\golf\3Dcam)
3. 스크립트 실행:

   .\run_in_venv.ps1

설명
- 스크립트는 다음을 자동으로 수행합니다:
  - `.venv` 가상환경 생성(존재하지 않으면)
  - `requirements.txt`와 `requests` 설치
  - `realsense_pack_and_upload.py` 실행

사용자 조작
- 프로그램 시작 후 GUI에서 Server base(예: http://<ip>:<port>)와 Object name을 입력하고
  Start Recording → Stop Recording → Package & Upload를 누르면 파일들이 S3로 업로드됩니다.

문제 발생 시 한 줄 점검
- pyrealsense2 ImportError: RealSense 런타임이 설치되어 있는지 확인하세요.
- 403 에러: presigned URL 만료/잘못된 key 확인
