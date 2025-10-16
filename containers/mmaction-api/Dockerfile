# ------------------------------------------------------------------------------------------------
# Dockerfile: api_src 폴더 구조 유지 및 경로 수정
# ------------------------------------------------------------------------------------------------
FROM dannymaq/mmaction:v6-nano

# 1. API 서버 파일을 /mmaction2 폴더 내부로 복사합니다.
#    => 컨테이너 내부 경로: /mmaction2/api_src/
COPY ./api_src /mmaction2/api_src/ 

# 💥 추가: 체크포인트 파일을 컨테이너 내부의 검색 경로로 복사
COPY ./api_src/stgcn_70p.pth /mmaction2/api_src/stgcn_70p.pth

# 2. 기본 작업 디렉토리를 /mmaction2/api_src로 변경합니다.
#    이제 모든 상대 경로는 /mmaction2/api_src를 기준으로 합니다.
WORKDIR /mmaction2/api_src

# 3. API 실행에 필요한 추가 Python 패키지를 설치합니다.
#    requirements.txt 파일이 이제 /mmaction2/api_src 폴더에 있습니다.
RUN pip install --no-cache-dir -r /mmaction2/api_src/requirements.txt

# 4. 컨테이너가 API 서버를 실행하도록 설정합니다.
EXPOSE 19031

# 5. 컨테이너 시작 시 API 서버 실행
#    WORKDIR 덕분에 api_server.py 파일만 지정하면 됩니다.
CMD ["python", "api_server.py"]