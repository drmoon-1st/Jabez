# ----------------------------------------
# MMACTION + FastAPI Layer
# ----------------------------------------
FROM mmaction2:orignal_base


# 1. 의존성 파일을 먼저 복사하여 캐시 활용도를 높입니다.
# requirements.txt만 복사합니다. 이 파일이 변경되지 않으면 다음 pip install은 캐시를 사용합니다.
COPY ./api_src/requirements.txt /mmaction2/api_src/requirements.txt

# FastAPI 의존성 설치 (캐시 활용 가능성 높음)
RUN pip install -r /mmaction2/api_src/requirements.txt

# install py-spy for runtime profiling/debugging (이것도 자주 바뀌지 않으니 여기에 유지)
RUN pip install py-spy

# default workdir 유지
WORKDIR /mmaction2

ENV FORCE_CUDA="1"
ENV PORT=19031
ENV PYTHONPATH="/mmaction2:/mmaction2/mmaction2:/mmaction2/api_src:${PYTHONPATH}"

# 3. 나머지 FastAPI 서버 코드 복사
# 이제 이 COPY 명령은 requirements.txt가 변경되지 않은 경우에도 이후의 레이어에만 영향을 미칩니다.
COPY ./api_src /mmaction2/api_src/

# expose an extra test port (if needed)
EXPOSE 19031

WORKDIR /mmaction2/api_src

CMD uvicorn api_src.api_server:app --host 0.0.0.0 --port ${PORT}