# Dockerfile for mmaction project
FROM condaforge/miniforge3

# 기본 유틸
RUN apt-get update && apt-get install -y git ffmpeg build-essential cmake && rm -rf /var/lib/apt/lists/*

# mamba 최신화
RUN conda install -y -n base -c conda-forge mamba && conda clean -a -y

WORKDIR /workspace
COPY ../../docker/mmaction-explicit.txt /tmp/mmaction-explicit.txt
COPY ../../docker/mmaction-pip.txt      /tmp/mmaction-pip.txt

# conda 스냅샷 1:1 복원 (환경 이름: mmaction)
RUN mamba create -y -n mmaction --file /tmp/mmaction-explicit.txt && conda clean -a -y

# pip 패키지도 1:1 복원
SHELL ["/bin/bash","-lc"]
RUN conda run -n mmaction pip install --no-cache-dir -r /tmp/mmaction-pip.txt

# 편의: 자동 활성화
RUN echo "conda activate mmaction" >> /root/.bashrc
ENV TZ=Asia/Seoul
