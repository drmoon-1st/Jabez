FROM condaforge/miniforge3

# 기본 툴
RUN apt-get update && apt-get install -y git ffmpeg build-essential && rm -rf /var/lib/apt/lists/*

# mamba
RUN conda install -y -n base -c conda-forge mamba && conda clean -a -y

WORKDIR /workspace

# 원본에서 export한 스냅샷 파일 복사
COPY docker/classifier_fusion-explicit.txt /tmp/classifier_fusion-explicit.txt
COPY docker/classifier_fusion-pip.txt      /tmp/classifier_fusion-pip.txt

# conda + pip 1:1 복원
RUN mamba create -y -n classifier_fusion --file /tmp/classifier_fusion-explicit.txt && conda clean -a -y
SHELL ["/bin/bash","-lc"]
RUN conda run -n classifier_fusion pip install --no-cache-dir -r /tmp/classifier_fusion-pip.txt

# 편의: 자동 활성화
RUN echo "conda activate classifier_fusion" >> /root/.bashrc
ENV TZ=Asia/Seoul
