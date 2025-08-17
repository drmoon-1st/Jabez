FROM condaforge/miniforge3

RUN apt-get update && apt-get install -y git ffmpeg build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN conda install -y -n base -c conda-forge mamba && conda clean -a -y

WORKDIR /workspace

# 빌드 컨텍스트는 D:\Jabez 이므로, docker/ 경로를 그대로 COPY 가능
COPY docker/mmaction-explicit.txt /tmp/mmaction-explicit.txt
COPY docker/mmaction-pip.txt      /tmp/mmaction-pip.txt

RUN mamba create -y -n mmaction --file /tmp/mmaction-explicit.txt && conda clean -a -y
SHELL ["/bin/bash","-lc"]
RUN conda run -n mmaction pip install --no-cache-dir -r /tmp/mmaction-pip.txt \
	&& conda run -n mmaction pip install -U openmim \
	&& conda run -n mmaction mim install mmcv==2.1.0

RUN echo "conda activate mmaction" >> /root/.bashrc
ENV TZ=Asia/Seoul
