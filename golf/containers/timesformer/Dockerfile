FROM condaforge/miniforge3

RUN apt-get update && apt-get install -y git ffmpeg build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN conda install -y -n base -c conda-forge mamba && conda clean -a -y

WORKDIR /workspace

COPY docker/timesformer-explicit.txt /tmp/timesformer-explicit.txt
COPY docker/timesformer-pip.txt      /tmp/timesformer-pip.txt

RUN mamba create -y -n timesformer --file /tmp/timesformer-explicit.txt && conda clean -a -y
SHELL ["/bin/bash","-lc"]
RUN conda run -n timesformer pip install --no-cache-dir -r /tmp/timesformer-pip.txt

RUN echo "conda activate timesformer" >> /root/.bashrc
ENV TZ=Asia/Seoul
