FROM condaforge/miniforge3
RUN apt-get update && apt-get install -y git ffmpeg build-essential && rm -rf /var/lib/apt/lists/*
RUN conda install -y -n base -c conda-forge mamba && conda clean -a -y

WORKDIR /workspace
COPY ../../docker/classifier_fusion-explicit.txt /tmp/classifier_fusion-explicit.txt
COPY ../../docker/classifier_fusion-pip.txt      /tmp/classifier_fusion-pip.txt

RUN mamba create -y -n classifier_fusion --file /tmp/classifier_fusion-explicit.txt && conda clean -a -y
SHELL ["/bin/bash","-lc"]
RUN conda run -n classifier_fusion pip install --no-cache-dir -r /tmp/classifier_fusion-pip.txt

RUN echo "conda activate classifier_fusion" >> /root/.bashrc
ENV TZ=Asia/Seoul
