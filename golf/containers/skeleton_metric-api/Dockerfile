FROM bocker060/openpose-api:cuda12

# The base OpenPose image may include NVIDIA-specific ENV like
# NVIDIA_REQUIRE_CUDA="cuda>=12.4 brand=tesla" which can force a
# driver/brand check that breaks on some hosts (for example RunPod RTX
# 4000a where brand != tesla). Override these here so the container
# will not be blocked by that label at runtime. These values can be
# tightened or removed if you deploy to environments that require them.
# Emptying NVIDIA_REQUIRE_CUDA disables the brand check from the base image.
ENV NVIDIA_REQUIRE_CUDA=""
ENV NVIDIA_PRODUCT_NAME=""
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Ensure the repository workdir is on PYTHONPATH so local modules (openpose/*)
# can be imported with normal 'import openpose.*' even if another package named
# 'openpose' exists in site-packages. Set to the repo path (does not rely on
# a preexisting PYTHONPATH variable).
ENV PYTHONPATH=/opt/skeleton_metric-api

# Install system dependencies
RUN apt-get update && apt-get install -y \
	python3-pip \
	python3-dev \
	&& rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /opt/skeleton_metric-api

# Copy API server code
COPY api_server.py /opt/skeleton_metric-api/api_server.py
COPY ./openpose /opt/skeleton_metric-api/openpose
COPY controller.py /opt/skeleton_metric-api/controller.py
COPY ./metric_algorithm /opt/skeleton_metric-api/metric_algorithm
COPY .env /opt/skeleton_metric-api/.env
COPY mmaction_client.py /opt/skeleton_metric-api/mmaction_client.py


# Copy pose_iter_440000.caffemodel to the correct model directory
# (Assumes the file is present in the build context at ./pose_iter_440000.caffemodel)
RUN mkdir -p /opt/openpose/models/pose/coco
COPY pose_iter_440000.caffemodel /opt/openpose/models/pose/coco/pose_iter_440000.caffemodel

# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Optional runtime environment variables (can be overridden at docker run)
ENV RECEIVED_PAYLOAD_DIR=/opt/skeleton_metric-api/received_payloads

# Expose port
EXPOSE 19030

# Entrypoint script
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
