FROM d0ckaaa/openpose:cuda12

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


# Install system dependencies
RUN apt-get update && apt-get install -y \
	python3-pip \
	python3-dev \
	&& rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /opt/openpose-api

# Copy API server code
COPY api_server.py ./
COPY skeleton_interpolate.py ./

# Copy pose_iter_440000.caffemodel to the correct model directory
# (Assumes the file is present in the build context at ./pose_iter_440000.caffemodel)
RUN mkdir -p /opt/openpose/models/pose/coco
COPY pose_iter_440000.caffemodel /opt/openpose/models/pose/coco/pose_iter_440000.caffemodel

# Install Python dependencies
RUN pip3 install --no-cache-dir fastapi uvicorn numpy opencv-python pandas

# Expose port
EXPOSE 19030

# Entrypoint script
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
