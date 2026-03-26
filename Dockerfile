# --- STAGE 1: Build Stage ---
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake wget libopencv-dev && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime (Auto-detect Architecture)
WORKDIR /opt
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        URL="https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz"; \
    else \
        URL="https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-aarch64-1.17.1.tgz"; \
    fi && \
    wget $URL -O onnxruntime.tgz && \
    tar -zxf onnxruntime.tgz && \
    mv onnxruntime-linux-* /usr/local/onnxruntime && \
    rm onnxruntime.tgz

WORKDIR /app
COPY . .
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# --- STAGE 2: Runtime Stage ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install ALL opencv runtime libraries to avoid missing .so files
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/build/DepthAnythingCpp .

# Copy ONNX libraries and REFRESH the system library cache
COPY --from=builder /usr/local/onnxruntime/lib/libonnxruntime.so* /usr/lib/
RUN ldconfig

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["./DepthAnythingCpp"]