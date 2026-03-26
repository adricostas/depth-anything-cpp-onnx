# Depth-Anything C++ Inference Engine 🚀

This repository features a **Performance-Oriented C++17** implementation for monocular depth estimation, utilizing the **Depth-Anything** model architecture and **ONNX Runtime**.

The project is fully containerized using a **multi-stage Docker** workflow, ensuring a consistent and isolated environment for inference across different platforms (Apple Silicon and x86_64).

---

## 🛠 Engineering Highlights

- **Native C++ Implementation:** Developed to minimize the overhead associated with interpreted languages like Python, focusing on direct memory access and linear execution.
- **ONNX Runtime Integration:** Leverages the ONNX C++ API for efficient model loading and inference, supporting hardware acceleration (CUDA/TensorRT) with automatic CPU fallback.
- **Multi-Stage Docker Architecture:** Uses a specialized Dockerfile to separate the heavy build environment from a lightweight, production-ready runtime image (~500MB).
- **Hardware Agnostic:** Automatically detects host architecture during the Docker build process to pull the appropriate ONNX Runtime binaries.

---

## 📁 Project Structure

```text
.
├── main.cpp            # Core inference engine and tensor logic
├── CMakeLists.txt      # Cross-platform build configuration
├── Dockerfile          # Multi-stage deployment recipe
├── .dockerignore       # Build-context optimization
└── .gitignore          # Repository hygiene
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Docker installed and running.
- A Depth-Anything model in `.onnx` format (e.g., `depth_anything_vitb14.onnx`). Place it inside the project folder.

---

### 2. Build the Image

The following command will compile the C++ source and set up the runtime environment:

```bash
docker build -t depth-anything-app .
```

---

### 3. Run Inference

Use Docker volumes to process local images and save the results back to your machine:

```bash
docker run --rm \
  -v $(pwd):/app/data \
  -v $(pwd)/my_results:/app/outputs \
  depth-anything-app /app/data/YOUR_MODEL.onnx /app/data/YOUR_IMAGE.jpg output_depth.jpg
```

> **Note:** Processed depth maps will be saved in your local `my_results/` directory.


---

## 📌 Summary

Developed as a robust C++ solution for low-latency computer vision tasks.

