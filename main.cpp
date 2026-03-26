#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
namespace fs = std::filesystem;

class DepthEstimator {
public:
    DepthEstimator(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "DepthAnything") 
    {
        // 1. HARDWARE ACCELERATION: Check for available providers
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        bool gpu_found = false;

        // Try to enable CUDA if available in the system/build
        if (std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end()) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0; // Use first GPU
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "🚀 Hardware Acceleration: Using NVIDIA GPU (CUDA)" << std::endl;
            gpu_found = true;
        } 
        
        if (!gpu_found) {
            std::cout << "💻 Hardware Acceleration: GPU not found. Falling back to CPU." << std::endl;
        }

        // 2. SESSION OPTIMIZATION
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 3. MODEL INITIALIZATION
        try {
            session = Ort::Session(env, model_path.c_str(), session_options);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
        }
    }

    cv::Mat predict(const cv::Mat& input_image) {
        // --- STEP 1: PREPROCESSING ---
        // Efficiently convert HWC (OpenCV) to NCHW (ONNX)
        // Resize to 518x518, normalize to [0,1], swap BGR to RGB, subtract ImageNet mean
        cv::Mat blob;
        cv::dnn::blobFromImage(input_image, blob, 1.0/255.0, cv::Size(518, 518), 
                            cv::Scalar(0.485, 0.456, 0.406), true, false);

        // Apply ImageNet standard deviation normalization: (x - mean) / std
        cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

        // --- STEP 2: INFERENCE ---
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, 3, 518, 518};
        
        // Zero-copy: wrap the OpenCV blob data directly into an ONNX tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, reinterpret_cast<float*>(blob.data), 1 * 3 * 518 * 518, 
            input_shape.data(), input_shape.size());

        // Node names verified via Netron for DepthAnything-ONNX
        const char* input_names[] = {"image"};
        const char* output_names[] = {"depth"};

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        // --- STEP 3: POST-PROCESSING ---
        // Create a float matrix from the output tensor
        cv::Mat depth_map(518, 518, CV_32F, output_data);
        
        // Normalize to 0-255 (8-bit) for visualization and saving
        cv::normalize(depth_map, depth_map, 0, 255, cv::NORM_MINMAX, CV_8U);
        
        // Apply a colormap to make the depth map human-readable
        cv::Mat colored_depth;
        cv::applyColorMap(depth_map, colored_depth, cv::COLORMAP_MAGMA);
        
        return colored_depth;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session = Ort::Session(nullptr); // Initialized in constructor
};

int main(int argc, char* argv[]) {
    // Command line argument validation
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <input.jpg> [output.jpg]" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string output_dir = "outputs";
    std::string output_path = output_dir + "/" + ((argc > 3) ? argv[3] : "output_depth.jpg");
   

    try {

        // Check if directory exists, if not, create it
        if (!fs::exists(output_dir)) {
            fs::create_directory(output_dir);
            std::cout << "📁 Created directory: " << output_dir << std::endl;
        }
        // Initialize the inference engine
        DepthEstimator estimator(model_path);

        // Load image from disk
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            std::cerr << "Error: Unable to load image at " << image_path << std::endl;
            return -1;
        }

        std::cout << "Processing: " << image_path << "..." << std::endl;
        
        // Execute inference
        cv::Mat result = estimator.predict(frame);

        // Save result to disk (Headless/Docker friendly)
        if (cv::imwrite(output_path, result)) {
            std::cout << "Done! Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Error: Failed to write output image." << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}