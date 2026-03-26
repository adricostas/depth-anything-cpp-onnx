cmake_minimum_required(VERSION 3.10)
project(DepthAnythingCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. FIND OPENCV
find_package(OpenCV REQUIRED)

# 2. DEFINE SEARCH PATHS FOR ONNX RUNTIME
# We define the two possible worlds: Docker (Linux) and Local (Mac Brew)
set(ORT_DOCKER_PATH "/usr/local/onnxruntime")
set(ORT_BREW_PATH "/opt/homebrew/opt/onnxruntime")

# 3. DYNAMIC SEARCH FOR HEADERS
find_path(ORT_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS ${ORT_DOCKER_PATH}/include 
          ${ORT_BREW_PATH}/include
    PATH_SUFFIXES onnxruntime
)

# 4. DYNAMIC SEARCH FOR LIBRARY
# This is the key: it will find .so on Linux and .dylib on Mac
find_library(ORT_LIBRARY 
    NAMES onnxruntime
    PATHS ${ORT_DOCKER_PATH}/lib 
          ${ORT_BREW_PATH}/lib
)

# Safety check: Stop if not found
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found! 
    Checked: ${ORT_DOCKER_PATH} and ${ORT_BREW_PATH}")
endif()

# 5. STATUS MESSAGES
message(STATUS "--------------------------------------------")
message(STATUS "🚀 System: ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})")
message(STATUS "📍 ONNX Include: ${ORT_INCLUDE_DIR}")
message(STATUS "📚 ONNX Library: ${ORT_LIBRARY}")
message(STATUS "--------------------------------------------")

# 6. CREATE EXECUTABLE
add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${OpenCV_INCLUDE_DIRS} 
    ${ORT_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS} 
    ${ORT_LIBRARY}
)

# 7. LINUX-SPECIFIC FIXES (Only applied inside Docker/Linux)
if(UNIX AND NOT APPLE)
    # Link filesystem for older GCC versions
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
    
    # Set RPATH so the executable finds the .so at runtime
    set_target_properties(${PROJECT_NAME} PROPERTIES 
        INSTALL_RPATH "${ORT_DOCKER_PATH}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()