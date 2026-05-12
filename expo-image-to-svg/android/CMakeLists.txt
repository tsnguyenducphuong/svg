cmake_minimum_required(VERSION 3.18.1)

project(expo-image-to-svg)

# 1. Set the path to the root of your module
set(PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# 2. Set the C++ standard to 17 (Required for JSI and modern C++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Add the library
add_library(
    # This name MUST match System.loadLibrary("expo-image-to-svg") in your Kotlin code
    expo-image-to-svg
    SHARED

    # The Android-specific JNI wrapper
    src/main/cpp/ExpoImageToSvgModule.cpp

    # The shared core engine
    ${PACKAGE_DIR}/cpp/VTracerEngine.cpp
)

# 4. Include Directories
target_include_directories(
    expo-image-to-svg
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp
    ${PACKAGE_DIR}/cpp
)

# 5. Link with the Android log library (essential for debugging LOGD)
find_library(log-lib log)

target_link_libraries(
    expo-image-to-svg
    ${log-lib}
)
