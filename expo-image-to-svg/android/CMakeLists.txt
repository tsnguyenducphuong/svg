cmake_minimum_required(VERSION 3.18.1)

project(expo-image-to-svg)

# 1. Set the path to the root of your module
set(PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# 2. Set the C++ standard to 17 (Required for JSI and modern C++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set REACT_NATIVE_DIR — passed in from Gradle, but provide a fallback
# so CMake can also be invoked standalone during IDE inspection.
if(NOT DEFINED REACT_NATIVE_DIR)
    set(REACT_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../node_modules/react-native")
endif()

# Explicitly add React Native's Prefab CMake package path.
# Without this, find_package(ReactAndroid) fails because CMake doesn't
# know where to look — even with prefab=true in Gradle.
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)

file(TO_CMAKE_PATH
    "${REACT_NATIVE_DIR}/ReactAndroid/build/intermediates/cxx"
    RN_CXX_DIR
)

# Add the Prefab package directory to CMAKE_PREFIX_PATH so
# find_package() can resolve ReactAndroid::jsi
list(APPEND CMAKE_PREFIX_PATH
    "${REACT_NATIVE_DIR}/ReactAndroid/build/intermediates/prefab/debug/prefab"
    "${REACT_NATIVE_DIR}/ReactAndroid/build/intermediates/prefab/release/prefab"
)


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
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
    ${PACKAGE_DIR}/cpp
    # Directly include JSI headers as a fallback if Prefab is unavailable
    "${REACT_NATIVE_DIR}/ReactCommon/jsi"
    "${REACT_NATIVE_DIR}/ReactCommon"
    "${REACT_NATIVE_DIR}/ReactAndroid/src/main/jni/react/turbomodule"
)

# 5. Link with the Android log library (essential for debugging LOGD)
find_library(log-lib log)

# 5. Link libraries (Android Log and JS Engine libraries)
# Important: React Native uses Prefab, so we need to find the ReactAndroid package
find_package(ReactAndroid REQUIRED CONFIG)

target_link_libraries(
    expo-image-to-svg
    ${log-lib}
    log
    ReactAndroid::jsi
    android
)

# 6. 16KB Page Alignment for Modern Android
# Ensures compatibility with Android 15+ and specialized devices.
target_link_options(
  expo-image-to-svg
  PRIVATE
  "-Wl,-z,max-page-size=16384"
)
