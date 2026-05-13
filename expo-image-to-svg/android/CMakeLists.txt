cmake_minimum_required(VERSION 3.10.2)
project(expo-image-to-svg)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# 1. Define the shared library
# ---------------------------------------------------------------------------
add_library(
    expo-image-to-svg
    SHARED
    src/main/cpp/ExpoImageToSvgModule.cpp
    ../cpp/VTracerEngine.cpp
)

# ---------------------------------------------------------------------------
# 2. Include directories
# ---------------------------------------------------------------------------
target_include_directories(
    expo-image-to-svg
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
)

# ---------------------------------------------------------------------------
# 3. Find ReactAndroid via Prefab (works because prefab=true in build.gradle
#    and react-android is in dependencies — same as the working project)
# ---------------------------------------------------------------------------
find_package(ReactAndroid REQUIRED CONFIG)

# ---------------------------------------------------------------------------
# 4. Link libraries
# ---------------------------------------------------------------------------
target_link_libraries(
    expo-image-to-svg
    log
    android
    ReactAndroid::jsi
)

# ---------------------------------------------------------------------------
# 5. 16KB page alignment (Android 15+)
# ---------------------------------------------------------------------------
target_link_options(
    expo-image-to-svg
    PRIVATE
    "-Wl,-z,max-page-size=16384"
)