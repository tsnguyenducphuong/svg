// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.hpp  —  public API for the enhanced SVG vectoriser v2
//
//  This header is designed to integrate cleanly into an Expo Native Module
//  (React Native / TypeScript) as a C++ implementation file compiled by
//  the Hermes/JSI or turbo-module build system.
//
//  Compile flags recommended for mobile targets:
//   iOS/Android arm64:  -O2 -ffast-math -march=armv8-a+simd
//   Android x86_64:     -O2 -ffast-math -msse4.2
//   macOS/Simulator:    -O2 -ffast-math -mavx2
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstdint>
#include <string>

namespace vtracer {

// ───────────────────────────────────────────────────────────────────────────
//  Color mode
// ───────────────────────────────────────────────────────────────────────────
enum class ColorMode {
    Color,          // Full RGB (default)
    BlackAndWhite,  // Threshold at luminance 128
};

// ───────────────────────────────────────────────────────────────────────────
//  Vectorisation options
//  All fields have sensible defaults inside vectorize() when set to 0 / -1.
// ───────────────────────────────────────────────────────────────────────────
struct Options {
    // Colour quantisation — number of palette entries = 2^color_precision
    // Range [1,8].  Default 6 (64 colours).
    int color_precision = 6;

    // Perceptual merge step in CIE-Lab ΔE units.
    // Palette entries closer than this are merged into one colour.
    // 0 = disabled.  Typical range [0, 10].
    float gradient_step = 0.f;

    // Corner threshold in degrees.  Turns sharper than (180 - threshold)°
    // become hard corners; all other vertices are smooth.
    // Default 120 (corners sharper than 60° are hard).
    float corner_threshold = 120.f;

    // Minimum connected-component size in pixels.  Smaller blobs are dropped
    // (speckle filter).  Default 4.
    int filter_speckle = 4;

    // Ramer-Douglas-Peucker simplification tolerance in pixels.
    // Smaller = more vertices, higher fidelity.  Default 1.5.
    float rdp_epsilon = 1.5f;

    // SVG coordinate decimal places [0,6].  Default 1.
    int path_precision = 1;

    // Gaussian pre-blur radius (sigma in pixels).
    // 0 = no blur.  Capped at 3.  Default 1.0.
    float blur_radius = 1.0f;

    // Color mode.  Default: full color.
    ColorMode color_mode = ColorMode::Color;

    // [E3] Maximum Bézier fitting residual tolerance (pixels).
    // Segments with error above this are recursively subdivided.
    // Default: 0.5 px.
    float fit_tolerance = 0.5f;
};

// ───────────────────────────────────────────────────────────────────────────
//  vectorize()
//
//  Convert a raster image to an SVG string.
//
//  pixels  — packed RGBA bytes, row-major, top-left origin.
//             Length must be exactly width * height * 4 bytes.
//  width   — image width  in pixels (> 0)
//  height  — image height in pixels (> 0)
//  options — tuning parameters (defaults applied for zero/negative values)
//
//  Returns a complete SVG document as a UTF-8 std::string.
//  Thread-safe: all state is local to the call (global LUTs are read-only
//  after initialisation).
// ───────────────────────────────────────────────────────────────────────────
std::string vectorize(
    const uint8_t* pixels,
    int            width,
    int            height,
    Options        options = {});

} // namespace vtracer