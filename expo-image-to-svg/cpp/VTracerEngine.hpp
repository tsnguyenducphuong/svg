#pragma once
#include <cstdint>
#include <string>

namespace vtracer {

// ───────────────────────────────────────────────────────────────────────────
//  Color mode
// ───────────────────────────────────────────────────────────────────────────
enum class ColorMode {
    Color,         // Full RGB (default)
    BlackAndWhite, // Threshold at luminance 128
};

// ───────────────────────────────────────────────────────────────────────────
//  Vectorisation options
//  Negative values trigger built-in defaults inside vectorize().
//  Zero is a valid input for most fields (e.g. blur_radius=0 = no blur).
// ───────────────────────────────────────────────────────────────────────────
struct Options {
    // Colour quantisation — palette entries = 2^color_precision.
    // Range [1, 8]. Default 6 (64 colours).
    int color_precision = 6;

    // Perceptual merge step in CIE-Lab ΔE units.
    // Entries closer than this are merged. 0 = disabled. Range [0, 10].
    float gradient_step = 0.f;

    // Internal corner threshold in degrees, stored as (180 - JS_threshold).
    // A vertex is a hard corner when its turning angle < corner_threshold.
    // Default 120 = JS threshold of 60° (turns sharper than 60° are hard).
    float corner_threshold = 120.f;

    // Minimum connected-component size in pixels (speckle filter).
    // Components smaller than this are dropped. Default 4.
    int filter_speckle = 4;

    // Ramer-Douglas-Peucker tolerance in pixels.
    // Smaller = more vertices, higher fidelity. Default 1.5.
    float rdp_epsilon = 1.5f;

    // SVG coordinate decimal places [0, 6]. Default 1.
    int path_precision = 1;

    // Gaussian pre-blur sigma in pixels. 0 = no blur. Capped at 3. Default 1.0.
    float blur_radius = 1.0f;

    // Color mode. Default: full colour.
    ColorMode color_mode = ColorMode::Color;

    // [E3] Maximum Bézier fitting residual in pixels.
    // Segments with error above this are recursively subdivided. Default 0.5.
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
//  options — tuning parameters (negative values use built-in defaults)
//
//  Returns a complete SVG document as a UTF-8 std::string.
//  Thread-safe: all state is local to the call (global LUTs are read-only
//  after first initialisation).
// ───────────────────────────────────────────────────────────────────────────
[[nodiscard]] std::string vectorize(
    const uint8_t* pixels,
    int            width,
    int            height,
    Options        options = {});

} // namespace vtracer