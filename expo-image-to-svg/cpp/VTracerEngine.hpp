// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.hpp
//  C++ port of visioncortex/vtracer  (https://github.com/visioncortex/vtracer)
//
//  Pipeline (mirrors the Rust source):
//    1. Colour quantisation            – color_precision bits per channel
//    2. Gradient merging               – union-find on colours within gradient_step
//    3. BFS connected-component labelling – 4-connectivity, single label map
//    4. Speckle filtering              – discard clusters < filter_speckle pixels
//    5. Moore Neighbourhood tracing    – Jacob's stopping criterion (correct loop)
//    6. Collinear-step collapse        – remove redundant walk pixels
//    7. Angle-based corner detection   – corner_threshold in degrees
//    8. Cubic Bézier spline fitting    – Catmull-Rom → Bézier, tension 0.5
//    9. SVG serialisation              – darkest layer first; path_precision dp
//
//  Requires C++17.  No external dependencies.
//
//  Quick-start:
//      vtracer::Options opt;
//      std::string svg = vtracer::vectorize(rgba_ptr, w, h, opt);
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstdint>
#include <string>

namespace vtracer {

// ── Processing mode ────────────────────────────────────────────────────────
enum class ColorMode {
    Color,           ///< Full-colour output (default).
    BlackAndWhite    ///< Luminance-threshold binarisation at L = 128.
};

// ── Tuning parameters ──────────────────────────────────────────────────────
struct Options {
    // Colour processing
    ColorMode color_mode     = ColorMode::Color;

    /// Significant bits kept per RGB channel [1–8].
    /// 8 = lossless; 6 = 64 steps/channel (VTracer default); 4 = coarse.
    int   color_precision    = 6;

    /// Merge quantised colours whose Euclidean RGB distance ≤ this value.
    /// Bridges staircase artefacts from fixed-step quantisation. 0 = off.
    float gradient_step      = 16.f;

    // Noise filtering
    /// Discard connected components smaller than this pixel count.
    int   filter_speckle     = 4;

    // Path quality
    /// Interior turning angle (degrees) above which a vertex becomes a hard
    /// corner (line-to command).  Higher → fewer corners, rounder output.
    /// Matches VTracer's --corner_threshold flag.  Default: 60.
    float corner_threshold   = 60.f;

    // SVG output
    /// Decimal places for SVG path coordinates [0–6].
    int   path_precision     = 2;
};

// ── Public API ─────────────────────────────────────────────────────────────

/// Convert a row-major RGBA8888 pixel buffer to a self-contained SVG string.
///
/// @param pixels   Pointer to width×height×4 bytes (R, G, B, A order).
///                 Pixels with alpha < 128 are treated as transparent.
/// @param width    Image width in pixels  (must be > 0).
/// @param height   Image height in pixels (must be > 0).
/// @param options  Tuning parameters — all fields have working defaults.
/// @return         Complete SVG document. viewBox = "0 0 width height".
///                 Layers ordered darkest-first; curves use cubic Bézier.
std::string vectorize(const uint8_t* pixels, int width, int height,
                      Options options = Options{});

} // namespace vtracer
