// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.hpp — production-grade raster→SVG vectoriser
//  14-Enhancement Edition
// ═══════════════════════════════════════════════════════════════════════════
#pragma once
#include <cstdint>
#include <string>

namespace vtracer {

enum class ColorMode { Color, BlackAndWhite };

struct Options {
    // ── Core options ────────────────────────────────────────────────────────
    int       color_precision      = 6;      // 1-8; palette = 2^precision colours
    float     corner_threshold     = 120.f;  // corner angle threshold (degrees)
    int       filter_speckle       = 4;      // min component pixel count
    float     rdp_epsilon          = 1.5f;   // RDP simplification tolerance (px)
    int       path_precision       = 1;      // decimal places in SVG coords
    float     blur_radius          = 1.0f;   // pre-filter spatial sigma
    float     gradient_step        = 0.f;    // Lab gradient-merge step (0 = off)
    float     fit_tolerance        = 0.5f;   // Bézier fit tolerance (px)
    ColorMode color_mode           = ColorMode::Color;

    // ── [E4] Bilateral filter range sigma ────────────────────────────────
    // Controls edge preservation strength. Higher = more blur across edges.
    // Typical range: 15 (strong edge preservation) – 50 (soft).
    float     bilateral_sigma_r    = 30.f;

    // ── [E9] Gradient region detection ───────────────────────────────────
    // ΔE (CIE-Lab) threshold for grouping perceptually similar adjacent
    // colour regions into a single SVG linearGradient fill.
    // Set to 0 to disable gradient detection entirely.
    float     gradient_detect_thresh = 22.f;
};

/// Vectorise a raw RGBA image (width×height pixels, 4 bytes/pixel RGBA).
/// Returns a UTF-8 SVG string. Thread-safe (no mutable globals after init).
std::string vectorize(const uint8_t* pixels, int width, int height,
                      Options options = {});

} // namespace vtracer