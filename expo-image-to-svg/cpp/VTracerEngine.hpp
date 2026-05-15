// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.hpp  —  Public API for the Premium Enhanced SVG Vectoriser
//
//  Supports two entry-points:
//
//   1. vectorize()           — original single-pass API (unchanged)
//   2. vectorizeMultiPass() — NEW Multi-Pass Frequency Separation workflow
//      Accepts pre-computed image layers from React Native side:
//        • original   : RGBA Uint8 buffer (W×H×4)
//        • blurImage  : Gaussian/bilateral blurred RGBA buffer
//        • highPass   : High-pass (detail) RGBA buffer
//        • maskImage  : Subject mask — white = foreground, black = background
//        • edgeMap    : High-quality Canny/Sobel edge map (grayscale in R channel)
//      Returns a layered SVG string with 4 composited groups.
//
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstdint>
#include <string>

namespace vtracer {

// ─────────────────────────────────────────────────────────────────────────────
//  Color mode
// ─────────────────────────────────────────────────────────────────────────────
enum class ColorMode { Color, BlackAndWhite };

// ─────────────────────────────────────────────────────────────────────────────
//  Options — shared across single-pass and per-pass overrides
// ─────────────────────────────────────────────────────────────────────────────
struct Options {
    // Quantisation
    int        color_precision        = 6;    // 2^n palette size
    ColorMode  color_mode             = ColorMode::Color;

    // Tracing
    float      corner_threshold       = 120.f; // degrees; lower = more corners
    int        filter_speckle         = 4;     // min component pixel size
    int        path_precision         = 1;     // SVG decimal places
    float      rdp_epsilon            = 1.5f;  // RDP simplification threshold
    float      fit_tolerance          = 0.5f;  // Bézier fit residual limit

    // Pre-filter (used only in single-pass mode)
    float      blur_radius            = 1.0f;
    float      bilateral_sigma_r      = 30.f;

    // Gradient detection
    float      gradient_detect_thresh = 16.f;

    // Segment threshold for merging nearly-identical palette colours
    float      segment_threshold      = 0.f;  // 0 = disabled
};

// ─────────────────────────────────────────────────────────────────────────────
//  MultiPassOptions — per-pass tuning for the frequency separation workflow
// ─────────────────────────────────────────────────────────────────────────────
struct MultiPassOptions {
    // ── Pass 1: Blur / Base layer ─────────────────────────────────────────
    //   Painterly foundation. Large smooth paths, tiny palette.
    Options pass1;   // caller can override; sensible defaults applied below

    // ── Pass 2: High-Pass / Texture layer ────────────────────────────────
    //   Fine lines, texture, highlights. Low corner threshold.
    Options pass2;

    // ── Pass 3: Foreground / Subject layer ────────────────────────────────
    //   Highest fidelity on the masked subject. Large palette.
    Options pass3;

    // ── Pass 4 is purely SVG composition (no extra tracing options) ───────

    // Dilation radius applied to Pass 1 (Base) paths to eliminate pinholes.
    // Default 0.75 px. Set to 0 to disable.
    float   baseDilateRadius          = 0.75f;

    // Opacity of the High-Pass group in the final composite (0–1).
    float   highPassGroupOpacity      = 0.50f;

    // Edge-map stroke width (Pass 4 structural lines).
    float   edgeStrokeWidth           = 0.8f;

    // Minimum luminance of an edge pixel to be included (0–255).
    int     edgeMinLuminance          = 80;

    // Whether to generate the AO vignette overlay on each sub-pass.
    bool    enableSubPassAO           = true;

    MultiPassOptions() {
        // ── Pass 1 defaults: painterly base ─────────────────────────────
        pass1.color_precision         = 4;    // 2^4 = 16 → 8–12 effective colours
        pass1.corner_threshold        = 60.f; // high → smooth curves
        pass1.segment_threshold       = 2.0f; // aggressive merge = fewer, larger paths
        pass1.filter_speckle          = 8;
        pass1.rdp_epsilon             = 2.5f;
        pass1.fit_tolerance           = 1.0f;
        pass1.gradient_detect_thresh  = 10.f;
        pass1.blur_radius             = 0.f;  // input already blurred
        pass1.bilateral_sigma_r       = 30.f;

        // ── Pass 2 defaults: texture / high-pass ────────────────────────
        pass2.color_precision         = 5;    // 32 colours
        pass2.corner_threshold        = 30.f; // low → preserve edges
        pass2.segment_threshold       = 0.f;
        pass2.filter_speckle          = 12;   // ignore noise
        pass2.rdp_epsilon             = 0.8f;
        pass2.fit_tolerance           = 0.5f;
        pass2.path_precision          = 1;
        pass2.blur_radius             = 0.f;
        pass2.bilateral_sigma_r       = 20.f;

        // ── Pass 3 defaults: foreground subject ─────────────────────────
        pass3.color_precision         = 6;    // 64 colours
        pass3.corner_threshold        = 120.f;
        pass3.segment_threshold       = 0.5f;
        pass3.filter_speckle          = 4;
        pass3.rdp_epsilon             = 1.0f;
        pass3.fit_tolerance           = 0.5f;
        pass3.gradient_detect_thresh  = 12.f;
        pass3.blur_radius             = 0.f;
        pass3.bilateral_sigma_r       = 25.f;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Single-pass entry point (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
std::string vectorize(
    const uint8_t* pixels,   // RGBA, W×H×4
    int width, int height,
    Options options = Options{});

// ─────────────────────────────────────────────────────────────────────────────
//  Multi-Pass Frequency Separation entry point (NEW)
//
//  All five buffers must be RGBA (W×H×4).
//  maskImage :  R≥128 → foreground pixel; else background (alpha=0 sentinel).
//  edgeMap   :  R channel encodes edge strength 0–255.
// ─────────────────────────────────────────────────────────────────────────────
std::string vectorizeMultiPass(
    const uint8_t* originalPixels,   // original RGBA
    const uint8_t* blurPixels,       // bilateral/Gaussian blurred RGBA
    const uint8_t* highPassPixels,   // high-pass detail RGBA
    const uint8_t* maskPixels,       // subject mask (white=fg, black=bg) RGBA
    const uint8_t* edgeMapPixels,    // Canny/Sobel edge map RGBA
    int width, int height,
    MultiPassOptions options = MultiPassOptions{});

} // namespace vtracer