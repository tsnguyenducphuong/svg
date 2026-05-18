// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.hpp  —  public API for the VTracer SVG vectoriser
//
//  Expo / React Native module interface.
//  All types and entry points used by the JS bridge are declared here.
//
//  ENH-12: Stochastic Painterly Rendering — 6-Pass Pipeline
//    vectorizeMultiPass() now accepts per-pass options that drive the full
//    6-pass compositing stack:
//      Pass 1  Base        — blurred image, solid painterly fills
//      Pass 2  Mid-Tones   — 16×16 Local Color Quantization (ENH-12a)
//      Pass 3  Micro-Detail— High-Pass + Adaptive Threshold (ENH-12b/c)
//      Pass 4  Highlights  — screen-blend shimmer (ENH-12c)
//      Pass 5  Low-Lights  — multiply-blend shadows (ENH-12c)
//      Pass 6  Edge/Ink    — Sobel/Canny strokes (multiply)
//
//  ENH-16: Variance-Driven Adaptive Tile Quantization
//    buildLocalColorQuantization() now measures per-tile luminance variance
//    and maps σ² → independent {color_precision, filter_speckle, min_area}
//    for each tile.  Two tunable thresholds (kVarFlat, kVarMid) control the
//    three-tier classification and can be set to 0 to fully disable the
//    enhancement for backward compatibility.
//
// ═══════════════════════════════════════════════════════════════════════════
#pragma once


#include <cstdint>
#include <string>


namespace vtracer {


// ─────────────────────────────────────────────────────────────────────────────
//  Color mode
// ─────────────────────────────────────────────────────────────────────────────
enum class ColorMode : uint8_t {
   Color        = 0,  // full RGB palette
   BlackAndWhite = 1  // 1-bit luminance threshold
};


// ─────────────────────────────────────────────────────────────────────────────
//  Single-pass vectorizer options
//  All floating-point fields default to ≤ 0 → engine applies built-in default.
// ─────────────────────────────────────────────────────────────────────────────
struct Options {
   // Quantization
   ColorMode color_mode          = ColorMode::Color;
   int       color_precision     = 0;  // 0 → default (6 → 64 colours)
                                       // n → 2^n palette entries (1–8)

   // Path shape
   float     corner_threshold    = 0.f; // degrees; 0 → default (120°)
   int       filter_speckle      = 0;  // min component area in px; 0 → default (4)
   int       path_precision      = -1; // SVG decimal places; -1 → default (1)
   float     rdp_epsilon         = 0.f; // RDP simplification; 0 → default (1.5)
   float     fit_tolerance       = 0.f; // Bézier fit error; 0 → default (0.5)

   // Pre-filter
   float     blur_radius         = 0.f; // bilateral spatial sigma; 0 → no filter
   float     bilateral_sigma_r   = 0.f; // bilateral range sigma; 0 → default (30)

   // Gradient detection (ENH-2)
   float     gradient_detect_thresh = 4.f; // Lab distance; 0 → default (16)
};


// ─────────────────────────────────────────────────────────────────────────────
//  Multi-pass vectorizer options  (ENH-11 + ENH-12 + ENH-16)
//
//  The caller pre-computes 5 RGBA pixel buffers at the React Native / Expo
//  Module layer and passes them into vectorizeMultiPass().  The engine then
//  executes all 6 passes without additional pixel processing on the C++ side
//  (other than the highlight/shadow pixel extraction which is internal).
//
//  Recommended per-pass configurations for photorealistic output:
//
//    pass1 (Base):
//      color_precision=3, corner_threshold=60, filter_speckle=6, rdp_epsilon=2.5
//
//    pass2 (Mid-Tones / LCQ):
//      color_precision=8 (LCQ handles real quantization), corner_threshold=30,
//      filter_speckle=1, path_precision=2, rdp_epsilon=0.8
//
//    pass3 (Micro-Detail):
//      color_precision=6, corner_threshold=15, filter_speckle=1,
//      path_precision=1, rdp_epsilon=0.3, blur_radius=0 (no smoothing)
//
//    Passes 4, 5, 6 are fully internally configured; their Options fields
//    in this struct are reserved for future caller overrides but ignored
//    in the current release (the engine uses hardcoded optimal values).
// ─────────────────────────────────────────────────────────────────────────────
struct MultiPassOptions {
   // Per-pass options (pass4/pass5/pass6 reserved for future use)
   Options   pass1;                    // Base layer
   Options   pass2;                    // Mid-Tones (LCQ)
   Options   pass3;                    // Micro-Detail (High-Pass + AThresh)
   Options   pass4;                    // Highlights (internally configured)
   Options   pass5;                    // Low-Lights (internally configured)

   // ── ENH-12d: Variable dilation ─────────────────────────────────────
   // Base layer dilation radius.  Set to 2.0 for the full painterly effect
   // (seals sub-pixel seams); must be ≥ 0.  Detail passes use 0 px dilation.
   float     baseDilateRadius          = 2.0f;

   // ── ENH-12 Local Color Quantization (Pass 2) ───────────────────────
   // Grid dimensions for the 16×16 tile partition.  Change only if your
   // image dimensions are very small (e.g. 128×128 → try 8×8).
   int       lcqGridW                  = 16;
   int       lcqGridH                  = 16;
   // Palette entries per tile.  16–32 gives the best quality/size trade-off.
   int       lcqColorsPerTile          = 24;

   float    stitchThresh               = 3.5f;

   float    seamRepairThresh           = 2.5f;

   // ── ENH-12b Adaptive Threshold (Pass 3) ───────────────────────────
   // Minimum CIEDE2000 distance from underlying Pass-2 colour before a
   // micro-detail pixel is retained.  Lower → more paths (richer texture,
   // larger file).  Recommended range: 4–10.
   float     microDetailDeltaEThresh   = 6.0f;

   // ── ENH-12c Highlight / Shadow thresholds ─────────────────────────
   // Pass 4 extracts pixels with CIE L* ≥ highlightLStar (top ~10%).
   float     highlightLStar            = 85.0f;
   // Pass 5 extracts pixels with CIE L* ≤ shadowLStar (darkest ~15%).
   float     shadowLStar               = 28.0f;

   // ── Pass 6 (Edge/Ink) configuration ───────────────────────────────
   float     edgeStrokeWidth           = 0.5f;
   int       edgeMinLuminance          = 80;  // R channel threshold in edge map

   // ── ENH-16: Variance-Driven Adaptive Tile Quantization ────────────
   // Two luminance-variance thresholds that drive the three-tier tile
   // classification in buildLocalColorQuantization():
   //
   //   σ² < kVarFlat  → flat / sky tile:
   //                     color_precision=3, filter_speckle=32, min_area=200
   //   σ² < kVarMid   → midtone tile:
   //                     color_precision=6, filter_speckle=4,  min_area=8
   //   σ² ≥ kVarMid   → high-detail tile (reflections, edges, petals):
   //                     color_precision=8, filter_speckle=1,  min_area=1
   //
   // Set both to 0.0 to disable ENH-16 entirely (backward-compatible).
   // Recommended defaults: kVarFlat=20.0, kVarMid=150.0
   float     varFlat                   = 20.0f;
   float     varMid                    = 150.0f;

   // ── Legacy field (ENH-11 compat) ──────────────────────────────────
   // highPassGroupOpacity is superseded by the 6-pass opacity constants
   // but kept for source compatibility with existing callers.
   float     highPassGroupOpacity      = 0.6f;
};


// ─────────────────────────────────────────────────────────────────────────────
//  Single-pass entry point (unchanged API)
// ─────────────────────────────────────────────────────────────────────────────
// Accepts a single RGBA pixel buffer (W×H×4 bytes, row-major).
// Returns a self-contained SVG string.
std::string vectorize(
   const uint8_t* pixels,
   int            width,
   int            height,
   Options        options = {});


// ─────────────────────────────────────────────────────────────────────────────
//  6-Pass Stochastic Painterly Rendering entry point  (ENH-12)
// ─────────────────────────────────────────────────────────────────────────────
//
//  All five pixel buffers must be RGBA, W×H×4 bytes, row-major.
//  The caller is responsible for pre-computing:
//
//    originalPixels  — The raw camera/source image (no filtering).
//    blurPixels      — Gaussian/bilateral blur of the original (Pass 1 input).
//                      Recommended: σ = 2–4 px Gaussian.
//    highPassPixels  — High-frequency residual: clamp(Original − Blur + 128).
//                      Each channel: out[c] = clamp(orig[c] − blur[c] + 128, 0, 255).
//                      Pass 3 (Micro-Detail) uses this after Adaptive Threshold.
//    maskPixels      — Subject segmentation mask.  R channel: 255 = foreground,
//                      0 = background.  Used to isolate the subject for Pass 2.
//    edgeMapPixels   — Sobel or Canny edge strength map.  R channel encodes
//                      edge magnitude [0–255].  Pass 6 strokes are drawn here.
//
//  Returns a self-contained SVG string with 6 named <g> layers:
//    layer-base, layer-midtones, layer-microdetail,
//    layer-highlights, layer-lowlights, layer-edges
//
std::string vectorizeMultiPass(
   const uint8_t*    originalPixels,
   const uint8_t*    blurPixels,
   const uint8_t*    highPassPixels,
   const uint8_t*    maskPixels,
   const uint8_t*    edgeMapPixels,
   int               width,
   int               height,
   MultiPassOptions  options = {});


} // namespace vtracer