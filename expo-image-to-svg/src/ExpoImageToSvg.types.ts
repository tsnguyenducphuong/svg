import type { StyleProp, ViewStyle } from 'react-native';


export type OnLoadEventPayload = {
  url: string;
};


export type ExpoImageToSvgModuleEvents = {
  onChange: (params: ChangeEventPayload) => void;
};


export type ChangeEventPayload = {
  value: string;
};


export type ExpoImageToSvgViewProps = {
  url: string;
  onLoad: (event: { nativeEvent: OnLoadEventPayload }) => void;
  style?: StyleProp<ViewStyle>;
};


// ═══════════════════════════════════════════════════════════════════════════
//  ExpoImageToSvg.types.ts — TypeScript bindings for VTracerEngine (v3)
//
//  Two public entry-points are supported:
//
//   vectorize()          — original single-pass API (unchanged)
//   vectorizeMultiPass() — Multi-Pass Frequency Separation workflow
//
//  Single-pass JS/C++ option mapping (VectorizeOptions → vtracer::Options):
//
//  JS key                  C++ field                  Type
//  ────────────────────    ─────────────────────────  ──────────────────────
//  buffer                  (raw pixel data)           Uint8Array  (required)
//  width                   (image width)              number      (required)
//  height                  (image height)             number      (required)
//  precision               color_precision            int         default 6
//  gradientStep            gradient_step              float       default 0
//  colorMode               color_mode                 enum        default "color"
//  filterSpeckle           filter_speckle             int         default 4
//  rdpEpsilon              rdp_epsilon                float       default 1.5
//  threshold               corner_threshold*          float       default 60
//  fitTolerance            fit_tolerance              float       default 0.5
//  blurRadius              blur_radius                float       default 1.0
//  pathPrecision           path_precision             int         default 1
//  bilateral_sigma_r       bilateral_sigma_r          float       default 30.0
//  gradient_detect_thresh  gradient_detect_thresh     float       default 22.0
//  segmentThreshold        segment_threshold          float       default 0
//
//  * JS `threshold` is the intuitive "sharpness" angle; the C++ layer
//    stores (180 - threshold) as `corner_threshold` internally.
//
//  Multi-pass JS/C++ option mapping (VectorizeMultiPassOptions → vtracer::MultiPassOptions):
//
//  JS key                  C++ field                  Notes
//  ────────────────────    ─────────────────────────  ──────────────────────
//  pass1                   pass1 (Options)            Blur/base layer tuning
//  pass2                   pass2 (Options)            High-pass texture tuning
//  pass3                   pass3 (Options)            Subject foreground tuning
//  baseDilateRadius        baseDilateRadius           float default 0.75
//  highPassGroupOpacity    highPassGroupOpacity       float default 0.50
//  edgeStrokeWidth         edgeStrokeWidth            float default 0.8
//  edgeMinLuminance        edgeMinLuminance           int   default 80
//  enableSubPassAO         enableSubPassAO            bool  default true
// ═══════════════════════════════════════════════════════════════════════════


// ---------------------------------------------------------------------------
//  Color mode
//  Mirrors: enum class ColorMode { Color, BlackAndWhite }
// ---------------------------------------------------------------------------
export type ColorMode = 'color' | 'blackAndWhite';


// ---------------------------------------------------------------------------
//  Per-pass options — a subset of VectorizeOptions used inside MultiPass.
//  Mirrors: vtracer::Options (all fields optional; C++ defaults apply).
// ---------------------------------------------------------------------------
export interface PassOptions {
  /**
   * Palette size = 2^precision.
   * Pass 1 (base): 4 → 16 colours. Pass 3 (subject): 6 → 64 colours.
   * @default 6
   */
  precision?: number;

  /**
   * Perceptual merge step (CIE-Lab ΔE). 0 = disabled.
   * Maps to C++: `segment_threshold`.
   * Pass 1: use ~2.0 to force large smooth regions.
   * @default 0
   */
  segmentThreshold?: number;

  /**
   * Hard-corner threshold in degrees (same semantics as VectorizeOptions.threshold).
   * Pass 1: ~60° (smooth curves). Pass 2: ~30° (preserve edges).
   * Maps to C++: `corner_threshold` as (180 - threshold).
   * @default 60
   */
  threshold?: number;

  /**
   * Minimum connected-component size in pixels. Larger = more noise rejection.
   * Pass 2: raise to ~12 to suppress high-pass noise.
   * Maps to C++: `filter_speckle`.
   * @default 4
   */
  filterSpeckle?: number;

  /**
   * RDP simplification tolerance in pixels.
   * Pass 1: ~2.5 (very smooth). Pass 2: ~0.8 (preserve fine detail).
   * Maps to C++: `rdp_epsilon`.
   * @default 1.5
   */
  rdpEpsilon?: number;

  /**
   * Maximum Bézier fit residual in pixels.
   * Maps to C++: `fit_tolerance`.
   * @default 0.5
   */
  fitTolerance?: number;

  /**
   * SVG coordinate decimal places [0–6].
   * Maps to C++: `path_precision`.
   * @default 1
   */
  pathPrecision?: number;

  /**
   * Gradient group detection ΔE threshold. 0 = disabled.
   * Maps to C++: `gradient_detect_thresh`.
   * @default 16.0
   */
  gradientDetectThresh?: number;

  /**
   * Bilateral filter range sigma. Higher = more smoothing across edges.
   * Maps to C++: `bilateral_sigma_r`.
   * @default 30.0
   */
  bilateralSigmaR?: number;

  /**
   * Color mode for this pass.
   * Maps to C++: `color_mode`.
   * @default "color"
   */
  colorMode?: ColorMode;
}


// ---------------------------------------------------------------------------
//  VectorizeOptions — single-pass entry point
//  Mirrors: vtracer::Options
// ---------------------------------------------------------------------------
export interface VectorizeOptions {
  // ── Required ─────────────────────────────────────────────────────────────

  /**
   * Packed RGBA pixel buffer (row-major, top-left origin).
   * Length must equal `width * height * 4` bytes.
   */
  buffer: Uint8Array;

  /** Image width in pixels. Must be > 0. */
  width: number;

  /** Image height in pixels. Must be > 0. */
  height: number;

  // ── Colour quantisation ──────────────────────────────────────────────────

  /**
   * Palette size = 2^precision entries. Range [1, 8].
   * Maps to C++: `color_precision`.
   * @default 6  (64 colours)
   */
  precision?: number;

  /**
   * Perceptual merge step in CIE-Lab ΔE units. 0 = disabled.
   * Maps to C++: `gradient_step`.
   * @default 0
   */
  gradientStep?: number;

  /**
   * Color mode. `"blackAndWhite"` thresholds at luminance 128.
   * Maps to C++: `color_mode`.
   * @default "color"
   */
  colorMode?: ColorMode;

  // ── Path tracing ─────────────────────────────────────────────────────────

  /**
   * Speckle filter: discard components smaller than this many pixels.
   * Maps to C++: `filter_speckle`.
   * @default 4
   */
  filterSpeckle?: number;

  /**
   * RDP simplification tolerance in pixels.
   * Maps to C++: `rdp_epsilon`.
   * @default 1.5
   */
  rdpEpsilon?: number;

  // ── Curve fitting ────────────────────────────────────────────────────────

  /**
   * Hard-corner detection threshold in degrees.
   * Vertices with a turning angle sharper than this become C0 corners.
   * Maps to C++: `corner_threshold` as (180 - threshold).
   * @default 60
   */
  threshold?: number;

  /**
   * Maximum Bézier fit residual in pixels.
   * Maps to C++: `fit_tolerance`.
   * @default 0.5
   */
  fitTolerance?: number;

  // ── Pre-processing ───────────────────────────────────────────────────────

  /**
   * Gaussian pre-blur sigma in pixels. 0 = skip.
   * Maps to C++: `blur_radius`.
   * @default 1.0
   */
  blurRadius?: number;

  // ── SVG output ───────────────────────────────────────────────────────────

  /**
   * SVG coordinate decimal places [0, 6].
   * Maps to C++: `path_precision`.
   * @default 1
   */
  pathPrecision?: number;

  /**
   * Bilateral filter range sigma. Higher = more blur across edges.
   * Maps to C++: `bilateral_sigma_r`.
   * @default 30.0
   */
  bilateralSigmaR?: number;

  /**
   * ΔE threshold for grouping adjacent similar colours into a gradient fill.
   * 0 = disabled.
   * Maps to C++: `gradient_detect_thresh`.
   * @default 22.0
   */
  gradientDetectThresh?: number;

  /**
   * CIE-Lab ΔE merge threshold for adjacent palette entries.
   * Drives aggressive colour merging for the painterly base pass.
   * Maps to C++: `segment_threshold`.
   * @default 0  (disabled)
   */
  segmentThreshold?: number;
}


// ---------------------------------------------------------------------------
//  VectorizeMultiPassOptions — multi-pass frequency separation entry point
//  Mirrors: vtracer::MultiPassOptions
//
//  The React Native side is responsible for pre-computing the five image
//  buffers before calling vectorizeMultiPass(). The engine runs four
//  sequential tracing passes and returns a single layered SVG:
//
//    Pass 1  blurBuffer   → <g id="layer-base">      — large dilated fills
//    Pass 3  original + maskBuffer → <g id="layer-subject"> — subject detail
//    Pass 2  highPassBuffer → <g id="layer-texture" opacity="…"> — texture
//    Pass 4  edgeMapBuffer  → <g id="layer-edges">   — structural strokes
// ---------------------------------------------------------------------------
export interface VectorizeMultiPassOptions {
  // ── Required: five pre-processed RGBA buffers ────────────────────────────

  /**
   * Original image RGBA pixels. Width × Height × 4 bytes.
   * Used as the source for the subject/foreground pass (Pass 3).
   */
  originalBuffer: Uint8Array;

  /**
   * Bilateral/Gaussian pre-blurred RGBA pixels.
   * Drives the painterly base layer (Pass 1).
   */
  blurBuffer: Uint8Array;

  /**
   * High-pass detail RGBA pixels (original minus blur).
   * Drives the texture/detail layer (Pass 2).
   */
  highPassBuffer: Uint8Array;

  /**
   * Subject mask RGBA pixels.
   * White pixels (R ≥ 128) = foreground; black = background (alpha → 0).
   * Applied to the original before Pass 3 to prevent ghost borders.
   */
  maskBuffer: Uint8Array;

  /**
   * Edge map RGBA pixels (Canny/Sobel output).
   * Edge strength is read from the R channel (0–255).
   * Drives the structural stroke layer (Pass 4).
   */
  edgeMapBuffer: Uint8Array;

  /** Image width in pixels. Must be > 0. */
  width: number;

  /** Image height in pixels. Must be > 0. */
  height: number;

  // ── Per-pass tuning (all optional — engine defaults are pre-configured) ──

  /**
   * Pass 1 (blur/base) tracing options.
   * Defaults: precision=4, threshold=60°, segmentThreshold=2.0, filterSpeckle=8.
   * Produces large smooth fills — the "painterly" foundation.
   */
  pass1?: PassOptions;

  /**
   * Pass 2 (high-pass/texture) tracing options.
   * Defaults: precision=5, threshold=30°, filterSpeckle=12, rdpEpsilon=0.8.
   * Low corner threshold preserves fine lines and texture edges.
   */
  pass2?: PassOptions;

  /**
   * Pass 3 (foreground/subject) tracing options.
   * Defaults: precision=6, threshold=60°, segmentThreshold=0.5.
   * CIEDE2000 refinement is active; alpha=0 mask pixels are skipped.
   */
  pass3?: PassOptions;

  // ── Composite layer controls ─────────────────────────────────────────────

  /**
   * Extra ENH-4 dilation radius applied specifically to the base (Pass 1)
   * layer in pixels. Expands fills outward to seal sub-pixel pinhole gaps
   * before the detail layers are stacked on top.
   * Maps to C++: `baseDilateRadius`.
   * @default 0.75
   */
  baseDilateRadius?: number;

  /**
   * Group opacity for the texture/high-pass layer (Pass 2).
   * The entire pass is wrapped in `<g opacity="…">` so base colours
   * bleed through at full strength.
   * Maps to C++: `highPassGroupOpacity`.
   * @default 0.50
   */
  highPassGroupOpacity?: number;

  // ── Edge layer (Pass 4) controls ─────────────────────────────────────────

  /**
   * Stroke width of structural edge lines in SVG user units.
   * Maps to C++: `edgeStrokeWidth`.
   * @default 0.8
   */
  edgeStrokeWidth?: number;

  /**
   * Minimum R-channel value (0–255) for an edge pixel to be included.
   * Raise to suppress weaker edges; lower to include faint detail lines.
   * Maps to C++: `edgeMinLuminance`.
   * @default 80
   */
  edgeMinLuminance?: number;

  /**
   * Whether to emit the ambient-occlusion vignette overlay (ENH-10) on
   * each sub-pass.
   * Maps to C++: `enableSubPassAO`.
   * @default true
   */
  enableSubPassAO?: boolean;
}


// ---------------------------------------------------------------------------
//  Default values for the single-pass entry point.
// ---------------------------------------------------------------------------
export const VECTORIZE_DEFAULTS = {
  precision:              6,
  gradientStep:           0,
  colorMode:              'color' as ColorMode,
  filterSpeckle:          4,
  rdpEpsilon:             1.5,
  threshold:              60,
  fitTolerance:           0.5,
  blurRadius:             1.0,
  pathPrecision:          1,
  bilateralSigmaR:        30.0,
  gradientDetectThresh:   22.0,
  segmentThreshold:       0,
} as const satisfies Partial<VectorizeOptions>;


// ---------------------------------------------------------------------------
//  Default values for the multi-pass entry point.
//  These mirror the MultiPassOptions constructor defaults in VTracerEngine.hpp.
// ---------------------------------------------------------------------------
export const MULTI_PASS_DEFAULTS = {
  baseDilateRadius:    0.75,
  highPassGroupOpacity: 0.50,
  edgeStrokeWidth:     0.8,
  edgeMinLuminance:    80,
  enableSubPassAO:     true,

  pass1: {
    precision:          4,
    threshold:          60,
    segmentThreshold:   2.0,
    filterSpeckle:      8,
    rdpEpsilon:         2.5,
    fitTolerance:       1.0,
    gradientDetectThresh: 10.0,
  } satisfies PassOptions,

  pass2: {
    precision:          5,
    threshold:          30,
    filterSpeckle:      12,
    rdpEpsilon:         0.8,
    fitTolerance:       0.5,
    pathPrecision:      1,
  } satisfies PassOptions,

  pass3: {
    precision:          6,
    threshold:          60,
    segmentThreshold:   0.5,
    filterSpeckle:      4,
    rdpEpsilon:         1.0,
    fitTolerance:       0.5,
    gradientDetectThresh: 12.0,
  } satisfies PassOptions,
} as const satisfies Partial<VectorizeMultiPassOptions>;