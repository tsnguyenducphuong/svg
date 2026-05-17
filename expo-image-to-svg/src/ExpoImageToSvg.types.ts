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
//  ExpoImageToSvg.types.ts — TypeScript bindings for VTracerEngine (v4)
//
//  Two public entry-points are supported:
//
//   vectorize()          — original single-pass API (unchanged)
//   vectorizeMultiPass() — ENH-12 Stochastic Painterly 6-Pass pipeline
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
//  bilateralSigmaR         bilateral_sigma_r          float       default 30.0
//  gradientDetectThresh    gradient_detect_thresh     float       default 22.0
//  segmentThreshold        segment_threshold          float       default 0
//
//  * JS `threshold` is the intuitive "sharpness" angle; the C++ layer
//    stores (180 - threshold) as `corner_threshold` internally.
//
//  Multi-pass JS/C++ option mapping (VectorizeMultiPassOptions → vtracer::MultiPassOptions):
//
//  JS key                    C++ field                    ENH      Default
//  ──────────────────────    ─────────────────────────    ───────  ──────────
//  pass1                     pass1 (Options)              —        see below
//  pass2                     pass2 (Options)              —        see below
//  pass3                     pass3 (Options)              —        see below
//  pass4                     pass4 (Options)              —        reserved
//  pass5                     pass5 (Options)              —        reserved
//  baseDilateRadius          baseDilateRadius             12d      2.0
//  edgeStrokeWidth           edgeStrokeWidth              —        0.5
//  edgeMinLuminance          edgeMinLuminance             —        80
//  highPassGroupOpacity      highPassGroupOpacity         legacy   0.5
//  lcqGridW                  lcqGridW                     12a      16
//  lcqGridH                  lcqGridH                     12a      16
//  lcqColorsPerTile          lcqColorsPerTile             12a      24
//  microDetailDeltaEThresh   microDetailDeltaEThresh      12b      6.0
//  highlightLStar            highlightLStar               12c      85.0
//  shadowLStar               shadowLStar                  12c      28.0
// ═══════════════════════════════════════════════════════════════════════════


// ---------------------------------------------------------------------------
//  Color mode
//  Mirrors: enum class ColorMode { Color, BlackAndWhite }
// ---------------------------------------------------------------------------
export type ColorMode = 'color' | 'blackAndWhite';


// ---------------------------------------------------------------------------
//  Per-pass options — used inside VectorizeMultiPassOptions for pass1–pass5.
//  Mirrors: vtracer::Options (all fields optional; C++ defaults apply).
// ---------------------------------------------------------------------------
export interface PassOptions {
  /**
   * Palette size = 2^precision. Range [1, 8].
   * Pass 1 (base): 3 → 8 colours. Pass 2 (mid-tones): 8 (LCQ handles real
   * quantization). Pass 3 (micro-detail): 6 → 64 colours.
   * Maps to C++: `color_precision`.
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
   * Pass 1: ~60° (smooth curves). Pass 2: 30° (LCQ spec). Pass 3: 15° (micro-detail).
   * Maps to C++: `corner_threshold` as (180 - threshold).
   * @default 60
   */
  threshold?: number;

  /**
   * Minimum connected-component size in pixels. Larger = more noise rejection.
   * Pass 3: set to 1 to keep every grain/vein (ENH-12 spec).
   * Maps to C++: `filter_speckle`.
   * @default 4
   */
  filterSpeckle?: number;

  /**
   * RDP simplification tolerance in pixels.
   * Pass 1: ~2.5 (very smooth). Pass 2: ~0.8. Pass 3: ~0.3 (no smoothing).
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
   * Pass 3: use a low value (e.g. 5.0) to preserve micro-texture sharpness.
   * Maps to C++: `bilateral_sigma_r`.
   * @default 30.0
   */
  bilateralSigmaR?: number;

  /**
   * Gaussian/bilateral spatial blur radius applied before this pass.
   * Pass 3: set to 0 to disable — micro-detail must not be re-blurred.
   * Maps to C++: `blur_radius`.
   * @default 0  (pre-blurred buffers are passed directly by the caller)
   */
  blurRadius?: number;

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
  // ── Required ───────────────────────────────────────────────────────────────

  /**
   * Packed RGBA pixel buffer (row-major, top-left origin).
   * Length must equal `width * height * 4` bytes.
   */
  buffer: Uint8Array;

  /** Image width in pixels. Must be > 0. */
  width: number;

  /** Image height in pixels. Must be > 0. */
  height: number;

  // ── Colour quantisation ────────────────────────────────────────────────────

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

  // ── Path tracing ───────────────────────────────────────────────────────────

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

  // ── Curve fitting ──────────────────────────────────────────────────────────

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

  // ── Pre-processing ─────────────────────────────────────────────────────────

  /**
   * Gaussian pre-blur sigma in pixels. 0 = skip.
   * Maps to C++: `blur_radius`.
   * @default 1.0
   */
  blurRadius?: number;

  // ── SVG output ─────────────────────────────────────────────────────────────

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
//  VectorizeMultiPassOptions — ENH-12 Stochastic Painterly 6-Pass pipeline
//  Mirrors: vtracer::MultiPassOptions
//
//  The React Native side is responsible for pre-computing the five image
//  buffers before calling vectorizeMultiPass(). The engine executes six
//  sequential tracing passes and returns a single layered SVG:
//
//    Pass 1  blurBuffer                → <g id="layer-base">        solid fills
//    Pass 2  originalBuffer + mask     → <g id="layer-midtones">    LCQ fills
//    Pass 3  highPassBuffer (filtered) → <g id="layer-microdetail"> texture
//    Pass 4  highlights (internal)     → <g id="layer-highlights">  screen shimmer
//    Pass 5  shadows   (internal)      → <g id="layer-lowlights">   multiply depth
//    Pass 6  edgeMapBuffer             → <g id="layer-edges">       ink strokes
// ---------------------------------------------------------------------------
export interface VectorizeMultiPassOptions {
  // ── Required: five pre-processed RGBA buffers ──────────────────────────────

  /**
   * Original image RGBA pixels. `width × height × 4` bytes, row-major.
   * Used as the subject source for Pass 2 (masked by maskBuffer).
   * Also used as the reference for the internal highlight/shadow extraction
   * in Passes 4 and 5.
   */
  originalBuffer: Uint8Array;

  /**
   * Gaussian or bilateral pre-blurred RGBA pixels.
   * Drives the painterly base layer (Pass 1).
   * Recommended: Gaussian σ = 2–4 px applied to originalBuffer.
   */
  blurBuffer: Uint8Array;

  /**
   * High-pass residual RGBA pixels: `clamp(original − blur + 128, 0, 255)`
   * per channel. Drives the micro-detail layer (Pass 3) after the engine
   * applies the Adaptive Threshold filter (ENH-12b).
   */
  highPassBuffer: Uint8Array;

  /**
   * Subject segmentation mask RGBA pixels.
   * R channel: 255 = foreground, 0 = background (alpha set to 0).
   * Applied to originalBuffer before Pass 2 to prevent ghost borders.
   */
  maskBuffer: Uint8Array;

  /**
   * Sobel or Canny edge-strength map RGBA pixels.
   * R channel encodes edge magnitude [0–255].
   * Drives the structural stroke layer (Pass 6).
   */
  edgeMapBuffer: Uint8Array;

  /** Image width in pixels. Must be > 0. */
  width: number;

  /** Image height in pixels. Must be > 0. */
  height: number;

  // ── Per-pass tuning (all optional — engine applies optimal defaults) ────────

  /**
   * Pass 1 (Base / blur layer) tracing options.
   * Engine defaults: precision=3, threshold=60°, filterSpeckle=6, rdpEpsilon=2.5.
   * Produces large smooth fills — the solid painterly foundation.
   */
  pass1?: PassOptions;

  /**
   * Pass 2 (Mid-Tones / LCQ layer) tracing options.
   * Engine defaults: precision=8 (LCQ handles real quantization),
   * threshold=30°, filterSpeckle=1, pathPrecision=2, rdpEpsilon=0.8.
   * Local Color Quantization runs 16×16 tiles of KMeans++ in Linear RGB.
   */
  pass2?: PassOptions;

  /**
   * Pass 3 (Micro-Detail / Adaptive Threshold layer) tracing options.
   * Engine defaults: precision=6, threshold=15°, filterSpeckle=1,
   * rdpEpsilon=0.3, blurRadius=0 (no smoothing), bilateralSigmaR=5.
   * Only pixels where ΔE(highPass, pass2Color) ≥ microDetailDeltaEThresh
   * are retained, keeping file size on visible texture.
   */
  pass3?: PassOptions;

  /**
   * Pass 4 (Highlights) tracing options — reserved for future caller override.
   * Currently the engine uses hardcoded optimal values (precision=3, high blur,
   * soft curves, fill-opacity 0.3, mix-blend-mode: screen).
   */
  pass4?: PassOptions;

  /**
   * Pass 5 (Low-Lights / Shadows) tracing options — reserved for future caller
   * override. Currently the engine uses hardcoded optimal values (precision=3,
   * 1.5 px dilation, opacity 0.7, mix-blend-mode: multiply).
   */
  pass5?: PassOptions;

  // ── Composite layer controls ────────────────────────────────────────────────

  /**
   * ENH-12d: Dilation radius for the base layer (Pass 1) in pixels.
   * Expands fills outward to seal sub-pixel pinhole seams before detail
   * layers are composited on top. Detail passes use 0 px so fine lines
   * stay crisp.
   * Maps to C++: `baseDilateRadius`.
   * @default 2.0
   */
  baseDilateRadius?: number;

  /**
   * Stroke width for the structural edge/ink lines emitted by Pass 6.
   * SVG user units; applies to all `<path>` elements in layer-edges.
   * Maps to C++: `edgeStrokeWidth`.
   * @default 0.5
   */
  edgeStrokeWidth?: number;

  /**
   * Minimum R-channel value [0–255] for an edge pixel to be included in
   * Pass 6. Raise to suppress weak edges; lower to capture faint detail lines.
   * Maps to C++: `edgeMinLuminance`.
   * @default 80
   */
  edgeMinLuminance?: number;

  /**
   * Legacy field kept for source compatibility. In the 6-pass pipeline each
   * pass opacity is controlled by the engine's ENH-12 constants. This field
   * is stored but has no effect on Passes 1–6.
   * Maps to C++: `highPassGroupOpacity`.
   * @default 0.5
   */
  highPassGroupOpacity?: number;

  // ── ENH-12a: Local Color Quantization (Pass 2) ───────────────────────────

  /**
   * Number of horizontal tiles in the Local Color Quantization grid.
   * The image width is divided into `lcqGridW` equal columns. For very
   * small images (< 256 px wide) consider reducing to 8.
   * Maps to C++: `lcqGridW`.
   * @default 16
   */
  lcqGridW?: number;

  /**
   * Number of vertical tiles in the Local Color Quantization grid.
   * Maps to C++: `lcqGridH`.
   * @default 16
   */
  lcqGridH?: number;

  /**
   * KMeans++ palette budget per tile. Range [8, 64].
   * Higher values produce richer colour transitions between adjacent tiles
   * at the cost of a larger SVG file. 16–32 is the recommended range for
   * photorealistic output.
   * Maps to C++: `lcqColorsPerTile`.
   * @default 24
   */
  lcqColorsPerTile?: number;

  // ── ENH-12b: Adaptive Threshold (Pass 3) ─────────────────────────────────

  /**
   * Minimum CIEDE2000 colour distance between a high-pass pixel and the
   * underlying Pass-2 fill colour before the micro-detail path is emitted.
   * Lower → richer texture, larger SVG file.
   * Higher → only the most visually distinct details survive.
   * Recommended range: 4–10. Use 4 for maximum flower/foliage detail.
   * Maps to C++: `microDetailDeltaEThresh`.
   * @default 6.0
   */
  microDetailDeltaEThresh?: number;

  // ── ENH-12c: Highlight / Shadow extraction (Passes 4 & 5) ────────────────

  /**
   * CIE L* threshold for highlight extraction (Pass 4).
   * Pixels with L* ≥ this value are isolated and vectorised with
   * `mix-blend-mode: screen` at `fill-opacity: 0.3`, creating the
   * translucent shimmer seen in professional digital art.
   * Range: [70, 95]. Lower = more pixels captured as highlights.
   * Maps to C++: `highlightLStar`.
   * @default 85.0  (~top 10% of luminance)
   */
  highlightLStar?: number;

  /**
   * CIE L* threshold for shadow extraction (Pass 5).
   * Pixels with L* ≤ this value are isolated and vectorised with
   * `mix-blend-mode: multiply` at `opacity: 0.7`, deepening shadows
   * without a Z-buffer.
   * Range: [15, 45]. Higher = more pixels captured as shadows.
   * Maps to C++: `shadowLStar`.
   * @default 28.0  (~darkest 15% of luminance)
   */
  shadowLStar?: number;
}


// ---------------------------------------------------------------------------
//  Default values — single-pass entry point.
// ---------------------------------------------------------------------------
export const VECTORIZE_DEFAULTS = {
  precision:            6,
  gradientStep:         0,
  colorMode:            'color' as ColorMode,
  filterSpeckle:        4,
  rdpEpsilon:           1.5,
  threshold:            60,
  fitTolerance:         0.5,
  blurRadius:           1.0,
  pathPrecision:        1,
  bilateralSigmaR:      30.0,
  gradientDetectThresh: 22.0,
  segmentThreshold:     0,
} as const satisfies Partial<VectorizeOptions>;


// ---------------------------------------------------------------------------
//  Default values — ENH-12 6-pass Stochastic Painterly pipeline.
//  These mirror the MultiPassOptions constructor defaults in VTracerEngine.hpp.
//
//  Layer stack produced:
//    layer-base        Pass 1  solid painterly fills          opacity: 1.0
//    layer-midtones    Pass 2  LCQ rich colour fills          opacity: 0.8
//    layer-microdetail Pass 3  texture / vein detail          opacity: 0.6
//    layer-highlights  Pass 4  screen shimmer                 fill-opacity: 0.3
//    layer-lowlights   Pass 5  multiply shadow depth          opacity: 0.7
//    layer-edges       Pass 6  ink strokes                    mix-blend: multiply
// ---------------------------------------------------------------------------
export const MULTI_PASS_DEFAULTS = {
  // ── Composite controls ──────────────────────────────────────────────────
  baseDilateRadius:       2.0,   // ENH-12d: 2 px seals all base-layer seams
  edgeStrokeWidth:        0.5,   // Pass 6 ink stroke width
  edgeMinLuminance:       80,    // Pass 6 minimum edge R-channel
  highPassGroupOpacity:   0.5,   // legacy compat

  // ── ENH-12a: Local Color Quantization ──────────────────────────────────
  lcqGridW:               24,    // 16×16 tile grid to 24x24 grid
  lcqGridH:               24,
  lcqColorsPerTile:       24,    // 16–32 per tile recommended

  // ── ENH-12b: Adaptive Threshold ────────────────────────────────────────
  microDetailDeltaEThresh: 2.0,  // ΔE gate for micro-detail pass: down from 6.0 to 2.0

  // ── ENH-12c: Highlight / Shadow thresholds ─────────────────────────────
  highlightLStar:         85.0,  // top ~10% brightness → screen blend
  shadowLStar:            28.0,  // darkest ~15% → multiply blend

  // ── Per-pass defaults ───────────────────────────────────────────────────

  // Pass 1: solid painterly undercoat from the pre-blurred image.
  pass1: {
    precision:            3,     // 8 colours — smooth broad fills
    threshold:            60,    // smooth curves
    filterSpeckle:        6,
    rdpEpsilon:           2.5,
    fitTolerance:         1.0,
    gradientDetectThresh: 10.0,
  } satisfies PassOptions,

  // Pass 2: mid-tone subject fills via 16×16 Local Color Quantization.
  // precision=8 is passed so the global zone-palette doesn't under-allocate;
  // the real quantization is done by the LCQ engine per-tile.
  pass2: {
    precision:            8,     // LCQ handles true quantization
    threshold:            30,    // ENH-12 spec: sharp 30° corners
    filterSpeckle:        1,     // min_area = 1 (keep every region)
    pathPrecision:        4,     // high path precision
    rdpEpsilon:           0.25,   //0.8 down to 0.25
    fitTolerance:         0.5,
    gradientDetectThresh: 12.0,
  } satisfies PassOptions,

  // Pass 3: micro-detail from the high-pass residual.
  // blurRadius: 0 — no re-blurring; bilateralSigmaR: 5 — tight range filter.
  pass3: {
    precision:            7,     // 64 colours for rich micro-palette to 128 colors
    threshold:            15,    // very sharp — micro-detail is angular
    filterSpeckle:        1,     // min_area = 1 — keep every grain
    pathPrecision:        1,
    rdpEpsilon:           0.2,   // minimal simplification
    fitTolerance:         0.5,
    blurRadius:           0,     // NO smoothing (ENH-12 spec)
    bilateralSigmaR:      5.0,   // tight range filter
    gradientDetectThresh: 8.0,
  } satisfies PassOptions,

  // pass4 / pass5: engine uses hardcoded optimal values; reserved here for
  // documentation and future caller overrides.
  pass4: {} satisfies PassOptions,
  pass5: {} satisfies PassOptions,

} as const satisfies Partial<VectorizeMultiPassOptions>;