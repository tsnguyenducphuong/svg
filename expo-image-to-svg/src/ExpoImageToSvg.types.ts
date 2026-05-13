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
//  VTracerEngine.ts  —  TypeScript bindings for the enhanced SVG vectoriser v2
//
//  Maps 1-to-1 with vtracer::Options in VTracerEngine.hpp.
//  All optional fields mirror the C++ defaults so callers only need to
//  provide what they want to override.
//
//  JS key           C++ field             Type
//  ──────────────   ────────────────────  ──────────────────────────────────
//  buffer           (raw pixel data)      Uint8Array  (required)
//  width            (image width)         number      (required)
//  height           (image height)        number      (required)
//  precision        color_precision       int         default 6
//  gradientStep     gradient_step         float       default 0
//  colorMode        color_mode            enum        default "color"
//  filterSpeckle    filter_speckle        int         default 4
//  rdpEpsilon       rdp_epsilon           float       default 1.5
//  threshold        corner_threshold*     float       default 60
//  fitTolerance     fit_tolerance         float       default 0.5
//  blurRadius       blur_radius           float       default 1.0
//  pathPrecision    path_precision        int         default 1
//
//  * JS `threshold` is the intuitive "sharpness" angle; the C++ layer
//    stores (180 - threshold) as `corner_threshold` internally.
// ═══════════════════════════════════════════════════════════════════════════


// ---------------------------------------------------------------------------
//  Color mode
//  Mirrors: enum class ColorMode { Color, BlackAndWhite }
// ---------------------------------------------------------------------------
export type ColorMode = "color" | "blackAndWhite";


// ---------------------------------------------------------------------------
//  Vectorise options
//  Mirrors: vtracer::Options
// ---------------------------------------------------------------------------
export interface VectorizeOptions {
  // ── Required image data ──────────────────────────────────────────────────

  /**
   * Packed RGBA pixel buffer (row-major, top-left origin).
   * Pass the `.buffer` property of a `Uint8Array` to the JSI layer.
   * Length must be exactly `width * height * 4` bytes.
   */
  buffer: Uint8Array;

  /** Image width in pixels. Must be > 0. */
  width: number;

  /** Image height in pixels. Must be > 0. */
  height: number;

  // ── Colour quantisation ──────────────────────────────────────────────────

  /**
   * Colour precision: palette size = 2^precision entries.
   * Maps to C++: `color_precision`.
   * Range [1, 8].
   * @default 6  (64 colours)
   */
  precision?: number;

  /**
   * Perceptual merge step in CIE-Lab ΔE units.
   * Maps to C++: `gradient_step`.
   * Adjacent palette entries closer than this value are merged into a
   * single colour, reducing banding in smooth gradients.
   * Set to 0 to disable merging.
   * Typical range [0, 10].
   * @default 0  (disabled)
   */
  gradientStep?: number;

  /**
   * Color mode. Use `"blackAndWhite"` to threshold the image at
   * luminance 128 before vectorising.
   * Maps to C++: `color_mode`.
   * @default "color"
   */
  colorMode?: ColorMode;

  // ── Path tracing ─────────────────────────────────────────────────────────

  /**
   * Speckle filter: discard connected components smaller than this many
   * pixels. Increase to suppress noise in low-resolution or noisy images.
   * Maps to C++: `filter_speckle`.
   * @default 4
   */
  filterSpeckle?: number;

  /**
   * Ramer-Douglas-Peucker simplification tolerance in pixels.
   * Smaller values preserve more vertices (higher fidelity, larger SVG).
   * Larger values produce simpler, smaller paths.
   * Maps to C++: `rdp_epsilon`.
   * @default 1.5
   */
  rdpEpsilon?: number;

  // ── Curve fitting (E3 — Constrained LS Bézier) ──────────────────────────

  /**
   * Hard-corner detection threshold in degrees.
   * Vertices where the turning angle is sharper than this value are treated
   * as hard corners (C0 continuity); all others are smoothed to G1.
   *
   * Maps to C++: `corner_threshold` as (180 - threshold).
   * The C++ layer stores 120 by default (= 180 - 60), so passing 60 here
   * preserves the engine's default behaviour.
   *
   * Examples:
   *   60  → only turns sharper than 60° become corners  (default, relaxed)
   *   90  → right-angle corners and sharper become hard
   *   30  → almost everything is smoothed
   *
   * @default 60
   */
  threshold?: number;

  /**
   * Maximum allowed residual error for the Constrained Least-Squares
   * Bézier fitting stage, in pixels.
   * When a fitted cubic segment deviates from the original boundary by
   * more than this amount it is recursively subdivided.
   * Smaller = more segments, higher fidelity.
   * Larger = fewer segments, smaller SVG, slightly lower fidelity.
   * Maps to C++: `fit_tolerance`.
   * @default 0.5
   */
  fitTolerance?: number;

  // ── Pre-processing ───────────────────────────────────────────────────────

  /**
   * Gaussian pre-blur sigma in pixels applied before quantisation.
   * Smooths pixel-level noise that would otherwise produce ragged edges.
   * Set to 0 to skip blurring entirely.
   * Capped internally at 3.0.
   * Maps to C++: `blur_radius`.
   * @default 1.0
   */
  blurRadius?: number;

  // ── SVG output ───────────────────────────────────────────────────────────

  /**
   * Number of decimal places for SVG path coordinates [0, 6].
   * Lower values produce smaller files; 0 snaps all coordinates to integers.
   * Maps to C++: `path_precision`.
   * @default 1
   */
  pathPrecision?: number;
}


// ---------------------------------------------------------------------------
//  Default values — exported so callers can reference them explicitly.
//  Every optional field in VectorizeOptions must appear here.
// ---------------------------------------------------------------------------
export const VECTORIZE_DEFAULTS = {
  precision:     6,
  gradientStep:  0,
  colorMode:     "color" as ColorMode,
  filterSpeckle: 4,
  rdpEpsilon:    1.5,
  threshold:     60,
  fitTolerance:  0.5,
  blurRadius:    1.0,
  pathPrecision: 1,
} as const satisfies Partial<VectorizeOptions>;