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
   * Range [1, 8].
   * @default 6  (64 colours)
   */
  precision?: number;

  /**
   * Perceptual merge step in CIE-Lab ΔE units.
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
   * @default "color"
   */
  colorMode?: ColorMode;

  // ── Path tracing ─────────────────────────────────────────────────────────

  /**
   * Speckle filter: discard connected components smaller than this many
   * pixels. Increase to suppress noise in low-resolution or noisy images.
   * @default 4
   */
  filterSpeckle?: number;

  /**
   * Ramer-Douglas-Peucker simplification tolerance in pixels.
   * Smaller values preserve more vertices (higher fidelity, larger SVG).
   * Larger values produce simpler, smaller paths.
   * @default 1.5
   */
  rdpEpsilon?: number;

  // ── Curve fitting (E3 — Constrained LS Bézier) ──────────────────────────

  /**
   * Hard-corner detection threshold in degrees.
   * Vertices where the turning angle is sharper than this value are treated
   * as hard corners (C0 continuity); all others are smoothed to G1.
   *
   * Note: the C++ layer stores this as `corner_threshold` which equals
   * `180 - threshold` internally, so the JS-facing value is the intuitive
   * "sharpness" angle rather than the complement.
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
   * @default 0.5
   */
  fitTolerance?: number;

  // ── Pre-processing ───────────────────────────────────────────────────────

  /**
   * Gaussian pre-blur sigma in pixels applied before quantisation.
   * Smooths pixel-level noise that would otherwise produce ragged edges.
   * Set to 0 to skip blurring entirely.
   * Capped internally at 3.0.
   * @default 1.0
   */
  blurRadius?: number;

  // ── SVG output ───────────────────────────────────────────────────────────

  /**
   * Number of decimal places for SVG path coordinates [0, 6].
   * Lower values produce smaller files; 0 snaps all coordinates to integers.
   * @default 1
   */
  pathPrecision?: number;
}

// ---------------------------------------------------------------------------
//  Result
// ---------------------------------------------------------------------------
export interface VectorizeResult {
  /** Complete SVG document as a UTF-8 string. */
  svg: string;
}

// ---------------------------------------------------------------------------
//  Default values — exported so callers can reference them explicitly
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

// ---------------------------------------------------------------------------
//  Option mapping helper
//  Converts the JS-facing VectorizeOptions into the flat object that the
//  JSI / NativeModule bridge expects to receive.
// ---------------------------------------------------------------------------
export interface NativeBridgeOptions {
  buffer:          ArrayBuffer;
  width:           number;
  height:          number;
  colorPrecision:  number;
  gradientStep:    number;
  colorMode:       number; // 0 = Color, 1 = BlackAndWhite
  filterSpeckle:   number;
  rdpEpsilon:      number;
  /** Stored as (180 - threshold) to match vtracer::Options::corner_threshold */
  cornerThreshold: number;
  fitTolerance:    number;
  blurRadius:      number;
  pathPrecision:   number;
}

/**
 * Narrows `ArrayBufferLike` to a plain `ArrayBuffer`.
 *
 * `Uint8Array.buffer` is typed as `ArrayBufferLike` because TypeScript
 * allows typed arrays to be backed by a `SharedArrayBuffer`. The JSI layer
 * only accepts a plain `ArrayBuffer`, so we assert and guard here rather
 * than scattering casts across every call site.
 *
 * A `SharedArrayBuffer` will never arrive from normal image-decoding APIs
 * (expo-image-manipulator, react-native-fs, fetch + arrayBuffer(), etc.),
 * so the runtime throw is a meaningful programming-error signal, not a
 * routine failure path.
 */
function toArrayBuffer(buf: ArrayBufferLike): ArrayBuffer {
  if (buf instanceof ArrayBuffer) return buf;
  throw new TypeError(
    "VTracer: buffer must be backed by a plain ArrayBuffer, not a SharedArrayBuffer. " +
    "Use Uint8Array.slice() to create an unshared copy if needed."
  );
}

/**
 * Map a `VectorizeOptions` object (JS-friendly naming + defaults) to the
 * flat `NativeBridgeOptions` struct that the C++ JSI module reads directly.
 *
 * Usage:
 *   const native = toNativeBridgeOptions(opts);
 *   NativeVTracer.vectorize(native);
 */
export function toNativeBridgeOptions(opts: VectorizeOptions): NativeBridgeOptions {
  const d = VECTORIZE_DEFAULTS;
  return {
    buffer:          toArrayBuffer(opts.buffer.buffer),
    width:           opts.width,
    height:          opts.height,
    colorPrecision:  opts.precision     ?? d.precision,
    gradientStep:    opts.gradientStep  ?? d.gradientStep,
    colorMode:       opts.colorMode === "blackAndWhite" ? 1 : 0,
    filterSpeckle:   opts.filterSpeckle ?? d.filterSpeckle,
    rdpEpsilon:      opts.rdpEpsilon    ?? d.rdpEpsilon,
    // Convert JS "sharpness" angle to the C++ complement (corner_threshold)
    cornerThreshold: 180 - (opts.threshold ?? d.threshold),
    fitTolerance:    opts.fitTolerance  ?? d.fitTolerance,
    blurRadius:      opts.blurRadius    ?? d.blurRadius,
    pathPrecision:   opts.pathPrecision ?? d.pathPrecision,
  };
}