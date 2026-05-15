import { requireNativeModule } from 'expo-modules-core';
import {
  VectorizeOptions,
  VectorizeMultiPassOptions,
  VECTORIZE_DEFAULTS,
  MULTI_PASS_DEFAULTS,
} from './ExpoImageToSvg.types';


// ---------------------------------------------------------------------------
//  Native JSI bindings
//
//  The C++ installJSIBindings() call installs two functions on the JS global:
//    • nativeVectorize        — single-pass pipeline
//    • nativeVectorizeMultiPass — four-pass frequency separation pipeline
//
//  Both are synchronous JSI host functions; no Promises or bridge marshalling.
// ---------------------------------------------------------------------------
declare global {
  var nativeVectorize: (options: VectorizeOptions) => string;
  var nativeVectorizeMultiPass: (options: VectorizeMultiPassOptions) => string;
}


// 1. Load the native module.
//    This triggers OnCreate / installJSIBindings on the native side, which
//    installs both globals before any JS code runs.
requireNativeModule('ExpoImageToSvg');


// ---------------------------------------------------------------------------
//  vectorize()
//
//  Converts a single RGBA pixel buffer into an SVG string synchronously.
//
//  All engine options (gradientStep, rdpEpsilon, blurRadius, colorMode,
//  fitTolerance, pathPrecision, bilateral_sigma_r, gradient_detect_thresh,
//  segmentThreshold) are forwarded to the C++ vtracer::vectorize() layer.
//  Omitted options fall back to VECTORIZE_DEFAULTS.
//
//  Example:
//    const svg = vectorize({
//      buffer:        pixelData,  // Uint8Array of RGBA bytes
//      width:         512,
//      height:        512,
//      precision:     6,          // 64-colour palette
//      threshold:     60,         // corners sharper than 60° are hard
//      filterSpeckle: 4,          // drop blobs < 4 px
//      rdpEpsilon:    1.5,        // RDP tolerance in pixels
//      blurRadius:    1.0,        // Gaussian pre-blur sigma
//      gradientStep:  0,          // Lab merge disabled
//      fitTolerance:  0.5,        // Bézier fit residual cap (px)
//      pathPrecision: 1,          // 1 decimal place in SVG coords
//      colorMode:     'color',    // full RGB
//    });
// ---------------------------------------------------------------------------
export function vectorize(options: VectorizeOptions): string {
  const nativeFn = globalThis.nativeVectorize;
  if (typeof nativeFn !== 'function') {
    throw new Error(
      '[ExpoImageToSvg] Native JSI function "nativeVectorize" not found. ' +
      'Ensure the native module is linked correctly and that ' +
      'installJSIBindings() has been called on the native side.',
    );
  }
  return nativeFn(options);
}


// ---------------------------------------------------------------------------
//  vectorizeMultiPass()
//
//  Multi-Pass Frequency Separation workflow.
//
//  Accepts five pre-processed RGBA buffers produced by the React Native side
//  (e.g. via Skia or a native image-processing library) and runs four
//  sequential tracing passes, returning a single layered SVG string.
//
//  SVG layer stack (bottom → top):
//    <g id="layer-base">                    — Pass 1: large dilated fills
//    <g id="layer-subject">                 — Pass 3: high-fidelity subject
//    <g id="layer-texture" opacity="0.5">   — Pass 2: texture / high-pass
//    <g id="layer-edges">                   — Pass 4: structural strokes
//
//  Example:
//    const svg = vectorizeMultiPass({
//      originalBuffer:  originalRGBA,   // Uint8Array — source image
//      blurBuffer:      blurRGBA,       // Uint8Array — bilateral/gaussian blur
//      highPassBuffer:  highPassRGBA,   // Uint8Array — original minus blur
//      maskBuffer:      maskRGBA,       // Uint8Array — white=fg, black=bg
//      edgeMapBuffer:   edgesRGBA,      // Uint8Array — Canny edge strength in R
//      width:           512,
//      height:          512,
//      // Optional per-pass overrides (engine defaults are well-tuned):
//      baseDilateRadius:    0.75,       // expand base fills to seal gaps
//      highPassGroupOpacity: 0.50,      // let base colours bleed through
//      edgeStrokeWidth:     0.8,
//      edgeMinLuminance:    80,
//      pass1: { precision: 4, threshold: 60, segmentThreshold: 2.0 },
//      pass2: { precision: 5, threshold: 30, filterSpeckle: 12 },
//      pass3: { precision: 6, threshold: 60 },
//    });
// ---------------------------------------------------------------------------
export function vectorizeMultiPass(options: VectorizeMultiPassOptions): string {
  const nativeFn = globalThis.nativeVectorizeMultiPass;
  if (typeof nativeFn !== 'function') {
    throw new Error(
      '[ExpoImageToSvg] Native JSI function "nativeVectorizeMultiPass" not found. ' +
      'Ensure the native module is linked correctly and that ' +
      'installJSIBindings() has been called on the native side.',
    );
  }

  // Validate that all five required buffers are present before crossing
  // the JSI boundary — a missing buffer produces a hard C++ crash.
  const requiredBuffers: Array<keyof VectorizeMultiPassOptions> = [
    'originalBuffer',
    'blurBuffer',
    'highPassBuffer',
    'maskBuffer',
    'edgeMapBuffer',
  ];
  for (const key of requiredBuffers) {
    if (!(options[key] instanceof Uint8Array)) {
      throw new Error(
        `[ExpoImageToSvg] vectorizeMultiPass: "${key}" must be a Uint8Array.`,
      );
    }
  }

  if (!Number.isInteger(options.width) || options.width <= 0) {
    throw new Error('[ExpoImageToSvg] vectorizeMultiPass: "width" must be a positive integer.');
  }
  if (!Number.isInteger(options.height) || options.height <= 0) {
    throw new Error('[ExpoImageToSvg] vectorizeMultiPass: "height" must be a positive integer.');
  }

  const expectedBytes = options.width * options.height * 4;
  for (const key of requiredBuffers) {
    const buf = options[key] as Uint8Array;
    if (buf.byteLength < expectedBytes) {
      throw new Error(
        `[ExpoImageToSvg] vectorizeMultiPass: "${key}" is too small ` +
        `(got ${buf.byteLength}, expected ${expectedBytes} bytes for ` +
        `${options.width}×${options.height} RGBA).`,
      );
    }
  }

  return nativeFn(options);
}


// ---------------------------------------------------------------------------
//  Re-exports
// ---------------------------------------------------------------------------
export { default }                        from './ExpoImageToSvgModule';
export { default as ExpoImageToSvgView }  from './ExpoImageToSvgView';
export * from './ExpoImageToSvg.types';


// Convenience re-exports so callers can inspect defaults without importing
// the types file directly.
export { VECTORIZE_DEFAULTS, MULTI_PASS_DEFAULTS };