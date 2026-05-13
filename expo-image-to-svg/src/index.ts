import { requireNativeModule } from 'expo-modules-core';
import {
  VectorizeOptions,
  NativeBridgeOptions,
  VECTORIZE_DEFAULTS,
  toNativeBridgeOptions,
} from './ExpoImageToSvg.types';

// ---------------------------------------------------------------------------
//  Native JSI binding
//  The C++ OnCreate/installJSIBindings call installs `nativeVectorize` on
//  the JS global. We declare it here so TypeScript knows the shape.
// ---------------------------------------------------------------------------
declare global {
  var nativeVectorize: (options: NativeBridgeOptions) => string;
}

// 1. Load the native module.
//    This triggers OnCreate / installJSIBindings on the native side, which
//    installs `globalThis.nativeVectorize` before any JS code runs.
requireNativeModule('ExpoImageToSvg');

// ---------------------------------------------------------------------------
//  vectorize()
//
//  Converts a bitmap pixel buffer into an SVG string synchronously via JSI.
//
//  All new engine options (gradientStep, rdpEpsilon, blurRadius, colorMode,
//  fitTolerance, pathPrecision) are forwarded to the C++ layer.
//  Omitted options fall back to VECTORIZE_DEFAULTS, which mirror the C++
//  defaults defined in VTracerEngine.hpp.
//
//  Example:
//    const svg = vectorize({
//      buffer: pixelData,       // Uint8Array of RGBA bytes
//      width:  512,
//      height: 512,
//      precision:    6,         // 64-colour palette
//      threshold:    60,        // corners sharper than 60° are hard
//      filterSpeckle: 4,        // drop blobs < 4 px
//      rdpEpsilon:   1.5,       // RDP tolerance in pixels
//      blurRadius:   1.0,       // Gaussian pre-blur sigma
//      gradientStep: 0,         // Lab merge disabled
//      fitTolerance: 0.5,       // Bézier fit residual cap (px)
//      pathPrecision: 1,        // 1 decimal place in SVG coords
//      colorMode: 'color',      // full RGB
//    });
// ---------------------------------------------------------------------------
export function vectorize(options: VectorizeOptions): string {
  const nativeFn = globalThis.nativeVectorize;
  if (typeof nativeFn !== 'function') {
    throw new Error(
      '[ExpoImageToSvg] Native JSI function not found. ' +
      'Ensure the native module is linked correctly and that ' +
      'installJSIBindings() has been called on the native side.'
    );
  }

  // toNativeBridgeOptions applies all defaults, converts the buffer to a
  // plain ArrayBuffer, and flips `threshold` → `cornerThreshold`.
  return nativeFn(toNativeBridgeOptions(options));
}

// ---------------------------------------------------------------------------
//  Re-exports
// ---------------------------------------------------------------------------
export { default }                        from './ExpoImageToSvgModule';
export { default as ExpoImageToSvgView }  from './ExpoImageToSvgView';
export * from './ExpoImageToSvg.types';

// Convenience re-export so callers can inspect defaults without importing
// the types file directly.
export { VECTORIZE_DEFAULTS };